"""Memory-conscious sequential smoke execution, with per-step durable evidence."""
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import runpy
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase1.common import ROOT


def stage(name, function, params):
    (ROOT / 'logs').mkdir(exist_ok=True)
    print(f'START {name}', flush=True)
    start = time.perf_counter()
    record = dict(name=name, utc_started=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], parameters=params,
                  log=f'logs/{name}.log')
    try:
        with (ROOT / record['log']).open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                function()
        record.update(returncode=0, outcome='passed')
    except BaseException as e:
        import traceback
        with (ROOT / record['log']).open('a', encoding='utf-8') as log:
            traceback.print_exc(file=log)
        record.update(returncode=1, outcome='failed', error=str(e))
        raise
    finally:
        record['seconds'] = time.perf_counter() - start
        with (ROOT / 'reports/command_runs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        gc.collect()


def main():
    from phase1.train import train, parser as training_parser, require_cuda
    from phase1.extract import extract
    from phase1.validate import validate
    from phase1.data import prepare
    from scripts.verify_resume import verify
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='smoke_adam')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    a = p.parse_args()
    require_cuda(a.gpu)
    run, training_run = a.run, a.run + '_resnet'
    manifest = f'data/manifests/{run}_sources.csv'
    if not (ROOT / f'data/manifests/{run}_variants.csv').exists():
        stage('prepare-' + run, lambda: prepare(run, 5, 5, True), {'sources': 25, 'conditions': 16})
    def training_args(name, epochs):
        args = training_parser().parse_args(['--manifest', manifest, '--run', name,
            '--epochs', str(epochs), '--gpu', str(a.gpu), '--optimizer', a.optimizer])
        path = f'checkpoints/{name}/last.pt'
        args.resume = path if (ROOT / path).exists() else None
        return args
    if not (ROOT / f'checkpoints/{training_run}/last.pt').exists():
        stage('train-' + run, lambda: train(training_args(training_run, 1)), vars(a))
    stage('resume-' + run, lambda: train(training_args(training_run, 2)), vars(a))
    continuous = training_run + '_continuous'
    stage('continuous-' + run, lambda: train(training_args(continuous, 2)), vars(a))
    checkpoint = f'checkpoints/{training_run}/last.pt'
    stage('verify-resume-' + run, lambda: verify(checkpoint,
        f'checkpoints/{continuous}/last.pt', f'reports/{run}_resume_verification.json'), vars(a))
    for model in ['farl', 'lvface', 'resnet50']:
        params = argparse.Namespace(model=model, run=run, backend='torch', batch_size=8,
                                    threads=4, checkpoint=checkpoint, gpu=a.gpu)
        stage('extract-' + run + '-' + model, lambda args=params: extract(args), vars(params))
    stage('validate-' + run, lambda: validate(run), {'run': run})


if __name__ == '__main__':
    main()
