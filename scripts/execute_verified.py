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
    import torch
    torch.set_num_threads(2)
    from phase1.train import train
    from phase1.extract import extract
    from phase1.validate import validate
    from phase1.data import prepare
    import pytest
    from argparse import Namespace
    # Preparation is already reusable; avoid rewriting immutable extraction
    # inputs if they exist. Fresh execution can generate them here.
    if not (ROOT / 'data/manifests/smoke_variants.csv').exists():
        stage('prepare-smoke', lambda: prepare('smoke', 5, 5, True), {'sources': 25, 'conditions': 16})
    def tests():
        code = pytest.main(['-q', 'tests'])
        if code:
            raise RuntimeError(f'pytest exit {code}')
    stage('tests', tests, {'test_directory': 'tests'})
    base = dict(manifest='data/manifests/smoke_sources.csv', batch_size=2, threads=2,
                device='cpu', lr=.001, initialization='imagenet')
    ck = ROOT / 'checkpoints/smoke_resnet/last.pt'
    if not ck.exists():
        stage('train-smoke', lambda: train(Namespace(**base, run='smoke_resnet', epochs=1, resume=None)), base)
    stage('resume-smoke', lambda: train(Namespace(**base, run='smoke_resnet', epochs=2,
          resume='checkpoints/smoke_resnet/last.pt')), {**base, 'epochs': 2, 'resume': True})
    stage('train-continuous', lambda: train(Namespace(**base, run='smoke_continuous', epochs=2, resume=None)),
          {**base, 'epochs': 2})
    stage('verify-resume', lambda: runpy.run_path(str(ROOT / 'scripts/verify_resume.py'), run_name='__main__'), {})
    for model in ['farl', 'lvface', 'resnet50']:
        params = dict(model=model, run='smoke', backend='torch', batch_size=8, threads=4,
                      checkpoint='checkpoints/smoke_resnet/last.pt')
        stage('extract-smoke-' + model, lambda p=params: extract(Namespace(**p)), params)
    stage('validate-smoke', lambda: validate('smoke'), {'run': 'smoke'})
    stage('verify-models', lambda: runpy.run_path(str(ROOT / 'scripts/verify_models.py'), run_name='__main__'), {})
    stage('environment', lambda: runpy.run_path(str(ROOT / 'scripts/environment.py'), run_name='__main__'), {})


if __name__ == '__main__':
    main()
