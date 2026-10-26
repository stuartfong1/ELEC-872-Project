"""Prepare, train and extract a separate CUDA pilot run."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.execute_verified import stage


def main():
    from phase1.common import ROOT
    from phase1.data import prepare
    from phase1.train import train, parser as training_parser, require_cuda
    from phase1.extract import extract
    from phase1.validate import validate
    p = argparse.ArgumentParser()
    p.add_argument('--identities', type=int, default=20)
    p.add_argument('--per-identity', type=int, default=5)
    p.add_argument('--run', default='pilot_adam')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    a = p.parse_args()
    require_cuda(a.gpu)
    if not (ROOT / f'data/manifests/{a.run}_variants.csv').exists():
        stage('prepare-' + a.run, lambda: prepare(a.run, a.identities, a.per_identity, False), vars(a))
    else:
        from phase1.common import read_csv
        import json
        existing = read_csv(f'data/manifests/{a.run}_sources.csv')
        processing = json.loads((ROOT / f'reports/{a.run}_processing.json').read_text())
        if len({r['person_id'] for r in existing}) != a.identities or processing.get('target_sources', processing['sources']) != a.identities * a.per_identity:
            raise ValueError('Existing reviewed pilot differs; use a distinct run for a new sample')
    training_run = a.run + '_resnet'
    checkpoint = f'checkpoints/{training_run}/last.pt'
    args = training_parser().parse_args(['--manifest', f'data/manifests/{a.run}_sources.csv',
        '--run', training_run, '--epochs', '2', '--gpu', str(a.gpu), '--optimizer', a.optimizer])
    args.resume = checkpoint if (ROOT / checkpoint).exists() else None
    stage('train-' + a.run, lambda: train(args), vars(a))
    for model in ['farl', 'lvface', 'resnet50']:
        params = argparse.Namespace(model=model, run=a.run, backend='torch', batch_size=8,
                                    threads=4, checkpoint=checkpoint, gpu=a.gpu)
        stage('extract-' + a.run + '-' + model, lambda args=params: extract(args), vars(params))
    stage('validate-' + a.run, lambda: validate(a.run), {'run': a.run})


if __name__ == '__main__':
    main()
