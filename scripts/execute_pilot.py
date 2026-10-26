"""Feasible Phase 1 expansion, not a probe/attack/fairness experiment."""
import argparse
import runpy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.execute_verified import stage


def main():
    from argparse import Namespace
    from phase1.common import ROOT
    from phase1.data import prepare
    from phase1.train import train
    from phase1.extract import extract
    from phase1.validate import validate
    import pytest
    p = argparse.ArgumentParser()
    p.add_argument('--identities', type=int, default=20)
    p.add_argument('--per-identity', type=int, default=5)
    a = p.parse_args()
    if not (ROOT / 'data/manifests/pilot_variants.csv').exists():
        stage('prepare-pilot', lambda: prepare('pilot', a.identities, a.per_identity, False), vars(a))
    else:
        from phase1.common import read_csv
        import json
        existing = read_csv('data/manifests/pilot_sources.csv')
        processing = json.loads((ROOT / 'reports/pilot_processing.json').read_text())
        if len({r['person_id'] for r in existing}) != a.identities or processing.get('target_sources', processing['sources']) != a.identities * a.per_identity:
            raise ValueError('Existing reviewed pilot differs from requested target; use a distinct run for a new sample')
    def tests():
        code = pytest.main(['-q', 'tests'])
        if code:
            raise RuntimeError(f'pytest exit {code}')
    stage('tests-final', tests, {'test_directory': 'tests', 'includes_immutable_cache_check': True})
    base = dict(manifest='data/manifests/pilot_sources.csv', run='pilot_resnet', batch_size=2,
                threads=2, device='cpu', lr=.001, initialization='imagenet', epochs=2)
    ck = ROOT / 'checkpoints/pilot_resnet/last.pt'
    stage('train-pilot', lambda: train(Namespace(**base, resume='checkpoints/pilot_resnet/last.pt' if ck.exists() else None)), base)
    stage('verify-backbone-update', lambda: runpy.run_path(str(ROOT / 'scripts/verify_backbone_update.py'), run_name='__main__'), {})
    for model in ['farl', 'lvface', 'resnet50']:
        params = dict(model=model, run='pilot', backend='torch', batch_size=8, threads=4,
                      checkpoint='checkpoints/pilot_resnet/last.pt')
        stage('extract-pilot-' + model, lambda p=params: extract(Namespace(**p)), params)
    stage('validate-pilot', lambda: validate('pilot'), {'run': 'pilot'})


if __name__ == '__main__':
    main()
