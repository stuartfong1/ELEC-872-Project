"""Inspect a saved CUDA training run without launching training or changing weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
import torch
from phase1.common import ROOT, read_csv, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='main_resnet_adam')
    args = p.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        p.error('Run must be a single directory name')
    path = ROOT / 'checkpoints' / args.run / 'last.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    rows = read_csv(checkpoint['manifest'])
    train = [r for r in rows if r['face_split'] == 'face_train']
    val = [r for r in rows if r['face_split'] == 'face_val']
    optimizer = checkpoint.get('optimizer_name', checkpoint['config'].get('optimizer', 'sgd'))
    required = {'step', 'exp_avg', 'exp_avg_sq'} if optimizer == 'adam' else {'momentum_buffer'}
    states = checkpoint['optimizer']['state'].values()
    progress_path = ROOT / f'reports/{args.run}_progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8-sig')) if progress_path.exists() else {}
    alive = False
    try:
        alive = 'phase1.train' in psutil.Process(progress['pid']).cmdline()
    except (psutil.Error, KeyError):
        pass
    checks = dict(
        original_train_only=all(r['original_split'] == 'train' for r in rows),
        photo_group_overlap_absent=not ({r['photo_group'] for r in train} & {r['photo_group'] for r in val}),
        classifier_matches_mapping=checkpoint['model']['fc.weight'].shape[0] == len(checkpoint['identity_map']),
        all_model_tensors_finite=all(torch.isfinite(v).all().item() for v in checkpoint['model'].values()),
        optimizer_state_saved=bool(checkpoint['optimizer']['state']) and all(required <= set(s) for s in states),
        manifest_hash_matches=checkpoint['manifest_sha256'] == sha256(checkpoint['manifest']),
        actual_backward_steps=checkpoint.get('global_step', 0) > 0)
    result = dict(checks=checks, passed=all(checks.values()), run=args.run, optimizer=optimizer,
        saved_device=checkpoint.get('device', 'legacy'), cuda_rng_saved=checkpoint.get('cuda_rng_state') is not None,
        trainer_process_alive=alive, checkpoint_global_step=checkpoint.get('global_step'),
        checkpoint_sha256=sha256(path), train_images=len(train), validation_images=len(val),
        training_code_sha256=sha256('phase1/train.py'),
        note='Inspection of saved state only; see hardware-gated tests for CUDA execution verification.')
    write_json(f'reports/{args.run}_verification.json', result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
