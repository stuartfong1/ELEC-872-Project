"""Inspect the actual main checkpoint without changing the running trainer."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
import torch
from phase1.common import ROOT, read_csv, sha256, write_json


def main():
    torch.set_num_threads(1)
    path = ROOT / 'checkpoints/main_resnet/last.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    initial = torch.load(ROOT / 'weights/torch_cache/hub/checkpoints/resnet50-11ad3fa6.pth',
                         map_location='cpu', weights_only=True)
    progress = json.loads((ROOT / 'reports/main_resnet_progress.json').read_text(encoding='utf-8'))
    rows = read_csv(checkpoint['manifest'])
    train = [r for r in rows if r['face_split'] == 'face_train']
    validation = [r for r in rows if r['face_split'] == 'face_val']
    delta = (checkpoint['model']['conv1.weight'] - initial['conv1.weight']).abs().max().item()
    finite = all(torch.isfinite(v).all().item() for v in checkpoint['model'].values())
    original = read_csv('data/manifests/sources.csv')
    heldout_ids = {r['person_id'] for r in original if r['original_split'] == 'val'}
    exclusions = json.loads((ROOT / 'configs/excluded_sources.json').read_text(encoding='utf-8'))
    checks = dict(
        original_train_only=all(r['original_split'] == 'train' for r in rows),
        heldout_identity_overlap_absent=not (set(checkpoint['identity_map']) & heldout_ids),
        photo_group_overlap_absent=not ({r['photo_group'] for r in train} & {r['photo_group'] for r in validation}),
        reviewed_exclusions_absent=not (set(exclusions) & {r['source_id'] for r in rows}),
        identities_480=len(checkpoint['identity_map']) == 480,
        classifier_480=checkpoint['model']['fc.weight'].shape[0] == 480,
        all_model_tensors_finite=finite, backbone_updated_from_imagenet=delta > 0,
        optimizer_momentum_saved=bool(checkpoint['optimizer']['state']),
        actual_backward_steps=checkpoint['global_step'] > 0,
        manifest_hash_matches=checkpoint['manifest_sha256'] == sha256(checkpoint['manifest']),
        trainer_process_alive='phase1.main_train' in psutil.Process(progress['pid']).cmdline())
    paused_path = ROOT / 'logs/main_resnet_before_resume.json'
    if paused_path.exists():
        before = json.loads(paused_path.read_text(encoding='utf-8'))
        launch = json.loads((ROOT / 'logs/main_resnet_launch.json').read_text(encoding='utf-8-sig'))
        events = [json.loads(line) for line in Path(launch['stdout']).read_text(encoding='utf-8').splitlines() if line]
        restored = next(event for event in events if event.get('stage') == 'initialized')
        restart_checks = dict(
            checkpoint_step_restored=before['global_step'] == restored['global_step'],
            shuffle_offset_restored=before['images_in_epoch'] == restored['images_in_epoch'],
            partial_loss_restored=before['train_loss_so_far'] == restored['train_loss_so_far'],
            optimizer_steps_continued=progress['global_step'] > before['global_step'],
            automatic_idle_sleep_prevented=any(event.get('automatic_idle_sleep_prevented') for event in events))
        write_json('reports/main_resnet_resume_verification.json', dict(
            checks=restart_checks, passed=all(restart_checks.values()),
            paused_global_step=before['global_step'], restored_global_step=restored['global_step'],
            subsequent_global_step=progress['global_step'],
            note='Actual ResNet50 stopped after a batch, saved and resumed without resetting epoch position.'))
        checks.update(restart_checks)
    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
                  passed=all(checks.values()), checkpoint_global_step=checkpoint['global_step'],
                  checkpoint_saved_utc=checkpoint['saved_utc'],
                  checkpoint_bytes=path.stat().st_size, conv1_max_abs_change=delta,
                  train_images=len(train), validation_images=len(validation),
                  training_code_sha256=sha256('phase1/main_train.py'),
                  mid_epoch_resume_test='tests/test_main_training.py: exact weights, SGD momentum, RNG and losses verified',
                  full_face_training_completed=False,
                  note='Startup and actual optimization verified; the 10-epoch schedule is still running.')
    write_json('reports/main_training_verification.json', result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
