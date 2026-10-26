"""Full ResNet50 face training with atomic, mid-epoch CPU checkpoints."""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from .common import ROOT, read_csv, write_csv, write_json, sha256
from .models import new_resnet
from .train import FaceDataset


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_manifest():
    """Keep original train only, preserving its within-person photo-group split."""
    path = ROOT / 'data/manifests/main_sources.csv'
    if path.exists():
        return
    exclusions_path = ROOT / 'configs/excluded_sources.json'
    exclusions = json.loads(exclusions_path.read_text(encoding='utf-8'))
    sources = read_csv('data/manifests/sources.csv')
    eligible = [r for r in sources if r['original_split'] == 'train']
    rows = [r for r in eligible if r['source_id'] not in exclusions and r['decode_status'] != 'failed']
    retained = {r['source_id'] for r in rows}
    write_csv(path, rows)
    write_json('reports/main_manifest.json', dict(
        created_utc=now(), original_manifest_sha256=sha256('data/manifests/sources.csv'),
        manifest_sha256=sha256(path), identities=len({r['person_id'] for r in rows}),
        train_images=sum(r['face_split'] == 'face_train' for r in rows),
        validation_images=sum(r['face_split'] == 'face_val' for r in rows),
        original_val_images=0,
        excluded_sources=[dict(source_id=r['source_id'], reason=exclusions.get(r['source_id'], 'decode failed'))
                          for r in eligible if r['source_id'] not in retained],
        full_decode_audit_completed=False,
        input_policy='Original crop; EXIF transpose; ImageNet resize 232 / center crop 224. No augmentation.'))


class MainFaceDataset(FaceDataset):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as error:
            row = self.rows[index]
            raise RuntimeError(f"Source decode/transform failed: {row['source_id']} {row['archive_member']}") from error


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(872)
    random.seed(872)
    np.random.seed(872)
    if args.device != 'cpu':
        raise ValueError('This measured local main-training configuration requires CPU')
    output = ROOT / 'checkpoints' / args.run
    output.mkdir(parents=True, exist_ok=True)
    progress_path = ROOT / 'reports' / f'{args.run}_progress.json'
    report_path = ROOT / 'reports' / f'{args.run}_training.json'
    stop_path = output / 'STOP'
    ck_path = output / 'last.pt'
    if ck_path.exists() and not args.resume:
        raise ValueError('Existing checkpoint: use --resume rather than overwriting it')
    rows = read_csv(args.manifest)
    if any(r['original_split'] != 'train' for r in rows):
        raise ValueError('Main training manifest must contain only original train identities')
    train_rows = [r for r in rows if r['face_split'] == 'face_train']
    val_rows = [r for r in rows if r['face_split'] == 'face_val']
    mapping = {p: i for i, p in enumerate(sorted({r['person_id'] for r in train_rows}))}
    if not train_rows or not val_rows or not {r['person_id'] for r in val_rows} <= set(mapping):
        raise ValueError('Need nonempty within-identity train / validation splits')
    if {r['photo_group'] for r in train_rows} & {r['photo_group'] for r in val_rows}:
        raise ValueError('Photo-group overlap between train and validation')
    manifest_sha = sha256(args.manifest)
    epoch, offset, global_step, history = 0, 0, 0, []
    stats = dict(loss_sum=0., correct=0, train_seconds=0.)
    ck = None
    if args.resume:
        ck = torch.load(ROOT / args.resume, map_location='cpu', weights_only=True)
        expected = dict(identity_map=mapping, manifest_sha256=manifest_sha,
                        batch_size=args.batch_size, initialization=args.initialization,
                        sampler='explicit_randperm_872_plus_epoch_v1')
        if any(ck.get(k) != v for k, v in expected.items()):
            raise ValueError('Checkpoint data, mapping, initialization or sampler differs')
        if ck['config']['lr'] != args.lr or ck['config']['threads'] != args.threads:
            raise ValueError('Keep learning rate and thread count unchanged on resume')
        epoch, offset, global_step = ck['epoch'], ck['offset'], ck['global_step']
        history, stats = ck['history'], ck['partial_stats']
        if epoch >= args.epochs:
            raise ValueError(f'Checkpoint already completed {epoch} epochs; requested limit is '
                             f'{args.epochs}. No training started and checkpoint preserved. '
                             f'Additional images in the next epoch: {offset}.')
    model = new_resnet(len(mapping), 'random' if ck else args.initialization)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=1e-4)
    if ck:
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['torch_rng_state'])
    del ck
    write_json(output / 'identity_map.json', mapping)
    train_data, val_data = MainFaceDataset(train_rows, mapping), MainFaceDataset(val_rows, mapping)
    # A private loader RNG avoids changing model RNG when an interrupted iterator is recreated.
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0,
                            generator=torch.Generator().manual_seed(872))
    started = time.perf_counter()
    last_saved_step = global_step if args.resume else None
    peak_rss = 0

    def progress(status, **extra):
        elapsed = time.perf_counter() - started
        value = dict(status=status, updated_utc=now(), pid=os.getpid(),
                     epoch_index=epoch, target_epochs=args.epochs,
                     epoch_display=min(epoch + 1, args.epochs),
                     images_in_epoch=offset, train_images_per_epoch=len(train_rows),
                     validation_images=len(val_rows), identities=len(mapping), global_step=global_step,
                     epoch_fraction=offset / len(train_rows),
                     train_loss_so_far=stats['loss_sum'] / offset if offset else None,
                     train_accuracy_so_far=stats['correct'] / offset if offset else None,
                     train_images_per_second=offset / stats['train_seconds'] if stats['train_seconds'] else None,
                     invocation_seconds=elapsed, checkpoint=str(ck_path),
                     checkpoint_global_step=last_saved_step,
                     full_face_training_completed=status == 'completed', **extra)
        write_json(progress_path, value)
        print(json.dumps(value), flush=True)

    def save():
        nonlocal last_saved_step
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, offset=offset, global_step=global_step, partial_stats=stats.copy(),
            history=history, identity_map=mapping, manifest=args.manifest,
            manifest_sha256=manifest_sha, batch_size=args.batch_size,
            initialization=args.initialization, sampler='explicit_randperm_872_plus_epoch_v1',
            torch_rng_state=torch.get_rng_state(), config=vars(args), saved_utc=now(),
            training_status='full_face_training_completed' if epoch >= args.epochs else 'full_face_training_in_progress')
        temporary = output / 'last.pt.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(ck_path)
        last_saved_step = global_step

    progress('running', stage='initialized', all_backbone_parameters_trainable=all(p.requires_grad for p in model.parameters()))
    try:
        while epoch < args.epochs:
            if offset == 0:
                torch.manual_seed(872 + epoch)
            order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(872 + epoch)).tolist()
            loader = DataLoader(train_data, batch_size=args.batch_size, sampler=order[offset:],
                                num_workers=0, generator=torch.Generator().manual_seed(1872 + epoch))
            model.train()
            batch_started = time.perf_counter()
            for images, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                if len(images) == 1:
                    for layer in model.modules():
                        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                            layer.eval()
                scores = model(images)
                loss = torch.nn.functional.cross_entropy(scores, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                global_step += 1
                offset += len(labels)
                stats['loss_sum'] += loss.item() * len(labels)
                stats['correct'] += (scores.argmax(1) == labels).sum().item()
                stats['train_seconds'] += time.perf_counter() - batch_started
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                stopping = stop_path.exists() or (args.max_steps and global_step >= args.max_steps)
                if global_step == 10 or global_step % args.checkpoint_every == 0 or stopping:
                    save()
                if global_step % args.progress_every == 0 or stopping:
                    progress('paused' if stopping else 'running', stage='train', last_batch_loss=loss.item(),
                             peak_process_rss_bytes=peak_rss)
                if stopping:
                    return
                batch_started = time.perf_counter()
            # Save all training updates before the potentially long validation pass.
            save()
            progress('running', stage='validation', validation_processed=0)
            model.eval()
            val_loss, val_correct, val_count = 0., 0, 0
            with torch.inference_mode():
                for batch_index, (images, labels) in enumerate(val_loader):
                    scores = model(images)
                    batch_loss = torch.nn.functional.cross_entropy(scores, labels)
                    if not torch.isfinite(batch_loss):
                        raise ValueError('Non-finite validation loss')
                    val_loss += batch_loss.item() * len(labels)
                    val_correct += (scores.argmax(1) == labels).sum().item()
                    val_count += len(labels)
                    if stop_path.exists():
                        progress('paused', stage='validation', validation_processed=val_count)
                        return
                    if (batch_index + 1) % 100 == 0:
                        progress('running', stage='validation', validation_processed=val_count)
            history.append(dict(epoch=epoch + 1, train_loss=stats['loss_sum'] / offset,
                train_accuracy=stats['correct'] / offset, train_images=offset,
                validation_loss=val_loss / val_count, validation_accuracy=val_correct / val_count,
                validation_images=val_count, train_seconds=stats['train_seconds'],
                images_per_second=offset / stats['train_seconds'], global_step=global_step))
            epoch += 1
            offset = 0
            stats = dict(loss_sum=0., correct=0, train_seconds=0.)
            save()
            write_json(report_path, dict(history=history, config=vars(args), identity_map=mapping,
                status='completed' if epoch == args.epochs else 'running',
                full_face_training_completed=epoch == args.epochs,
                note='Configured identity-training schedule only; not a face-recognition performance claim.'))
            progress('completed' if epoch == args.epochs else 'running', stage='epoch_complete')
    except BaseException as error:
        # Retain the last successful optimizer step, including its shuffle position.
        save()
        progress('paused' if isinstance(error, KeyboardInterrupt) else 'failed',
                 error=repr(error), stage='interrupted')
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifests/main_sources.csv')
    p.add_argument('--run', default='main_resnet')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cpu')
    p.add_argument('--initialization', choices=['imagenet', 'random'], default='imagenet')
    p.add_argument('--resume')
    p.add_argument('--checkpoint-every', type=int, default=300)
    p.add_argument('--progress-every', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=0, help='Controlled pause after this global step (0 = no limit)')
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.manifest == 'data/manifests/main_sources.csv':
        prepare_manifest()
    if min(arguments.epochs, arguments.batch_size, arguments.threads,
           arguments.checkpoint_every, arguments.progress_every) < 1:
        raise ValueError('Epoch, batch, thread and checkpoint/progress settings must be positive')
    # Keep an unattended CPU run active without changing persistent Windows settings.
    # The display may turn off; explicit sleep/lid/hibernate actions still take precedence.
    awake_requested = False
    if os.name == 'nt':
        import ctypes
        awake_requested = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        print(json.dumps(dict(automatic_idle_sleep_prevented=awake_requested)), flush=True)
    try:
        train(arguments)
    finally:
        if awake_requested:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
