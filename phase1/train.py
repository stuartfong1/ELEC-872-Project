"""CUDA-only ResNet50 identity training with atomic mid-epoch checkpoints."""
import argparse
import io
import json
import math
import os
import random
import re
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps
import psutil
import torch
from torch.utils.data import Dataset, DataLoader

from .common import ROOT, read_csv, write_csv, write_json, sha256
from .models import new_resnet, resnet_transform

SAMPLER = 'explicit_randperm_872_plus_epoch_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def require_cuda(gpu=0):
    if gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError('CUDA training requires an NVIDIA GPU and CUDA-enabled PyTorch; no CPU fallback is provided.')
    if gpu >= torch.cuda.device_count():
        raise ValueError(f'CUDA GPU index {gpu} is unavailable')
    torch.cuda.set_device(gpu)
    return torch.device('cuda', gpu)


def prepare_manifest():
    """Use original train identities and preserve photograph-group splits."""
    path = ROOT / 'data/manifests/main_sources.csv'
    if path.exists():
        return
    exclusions = json.loads((ROOT / 'configs/excluded_sources.json').read_text(encoding='utf-8'))
    sources = read_csv(ROOT / 'data/manifests/sources.csv')
    eligible = [r for r in sources if r['original_split'] == 'train']
    rows = [r for r in eligible if r['source_id'] not in exclusions and r['decode_status'] != 'failed']
    retained = {r['source_id'] for r in rows}
    write_csv(path, rows)
    write_json(ROOT / 'reports/main_manifest.json', dict(
        created_utc=now(), original_manifest_sha256=sha256(ROOT / 'data/manifests/sources.csv'),
        manifest_sha256=sha256(path), identities=len({r['person_id'] for r in rows}),
        train_images=sum(r['face_split'] == 'face_train' for r in rows),
        validation_images=sum(r['face_split'] == 'face_val' for r in rows), original_val_images=0,
        excluded_sources=[dict(source_id=r['source_id'], reason=exclusions.get(r['source_id'], 'decode failed'))
                          for r in eligible if r['source_id'] not in retained],
        full_decode_audit_completed=False,
        input_policy='Original crop; EXIF transpose; ImageNet resize 232 / center crop 224. No augmentation.'))


class FaceDataset(Dataset):
    def __init__(self, rows, mapping):
        self.rows, self.mapping = rows, mapping
        self.transform = resnet_transform()
        self.archives = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            ref = row.get('reference_path')
            if ref and (ROOT / ref).exists():
                with Image.open(ROOT / ref) as im:
                    image = im.convert('RGB')
            else:
                archive = row['archive']
                if archive not in self.archives:
                    self.archives[archive] = zipfile.ZipFile(ROOT / archive)
                with Image.open(io.BytesIO(self.archives[archive].read(row['archive_member']))) as im:
                    image = ImageOps.exif_transpose(im).convert('RGB')
            return self.transform(image), self.mapping[row['person_id']]
        except Exception as error:
            raise RuntimeError(f"Source decode/transform failed: {row['source_id']}") from error


def make_optimizer(model, args):
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
    raise ValueError('Unsupported optimizer')


def validate_resume(ck, args, mapping, manifest_sha):
    expected = dict(identity_map=mapping, manifest_sha256=manifest_sha,
                    batch_size=args.batch_size, initialization=args.initialization)
    if any(ck.get(key) != value for key, value in expected.items()):
        raise ValueError('Resume data, identity mapping, batch size or initialization differs')
    old_optimizer = ck.get('optimizer_name', ck['config'].get('optimizer', 'sgd'))
    if old_optimizer != args.optimizer:
        raise ValueError('Resume optimizer differs; use --initialize-from in a new run to load weights only')
    if ck['config']['lr'] != args.lr or ck['config'].get('weight_decay', 1e-4) != args.weight_decay:
        raise ValueError('Keep learning rate and weight decay unchanged on resume')
    if ck.get('sampler') not in (None, SAMPLER):
        raise ValueError('Unsupported checkpoint sampler')
    if ck.get('offset', 0) and ck.get('sampler') != SAMPLER:
        raise ValueError('Unknown mid-epoch sampler; cannot safely resume')


def train(args):
    if hasattr(args, 'device'):
        raise ValueError('Training is CUDA-only; select a GPU with --gpu, not --device')
    options = vars(parser().parse_args([]))
    options.update(vars(args))
    args = argparse.Namespace(**options)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run):
        raise ValueError('Run must be a single directory name')
    if min(args.epochs, args.batch_size, args.threads, args.checkpoint_every, args.progress_every) < 1:
        raise ValueError('Epoch, batch, thread and interval settings must be positive')
    if args.lr <= 0 or args.weight_decay < 0 or args.max_steps < 0:
        raise ValueError('Invalid learning rate, weight decay or step limit')
    if args.resume and args.initialize_from:
        raise ValueError('--resume and --initialize-from are mutually exclusive')
    device = require_cuda(args.gpu)
    torch.set_num_threads(args.threads)
    torch.manual_seed(872)
    random.seed(872)
    np.random.seed(872)
    if args.manifest == 'data/manifests/main_sources.csv':
        prepare_manifest()
    output = ROOT / 'checkpoints' / args.run
    ck_path, stop_path = output / 'last.pt', output / 'STOP'
    if ck_path.exists() and not args.resume:
        raise ValueError('Existing checkpoint: use --resume or select a new --run')
    rows = read_csv(args.manifest)
    if any(r['original_split'] != 'train' for r in rows):
        raise ValueError('Training manifest must contain only original train identities')
    if any(r.get('decode_status') == 'failed' for r in rows):
        raise ValueError('Resolve failed source images before training')
    train_rows = [r for r in rows if r['face_split'] == 'face_train']
    val_rows = [r for r in rows if r['face_split'] == 'face_val']
    mapping = {p: i for i, p in enumerate(sorted({r['person_id'] for r in train_rows}))}
    if not train_rows or not val_rows or not {r['person_id'] for r in val_rows} <= set(mapping):
        raise ValueError('Need nonempty within-identity train and validation splits')
    if {r['photo_group'] for r in train_rows} & {r['photo_group'] for r in val_rows}:
        raise ValueError('Photo-group overlap between train and validation')
    manifest_sha = sha256(args.manifest)
    is_main = (ROOT / args.manifest).resolve() == (ROOT / 'data/manifests/main_sources.csv').resolve()
    epoch, offset, global_step, history = 0, 0, 0, []
    stats = dict(loss_sum=0., correct=0, train_seconds=0.)
    ck, initialized_from = None, None
    if args.resume:
        # Stage on the host so CPU RNG tensors stay valid and GPU memory is bounded.
        ck = torch.load(ROOT / args.resume, map_location='cpu', weights_only=True)
        validate_resume(ck, args, mapping, manifest_sha)
        epoch, offset = ck['epoch'], ck.get('offset', 0)
        history = ck['history']
        global_step = ck.get('global_step', sum(h.get('backward_steps', math.ceil(h['train_images'] / args.batch_size)) for h in history))
        stats = ck.get('partial_stats', stats)
        if epoch < 0 or not 0 <= offset <= len(train_rows) or global_step < 0:
            raise ValueError('Invalid checkpoint training position')
        initialized_from = ck.get('initialized_from')
        if epoch >= args.epochs:
            if offset or epoch > args.epochs:
                raise ValueError('Checkpoint exceeds the requested epoch limit; weights are preserved')
            print(json.dumps(dict(status='completed', stage='already_complete', run=args.run)), flush=True)
            return
    elif args.initialize_from:
        ck = torch.load(ROOT / args.initialize_from, map_location='cpu', weights_only=True)
        if ck['identity_map'] != mapping:
            raise ValueError('Weight initialization requires the same identity mapping')
        initialized_from = dict(path=args.initialize_from, sha256=sha256(args.initialize_from),
                                optimizer_reset=True, counters_reset=True)
    model = new_resnet(len(mapping), 'random' if ck else args.initialization).to(device)
    if ck:
        model.load_state_dict(ck['model'])
    optimizer = make_optimizer(model, args)
    if ck and args.resume:
        optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['torch_rng_state'])
        if ck.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(ck['cuda_rng_state'], device)
    resumed_from_device = ck.get('device', ck['config'].get('device', 'cpu')) if ck and args.resume else None
    del ck
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'identity_map.json', mapping)
    train_data, val_data = FaceDataset(train_rows, mapping), FaceDataset(val_rows, mapping)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                            generator=torch.Generator().manual_seed(872))
    started = time.perf_counter()
    last_saved_step = global_step if args.resume else None
    peak_rss = 0
    torch.cuda.reset_peak_memory_stats(device)

    def progress(status, **extra):
        value = dict(status=status, updated_utc=now(), pid=os.getpid(), run=args.run,
                     device=str(device), gpu_name=torch.cuda.get_device_name(device), optimizer=args.optimizer,
                     epoch_index=epoch, target_epochs=args.epochs, epoch_display=min(epoch + 1, args.epochs),
                     images_in_epoch=offset, train_images_per_epoch=len(train_rows), validation_images=len(val_rows),
                     identities=len(mapping), global_step=global_step, epoch_fraction=offset / len(train_rows),
                     train_loss_so_far=stats['loss_sum'] / offset if offset else None,
                     train_accuracy_so_far=stats['correct'] / offset if offset else None,
                     train_images_per_second=offset / stats['train_seconds'] if stats['train_seconds'] else None,
                     invocation_seconds=time.perf_counter() - started, checkpoint=str(ck_path),
                     checkpoint_global_step=last_saved_step, peak_process_rss_bytes=peak_rss,
                     peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
                     schedule_completed=status == 'completed',
                     full_face_training_completed=is_main and status == 'completed', **extra)
        write_json(ROOT / f'reports/{args.run}_progress.json', value)
        print(json.dumps(value), flush=True)

    def save():
        nonlocal last_saved_step
        status = ('full_face_training_completed' if epoch >= args.epochs else 'full_face_training_in_progress') if is_main else (
            'smoke_face_training_only' if args.run.startswith('smoke') else 'subset_face_training')
        checkpoint = dict(schema_version=2, model=model.state_dict(), optimizer=optimizer.state_dict(),
            optimizer_name=args.optimizer, device=str(device), initialized_from=initialized_from,
            resumed_from_device=resumed_from_device, epoch=epoch, offset=offset, global_step=global_step,
            partial_stats=stats.copy(), history=history, identity_map=mapping, manifest=args.manifest,
            manifest_sha256=manifest_sha, batch_size=args.batch_size, initialization=args.initialization,
            sampler=SAMPLER, torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device), config=vars(args), saved_utc=now(), training_status=status)
        temporary = output / 'last.pt.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(ck_path)
        last_saved_step = global_step

    progress('running', stage='initialized', all_backbone_parameters_trainable=all(p.requires_grad for p in model.parameters()))
    try:
        if stop_path.exists() or (args.max_steps and global_step >= args.max_steps):
            save()
            progress('paused', stage='train')
            return
        while epoch < args.epochs:
            if offset == 0:
                torch.manual_seed(872 + epoch)
            order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(872 + epoch)).tolist()
            loader = DataLoader(train_data, batch_size=args.batch_size, sampler=order[offset:],
                                num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(1872 + epoch))
            model.train()
            batch_started = time.perf_counter()
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
                    progress('paused' if stopping else 'running', stage='train', last_batch_loss=loss.item())
                if stopping:
                    return
                batch_started = time.perf_counter()
            save()
            progress('running', stage='validation', validation_processed=0)
            model.eval()
            val_loss, val_correct, val_count = 0., 0, 0
            with torch.inference_mode():
                for batch_index, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
                images_per_second=offset / stats['train_seconds'], global_step=global_step,
                optimizer=args.optimizer, device=str(device), peak_process_rss_bytes=peak_rss,
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device)))
            epoch += 1
            offset = 0
            stats = dict(loss_sum=0., correct=0, train_seconds=0.)
            save()
            write_json(ROOT / f'reports/{args.run}_training.json', dict(history=history, config=vars(args), identity_map=mapping,
                status='completed' if epoch == args.epochs else 'running',
                schedule_completed=epoch == args.epochs, full_face_training_completed=is_main and epoch == args.epochs,
                note='Configured identity-training schedule only; not a face-recognition performance claim.'))
            progress('completed' if epoch == args.epochs else 'running', stage='epoch_complete')
    except BaseException as error:
        save()
        progress('paused' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error), stage='interrupted')
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifests/main_sources.csv')
    p.add_argument('--run', default='main_resnet_adam')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=2, help='Host preprocessing threads')
    p.add_argument('--gpu', type=int, default=0, help='CUDA device index; CUDA is required')
    p.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--initialization', choices=['imagenet', 'random'], default='imagenet')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--resume', help='Restore the same optimizer and training position')
    source.add_argument('--initialize-from', help='Load matching identity weights into a new run; reset optimizer and counters')
    p.add_argument('--checkpoint-every', type=int, default=300)
    p.add_argument('--progress-every', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=0, help='Pause at this global step (0 = no limit)')
    return p


def main():
    args = parser().parse_args()
    awake_requested = False
    if os.name == 'nt':
        import ctypes
        awake_requested = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        train(args)
    finally:
        if awake_requested:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
