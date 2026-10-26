import argparse
import json
import time
import numpy as np
from PIL import Image
import psutil
import torch
from .common import ROOT, read_csv, write_csv, write_json, sha256
from .models import load_farl, load_lvface, load_resnet
from .train import require_cuda


def extract(args):
    device = require_cuda(getattr(args, 'gpu', 0))
    torch.set_num_threads(args.threads)
    rows = read_csv(f'data/manifests/{args.run}_variants.csv')
    if any(r['status'] != 'ok' for r in rows):
        raise ValueError('Failed variants present; resolve explicitly before feature extraction')
    if args.model == 'farl':
        model, preprocess, metadata = load_farl()
    elif args.model == 'lvface':
        model, preprocess, metadata = load_lvface(args.backend, gpu=device.index if args.backend == 'onnx' else None)
    else:
        model, preprocess, metadata = load_resnet(args.checkpoint)
    if args.backend == 'torch':
        model = model.to(device)
    dest = ROOT / 'features' / args.run / (args.model + ('_onnx' if args.backend == 'onnx' else ''))
    dest.mkdir(parents=True, exist_ok=True)
    input_sha = sha256(f'data/manifests/{args.run}_variants.csv')
    metadata.update(manifest_sha256=input_sha, raw_features='raw.npy', normalized_features='l2.npy',
                    feature_normalization='rowwise L2; raw preserved', source_run=args.run,
                    inference_backend='cuda', gpu_name=torch.cuda.get_device_name(device))
    metadata = json.loads(json.dumps(metadata))
    # Resume immutable batches only if the checkpoint, inputs and preprocessing match.
    meta_path = dest / 'model.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError('Feature resume metadata differs; use a new run directory')
    write_json(meta_path, metadata)
    shards = dest / 'shards'
    shards.mkdir(exist_ok=True)
    start = time.perf_counter()
    cache = {}
    for existing in sorted(shards.glob('*.npz'), key=lambda p: (p.stat().st_mtime_ns, p.name)):
        if existing.stem.endswith('.tmp'):
            continue
        with np.load(existing, allow_pickle=False) as saved:
            if saved['raw'].ndim != 2 or len(saved['ids']) != len(saved['raw']) or not np.isfinite(saved['raw']).all():
                raise ValueError(f'Invalid feature cache: {existing}')
            for variant_id, feature_row in zip(saved['ids'].tolist(), saved['raw']):
                cache[variant_id] = feature_row.copy()
    arrays, mapped, peak_rss, computed = [], [], 0, 0
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset:offset + args.batch_size]
        part = shards / f'{offset:08d}_{len(batch_rows):04d}.npz'
        ids = [r['variant_id'] for r in batch_rows]
        if all(variant_id in cache for variant_id in ids):
            feature = np.stack([cache[variant_id] for variant_id in ids])
        else:
            inputs = []
            for row in batch_rows:
                with Image.open(ROOT / row['variant_path']) as im:
                    inputs.append(preprocess(im.convert('RGB')))
            batch = torch.stack(inputs)
            with torch.inference_mode():
                if args.backend == 'onnx':
                    # Official export may have a static singleton batch.
                    feature = np.concatenate([model.run(None, {model.get_inputs()[0].name: x.numpy()[None]})[0]
                                              for x in batch], axis=0)
                else:
                    feature = model(batch.to(device)).cpu().numpy()
            feature = np.asarray(feature, dtype=np.float32)
            if feature.ndim != 2 or len(feature) != len(batch_rows) or not np.isfinite(feature).all():
                raise ValueError(f'Invalid model output at offset {offset}')
            temp = part.with_suffix('.tmp.npz')
            np.savez(temp, raw=feature, ids=np.asarray(ids))
            temp.replace(part)
            computed += len(batch_rows)
        arrays.append(feature)
        for i, row in enumerate(batch_rows):
            mapped.append(dict(feature_row=offset + i, **row, model=metadata['model'],
                               checkpoint=metadata['checkpoint'], checkpoint_sha256=metadata['checkpoint_sha256'],
                               feature_layer=metadata['feature_layer']))
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        if offset % (args.batch_size * 10) == 0:
            print(f'{args.model}: {offset + len(batch_rows)}/{len(rows)}', flush=True)
    raw = np.concatenate(arrays)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.isfinite(raw).all() or not (norms > 0).all():
        raise ValueError('Non-finite or zero-norm features')
    np.save(dest / 'raw.npy', raw)
    np.save(dest / 'l2.npy', raw / norms)
    write_csv(dest / 'rows.csv', mapped)
    elapsed = time.perf_counter() - start
    report = dict(model=metadata, shape=list(raw.shape), seconds=elapsed, computed_rows= computed,
                  cached_rows=len(rows)-computed, execution_threads=args.threads, execution_batch_size=args.batch_size,
                  execution_device=str(device),
                  images_per_second=computed / elapsed, peak_process_rss_bytes=peak_rss,
                  all_finite=bool(np.isfinite(raw).all()), row_alignment=True,
                  rows_sha256=sha256(dest / 'rows.csv'), raw_sha256=sha256(dest / 'raw.npy'))
    write_json(dest / 'validation.json', report)
    # Keep first real computation timing when a later resume computes zero rows.
    if computed:
        write_json(dest / 'timing_original.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'model'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['farl', 'lvface', 'resnet50'])
    p.add_argument('--run', default='smoke_adam')
    p.add_argument('--backend', choices=['torch', 'onnx'], default='torch')
    p.add_argument('--checkpoint', default='checkpoints/smoke_adam_resnet/last.pt')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    if a.backend == 'onnx' and a.model != 'lvface':
        p.error('ONNX backend is only for LVFace')
    extract(a)
