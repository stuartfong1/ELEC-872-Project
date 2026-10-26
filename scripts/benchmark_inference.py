"""Tune throughput only, using one original train reference and frozen FaRL."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gc
import json
import time
import numpy as np
from PIL import Image
import psutil
import torch
from phase1.common import ROOT, read_csv, write_json
from phase1.models import load_farl

torch.set_num_threads(4)
model, preprocess, meta = load_farl()
row = read_csv('data/manifests/smoke_sources.csv')[0]
with Image.open(ROOT / row['reference_path']) as im:
    tensor = preprocess(im.convert('RGB'))
results = []
with torch.inference_mode():
    for threads, batch_size in [(2, 2), (4, 4), (4, 8), (8, 8)]:
        torch.set_num_threads(threads)
        batch = tensor[None].repeat(batch_size, 1, 1, 1)
        start = time.perf_counter()
        result = model(batch)
        elapsed = time.perf_counter() - start
        assert torch.isfinite(result).all()
        results.append(dict(threads=threads, batch_size=batch_size, seconds=elapsed,
                            images_per_second=batch_size/elapsed,
                            sampled_rss_bytes=psutil.Process().memory_info().rss))
        print(json.dumps(results[-1]), flush=True)
        del batch, result
        gc.collect()
best = max(results, key=lambda r: r['images_per_second'])
report = dict(scope='throughput tuning only; one original-train source; no recognition evaluation',
              source_id=row['source_id'], model=meta, warmup_note='first case includes cold execution overhead',
              measurements=results, selected=best)
write_json('reports/inference_benchmark.json', report)
