"""Bounded decode audit: full listing is separate; no image tuning/evaluation."""
import csv
import io
import json
from pathlib import Path
import sys
import time
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from phase1.common import ROOT, digest, write_csv, write_json


def main(limit=2000):
    start = time.perf_counter()
    # Fixed bounded hash sample from original training data. Keep records so
    # actual decode coverage can be counted without extrapolation.
    candidates = []
    with (ROOT / 'data/manifests/sources.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['original_split'] == 'train' and int(digest('audit/' + row['source_id'])[:8], 16) % 100 == 0:
                candidates.append(row)
    candidates = sorted(candidates, key=lambda r: digest('audit-order/' + r['source_id']))[:limit]
    results = []
    with zipfile.ZipFile(ROOT / 'data/raw/vggface2.zip') as z:
        for r in candidates:
            record = dict(source_id=r['source_id'], person_id=r['person_id'], archive_member=r['archive_member'],
                          decode_status='ok', width='', height='', error='')
            try:
                with Image.open(io.BytesIO(z.read(r['archive_member']))) as im:
                    im.load()
                    record.update(width=im.width, height=im.height)
            except Exception as e:
                record.update(decode_status='failed', error=str(e))
            results.append(record)
    write_csv('data/manifests/decode_audit.csv', results)
    report = dict(scope='hash-selected original train only; no model evaluation', requested_maximum=limit,
                  actual_decoded=len(results),
                  valid=sum(r['decode_status'] == 'ok' for r in results),
                  failures=[r for r in results if r['decode_status'] != 'ok'],
                  identities=len({r['person_id'] for r in results}), seconds=time.perf_counter() - start,
                  all_images_decoded=False)
    write_json('reports/decode_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
