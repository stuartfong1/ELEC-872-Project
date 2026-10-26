import argparse
from collections import Counter, defaultdict
import io
import json
import time
import zipfile
from pathlib import PurePosixPath
import numpy as np
import psutil
from PIL import Image, ImageOps, ImageDraw
from .common import ROOT, digest, write_csv, read_csv, write_json


def save_immutable_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as existing:
            if not np.array_equal(np.asarray(existing.convert('RGB')), np.asarray(image.convert('RGB'))):
                raise ValueError(f'Existing generated PNG differs; use a new run: {path}')
    else:
        image.save(path)


def inventory(archive, decode_all=False):
    start = time.perf_counter()
    rows, failures = [], []
    counts = Counter()
    people = defaultdict(set)
    with zipfile.ZipFile(ROOT / archive) as z:
        for entry in z.infolist():
            p = PurePosixPath(entry.filename)
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            parts = p.parts
            split_index = next((i for i, x in enumerate(parts) if x in ('train', 'val')), None)
            if split_index is None:
                raise ValueError(f'Unrecognized source split: {p}')
            split, person = parts[split_index:split_index + 2]
            source_id = digest(str(p))[:24]
            photo_group = person + '/' + p.stem.split('_')[0]
            face_split = ('heldout_identity' if split == 'val' else
                          ('face_val' if int(digest(photo_group)[:8], 16) % 5 == 0 else 'face_train'))
            probe_split = ('probe_test' if split == 'val' else
                           ('probe_val' if int(digest(person)[:8], 16) % 5 == 0 else 'probe_train'))
            row = dict(source_id=source_id, person_id=person, original_split=split,
                       archive=archive, archive_member=str(p), source_path=f'{archive}::{p}',
                       source_bytes=entry.file_size, archive_crc32=f'{entry.CRC:08x}',
                       photo_group=photo_group, face_split=face_split, probe_split=probe_split,
                       decode_status='not_checked')
            if decode_all:
                try:
                    with Image.open(io.BytesIO(z.read(entry))) as im:
                        im.load()
                    row['decode_status'] = 'ok'
                except Exception as e:
                    row['decode_status'] = 'failed'
                    failures.append({'source_id': source_id, 'error': str(e)})
            rows.append(row)
            counts[split] += 1
            people[split].add(person)
    assert not (people['train'] & people['val'])
    assert len({r['source_id'] for r in rows}) == len(rows)
    write_csv('data/manifests/sources.csv', rows)
    report = {'file_listing_images': dict(counts), 'file_listing_identities': {k: len(v) for k, v in people.items()},
              'identity_overlap': [], 'decode_coverage': len(rows) if decode_all else 0,
              'decode_failures': failures, 'seconds': time.perf_counter() - start,
              'dataset_is_full_original_vggface2': False}
    write_json('reports/data_inventory.json', report)
    print(json.dumps(report), flush=True)


def prepare(run, identities, per_identity, extras=False):
    start = time.perf_counter()
    sources = read_csv('data/manifests/sources.csv')
    eligible = defaultdict(list)
    for row in sources:
        if row['original_split'] == 'train' and row['decode_status'] != 'failed':
            eligible[row['person_id']].append(row)
    selected, exclusions = [], []
    archives = {}
    exclusion_path = ROOT / 'configs/excluded_sources.json'
    excluded_sources = json.loads(exclusion_path.read_text(encoding='utf-8')) if exclusion_path.exists() else {}
    # Train partition only; stable identity and photograph selection.
    for person in sorted(eligible, key=lambda p: digest('872/' + p))[:identities]:
        candidates = sorted(eligible[person], key=lambda r: digest('872/' + r['source_id']))
        targets = {'face_val': max(1, per_identity // 5), 'face_train': per_identity - max(1, per_identity // 5)}
        found = Counter()
        groups = set()
        for r in candidates:
            if r['source_id'] in excluded_sources:
                exclusions.append({'source_id': r['source_id'], 'reason': excluded_sources[r['source_id']]})
                continue
            if found[r['face_split']] >= targets[r['face_split']] or r['photo_group'] in groups:
                continue
            try:
                if r['archive'] not in archives:
                    archives[r['archive']] = zipfile.ZipFile(ROOT / r['archive'])
                blob = archives[r['archive']].read(r['archive_member'])
                with Image.open(io.BytesIO(blob)) as im:
                    im.load()
                    ref = ImageOps.exif_transpose(im).convert('RGB')
                r = dict(r)
                r.update(decode_status='ok', source_sha256=__import__('hashlib').sha256(blob).hexdigest(),
                         width=ref.width, height=ref.height, reference_policy='existing_crop_full_extent',
                         review_status='contact_sheet_review_required',
                         reference_path=f'data/reference/{run}/{r["source_id"]}.png')
                path = ROOT / r['reference_path']
                save_immutable_png(ref, path)
                selected.append(r)
                found[r['face_split']] += 1
                groups.add(r['photo_group'])
            except Exception as e:
                exclusions.append({'source_id': r['source_id'], 'reason': str(e)})
        if sum(found.values()) != per_identity:
            raise RuntimeError(f'Insufficient valid sources for {person}: {found}')
    for archive in archives.values():
        archive.close()
    del sources, eligible, archives
    __import__('gc').collect()
    write_csv(f'data/manifests/{run}_sources.csv', selected)
    contact_sheet(selected[:100], f'reports/{run}_sources.png')
    print(f'References ready: {len(selected)} decoded sources', flush=True)
    from .processing import CORE, EXTRA, apply
    variants = []
    conditions = CORE + (EXTRA if extras else [])
    for source in selected:
        with Image.open(ROOT / source['reference_path']) as im:
            reference = im.convert('RGB')
        for label, condition in enumerate(conditions):
            variant_id = source['source_id'] + '_' + condition
            row = dict(source, variant_id=variant_id, processing_label=condition,
                       processing_class=label, quality='', subsampling='', processing_order='',
                       parameters_json='', variant_path=f'data/processed/{run}/{variant_id}.png', status='pending', error='')
            try:
                result, meta = apply(reference, condition, source['source_id'])
                row['processing_order'] = ' -> '.join(meta['operations'])
                row['parameters_json'] = json.dumps(meta)
                if condition.startswith('JPEG'):
                    row['quality'], row['subsampling'] = condition[4:].split('_')
                elif condition != 'IDENTITY':
                    row['quality'], row['subsampling'] = '95', '444'
                dest = ROOT / row['variant_path']
                save_immutable_png(result, dest)
                row['status'] = 'ok'
            except Exception as e:
                row['status'], row['error'] = 'failed', str(e)
            variants.append(row)
    write_csv(f'data/manifests/{run}_variants.csv', variants)
    comparison(selected[0], variants, run)
    report = dict(run=run, sources=len(selected), identities=len({r['person_id'] for r in selected}),
                  conditions=conditions, variants=len(variants), condition_counts=dict(Counter(r['processing_label'] for r in variants)),
                  failures=[r for r in variants if r['status'] != 'ok'], exclusions=exclusions,
                  source_decode_coverage=len(selected), contact_sheet_sources=min(len(selected), 100),
                  seconds=time.perf_counter() - start,
                  peak_process_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss))
    write_json(f'reports/{run}_processing.json', report)
    print(json.dumps(report), flush=True)


def contact_sheet(rows, dest):
    cols = 5
    cell_w, cell_h = 200, 220
    out = Image.new('RGB', (cols * cell_w, ((len(rows) + cols - 1) // cols) * cell_h), 'white')
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(rows):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        with Image.open(ROOT / row['reference_path']) as im:
            thumb = ImageOps.contain(im, (194, 188))
            out.paste(thumb, (x + 3, y + 24))
        draw.text((x + 4, y + 4), f'{i}: {row["person_id"]} / {row["face_split"]}', fill='black')
    out.save(ROOT / dest)


def comparison(source, variants, run):
    rows = [r for r in variants if r['source_id'] == source['source_id'] and r['status'] == 'ok']
    w, h, cols = 250, 290, 4
    out = Image.new('RGB', (cols * w, ((len(rows) + cols - 1) // cols) * h), '#eef1f5')
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(rows):
        x, y = i % cols * w, i // cols * h
        with Image.open(ROOT / row['variant_path']) as im:
            thumb = ImageOps.contain(im, (240, 240))
            out.paste(thumb, (x + 5, y + 32))
        draw.text((x + 5, y + 5), row['processing_label'], fill='black')
        draw.text((x + 5, y + 273), 'PNG of processed pixels', fill='black')
    out.save(ROOT / f'reports/{run}_comparison.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--archive', default='data/raw/vggface2.zip')
    inv.add_argument('--decode-all', action='store_true')
    prep = sub.add_parser('prepare')
    prep.add_argument('--run', default='smoke')
    prep.add_argument('--identities', type=int, default=5)
    prep.add_argument('--per-identity', type=int, default=5)
    prep.add_argument('--extras', action='store_true')
    a = p.parse_args()
    if a.cmd == 'inventory':
        inventory(a.archive, a.decode_all)
    else:
        prepare(a.run, a.identities, a.per_identity, a.extras)
