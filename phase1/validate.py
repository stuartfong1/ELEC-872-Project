"""Phase 1 integrity checks and descriptive similarities, never probe training."""
import argparse
from collections import Counter, defaultdict
import json
import numpy as np
from PIL import Image
from .common import ROOT, read_csv, write_json, sha256
from .processing import CORE


def validate(run):
    sources = read_csv(f'data/manifests/{run}_sources.csv')
    rows = read_csv(f'data/manifests/{run}_variants.csv')
    by_source = defaultdict(list)
    checks = {}
    for r in rows:
        by_source[r['source_id']].append(r)
    source_index = {r['source_id']: r for r in sources}
    assert len(source_index) == len(sources)
    assert set(by_source) == set(source_index)
    checks['source_variant_linkage'] = True
    conditions = set(r['processing_label'] for r in rows)
    for sid, variants in by_source.items():
        assert len(variants) == len(conditions)
        assert set(r['processing_label'] for r in variants) == conditions
        for r in variants:
            assert r['status'] == 'ok'
            assert r['person_id'] == source_index[sid]['person_id']
            for field in ('face_split', 'probe_split', 'original_split', 'photo_group'):
                assert r[field] == source_index[sid][field]
    checks['balanced_conditions_and_labels'] = True
    face_train = {r['photo_group'] for r in sources if r['face_split'] == 'face_train'}
    face_val = {r['photo_group'] for r in sources if r['face_split'] == 'face_val'}
    assert not face_train & face_val
    checks['face_photo_group_disjoint'] = True
    people = defaultdict(set)
    for row in sources:
        people[row['probe_split']].add(row['person_id'])
    assert not people['probe_train'] & people['probe_val']
    checks['probe_identity_disjoint'] = True
    # Exact selected-source duplicates cannot cross a split.
    hashes = defaultdict(list)
    for r in sources:
        hashes[r['source_sha256']].append(r)
    for duplicates in hashes.values():
        for field in ('face_split', 'probe_split'):
            assert len({r[field] for r in duplicates}) == 1
    checks['selected_exact_duplicate_split_check'] = True
    identity_equal = 0
    quantizations = {}
    for sid, variants in by_source.items():
        with Image.open(ROOT / source_index[sid]['reference_path']) as im:
            reference = np.asarray(im).copy()
        identity = next(r for r in variants if r['processing_label'] == 'IDENTITY')
        with Image.open(ROOT / identity['variant_path']) as im:
            assert np.array_equal(reference, np.asarray(im))
        identity_equal += 1
        for r in variants:
            if r['processing_label'].startswith('JPEG'):
                params = json.loads(r['parameters_json'])
                assert params['jpeg_sampling_code'] == (2 if r['subsampling'] == '420' else 0)
                assert params['quantization_tables']
                quantizations[r['processing_label']] = params['quantization_tables']
    checks['identity_pixel_exact_count'] = identity_equal
    checks['jpeg_sampling_and_quantization_verified'] = True
    features = {}
    expected_ids = [r['variant_id'] for r in rows]
    for folder in sorted((ROOT / 'features' / run).glob('*')):
        if not (folder / 'raw.npy').exists():
            continue
        raw, l2 = np.load(folder / 'raw.npy'), np.load(folder / 'l2.npy')
        mapping = read_csv(folder / 'rows.csv')
        assert [r['variant_id'] for r in mapping] == expected_ids
        assert [int(r['feature_row']) for r in mapping] == list(range(len(rows)))
        assert len(raw) == len(rows) and raw.ndim == 2 and np.isfinite(raw).all()
        assert np.isfinite(l2).all() and np.allclose(np.linalg.norm(l2, axis=1), 1, atol=1e-5)
        assert np.allclose(l2, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-6)
        meta = json.loads((folder / 'model.json').read_text())
        assert meta['checkpoint_sha256'] == sha256(meta['checkpoint'])
        for i, r in enumerate(mapping):
            for field in ('source_id', 'person_id', 'processing_label', 'face_split', 'probe_split'):
                assert r[field] == rows[i][field]
        # Descriptive only; no classifier or final evaluation is performed.
        groups = defaultdict(list)
        identity_indices = [i for i, r in enumerate(rows) if r['processing_label'] == 'IDENTITY']
        id_by_source = {rows[i]['source_id']: i for i in identity_indices}
        for i, r in enumerate(rows):
            if r['processing_label'] != 'IDENTITY':
                groups['same_source_variants'].append(float(l2[i] @ l2[id_by_source[r['source_id']]]))
        for n, i in enumerate(identity_indices):
            for j in identity_indices[n + 1:]:
                group = 'same_identity_different_photo' if rows[i]['person_id'] == rows[j]['person_id'] else 'different_identity'
                groups[group].append(float(l2[i] @ l2[j]))
        similarities = {k: dict(count=len(v), mean=float(np.mean(v)), std=float(np.std(v)),
                               min=float(np.min(v)), max=float(np.max(v))) for k, v in groups.items()}
        features[folder.name] = dict(shape=list(raw.shape), all_finite=True, row_alignment=True,
                                     l2_unit_norm=True, similarities=similarities, model=meta)
    if not features:
        raise ValueError('No actual model features found')
    report = dict(run=run, checks=checks, source_count=len(sources), variant_count=len(rows),
                  original_splits=dict(Counter(r['original_split'] for r in sources)),
                  face_splits=dict(Counter(r['face_split'] for r in sources)),
                  probe_splits=dict(Counter(r['probe_split'] for r in sources)),
                  condition_counts=dict(Counter(r['processing_label'] for r in rows)), features=features,
                  limitations=['cosine similarity is not identity probability',
                               'no shortcut reliance, camera identification, fairness or attack conclusion',
                               'pretraining overlap with FaRL/LVFace is unknown',
                               'cross-model coordinates are not directly comparable'], passed=True)
    write_json(f'reports/{run}_validation.json', report)
    print(json.dumps(dict(run=run, passed=True, sources=len(sources), variants=len(rows),
                          feature_shapes={k: v['shape'] for k, v in features.items()})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='smoke')
    validate(p.parse_args().run)
