"""Apply the observed manual review decision; retain original/generated files."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
import json
from phase1.common import ROOT, read_csv, write_csv, write_json
from phase1.data import contact_sheet

sources = read_csv('data/manifests/pilot_sources.csv')
variants = read_csv('data/manifests/pilot_variants.csv')
replacements = json.loads((ROOT / 'reports/pilot_final_replacement_ids.json').read_text())
excluded_id = replacements[0]['source_id']
reason = 'final visual review: secondary background faces; exclude without selecting largest face; retain 99 valid sources'
blacklist = json.loads((ROOT / 'configs/excluded_sources.json').read_text())
blacklist[excluded_id] = reason
write_json('configs/excluded_sources.json', blacklist)
retained = [r for r in sources if r['source_id'] != excluded_id]
assert len(retained) == 99 and len({r['person_id'] for r in retained}) == 20
for row in retained:
    row['review_status'] = 'visually_reviewed_existing_crop'
retained_ids = {r['source_id'] for r in retained}
variants = [r for r in variants if r['source_id'] in retained_ids]
for row in variants:
    row['review_status'] = 'visually_reviewed_existing_crop'
write_csv('data/manifests/pilot_sources.csv', retained)
write_csv('data/manifests/pilot_variants.csv', variants)
write_csv('data/manifests/review_exclusions.csv', [dict(source_id=k, reason=v) for k,v in blacklist.items()])
processing = json.loads((ROOT / 'reports/pilot_processing.json').read_text())
write_json('reports/pilot_processing_before_final_review.json', processing)
processing.update(target_sources=100, generated_sources_before_final_review=100, sources=99,
                  generated_variants_before_final_review=700, variants=len(variants),
                  condition_counts=dict(Counter(r['processing_label'] for r in variants)),
                  final_manual_exclusion=dict(source_id=excluded_id, reason=reason),
                  exclusions=[dict(source_id=k, reason=v) for k,v in blacklist.items()])
write_json('reports/pilot_processing.json', processing)
review = dict(retained_sources=99, retained_identities=20, target_sources=100,
              unique_pilot_candidates_visually_reviewed=111, excluded_pilot_candidates=12,
              retained_source_ids=sorted(retained_ids), excluded_sources=blacklist,
              geometry='existing crop, full RGB extent; no redetection/realignment',
              source_contact_sheets=['pilot_review_1.png', 'pilot_review_2.png', 'pilot_review_3.png',
                                     'pilot_review_4.png', 'pilot_replacements.png', 'pilot_replacements_final.png'],
              identity_ground_truth_independently_verified=False,
              limitations='dataset labels trusted; no full-dataset visual certification; background partial person in smoke review retained as noted')
write_json('reports/pilot_reference_review.json', review)
contact_sheet(retained, 'reports/pilot_sources.png')
print(json.dumps(dict(retained_sources=99, variants=len(variants), identities=20)))
