from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from PIL import Image
import torch
from phase1.common import ROOT, read_csv, write_csv, write_json
from phase1.models import load_resnet

torch.set_num_threads(4)
known = json.loads((ROOT / 'checkpoints/smoke_resnet/identity_map.json').read_text())
rows = [r for r in read_csv('data/manifests/pilot_sources.csv') if r['person_id'] not in known]
first_person = rows[0]['person_id']
same = [r for r in rows if r['person_id'] == first_person][:2]
other = next(r for r in rows if r['person_id'] != first_person)
selected = same + [other]
assert len(same) == 2 and all(r['person_id'] not in known for r in selected)
model, preprocess, meta = load_resnet('checkpoints/smoke_resnet/last.pt')
inputs = []
for row in selected:
    with Image.open(ROOT / row['reference_path']) as im:
        inputs.append(preprocess(im.convert('RGB')))
with torch.inference_mode():
    raw = model(torch.stack(inputs)).numpy()
assert raw.shape == (3, 2048) and np.isfinite(raw).all()
l2 = raw / np.linalg.norm(raw, axis=1, keepdims=True)
dest = ROOT / 'features/unseen_demo'
dest.mkdir(parents=True, exist_ok=True)
np.save(dest / 'resnet50_raw.npy', raw)
np.save(dest / 'resnet50_l2.npy', l2)
write_csv(dest / 'rows.csv', [dict(feature_row=i, **r, model='ResNet50',
          checkpoint=meta['checkpoint'], checkpoint_sha256=meta['checkpoint_sha256'],
          feature_layer=meta['feature_layer']) for i,r in enumerate(selected)])
report = dict(passed=True, shape=list(raw.shape), all_outside_training_identity_map=True,
              original_split='train', no_final_evaluation_images_used=True,
              source_ids=[r['source_id'] for r in selected],
              same_identity_cosine=float(l2[0] @ l2[1]), different_identity_cosine=float(l2[0] @ l2[2]),
              limitation='embedding pathway demonstration only; not recognition accuracy or identity probability', model=meta)
write_json('reports/unseen_embedding_verification.json', report)
print(json.dumps(report, indent=2))
