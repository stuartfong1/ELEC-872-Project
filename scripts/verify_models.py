"""Real-image repeatability, frozen/eval and LVFace PyTorch/ONNX parity checks."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from phase1.common import ROOT, read_csv, write_json
from phase1.models import load_farl, load_lvface, load_resnet

torch.set_num_threads(2)
source = read_csv('data/manifests/smoke_sources.csv')[0]
with Image.open(ROOT / source['reference_path']) as im:
    image = im.convert('RGB')
report = {'source_id': source['source_id'], 'models': {}}
for name, loader in [('farl', load_farl), ('lvface', load_lvface),
                     ('resnet50', lambda: load_resnet('checkpoints/smoke_resnet/last.pt'))]:
    model, preprocess, meta = loader()
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    inputs = preprocess(image)[None]
    with torch.inference_mode():
        a = model(inputs).numpy()
        b = model(inputs).numpy()
    assert a.ndim == 2 and a.shape[0] == 1 and np.isfinite(a).all()
    assert np.allclose(a, b, atol=1e-6)
    detail = dict(shape=list(a.shape), frozen=True, eval=True, repeatable=True,
                  checkpoint_sha256=meta['checkpoint_sha256'])
    if name == 'lvface':
        session, prep, _ = load_lvface('onnx')
        onnx_result = session.run(None, {session.get_inputs()[0].name: prep(image).numpy()[None]})[0]
        detail['onnx_shape'] = list(onnx_result.shape)
        detail['onnx_max_absolute_error'] = float(np.max(np.abs(a - onnx_result)))
        detail['onnx_cosine'] = float(np.ravel(a) @ np.ravel(onnx_result) / (np.linalg.norm(a) * np.linalg.norm(onnx_result)))
        assert np.allclose(a, onnx_result, atol=1e-3, rtol=1e-3)
        detail['onnx_parity_passed'] = True
    report['models'][name] = detail
    del model
write_json('reports/model_verification.json', report)
print(json.dumps(report, indent=2))
