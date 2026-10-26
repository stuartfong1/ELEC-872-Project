"""Compare epoch-boundary resumed training with an uninterrupted real run."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from phase1.common import ROOT, write_json

resumed = torch.load(ROOT / 'checkpoints/smoke_resnet/last.pt', map_location='cpu', weights_only=True)
continuous = torch.load(ROOT / 'checkpoints/smoke_continuous/last.pt', map_location='cpu', weights_only=True)
assert resumed['epoch'] == continuous['epoch'] == 2
assert resumed['identity_map'] == continuous['identity_map']
assert resumed['manifest_sha256'] == continuous['manifest_sha256']
differences = {k: float((v.float() - continuous['model'][k].float()).abs().max())
               for k, v in resumed['model'].items()}
maximum = max(differences.values())
assert maximum <= 1e-6, f'Resume diverged: {maximum}'
for opt_key, values in resumed['optimizer']['state'].items():
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, continuous['optimizer']['state'][opt_key][key])
report = dict(passed=True, epochs=2, compared_model_tensors=len(differences),
              maximum_absolute_parameter_difference=maximum, optimizer_state_equal=True,
              manifest_sha256=resumed['manifest_sha256'])
write_json('reports/resume_verification.json', report)
print(json.dumps(report, indent=2))
