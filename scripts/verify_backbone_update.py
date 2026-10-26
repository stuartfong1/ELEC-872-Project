from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
from phase1.common import ROOT, write_json

initial = torch.load(ROOT / 'weights/torch_cache/hub/checkpoints/resnet50-11ad3fa6.pth',
                     weights_only=True, map_location='cpu', mmap=True)
report = {}
for run in ['smoke_resnet', 'pilot_resnet']:
    path = ROOT / 'checkpoints' / run / 'last.pt'
    if not path.exists():
        continue
    trained = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    delta = float((trained['model']['conv1.weight'] - initial['conv1.weight']).abs().max())
    classes = len(trained['identity_map'])
    assert delta > 0 and trained['model']['fc.weight'].shape[0] == classes
    report[run] = dict(backbone_conv1_max_change_from_imagenet=delta,
                       identity_head_classes=classes, epochs=trained['epoch'],
                       actual_face_backbone_update_verified=True)
write_json('reports/backbone_training_verification.json', report)
print(json.dumps(report, indent=2))
