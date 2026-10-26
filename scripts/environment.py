import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
import psutil
import torch
import torchvision
import cv2
import onnxruntime
from packaging.requirements import Requirement

root = Path(__file__).resolve().parents[1]
def hardware(class_name, fields):
    command = f'Get-CimInstance {class_name} | Select-Object {fields} | ConvertTo-Json -Compress'
    return json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True))

resolved_packages = {}
for dist in importlib.metadata.distributions():
    resolved_packages.setdefault(dist.metadata['Name'], dist.version)
report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
              memory_bytes=psutil.virtual_memory().total, available_memory_bytes=psutil.virtual_memory().available,
              cpu=hardware('Win32_Processor', 'Name,NumberOfCores,NumberOfLogicalProcessors'),
              gpu=hardware('Win32_VideoController', 'Name'), cuda_available=torch.cuda.is_available(),
              package_locations={'torch': torch.__file__, 'torchvision': torchvision.__file__,
                                 'opencv': cv2.__file__, 'onnxruntime': onnxruntime.__file__},
              runtime_versions={'torch': torch.__version__, 'torchvision': torchvision.__version__,
                                'opencv': cv2.__version__, 'onnxruntime': onnxruntime.__version__},
              disk_free_bytes=psutil.disk_usage(str(root)).free,
              packages=resolved_packages)
(root / 'reports/environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
pending = [Requirement(line).name for line in (root / 'requirements.txt').read_text().splitlines()
           if line.strip() and not line.startswith('#')]
locked = {}
while pending:
    name = pending.pop()
    normalized = name.lower().replace('_', '-')
    if normalized in locked:
        continue
    dist = importlib.metadata.distribution(name)
    locked[normalized] = f'{dist.metadata["Name"]}=={dist.version}'
    for requirement in dist.requires or []:
        req = Requirement(requirement)
        if req.marker is None or req.marker.evaluate({'extra': ''}):
            pending.append(req.name)
locked['torch'] = f'torch=={torch.__version__}'
locked['torchvision'] = f'torchvision=={torchvision.__version__}'
(root / 'requirements.lock.txt').write_text('\n'.join(locked[k] for k in sorted(locked)) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
