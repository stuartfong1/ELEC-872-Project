param([string]$Environment = '.venv-cuda')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (Test-Path -LiteralPath $Environment) { throw 'Choose a new environment directory; existing environments are preserved.' }
python -m venv $Environment
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$projectPython = Join-Path (Get-Location) "$Environment\Scripts\python.exe"
& $projectPython -m pip install -r requirements-cuda.txt
if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency installation failed' }
& $projectPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable: check NVIDIA hardware and driver'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed; no CPU fallback is configured.' }
