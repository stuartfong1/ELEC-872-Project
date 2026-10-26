param([string]$Environment = '.venv-run')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
# Reuse read-only system packages when available; pip writes into this venv.
# --without-pip avoids copying the bundled pip tree; inherited pip is sufficient.
python -S -m venv --system-site-packages --without-pip $Environment
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$projectPython = Join-Path (Get-Location) "$Environment\Scripts\python.exe"
& $projectPython -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed' }
