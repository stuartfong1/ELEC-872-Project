param([ValidateRange(0, 128)][int]$Gpu = 0, [string]$Run = 'smoke_adam')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$projectPython = Join-Path (Get-Location) '.venv-cuda\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run Setup-Environment.ps1 first.' }
& $projectPython scripts/acquire.py
if ($LASTEXITCODE -ne 0) { throw 'Data/model acquisition failed.' }
if (-not (Test-Path -LiteralPath 'data/manifests/sources.csv')) {
    & $projectPython -m phase1.data inventory
    if ($LASTEXITCODE -ne 0) { throw 'Inventory failed.' }
}
& $projectPython scripts/execute_verified.py --run $Run --gpu $Gpu
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification run failed.' }
