$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$projectPython = Join-Path (Get-Location) '.venv-run\Scripts\python.exe'
$phaseConfig = Get-Content -LiteralPath 'configs\phase1.json' -Raw | ConvertFrom-Json

function Invoke-Checked([string]$Name, [string[]]$PythonArgs) {
    & $projectPython scripts/run_checked.py $Name @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Name (see logs/$Name.log)" }
}

Invoke-Checked 'acquire' @('scripts/acquire.py')
Invoke-Checked 'environment' @('scripts/environment.py')
if (-not (Test-Path -LiteralPath 'data/manifests/sources.csv')) {
    Invoke-Checked 'inventory' @('-m', 'phase1.data', 'inventory')
}
if (-not (Test-Path -LiteralPath 'data/manifests/smoke_variants.csv')) {
    Invoke-Checked 'prepare-smoke' @('-m', 'phase1.data', 'prepare', '--run', 'smoke', '--identities', "$($phaseConfig.smoke.identities)", '--per-identity', "$($phaseConfig.smoke.images_per_identity)", '--extras')
}
Invoke-Checked 'tests' @('-m', 'pytest', '-q')
if (-not (Test-Path -LiteralPath 'checkpoints/smoke_resnet/last.pt')) {
    Invoke-Checked 'train-smoke' @('-m', 'phase1.train', '--run', 'smoke_resnet', '--epochs', '1')
}
Invoke-Checked 'resume-smoke' @('-m', 'phase1.train', '--run', 'smoke_resnet', '--epochs', '2', '--resume', 'checkpoints/smoke_resnet/last.pt')
foreach ($modelName in @('farl', 'lvface', 'resnet50')) {
    Invoke-Checked "extract-smoke-$modelName" @('-m', 'phase1.extract', '--run', 'smoke', '--model', $modelName, '--batch-size', "$($phaseConfig.feature_batch_size)", '--threads', "$($phaseConfig.torch_threads)")
}
Invoke-Checked 'validate-smoke' @('-m', 'phase1.validate', '--run', 'smoke')
