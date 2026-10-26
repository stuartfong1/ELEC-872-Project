param(
    [ValidateRange(1, 10000)][int]$Epochs = 2,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')][string]$Run = 'main_resnet_adam',
    [ValidateRange(0, 128)][int]$Gpu = 0,
    [ValidateSet('adam', 'sgd')][string]$Optimizer = 'adam',
    [string]$Manifest = 'data/manifests/main_sources.csv',
    [ValidateRange(1, 65536)][int]$BatchSize = 2,
    [double]$LearningRate = 0.001,
    [string]$InitializeFrom = '',
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot '.venv-cuda/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Create .venv-cuda with Setup-Environment.ps1, or pass -PythonPath for a CUDA environment.' }
$active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match '-m phase1\.(train|main_train)(\s|$)'
}
if ($active) { throw "Training is already running: PID $($active.ProcessId -join ', ')" }
$reportPath = Join-Path $projectRoot "reports/${Run}_training.json"
if (Test-Path -LiteralPath $reportPath) {
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $savedOptimizer = if ($report.config.optimizer) { $report.config.optimizer } else { 'sgd' }
    if ($savedOptimizer -ne $Optimizer) { throw 'Existing run uses another optimizer. Choose a new -Run.' }
    if (@($report.history | Where-Object { $_.epoch -ge $Epochs }).Count -gt 0) {
        Write-Output "Requested $Epochs epochs already completed with validation. No training started."
        return
    }
}
$checkpointPath = Join-Path $projectRoot "checkpoints/$Run/last.pt"
if ($InitializeFrom -and (Test-Path -LiteralPath $checkpointPath)) { throw '-InitializeFrom requires a new -Run.' }
# Check CUDA synchronously before launching a hidden background process.
& $PythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA is required; no CPU fallback'; assert $Gpu < torch.cuda.device_count(), 'Invalid GPU index'"
if ($LASTEXITCODE -ne 0) { throw 'CUDA is unavailable. No training was launched.' }
$stopPath = Join-Path $projectRoot "checkpoints/$Run/STOP"
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $projectRoot "logs/${Run}_$stamp.stdout.log"
$stderrPath = Join-Path $projectRoot "logs/${Run}_$stamp.stderr.log"
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'logs') | Out-Null
$trainingArguments = @('-u', '-m', 'phase1.train', '--epochs', "$Epochs", '--run', $Run,
    '--gpu', "$Gpu", '--optimizer', $Optimizer, '--manifest', $Manifest,
    '--batch-size', "$BatchSize", '--lr', "$LearningRate")
if (Test-Path -LiteralPath $checkpointPath) {
    $trainingArguments += @('--resume', "checkpoints/$Run/last.pt")
} elseif ($InitializeFrom) {
    $trainingArguments += @('--initialize-from', $InitializeFrom)
}
# Start-Process joins its arguments on Windows; quote file arguments containing spaces.
$quotedArguments = @($trainingArguments | ForEach-Object {
    if ($_ -match '[\s"]') {
        if ($_ -match '"') { throw 'Argument paths must not contain double quotes.' }
        '"' + $_ + '"'
    } else { $_ }
})
$process = Start-Process -FilePath $PythonPath -ArgumentList $quotedArguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record = [ordered]@{
    pid = $process.Id; launched_utc = (Get-Date).ToUniversalTime().ToString('o')
    python = $PythonPath; arguments = $trainingArguments; run = $Run
    stdout = $stdoutPath; stderr = $stderrPath
    progress = (Join-Path $projectRoot "reports/${Run}_progress.json"); checkpoint = $checkpointPath
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot "logs/${Run}_launch.json") -Encoding UTF8
$record | ConvertTo-Json
