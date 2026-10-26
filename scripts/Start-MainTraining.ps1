param([ValidateRange(1, 10000)][int]$Epochs = 2)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv-run/Scripts/python.exe'
$active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -like '*-m phase1.main_train*'
}
if ($active) { throw "Main training is already running: PID $($active.ProcessId -join ', ')" }
$reportPath = Join-Path $projectRoot 'reports/main_resnet_training.json'
if (Test-Path -LiteralPath $reportPath) {
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    if (@($report.history | Where-Object { $_.epoch -ge $Epochs }).Count -gt 0) {
        Write-Output "Requested $Epochs epochs already completed with validation. No training started."
        return
    }
}
$checkpointPath = Join-Path $projectRoot 'checkpoints/main_resnet/last.pt'
$stopPath = Join-Path $projectRoot 'checkpoints/main_resnet/STOP'
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $projectRoot "logs/main_resnet_$stamp.stdout.log"
$stderrPath = Join-Path $projectRoot "logs/main_resnet_$stamp.stderr.log"
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'logs') | Out-Null
$trainingArguments = @('-u', '-m', 'phase1.main_train', '--epochs', "$Epochs")
if (Test-Path -LiteralPath $checkpointPath) {
    $trainingArguments += @('--resume', 'checkpoints/main_resnet/last.pt')
}
$process = Start-Process -FilePath $pythonPath -ArgumentList $trainingArguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record = [ordered]@{
    pid = $process.Id
    launched_utc = (Get-Date).ToUniversalTime().ToString('o')
    python = $pythonPath
    arguments = $trainingArguments
    stdout = $stdoutPath
    stderr = $stderrPath
    progress = (Join-Path $projectRoot 'reports/main_resnet_progress.json')
    checkpoint = $checkpointPath
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'logs/main_resnet_launch.json') -Encoding UTF8
$record | ConvertTo-Json
