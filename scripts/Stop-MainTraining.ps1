$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$checkpointDirectory = Join-Path $projectRoot 'checkpoints/main_resnet'
if (-not (Test-Path -LiteralPath $checkpointDirectory)) { throw 'Main training has not started.' }
New-Item -ItemType File -Force -Path (Join-Path $checkpointDirectory 'STOP') | Out-Null
Write-Output 'Stop requested. Training saves after the current batch, then exits. Check reports/main_resnet_progress.json.'
