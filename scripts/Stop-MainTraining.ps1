param([ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')][string]$Run = 'main_resnet_adam')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$checkpointDirectory = Join-Path $projectRoot "checkpoints/$Run"
if (-not (Test-Path -LiteralPath $checkpointDirectory)) { throw "Training run $Run has not started." }
New-Item -ItemType File -Force -Path (Join-Path $checkpointDirectory 'STOP') | Out-Null
Write-Output "Stop requested for $Run. Check reports/${Run}_progress.json for saved state and process exit."
