# Shared environment for the MIREX PowerShell pipeline. Dot-source, don't run:
#   . .\scripts\win\env.ps1
$ErrorActionPreference = "Stop"

# PowerShell 7+ required: ForEach-Object -Parallel (03) and ?? (05).
if ($PSVersionTable.PSVersion.Major -lt 7) {
  Write-Host " fail PowerShell 7+ required (you have $($PSVersionTable.PSVersion))." -ForegroundColor Red
  Write-Host "      winget install Microsoft.PowerShell   then run with: pwsh" -ForegroundColor Red
  exit 1
}

$script:MirexRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$env:MIREX_ROOT = $MirexRoot

# Point these at your fast drive. THE DEFAULT PUTS 500 GB IN THE REPO.
if (-not $env:MIREX_DATA_DIR)       { $env:MIREX_DATA_DIR       = "$MirexRoot\data" }
if (-not $env:MIREX_CHECKPOINT_DIR) { $env:MIREX_CHECKPOINT_DIR = "$MirexRoot\checkpoints" }
if (-not $env:HF_HOME)              { $env:HF_HOME              = "$MirexRoot\hf_cache" }
if (-not $env:MTG_CAP_GB)           { $env:MTG_CAP_GB           = "190" }

$script:PY   = "$MirexRoot\.venv\Scripts\python.exe"
$script:PIP  = "$MirexRoot\.venv\Scripts\pip.exe"
$script:LOGS = "$MirexRoot\logs"
New-Item -ItemType Directory -Force -Path $LOGS | Out-Null

function Say  ($m) { Write-Host "==> $m" -ForegroundColor White }
function Ok   ($m) { Write-Host "  ok $m"  -ForegroundColor Green }
function Warn ($m) { Write-Host " warn $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host " fail $m" -ForegroundColor Red; exit 1 }
function Need-Venv { if (-not (Test-Path $PY)) { Die "no venv - run scripts\win\01_setup.ps1 first" } }
function Dir-SizeGB ($p) {
  if (-not (Test-Path $p)) { return 0 }
  [math]::Floor((Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object Length -Sum).Sum / 1GB)
}
