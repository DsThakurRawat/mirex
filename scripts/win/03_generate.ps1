# Step 03 - self-generation campaign. Mureka, MiniMax, YuE and ACE-Step have
# NO public training data; they are four of six graded strata.
. "$PSScriptRoot\env.ps1"
Need-Venv
if (-not $env:MUREKA_API_KEY)  { Die "set `$env:MUREKA_API_KEY first" }
if (-not $env:MINIMAX_API_KEY) { Die "set `$env:MINIMAX_API_KEY first" }
Set-Location "$MirexRoot\src"

Say "dry run - checks credentials, spends nothing"
& $PY -m generation.campaign --backend mureka --count 10 --dry-run

Say "ACE-Step 2000 (open model, local GPU)"
& $PY -m generation.campaign --backend ace_step --count 2000 --workers 1
Say "YuE 800"
& $PY -m generation.campaign --backend yue --count 800 --workers 1
Say "Mureka 400 (API)"
& $PY -m generation.campaign --backend mureka --count 400 --workers 4
Say "MiniMax 400 (API)"
& $PY -m generation.campaign --backend minimax --count 400 --workers 2

Say "transcoding generated WAV to FLAC (lossless, saves ~40%)"
Warn "never MP3 here - a lossy codec on the AI class only manufactures the"
Warn "exact confound the delivery-chain simulator exists to destroy"
$before = Dir-SizeGB "$env:MIREX_DATA_DIR\generated"
Get-ChildItem "$env:MIREX_DATA_DIR\generated" -Recurse -Filter *.wav |
  ForEach-Object -ThrottleLimit ([int]$env:NUMBER_OF_PROCESSORS) -Parallel {
    $flac = $_.FullName -replace '\.wav$', '.flac'
    & ffmpeg -v error -y -i $_.FullName $flac
    if ($LASTEXITCODE -eq 0) { Remove-Item $_.FullName -Force }
  }
Ok "generated pool: $before GB -> $(Dir-SizeGB "$env:MIREX_DATA_DIR\generated") GB"

Set-Location $MirexRoot
& $PY src\data_fetch.py --dataset all --register-only
Ok "generation complete - next: scripts\win\04_quarantine_gate.ps1"
