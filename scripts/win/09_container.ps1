# Step 09 - build the offline submission container and rehearse the runtime.
# Needs Docker Desktop with the WSL2 backend and GPU support enabled.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

Say "pre-populating hf_cache (the Dockerfile COPYs it; build fails if absent)"
New-Item -ItemType Directory -Force -Path "$MirexRoot\hf_cache" | Out-Null
$env:HF_HOME = "$MirexRoot\hf_cache"
& $PY -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m'); AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True); print('  hf cache populated')"

Say "building image"
docker build -t mirex2026-detector .
if ($LASTEXITCODE -ne 0) { Die "docker build failed" }
Ok "built mirex2026-detector"

$rehearsal = if ($env:REHEARSAL_DIR) { $env:REHEARSAL_DIR } else { "$env:MIREX_DATA_DIR\rehearsal" }
if (Test-Path $rehearsal) {
  $count = @(Get-ChildItem $rehearsal -Recurse -Filter *.wav).Count
  Say "runtime rehearsal: $count tracks, ONE gpu, --network none (must beat 24 h)"
  New-Item -ItemType Directory -Force -Path "$env:TEMP\mirex_out" | Out-Null
  $t0 = Get-Date
  docker run --gpus '"device=0"' --network none `
      -v "${rehearsal}:/data/input" -v "$env:TEMP\mirex_out:/data/output" `
      mirex2026-detector
  $el = ((Get-Date) - $t0).TotalSeconds
  if ($count -gt 0) {
    Ok "scored $count tracks in $([int]($el/60)) min - extrapolated 10k: $([math]::Round($el*10000/$count/3600,1)) h"
  }
  Get-Content "$env:TEMP\mirex_out\scores.csv" -TotalCount 3
} else {
  Warn "no rehearsal dir at $rehearsal - set `$env:REHEARSAL_DIR to a folder of WAVs"
  Warn "the 10k-track rehearsal on one GPU is a submission requirement"
}
