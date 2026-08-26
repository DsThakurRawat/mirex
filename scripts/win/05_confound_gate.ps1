# Step 05 - HARD GATE. Probe AUROC on non-content features must stay < 0.60.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

Say "running the confound audit (decodes AUDIT_N=$($env:AUDIT_N ?? '4000') tracks)"
& $PY scripts\lib\confound_gate.py | Tee-Object "$LOGS\confound_gate.log"
if ($LASTEXITCODE -ne 0) {
  Warn "confound gate FAILED"
  Warn "suspect the SUBSET COMPOSITION before the model:"
  Warn "  - MTG audio tier (audio-low is transcoded -> bitrate floor on the real class)"
  Warn "  - FMA clip length (fma_large/medium are 30 s -> duration signature)"
  Die "not proceeding to training"
}
Ok "confound gate PASSED"
