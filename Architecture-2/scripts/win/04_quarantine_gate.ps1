# Step 04 - HARD GATE. Zero SDD overlap required.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

Say "building the quarantine blocklist"
& $PY src\quarantine.py build | Tee-Object "$LOGS\quarantine_build.log"

Say "verifying zero overlap"
& $PY src\quarantine.py verify | Tee-Object "$LOGS\quarantine_verify.log"
if ($LASTEXITCODE -ne 0) { Die "quarantine gate FAILED - do not train" }
Ok "quarantine gate PASSED"
