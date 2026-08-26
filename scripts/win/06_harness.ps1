# Step 06 - freeze the dev set and materialize the strata grid.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot
Say "freezing dev set and materializing strata"
& $PY scripts\lib\harness_build.py | Tee-Object "$LOGS\harness.log"
Ok "harness ready - regenerable, delete $env:MIREX_DATA_DIR\harness_cache to reclaim space"
