# Step 08 - stacked fusion + isotonic calibration on LOGO out-of-fold scores.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot
$oof = if ($args.Count -gt 0) { $args[0] } else { "oof_scores.jsonl" }

if (-not (Test-Path $oof)) {
  Warn "MISSING PIECE: $oof does not exist."
  Warn "StackedFusion.fit_from_logo() needs rows shaped"
  Warn '  {"scores": {"a":0.9,...,"e":0.3}, "label": 1, "fold": "suno"}'
  Warn "but nothing in the repo generates them yet. You need a script that,"
  Warn "for each fold, loads that fold's held-out tracks and scores them with"
  Warn "that fold's five checkpoints. This is the one remaining gap."
  Die "see RUNBOOK.md step 09"
}
Say "fitting fusion on $oof"
& $PY scripts\lib\fit_fusion.py $oof | Tee-Object "$LOGS\fusion.log"
Ok "fusion fitted - select on worst-stratum and macro AUROC, never pooled accuracy"
