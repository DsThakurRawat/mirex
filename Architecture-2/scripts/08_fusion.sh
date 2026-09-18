#!/usr/bin/env bash
# Step 08 — stacked fusion + isotonic calibration on LOGO out-of-fold scores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

OOF="${1:-oof_scores.jsonl}"
if [ ! -f "$OOF" ]; then
  warn "MISSING PIECE: $OOF does not exist."
  warn "StackedFusion.fit_from_logo() needs rows shaped"
  warn '  {"scores": {"a":0.9,...,"e":0.3}, "label": 1, "fold": "suno"}'
  warn "but nothing in the repo generates them yet. You need a script that,"
  warn "for each fold, loads that fold's held-out tracks and scores them with"
  warn "that fold's five checkpoints. This is the one remaining gap."
  die "see RUNBOOK.md step 09"
fi
say "fitting fusion on $OOF"
"$PY" scripts/lib/fit_fusion.py "$OOF" 2>&1 | tee "$LOGS/fusion.log"
ok "fusion fitted — select on worst-stratum and macro AUROC, never pooled accuracy"
