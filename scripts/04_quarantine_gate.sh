#!/usr/bin/env bash
# Step 04 — HARD GATE. Blocks SDD (subset of MTG-Jamendo split-0 test) from
# contaminating training. Zero overlap required; nothing downstream is valid
# until this passes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

say "building the quarantine blocklist"
"$PY" src/quarantine.py build 2>&1 | tee "$LOGS/quarantine_build.log" | tail -5

say "verifying zero overlap"
if "$PY" src/quarantine.py verify 2>&1 | tee "$LOGS/quarantine_verify.log" | tail -12; then
  ok "quarantine gate PASSED"
else
  die "quarantine gate FAILED — do not train. See $LOGS/quarantine_verify.log"
fi
