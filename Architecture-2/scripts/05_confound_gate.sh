#!/usr/bin/env bash
# Step 05 — HARD GATE. Probes whether labels are predictable from non-content
# features (bitrate, sample rate, loudness...). Must stay under 0.60 AUROC.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

say "running the confound audit (a few minutes; decodes AUDIT_N=${AUDIT_N:-4000} tracks)"
if "$PY" scripts/lib/confound_gate.py 2>&1 | tee "$LOGS/confound_gate.log"; then
  ok "confound gate PASSED"
else
  warn "confound gate FAILED"
  warn "suspect the SUBSET COMPOSITION before the model:"
  warn "  - MTG audio tier (audio-low is transcoded -> bitrate floor on the real class)"
  warn "  - FMA clip length (fma_large/medium are 30 s -> duration signature)"
  warn "the per-feature leak table above names the culprit"
  die "not proceeding to training"
fi
