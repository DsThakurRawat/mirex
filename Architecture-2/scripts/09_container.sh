#!/usr/bin/env bash
# Step 09 — build the offline submission container and rehearse the runtime.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

say "pre-populating hf_cache (the Dockerfile COPYs it; build fails if absent)"
mkdir -p hf_cache
HF_HOME="$MIREX_ROOT/hf_cache" "$PY" - <<'PY'
from transformers import AutoModel
AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m")
AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
print("  hf cache populated")
PY

say "building image"
docker build -t mirex2026-detector . 2>&1 | tail -5
ok "built mirex2026-detector"

REHEARSAL="${REHEARSAL_DIR:-$MIREX_DATA_DIR/rehearsal}"
if [ -d "$REHEARSAL" ]; then
  N=$(find "$REHEARSAL" -name '*.wav' | wc -l)
  say "runtime rehearsal: $N tracks, ONE gpu, --network none (must beat 24 h)"
  mkdir -p /tmp/mirex_out
  START=$(date +%s)
  docker run --gpus '"device=0"' --network none \
      -v "$REHEARSAL:/data/input" -v /tmp/mirex_out:/data/output \
      mirex2026-detector
  EL=$(( $(date +%s) - START ))
  ok "scored $N tracks in $(( EL / 60 )) min — extrapolated 10k: $(( EL * 10000 / (N>0?N:1) / 3600 )) h"
  head -3 /tmp/mirex_out/scores.csv
else
  warn "no rehearsal dir at $REHEARSAL — set REHEARSAL_DIR to a folder of WAVs"
  warn "the 10k-track rehearsal on one GPU is a submission requirement"
fi
