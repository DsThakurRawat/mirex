#!/usr/bin/env bash
# Step 03 — self-generation campaign. Mureka, MiniMax, YuE and ACE-Step have
# NO public training data: they exist in your training set only if this runs,
# and they are four of six graded strata under macro-AUROC.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT/src"

: "${MUREKA_API_KEY:?export MUREKA_API_KEY first (or comment out the mureka line)}"
: "${MINIMAX_API_KEY:?export MINIMAX_API_KEY first (or comment out the minimax line)}"

say "dry run — checks credentials, spends nothing"
"$PY" -m generation.campaign --backend mureka --count 10 --dry-run 2>&1 | tail -3

# Counts scaled to the 501 GB budget (~79 GB as FLAC). Resumable: the campaign
# keys on a job index, so raising --count extends it rather than redoing work.
say "ACE-Step 2000 (open model, local GPU)"
"$PY" -m generation.campaign --backend ace_step --count 2000 --workers 1 2>&1 | tee -a "$LOGS/gen_ace.log" | tail -3
say "YuE 800 (open model, slower)"
"$PY" -m generation.campaign --backend yue --count 800 --workers 1 2>&1 | tee -a "$LOGS/gen_yue.log" | tail -3
say "Mureka 400 (API)"
"$PY" -m generation.campaign --backend mureka --count 400 --workers 4 2>&1 | tee -a "$LOGS/gen_mureka.log" | tail -3
say "MiniMax 400 (API)"
"$PY" -m generation.campaign --backend minimax --count 400 --workers 2 2>&1 | tee -a "$LOGS/gen_minimax.log" | tail -3

say "transcoding generated WAV to FLAC (lossless, saves ~40%)"
warn "never MP3 here — a lossy codec on the AI class only manufactures the"
warn "exact confound the delivery-chain simulator exists to destroy"
BEFORE=$(du -sb "$MIREX_DATA_DIR/generated" 2>/dev/null | cut -f1 || echo 0)
find "$MIREX_DATA_DIR/generated" -name '*.wav' -print0 \
  | xargs -0 -r -P "$(nproc)" -I{} sh -c 'ffmpeg -v error -y -i "$1" "${1%.wav}.flac" && rm "$1"' _ {}
AFTER=$(du -sb "$MIREX_DATA_DIR/generated" 2>/dev/null | cut -f1 || echo 0)
ok "generated pool: $(( BEFORE / 1024**3 )) GB -> $(( AFTER / 1024**3 )) GB"

say "re-registering generated tracks"
cd "$MIREX_ROOT" && "$PY" src/data_fetch.py --dataset all --register-only 2>&1 | tail -3
ok "generation complete — next: scripts/04_quarantine_gate.sh"
