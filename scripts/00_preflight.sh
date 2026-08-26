#!/usr/bin/env bash
# Step 00 — check the machine before committing to anything.
set -euo pipefail
source "$(dirname "$0")/env.sh"

say "GPUs"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
NGPU=$(nvidia-smi -L | wc -l)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ok "$NGPU GPU(s), compute capability $(echo "$CAP" | sed 's/./&./1')"
if [ "$CAP" -lt 80 ]; then
  warn "no native bf16 (pre-Ampere). ACE-Step will auto-select float32."
  warn "fp16 training is fine — Lightning precision=16-mixed works on Volta."
fi

say "Storage"
df -h "$MIREX_DATA_DIR" 2>/dev/null || df -h "$MIREX_ROOT"
AVAIL_GB=$(df -BG --output=avail "$(dirname "$MIREX_DATA_DIR")" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 600 ]; then
  warn "only ${AVAIL_GB} GB free — the 501 GB sample plus harness cache needs ~700 GB"
else
  ok "${AVAIL_GB} GB free"
fi
lsblk -d -o NAME,SIZE,TYPE,MODEL 2>/dev/null | grep -v '^loop' || true
warn "confirm the RAID level: RAID 0 has NO redundancy"

say "Tools"
command -v ffmpeg >/dev/null || die "ffmpeg missing — required by the simulator and tests"
ok "ffmpeg $(ffmpeg -version | head -1 | cut -d' ' -f3)"
ok "$(nproc) cores, $(free -g | awk '/^Mem:/{print $2}') GB RAM"

say "Paths"
printf '  MIREX_DATA_DIR       %s\n' "$MIREX_DATA_DIR"
printf '  MIREX_CHECKPOINT_DIR %s\n' "$MIREX_CHECKPOINT_DIR"
printf '  HF_HOME              %s\n' "$HF_HOME"
case "$MIREX_DATA_DIR" in
  "$MIREX_ROOT"/*) warn "MIREX_DATA_DIR is inside the repo — point it at your array" ;;
esac
ok "preflight done"
