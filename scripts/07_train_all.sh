#!/usr/bin/env bash
# Step 07 — train every branch on every LOGO fold.
#
# 5 branches x 7 folds = 35 jobs. train.py hardcodes devices=1, so one job owns
# one GPU; this schedules 35 jobs across however many GPUs the box has.
# Batch sizes come from config.BRANCHES (a=16 b=8 c=32 d=4 e=32) — do not
# override them unless a job OOMs. AMP is already on (Lightning 16-mixed).
#
# Resumable: a fold with an existing checkpoint is skipped.
#   BRANCHES=ab ./scripts/07_train_all.sh    # only branches a and b
#   EPOCHS=20   ./scripts/07_train_all.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

BRANCHES="${BRANCHES:-abcde}"
FOLDS=(suno udio mureka minimax yue ace-step none)
EPOCHS="${EPOCHS:-10}"
WORKERS="${WORKERS:-6}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
[ "$NGPU" -ge 1 ] || die "no GPUs visible"

say "smoke test first (CPU, seconds) — validates the pipeline before GPU-weeks"
"$PY" src/train.py --branch a --holdout suno --smoke 2>&1 | tail -3
ok "smoke passed"

# --- build the job list, skipping folds already trained -------------------
JOBS=()
for (( i=0; i<${#BRANCHES}; i++ )); do
  b="${BRANCHES:$i:1}"
  for f in "${FOLDS[@]}"; do
    dir="$MIREX_CHECKPOINT_DIR/$b/$([ "$f" = none ] && echo full || echo "logo_$f")"
    if compgen -G "$dir/*.ckpt" > /dev/null; then
      ok "skip $b/$f (checkpoint exists)"
    else
      JOBS+=("$b:$f")
    fi
  done
done

TOTAL=${#JOBS[@]}
[ "$TOTAL" -gt 0 ] || { ok "every fold already trained"; exit 0; }
say "$TOTAL job(s) to run across $NGPU GPU(s), $EPOCHS epochs each"

# --- GPU slot scheduler: a FIFO holds one token per free GPU --------------
FIFO=$(mktemp -u); mkfifo "$FIFO"; exec 3<>"$FIFO"; rm -f "$FIFO"
for (( g=0; g<NGPU; g++ )); do echo "$g" >&3; done

START=$(date +%s); n=0
for job in "${JOBS[@]}"; do
  read -r -u 3 gpu
  n=$(( n + 1 ))
  b="${job%%:*}"; f="${job##*:}"
  log="$LOGS/train_${b}_${f}.log"
  printf '  [%2d/%2d] gpu%s  branch %s  holdout %-9s -> %s\n' "$n" "$TOTAL" "$gpu" "$b" "$f" "$log"
  (
    if CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/train.py \
         --branch "$b" --holdout "$f" \
         --epochs "$EPOCHS" --workers "$WORKERS" > "$log" 2>&1
    then printf '%s  done%s  %s/%s\n' "$c_grn" "$c_off" "$b" "$f"
    else printf '%s  FAIL%s  %s/%s — see %s\n' "$c_red" "$c_off" "$b" "$f" "$log"
    fi
    echo "$gpu" >&3
  ) &
done
wait
exec 3>&-

say "training finished in $(( ($(date +%s) - START) / 60 )) min"
say "checkpoints"
find "$MIREX_CHECKPOINT_DIR" -name '*.ckpt' | sed 's|.*/checkpoints/|  |' | sort
FAILED=$(grep -l "Traceback" "$LOGS"/train_*.log 2>/dev/null | wc -l)
[ "$FAILED" -eq 0 ] || warn "$FAILED job log(s) contain tracebacks — check $LOGS"
ok "next: scripts/08_fusion.sh"
