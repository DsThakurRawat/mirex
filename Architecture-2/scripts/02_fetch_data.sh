#!/usr/bin/env bash
# Step 02 — the 501 GB sample. Resumable: re-run safely after any interruption.
#
# Allocation buys generator-family diversity, not volume. Small multi-generator
# sets go in full; the giant single-family dumps get cut hard.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

FETCH=("$PY" src/data_fetch.py)
run() { say "$*"; "${FETCH[@]}" "$@" 2>&1 | tee -a "$LOGS/fetch.log" | tail -2; }

say "AI pool — small, high family diversity, in full (109 GB)"
run --dataset echoes                       #   8.0 GB · ~10 systems
run --dataset fakemusiccaps                #  12.9 GB · 5 TTM models
run --dataset sonics                       #  30.4 GB · suno + udio
run --dataset aime                         #  58.0 GB · 12 models

say "AI pool — giant single-family dumps, cut hard (90 GB)"
run --dataset suno_audio --subset-gb 40    # keep version labels for the drift proxy
run --dataset muse       --subset-gb 25    # 576 GB on the hub, one family
run --dataset udio       --subset-gb 25    # 583 GB on the hub, one family

say "Real pool (33 GB + MTG below)"
run --dataset musicnet                     #  11.1 GB · full-length classical
run --dataset fma        --subset-gb 22    #  medium tier
run --dataset mtg_jamendo                  #  metadata TSVs only

say "SDD — metadata ONLY, audio is quarantined"
run --dataset sdd

# --- MTG-Jamendo audio: capped full-quality subset ------------------------
MTG_AUDIO="$MIREX_DATA_DIR/raw/mtg_jamendo/audio"
CAP_BYTES=$(( MTG_CAP_GB * 1024 * 1024 * 1024 ))
CUR=$( [ -d "$MTG_AUDIO" ] && du -sb "$MTG_AUDIO" 2>/dev/null | cut -f1 || echo 0 )

if [ "$CUR" -ge "$CAP_BYTES" ]; then
  ok "MTG audio already at $(( CUR / 1024**3 )) GB (cap ${MTG_CAP_GB} GB)"
else
  say "MTG-Jamendo audio — full quality, stopping at ${MTG_CAP_GB} GB"
  warn "NOT using the audio-low tier: transcoded real class fails the confound gate"
  [ -d /tmp/mtgj ] || git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git /tmp/mtgj
  mkdir -p "$MTG_AUDIO"
  python3 /tmp/mtgj/scripts/download/download.py \
      --dataset raw_30s --type audio "$MTG_AUDIO" \
      >> "$LOGS/mtg_download.log" 2>&1 &
  DL=$!
  while kill -0 "$DL" 2>/dev/null; do
    sleep 60
    SZ=$(du -sb "$MTG_AUDIO" 2>/dev/null | cut -f1 || echo 0)
    printf '\r  MTG audio: %d / %d GB' $(( SZ / 1024**3 )) "$MTG_CAP_GB"
    if [ "$SZ" -ge "$CAP_BYTES" ]; then
      echo; say "cap reached — stopping the MTG downloader"
      kill "$DL" 2>/dev/null || true; wait "$DL" 2>/dev/null || true
      break
    fi
  done
  echo; ok "MTG audio at $(( $(du -sb "$MTG_AUDIO" | cut -f1) / 1024**3 )) GB"
fi

say "registering everything into the metadata DB"
"$PY" src/data_fetch.py --dataset all --register-only 2>&1 | tail -5

say "census"
"$PY" - <<'PY'
import sys; sys.path.insert(0, 'src')
from metadata_db import MetadataDatabase
db = MetadataDatabase()
rows = db.fetch()
from collections import Counter
print(f"  total tracks: {len(rows)}")
print(f"  AI / real   : {sum(r['is_ai'] for r in rows)} / {sum(1-r['is_ai'] for r in rows)}")
for fam, n in Counter(r['generator_family'] for r in rows).most_common():
    print(f"    {str(fam):16s} {n}")
PY
ok "data step complete — next: scripts/03_generate.sh"
