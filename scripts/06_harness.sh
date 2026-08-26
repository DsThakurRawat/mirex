#!/usr/bin/env bash
# Step 06 — freeze the dev set (never trained on) and materialize the
# 6 conditions x 4 excerpt lengths strata grid. ~17 GB per family.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

say "freezing dev set and materializing strata"
"$PY" scripts/lib/harness_build.py 2>&1 | tee "$LOGS/harness.log"
ok "harness ready — regenerable, delete $MIREX_DATA_DIR/harness_cache to reclaim space"
