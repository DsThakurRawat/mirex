#!/usr/bin/env bash
# Step 01 — venv, dependencies, and the test suite.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MIREX_ROOT"

if [ ! -x "$PY" ]; then
  say "creating venv"
  python3 -m venv .venv
fi
say "installing dependencies"
"$MIREX_ROOT/.venv/bin/pip" install --upgrade pip -q
"$MIREX_ROOT/.venv/bin/pip" install -r requirements.txt -q
ok "dependencies installed"

say "creating directory tree"
"$PY" -c "import sys; sys.path.insert(0,'src'); import config; config.ensure_dirs(); print(config.DATA_DIR)"

say "running tests (expect 107 passed, ~2 min)"
"$PYTEST" tests/ -q 2>&1 | tail -3
ok "setup complete"
