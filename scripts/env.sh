#!/usr/bin/env bash
# Shared environment for every MIREX script. Sourced, not run.
# Override any of these by exporting them before calling a script.

MIREX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MIREX_ROOT

# Point these at your fast array. THE DEFAULT PUTS 500 GB IN THE REPO — change it.
export MIREX_DATA_DIR="${MIREX_DATA_DIR:-$MIREX_ROOT/data}"
export MIREX_CHECKPOINT_DIR="${MIREX_CHECKPOINT_DIR:-$MIREX_ROOT/checkpoints}"
export HF_HOME="${HF_HOME:-$MIREX_ROOT/hf_cache}"

PY="$MIREX_ROOT/.venv/bin/python"
PYTEST="$MIREX_ROOT/.venv/bin/pytest"
LOGS="$MIREX_ROOT/logs"
export PY PYTEST LOGS
mkdir -p "$LOGS"

# MTG-Jamendo full-quality cap for the 501 GB sample (GB).
export MTG_CAP_GB="${MTG_CAP_GB:-190}"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'
c_bld=$'\033[1m';  c_off=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$c_bld" "$c_off" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s warn%s %s\n' "$c_yel" "$c_off" "$*"; }
die()  { printf '%s fail%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

need_venv() { [ -x "$PY" ] || die "no venv — run scripts/01_setup.sh first"; }
