# MIREX 2026 — DGX-1 Runbook

Everything needed to run this project on another machine. Written 2026-08-26.
Companion docs: `HANDOFF.md` (what's left + storage budget),
`IMPLEMENTATION_PLAN.md` (methodology, §1.1 = task-page provenance).

---

## Just run it

One cross-platform driver, `run.py`. Stdlib-only, so it works on the system
Python before the venv it creates exists — same commands on Linux, macOS and
Windows.

```bash
export MIREX_DATA_DIR=/raid/mirex/data          # POSIX
export MIREX_CHECKPOINT_DIR=/raid/mirex/checkpoints
export HF_HOME=/raid/mirex/hf_cache

python3 run.py preflight     # GPUs, compute capability, disk, ffmpeg
python3 run.py setup         # venv, deps, 107 tests
python3 run.py fetch         # the 501 GB sample + registration
export MUREKA_API_KEY=... MINIMAX_API_KEY=...
python3 run.py generate      # ACE-Step / YuE / Mureka / MiniMax -> FLAC
python3 run.py quarantine    # HARD GATE
python3 run.py confound      # HARD GATE
python3 run.py harness       # freeze dev set, materialize strata
python3 run.py train         # 35 jobs across the GPUs
python3 run.py fusion        # stacked fusion + calibration
python3 run.py container     # offline Docker + runtime rehearsal
```

`python3 run.py all` chains everything from `setup` to `fusion`, stopping if a
gate fails. Every step is resumable — re-run after any interruption.

On Windows the only change is the export syntax:

```powershell
$env:MIREX_DATA_DIR       = "D:\mirex\data"
$env:MIREX_CHECKPOINT_DIR = "D:\mirex\checkpoints"
$env:HF_HOME              = "D:\mirex\hf_cache"
python run.py preflight
```

### Useful flags

```bash
python3 run.py fetch     --mtg-cap 120        # smaller MTG slice
python3 run.py confound  --audit-n 1500       # faster gate check
python3 run.py train     --branches ab --epochs 20 --gpus 4
python3 run.py train     --workers 2          # default 6, or 2 on Windows
python3 run.py container --rehearsal /raid/mirex/rehearsal
```

### What each step does

| Step | Does | Roughly |
|---|---|---|
| `preflight` | GPU count, compute capability, free space, ffmpeg | seconds |
| `setup` | venv, dependencies, 107 tests | 5 min |
| `fetch` | 501 GB sample, MTG capped at 190 GB, registration, census | hours–days |
| `generate` | dry-run creds, four backends, WAV→FLAC | days |
| `quarantine` | **hard gate** — zero SDD overlap | minutes |
| `confound` | **hard gate** — probe AUROC < 0.60, prints leak table | ~20 min |
| `harness` | freeze dev set, materialize strata | ~1 h |
| `train` | 35 jobs, one GPU each, skips finished folds | days–weeks |
| `fusion` | stacked fusion + isotonic calibration | minutes |
| `container` | hf_cache, Docker build, offline rehearsal + timing | ~1 h |

The shell scripts in `scripts/` (bash) and `scripts/win/` (PowerShell) do the
same thing and remain for reference; `run.py` supersedes both. The Python
helpers in `scripts/lib/` are shared by all three.

The rest of this document explains what each step does and why the two gates
matter. Every script is reproduced in full in Appendix A (bash) and Appendix B
(PowerShell) at the end of this file.

---

## Windows

**Use `run.py`** — it is cross-platform by construction and needs no ports.
The rest of this section covers the shell alternatives.

The bash scripts do **not** run on native Windows. They use `mkfifo` (the GPU
scheduler), `du -sb`, `nproc`, `free`, `lsblk`, and `VAR=value command`
prefixes. Git Bash does not implement `mkfifo` usefully either, so step 07
cannot work there. If you prefer shell:

### WSL2 — recommended, scripts run unchanged

CUDA passes through to WSL2, so the Linux scripts work exactly as written and
you avoid maintaining a second copy.

```powershell
wsl --install -d Ubuntu-24.04       # then reboot if prompted
```

```bash
# inside WSL2
sudo apt update && sudo apt install -y python3-venv ffmpeg git
nvidia-smi                           # must list your GPU; if not, update the
                                     # Windows NVIDIA driver (not a WSL driver)
git clone https://github.com/DsThakurRawat/mirex.git && cd mirex
./scripts/00_preflight.sh
```

Keep the data on the Linux filesystem (`/home/...`), **not** on `/mnt/c`.
Cross-filesystem I/O in WSL2 is roughly an order of magnitude slower, and this
pipeline is I/O-bound on 500 GB of audio.

### Native PowerShell — `scripts\win\*.ps1`

Ports of all ten steps, plus `env.ps1`. **PowerShell 7+ is required**
(`env.ps1` checks and exits otherwise) because step 03 uses
`ForEach-Object -Parallel` and step 05 uses `??`:

```powershell
winget install Microsoft.PowerShell   # gives you `pwsh`
winget install Gyan.FFmpeg            # ffmpeg must be on PATH
```

```powershell
$env:MIREX_DATA_DIR       = "D:\mirex\data"
$env:MIREX_CHECKPOINT_DIR = "D:\mirex\checkpoints"
$env:HF_HOME              = "D:\mirex\hf_cache"

.\scripts\win\00_preflight.ps1
.\scripts\win\01_setup.ps1
.\scripts\win\02_fetch_data.ps1
$env:MUREKA_API_KEY="..."; $env:MINIMAX_API_KEY="..."
.\scripts\win\03_generate.ps1
.\scripts\win\04_quarantine_gate.ps1
.\scripts\win\05_confound_gate.ps1
.\scripts\win\06_harness.ps1
.\scripts\win\07_train_all.ps1
.\scripts\win\08_fusion.ps1
.\scripts\win\09_container.ps1
```

The PowerShell scheduler replaces the FIFO semaphore with a polling loop over a
`gpu -> process` table; verified to hold exactly N concurrent jobs across all
35. Four differences to know about on native Windows:

| | Linux | Windows |
|---|---|---|
| DataLoader workers | fork, `--workers 6` | spawn, defaults to `--workers 2` — raising it is often slower |
| Long paths | fine | enable `LongPathsEnabled` or HF cache paths can exceed 260 chars |
| Docker (step 09) | native | needs Docker Desktop with the WSL2 backend and GPU support |
| Symlinks | fine | HF downloads may need Developer Mode on, or they fall back to copies |

The Python helpers in `scripts/lib/` are cross-platform and shared by both.

---

## 0. Before you leave this laptop

The three source fixes and both planning docs are **uncommitted**. Without
this step the new box gets none of them.

```bash
cd /home/divyansh-rawat/mirex
git add HANDOFF.md IMPLEMENTATION_PLAN.md RUNBOOK.md paper/main.md \
        src/data_fetch.py src/baselines.py src/generation/ace_step_runner.py
git commit -m "Add DGX-1 runbook, storage budget, task-page provenance; archive cleanup + ACE-Step dtype autodetect"
git push origin master
```

`rep.cpp` is unrelated to this project — leave it out (or `rm rep.cpp`).

---

## 1. On the new box — verify hardware first

```bash
nvidia-smi                       # count GPUs, note the model and VRAM
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
lsblk                            # find the NVMe array
df -h                            # confirm free space on it
nproc && free -g
ffmpeg -version | head -1        # REQUIRED — tests and the simulator need it
```

**Record the compute capability.** It decides three things:

| Capability | Card | bf16 | Consequence |
|---|---|---|---|
| 6.0 | P100 (original DGX-1) | no | ACE-Step runs fp32; no tensor cores at all |
| 7.0 | V100 (DGX-1V) | no | ACE-Step runs fp32; fp16 AMP available for training |
| 8.0+ | A100 and newer | yes | everything native |

A DGX-1 is 6.0 or 7.0, so `resolve_dtype()` will select **float32** for
ACE-Step — correct and automatic, but ~2x the memory and slower. There is no
fp16 option: the ACE-Step pipeline accepts only `bfloat16` or `float32`.

**Confirm the RAID level.** RAID 0 across 4x1.92 TB gives ~7 TB with no
redundancy — one SSD failure destroys the pool. RAID 5 leaves ~5.3 TB.

---

## 2. Setup

```bash
git clone https://github.com/DsThakurRawat/mirex.git
cd mirex
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Point the data dir at the array — **do this in every shell**, or the 500 GB
pool lands on the OS disk:

```bash
export MIREX_DATA_DIR=/raid/mirex/data          # <-- your array mount
export MIREX_CHECKPOINT_DIR=/raid/mirex/checkpoints
export HF_HOME=/raid/mirex/hf_cache             # HF blobs are large too
```

Verify the install:

```bash
.venv/bin/pytest tests/ -q                      # expect: 107 passed
```

---

## 3. Data — the 500 GB sample

`--subset-gb` is **per dataset, not global**, so run these individually. The
allocation buys generator-family diversity rather than volume: the small
multi-generator sets go in full, the giant single-family dumps get cut hard.

```bash
cd mirex
P=".venv/bin/python src/data_fetch.py"

# --- AI: small + high family diversity, take in full (109 GB) ---
$P --dataset echoes                    #   8.0 GB, ~10 systems
$P --dataset fakemusiccaps             #  12.9 GB, 5 TTM models
$P --dataset sonics                    #  30.4 GB, suno + udio
$P --dataset aime                      #  58.0 GB, 12 models

# --- AI: giant single-family dumps, subset hard (90 GB) ---
$P --dataset suno_audio --subset-gb 40 # keep version labels (drift proxy)
$P --dataset muse       --subset-gb 25
$P --dataset udio       --subset-gb 25

# --- Real (223 GB) ---
$P --dataset musicnet                  #  11.1 GB, full-length classical
$P --dataset fma        --subset-gb 22 #  fma_medium tier
$P --dataset mtg_jamendo               #  metadata TSVs only; audio below

# --- SDD: metadata ONLY. Never fetch its audio. ---
$P --dataset sdd
```

MTG-Jamendo audio is not auto-fetched. Use their downloader, and take a
**full-quality subset** — not the `audio-low` tier:

```bash
git clone https://github.com/MTG/mtg-jamendo-dataset.git /tmp/mtgj
python /tmp/mtgj/scripts/download/download.py \
    --dataset raw_30s --type audio \
    "$MIREX_DATA_DIR/raw/mtg_jamendo/audio"
# ^ full set is 508 GB; interrupt at ~190 GB for the sample, then re-run
#   data_fetch.py --dataset mtg_jamendo --register-only
```

**Why not `audio-low` (156 GB, the whole set):** it is transcoded, so the real
class would carry a systematic bitrate floor the AI class lacks. That is the
documented 83%-precision shortcut and it will trip your own confound gate in
§6. Same reason FMA is capped: `fma_large`/`medium` are 30 s clips, so a big
FMA share stamps a duration signature on the real class.

Archives are now deleted automatically after extraction (pass
`--keep-archives` to retain them). Re-running is safe: a sentinel at
`.extracted/<name>.done` means neither re-download nor re-extract.

Register everything and check the census:

```bash
.venv/bin/python src/data_fetch.py --dataset all --register-only
```

**Known bias to record in the paper:** subset shard selection is
lexicographic, not random, so a subset of Muse/Udio is the *earliest* shards —
which usually correlates with older model versions.

---

## 4. Generation campaign — four of six graded strata

Mureka, MiniMax, YuE and ACE-Step have **no public training data**. They exist
in your training set only if this runs. Scaled to the 500 GB budget (~79 GB):

```bash
export MUREKA_API_KEY=...
export MINIMAX_API_KEY=...
export ACE_STEP_CHECKPOINT=/raid/mirex/models/ace-step   # or leave unset to auto-download
cd src

python -m generation.campaign --backend ace_step --count 2000 --workers 1
python -m generation.campaign --backend yue      --count 800  --workers 1
python -m generation.campaign --backend mureka   --count 400  --workers 4
python -m generation.campaign --backend minimax  --count 400  --workers 2
```

Dry-run an API backend first to check credentials without spending:

```bash
python -m generation.campaign --backend mureka --count 10 --dry-run
```

The campaign is resumable — it keys on a job index, so re-running with a
larger `--count` extends it rather than redoing work.

**Convert generated WAV to FLAC** (ACE-Step writes `.wav`; saves ~40%):

```bash
find "$MIREX_DATA_DIR/generated" -name '*.wav' -print0 \
  | xargs -0 -P 16 -I{} sh -c 'ffmpeg -v error -i "$1" "${1%.wav}.flac" && rm "$1"' _ {}
```

**Never store generated audio as MP3.** This detector reads codec artifacts —
a lossy codec on the AI class only manufactures the exact confound the
delivery-chain simulator exists to destroy. FLAC is lossless; MP3 is not.

---

## 5. Quarantine gate (hard gate — must pass)

```bash
.venv/bin/python src/quarantine.py build
.venv/bin/python src/quarantine.py verify     # must report ZERO overlap
```

Blocks SDD ⊂ MTG-Jamendo split-0 test contamination. If `verify` reports any
overlap, **stop** — do not train. Nothing downstream is valid until this is
clean.

---

## 6. Confound gate (hard gate — probe AUROC must be < 0.60)

No CLI exists for this; run it as a script:

```bash
cd mirex && .venv/bin/python - <<'PY'
import sys; sys.path.insert(0, 'src')
from pathlib import Path
import config
from metadata_db import MetadataDatabase
from datasets import load_audio
from simulator import DeliveryChainSimulator
from confound_audit import run_audit

db, sim = MetadataDatabase(), DeliveryChainSimulator()
rows = db.fetch("split IS NULL OR split != 'dev_frozen'")[:4000]
waves, labels, groups = [], [], []
for r in rows:
    try:
        w, sr = load_audio(r["file_path"], max_s=60)
    except Exception:
        continue
    waves.append(sim.random_chain(w, sr, item_key=r["track_id"], excerpt=False))
    labels.append(int(r["is_ai"]))
    groups.append(r["source_dataset"])

rep = run_audit(waves, labels, groups, sr=sim.sr,
                report_path=config.PROCESSED_DATA_DIR / "confound_report.json")
print("worst probe AUROC:", rep["worst_auroc"],
      "| gate:", config.CONFOUND_GATE_AUROC,
      "| passed:", rep["gate_passed"])
for k, v in sorted(rep["per_feature_auroc"].items(), key=lambda kv: -kv[1])[:5]:
    print(f"  leak: {k:24s} {v:.3f}")
PY
```

If the gate fails, **suspect the subset composition before the model** — at
500 GB the likeliest causes are the MTG audio tier and the FMA clip length
(§3). The report's per-feature leak attribution names the culprit.

---

## 7. Harness — freeze the dev set, materialize strata

```bash
cd mirex && .venv/bin/python - <<'PY'
import sys; sys.path.insert(0, 'src')
from harness import freeze_dev_set, materialize_strata
n = freeze_dev_set(per_family=150, real_n=1500)
print("frozen dev tracks:", n)
print("manifest:", materialize_strata(max_per_cell=40))
PY
```

Writes 6 conditions x 4 excerpt lengths of WAV per family (~17 GB per family;
120–260 GB total). It is regenerable — delete it any time to reclaim space.

---

## 8. Training — one LOGO fold per GPU

`train.py` has **no DDP**: one job owns one GPU. With 8 GPUs, run the 6 LOGO
folds plus the full model in parallel (7 jobs).

```bash
cd mirex
FOLDS=(suno udio mureka minimax yue ace-step none)
BRANCH=a                                   # repeat for b, c, d, e

for i in "${!FOLDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python src/train.py \
      --branch "$BRANCH" --holdout "${FOLDS[$i]}" \
      --epochs 10 --workers 6 \
      > "logs/${BRANCH}_${FOLDS[$i]}.log" 2>&1 &
done
wait
```

Validate the plumbing before committing GPU-weeks:

```bash
.venv/bin/python src/train.py --branch a --holdout suno --smoke
```

Checkpoints land in `$MIREX_CHECKPOINT_DIR/<branch>/<fold>/`.

**On V100:**
- AMP is **already on** — `train.py` sets Lightning `precision="16-mixed"`,
  which is fp16 mixed precision with an automatic grad scaler. Volta has fp16
  tensor cores, so this works and no change is needed.
- Batch sizes come from `config.BRANCHES` (a=16, b=8, c=32, d=4, e=32) and are
  tuned per branch. Don't pass a flat `--batch-size` unless a job OOMs.
- `MIREX_SMALL=1` swaps branch B to the 95M MERT variant if you need headroom.
- No flash-attn is used, which is correct: FA2 requires Ampere or newer.

---

## 9. Fusion + calibration

**Gap to close first.** `StackedFusion.fit_from_logo()` expects rows shaped
`{"scores": {branch: float}, "label": 0/1, "fold": str}` — but **nothing in
the repo generates them**. You need a short script that loads each fold's
held-out tracks, scores them with that fold's five checkpoints, and emits the
rows. That is the one piece of pipeline code still missing.

Once the OOF rows exist:

```bash
cd mirex && .venv/bin/python - <<'PY'
import sys, json; sys.path.insert(0, 'src')
from fusion import StackedFusion
oof = [json.loads(l) for l in open('oof_scores.jsonl')]
import config
f = StackedFusion()
f.fit_from_logo(oof)          # logs the a..e weights and the intercept
f.save()                      # -> config.FUSION_DIR
print("saved to", config.FUSION_DIR)
PY
```

Select on **worst-stratum and macro AUROC**, never pooled accuracy.

---

## 10. Submission container

The Dockerfile copies `hf_cache/` — populate it first or the build fails:

```bash
mkdir -p hf_cache
HF_HOME=./hf_cache .venv/bin/python -c "
from transformers import AutoModel
AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m')
AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)"

docker build -t mirex2026-detector .
docker run --gpus all \
    -v /path/test:/data/input -v /path/out:/data/output \
    mirex2026-detector --input_dir /data/input --output_csv /data/output/scores.csv
```

The container must run **fully offline** — `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` are already set in the image. Verify by running it
with networking disabled (`--network none`).

Runtime rehearsal — 10k tracks must finish well under 24 h on ONE GPU:

```bash
docker run --gpus '"device=0"' --network none \
    -v /raid/mirex/rehearsal:/data/input -v /tmp/out:/data/output \
    mirex2026-detector
```

Modes: `ensemble` (default), `rank_average`, `tta`, `single:a`..`single:e` —
the four submission slots are a risk ladder (plan §12).

---

## 11. Gotchas, collected

| Thing | Detail |
|---|---|
| `--subset-gb` | Per dataset, not global. `--dataset all --subset-gb 500` fetches 500 GB **each**. |
| Shard order | Lexicographic, not random — subsets skew to early shards / older model versions. |
| MTG `audio-low` | Don't. Transcoded real class = confound gate failure. |
| Generated MP3 | Don't. Lossy codec on the AI class only = the confound you're defending against. |
| ACE-Step dtype | Auto-resolves; `ACE_STEP_DTYPE` forces it. No fp16 path exists. |
| `train.py` | No DDP (`devices=1`), so one GPU per job. AMP is on via Lightning `16-mixed`. |
| RAID 0 | No redundancy. Back up the metadata DB and generated audio off-array. |
| `sonics` | 30.4 GB for a claimed 97k songs — verify the count after registration. |
| Env vars | `MIREX_DATA_DIR` / `MIREX_CHECKPOINT_DIR` / `HF_HOME` must be set in every shell. |
| No deadlines | Registration and submission dates are still TBD on the task page. |

## 12. Still unreleased

The organizers have announced but not shipped a training dataset and a
baseline model + checkpoint. Stubs are wired (`mirex_provided`,
`mirex_baseline` in `data_fetch.py`; `MirexProvidedBaseline` in
`baselines.py`) and are skipped by `--dataset all`. When the training set
lands it must clear the §5 quarantine gate before use — the organizers'
candidate mix names SDD as a human-music source.

Task page: https://music-ir.org/mirex/wiki/2026:AI-Generated_Music_Detection

---

## Appendix A — every bash script, in full

Reproduced verbatim from `scripts/`. Source of truth is the files themselves.

### `scripts/env.sh`

```bash
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
```

### `scripts/00_preflight.sh`

```bash
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
```

### `scripts/01_setup.sh`

```bash
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
```

### `scripts/02_fetch_data.sh`

```bash
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
```

### `scripts/03_generate.sh`

```bash
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
```

### `scripts/04_quarantine_gate.sh`

```bash
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
```

### `scripts/05_confound_gate.sh`

```bash
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
```

### `scripts/06_harness.sh`

```bash
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
```

### `scripts/07_train_all.sh`

```bash
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
```

### `scripts/08_fusion.sh`

```bash
#!/usr/bin/env bash
# Step 08 — stacked fusion + isotonic calibration on LOGO out-of-fold scores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_venv
cd "$MIREX_ROOT"

OOF="${1:-oof_scores.jsonl}"
if [ ! -f "$OOF" ]; then
  warn "MISSING PIECE: $OOF does not exist."
  warn "StackedFusion.fit_from_logo() needs rows shaped"
  warn '  {"scores": {"a":0.9,...,"e":0.3}, "label": 1, "fold": "suno"}'
  warn "but nothing in the repo generates them yet. You need a script that,"
  warn "for each fold, loads that fold's held-out tracks and scores them with"
  warn "that fold's five checkpoints. This is the one remaining gap."
  die "see RUNBOOK.md step 09"
fi
say "fitting fusion on $OOF"
"$PY" scripts/lib/fit_fusion.py "$OOF" 2>&1 | tee "$LOGS/fusion.log"
ok "fusion fitted — select on worst-stratum and macro AUROC, never pooled accuracy"
```

### `scripts/09_container.sh`

```bash
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
```

### Shared Python helpers

Used by both the bash and PowerShell pipelines.

### `scripts/lib/confound_gate.py`

```python
"""Confound audit gate (plan §4.4). Probe AUROC on non-content features must
stay below config.CONFOUND_GATE_AUROC or training is not permitted."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import config
from metadata_db import MetadataDatabase
from datasets import load_audio
from simulator import DeliveryChainSimulator
from confound_audit import run_audit

N = int(os.environ.get("AUDIT_N", "4000"))
db, sim = MetadataDatabase(), DeliveryChainSimulator()
rows = db.fetch("split IS NULL OR split != 'dev_frozen'")[:N]
if not rows:
    sys.exit("no rows in the metadata DB — run scripts/02_fetch_data.sh first")

waves, labels, groups = [], [], []
for i, r in enumerate(rows):
    try:
        w, sr = load_audio(r["file_path"], max_s=60)
    except Exception:
        continue
    waves.append(sim.random_chain(w, sr, item_key=r["track_id"], excerpt=False))
    labels.append(int(r["is_ai"]))
    groups.append(r["source_dataset"])
    if i % 500 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

rep = run_audit(waves, labels, groups, sr=sim.sr,
                report_path=config.PROCESSED_DATA_DIR / "confound_report.json")
print(f"\n  worst probe AUROC : {rep['worst_auroc']:.4f}")
print(f"  gate threshold    : {config.CONFOUND_GATE_AUROC}")
print("  top feature leaks:")
for k, v in sorted(rep["per_feature_auroc"].items(), key=lambda kv: -kv[1])[:6]:
    print(f"    {k:26s} {v:.3f}")
sys.exit(0 if rep["gate_passed"] else 1)
```

### `scripts/lib/harness_build.py`

```python
"""Freeze the dev set and materialize the condition x excerpt strata grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from harness import freeze_dev_set, materialize_strata

n = freeze_dev_set(per_family=int(os.environ.get("DEV_PER_FAMILY", "150")),
                   real_n=int(os.environ.get("DEV_REAL_N", "1500")))
print(f"  frozen dev tracks: {n}")
manifest = materialize_strata(max_per_cell=int(os.environ.get("MAX_PER_CELL", "40")))
print(f"  manifest: {manifest}")
with open(manifest) as f:
    items = f.readlines()
print(f"  materialized cells: {len(items)} items across "
      f"{len(set(__import__('json').loads(l)['stratum'] for l in items))} strata")
```

### `scripts/lib/fit_fusion.py`

```python
"""Fit stacked fusion + isotonic calibration on LOGO out-of-fold scores.

Expects oof_scores.jsonl, one row per held-out track:
    {"scores": {"a": 0.9, "b": 0.7, ...}, "label": 1, "fold": "suno"}

NOTE: nothing in the repo generates that file yet — see RUNBOOK.md step 09.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import config
from fusion import StackedFusion

path = sys.argv[1] if len(sys.argv) > 1 else "oof_scores.jsonl"
if not os.path.exists(path):
    sys.exit(f"{path} not found — the OOF scoring script still needs writing "
             "(RUNBOOK.md step 09).")
oof = [json.loads(l) for l in open(path)]
print(f"  {len(oof)} out-of-fold rows, folds: "
      f"{sorted(set(r['fold'] for r in oof))}")
f = StackedFusion()
f.fit_from_logo(oof)          # logs the a..e weights and intercept
f.save()
print(f"  saved to {config.FUSION_DIR}")
```


---

## Appendix B — every PowerShell script, in full

Reproduced verbatim from `scripts/win/`. PowerShell 7+ required.

### `scripts/win/env.ps1`

```powershell
# Shared environment for the MIREX PowerShell pipeline. Dot-source, don't run:
#   . .\scripts\win\env.ps1
$ErrorActionPreference = "Stop"

# PowerShell 7+ required: ForEach-Object -Parallel (03) and ?? (05).
if ($PSVersionTable.PSVersion.Major -lt 7) {
  Write-Host " fail PowerShell 7+ required (you have $($PSVersionTable.PSVersion))." -ForegroundColor Red
  Write-Host "      winget install Microsoft.PowerShell   then run with: pwsh" -ForegroundColor Red
  exit 1
}

$script:MirexRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$env:MIREX_ROOT = $MirexRoot

# Point these at your fast drive. THE DEFAULT PUTS 500 GB IN THE REPO.
if (-not $env:MIREX_DATA_DIR)       { $env:MIREX_DATA_DIR       = "$MirexRoot\data" }
if (-not $env:MIREX_CHECKPOINT_DIR) { $env:MIREX_CHECKPOINT_DIR = "$MirexRoot\checkpoints" }
if (-not $env:HF_HOME)              { $env:HF_HOME              = "$MirexRoot\hf_cache" }
if (-not $env:MTG_CAP_GB)           { $env:MTG_CAP_GB           = "190" }

$script:PY   = "$MirexRoot\.venv\Scripts\python.exe"
$script:PIP  = "$MirexRoot\.venv\Scripts\pip.exe"
$script:LOGS = "$MirexRoot\logs"
New-Item -ItemType Directory -Force -Path $LOGS | Out-Null

function Say  ($m) { Write-Host "==> $m" -ForegroundColor White }
function Ok   ($m) { Write-Host "  ok $m"  -ForegroundColor Green }
function Warn ($m) { Write-Host " warn $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host " fail $m" -ForegroundColor Red; exit 1 }
function Need-Venv { if (-not (Test-Path $PY)) { Die "no venv - run scripts\win\01_setup.ps1 first" } }
function Dir-SizeGB ($p) {
  if (-not (Test-Path $p)) { return 0 }
  [math]::Floor((Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object Length -Sum).Sum / 1GB)
}
```

### `scripts/win/00_preflight.ps1`

```powershell
# Step 00 - check the machine before committing to anything.
. "$PSScriptRoot\env.ps1"

Say "GPUs"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Die "nvidia-smi not found" }
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
$caps  = nvidia-smi --query-gpu=compute_cap --format=csv,noheader
$nGpu  = @($caps).Count
$capNum = [int](@($caps)[0] -replace '[^0-9]','')
Ok "$nGpu GPU(s), compute capability $(@($caps)[0].Trim())"
if ($capNum -lt 80) {
  Warn "no native bf16 (pre-Ampere). ACE-Step will auto-select float32."
  Warn "fp16 training is fine - Lightning precision=16-mixed works on Volta."
}

Say "Storage"
$drive = (Split-Path -Qualifier $env:MIREX_DATA_DIR)
$free  = [math]::Floor((Get-PSDrive $drive.TrimEnd(':')).Free / 1GB)
if ($free -lt 600) { Warn "only $free GB free - the 501 GB sample plus harness cache needs ~700 GB" }
else               { Ok "$free GB free on $drive" }

Say "Tools"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Die "ffmpeg missing - install it and put it on PATH (winget install Gyan.FFmpeg)"
}
Ok "ffmpeg present"
Ok "$env:NUMBER_OF_PROCESSORS logical cores"

Say "Paths"
"  MIREX_DATA_DIR       $env:MIREX_DATA_DIR"
"  MIREX_CHECKPOINT_DIR $env:MIREX_CHECKPOINT_DIR"
"  HF_HOME              $env:HF_HOME"
if ($env:MIREX_DATA_DIR.StartsWith($MirexRoot)) {
  Warn "MIREX_DATA_DIR is inside the repo - point it at your data drive"
}
Warn "Native Windows runs ONE GPU per job with no scheduler parallelism gains"
Warn "beyond your GPU count. For a multi-GPU box, use WSL2 and the bash scripts."
Ok "preflight done"
```

### `scripts/win/01_setup.ps1`

```powershell
# Step 01 - venv, dependencies, and the test suite.
. "$PSScriptRoot\env.ps1"
Set-Location $MirexRoot

if (-not (Test-Path $PY)) { Say "creating venv"; python -m venv .venv }
Say "installing dependencies"
& $PIP install --upgrade pip -q
& $PIP install -r requirements.txt -q
Ok "dependencies installed"

Say "creating directory tree"
& $PY -c "import sys; sys.path.insert(0,'src'); import config; config.ensure_dirs(); print(config.DATA_DIR)"

Say "running tests (expect 107 passed)"
& "$MirexRoot\.venv\Scripts\pytest.exe" tests/ -q
if ($LASTEXITCODE -ne 0) { Die "tests failed" }
Ok "setup complete"
```

### `scripts/win/02_fetch_data.ps1`

```powershell
# Step 02 - the 501 GB sample. Resumable: re-run safely after any interruption.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

function Fetch { Say ("data_fetch " + ($args -join " ")); & $PY src\data_fetch.py @args }

Say "AI pool - small, high family diversity, in full (109 GB)"
Fetch --dataset echoes
Fetch --dataset fakemusiccaps
Fetch --dataset sonics
Fetch --dataset aime

Say "AI pool - giant single-family dumps, cut hard (90 GB)"
Fetch --dataset suno_audio --subset-gb 40
Fetch --dataset muse       --subset-gb 25
Fetch --dataset udio       --subset-gb 25

Say "Real pool (33 GB + MTG below)"
Fetch --dataset musicnet
Fetch --dataset fma        --subset-gb 22
Fetch --dataset mtg_jamendo

Say "SDD - metadata ONLY, audio is quarantined"
Fetch --dataset sdd

# --- MTG-Jamendo audio: capped full-quality subset ---
$mtgAudio = "$env:MIREX_DATA_DIR\raw\mtg_jamendo\audio"
$capGB    = [int]$env:MTG_CAP_GB
$cur      = Dir-SizeGB $mtgAudio
if ($cur -ge $capGB) {
  Ok "MTG audio already at $cur GB (cap $capGB GB)"
} else {
  Say "MTG-Jamendo audio - full quality, stopping at $capGB GB"
  Warn "NOT using the audio-low tier: transcoded real class fails the confound gate"
  if (-not (Test-Path "$env:TEMP\mtgj")) {
    git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git "$env:TEMP\mtgj"
  }
  New-Item -ItemType Directory -Force -Path $mtgAudio | Out-Null
  $p = Start-Process -PassThru -NoNewWindow -FilePath "python" -ArgumentList @(
        "$env:TEMP\mtgj\scripts\download\download.py",
        "--dataset","raw_30s","--type","audio",$mtgAudio)
  while (-not $p.HasExited) {
    Start-Sleep -Seconds 60
    $sz = Dir-SizeGB $mtgAudio
    Write-Host "`r  MTG audio: $sz / $capGB GB" -NoNewline
    if ($sz -ge $capGB) { Write-Host ""; Say "cap reached - stopping"; Stop-Process -Id $p.Id -Force; break }
  }
  Write-Host ""
  Ok "MTG audio at $(Dir-SizeGB $mtgAudio) GB"
}

Say "registering everything into the metadata DB"
& $PY src\data_fetch.py --dataset all --register-only

Say "census"
& $PY -c @"
import sys; sys.path.insert(0,'src')
from collections import Counter
from metadata_db import MetadataDatabase
rows = MetadataDatabase().fetch()
print(f'  total tracks: {len(rows)}')
print(f'  AI / real   : {sum(r[\"is_ai\"] for r in rows)} / {sum(1-r[\"is_ai\"] for r in rows)}')
for fam, n in Counter(r['generator_family'] for r in rows).most_common():
    print(f'    {str(fam):16s} {n}')
"@
Ok "data step complete - next: scripts\win\03_generate.ps1"
```

### `scripts/win/03_generate.ps1`

```powershell
# Step 03 - self-generation campaign. Mureka, MiniMax, YuE and ACE-Step have
# NO public training data; they are four of six graded strata.
. "$PSScriptRoot\env.ps1"
Need-Venv
if (-not $env:MUREKA_API_KEY)  { Die "set `$env:MUREKA_API_KEY first" }
if (-not $env:MINIMAX_API_KEY) { Die "set `$env:MINIMAX_API_KEY first" }
Set-Location "$MirexRoot\src"

Say "dry run - checks credentials, spends nothing"
& $PY -m generation.campaign --backend mureka --count 10 --dry-run

Say "ACE-Step 2000 (open model, local GPU)"
& $PY -m generation.campaign --backend ace_step --count 2000 --workers 1
Say "YuE 800"
& $PY -m generation.campaign --backend yue --count 800 --workers 1
Say "Mureka 400 (API)"
& $PY -m generation.campaign --backend mureka --count 400 --workers 4
Say "MiniMax 400 (API)"
& $PY -m generation.campaign --backend minimax --count 400 --workers 2

Say "transcoding generated WAV to FLAC (lossless, saves ~40%)"
Warn "never MP3 here - a lossy codec on the AI class only manufactures the"
Warn "exact confound the delivery-chain simulator exists to destroy"
$before = Dir-SizeGB "$env:MIREX_DATA_DIR\generated"
Get-ChildItem "$env:MIREX_DATA_DIR\generated" -Recurse -Filter *.wav |
  ForEach-Object -ThrottleLimit ([int]$env:NUMBER_OF_PROCESSORS) -Parallel {
    $flac = $_.FullName -replace '\.wav$', '.flac'
    & ffmpeg -v error -y -i $_.FullName $flac
    if ($LASTEXITCODE -eq 0) { Remove-Item $_.FullName -Force }
  }
Ok "generated pool: $before GB -> $(Dir-SizeGB "$env:MIREX_DATA_DIR\generated") GB"

Set-Location $MirexRoot
& $PY src\data_fetch.py --dataset all --register-only
Ok "generation complete - next: scripts\win\04_quarantine_gate.ps1"
```

### `scripts/win/04_quarantine_gate.ps1`

```powershell
# Step 04 - HARD GATE. Zero SDD overlap required.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

Say "building the quarantine blocklist"
& $PY src\quarantine.py build | Tee-Object "$LOGS\quarantine_build.log"

Say "verifying zero overlap"
& $PY src\quarantine.py verify | Tee-Object "$LOGS\quarantine_verify.log"
if ($LASTEXITCODE -ne 0) { Die "quarantine gate FAILED - do not train" }
Ok "quarantine gate PASSED"
```

### `scripts/win/05_confound_gate.ps1`

```powershell
# Step 05 - HARD GATE. Probe AUROC on non-content features must stay < 0.60.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

Say "running the confound audit (decodes AUDIT_N=$($env:AUDIT_N ?? '4000') tracks)"
& $PY scripts\lib\confound_gate.py | Tee-Object "$LOGS\confound_gate.log"
if ($LASTEXITCODE -ne 0) {
  Warn "confound gate FAILED"
  Warn "suspect the SUBSET COMPOSITION before the model:"
  Warn "  - MTG audio tier (audio-low is transcoded -> bitrate floor on the real class)"
  Warn "  - FMA clip length (fma_large/medium are 30 s -> duration signature)"
  Die "not proceeding to training"
}
Ok "confound gate PASSED"
```

### `scripts/win/06_harness.ps1`

```powershell
# Step 06 - freeze the dev set and materialize the strata grid.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot
Say "freezing dev set and materializing strata"
& $PY scripts\lib\harness_build.py | Tee-Object "$LOGS\harness.log"
Ok "harness ready - regenerable, delete $env:MIREX_DATA_DIR\harness_cache to reclaim space"
```

### `scripts/win/07_train_all.ps1`

```powershell
# Step 07 - train every branch on every LOGO fold.
#
# 5 branches x 7 folds = 35 jobs. train.py hardcodes devices=1, so one job owns
# one GPU; this schedules 35 jobs across however many GPUs the box has.
# Batch sizes come from config.BRANCHES - do not override unless a job OOMs.
# AMP is already on (Lightning precision="16-mixed").
#
# Resumable: a fold with an existing checkpoint is skipped.
#   $env:BRANCHES="ab"; $env:EPOCHS="20"; .\scripts\win\07_train_all.ps1
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

$branches = if ($env:BRANCHES) { $env:BRANCHES } else { "abcde" }
$folds    = @("suno","udio","mureka","minimax","yue","ace-step","none")
$epochs   = if ($env:EPOCHS)  { $env:EPOCHS }  else { "10" }
# Windows dataloader workers use spawn, not fork - keep this modest.
$workers  = if ($env:WORKERS) { $env:WORKERS } else { "2" }
$nGpu     = if ($env:NGPU) { [int]$env:NGPU } else { @(nvidia-smi --query-gpu=index --format=csv,noheader).Count }
if ($nGpu -lt 1) { Die "no GPUs visible" }

Say "smoke test first (CPU, seconds) - validates the pipeline before GPU-weeks"
& $PY src\train.py --branch a --holdout suno --smoke
if ($LASTEXITCODE -ne 0) { Die "smoke test failed" }
Ok "smoke passed"

# --- build the job list, skipping folds already trained ---
$queue = [System.Collections.Queue]::new()
foreach ($b in $branches.ToCharArray()) {
  foreach ($f in $folds) {
    $foldDir = if ($f -eq "none") { "full" } else { "logo_$f" }
    $dir = Join-Path $env:MIREX_CHECKPOINT_DIR "$b\$foldDir"
    if ((Test-Path $dir) -and (Get-ChildItem $dir -Filter *.ckpt -ErrorAction SilentlyContinue)) {
      Ok "skip $b/$f (checkpoint exists)"
    } else {
      $queue.Enqueue(@{ Branch = "$b"; Fold = $f })
    }
  }
}
$total = $queue.Count
if ($total -eq 0) { Ok "every fold already trained"; exit 0 }
Say "$total job(s) to run across $nGpu GPU(s), $epochs epochs each"

# --- GPU slot scheduler: poll for a free slot, launch, repeat ---
$running = @{}                       # gpu index -> @{Proc;Branch;Fold;Log}
$start = Get-Date; $n = 0
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  while ($running.Count -lt $nGpu -and $queue.Count -gt 0) {
    $free = (0..($nGpu-1) | Where-Object { -not $running.ContainsKey($_) })[0]
    $job  = $queue.Dequeue(); $n++
    $log  = Join-Path $LOGS "train_$($job.Branch)_$($job.Fold).log"
    Write-Host ("  [{0,2}/{1,2}] gpu{2}  branch {3}  holdout {4,-9} -> {5}" -f $n,$total,$free,$job.Branch,$job.Fold,$log)
    $env:CUDA_VISIBLE_DEVICES = "$free"     # inherited by the child at launch
    $p = Start-Process -PassThru -NoNewWindow -FilePath $PY -ArgumentList @(
          "src\train.py","--branch",$job.Branch,"--holdout",$job.Fold,
          "--epochs",$epochs,"--workers",$workers) `
          -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $running[$free] = @{ Proc = $p; Branch = $job.Branch; Fold = $job.Fold; Log = $log }
  }
  Start-Sleep -Seconds 10
  foreach ($gpu in @($running.Keys)) {
    if ($running[$gpu].Proc.HasExited) {
      $r = $running[$gpu]
      if ($r.Proc.ExitCode -eq 0) { Write-Host "  done  $($r.Branch)/$($r.Fold)" -ForegroundColor Green }
      else { Write-Host "  FAIL  $($r.Branch)/$($r.Fold) - see $($r.Log)" -ForegroundColor Red }
      $running.Remove($gpu)
    }
  }
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

Say "training finished in $([int]((Get-Date) - $start).TotalMinutes) min"
Say "checkpoints"
Get-ChildItem $env:MIREX_CHECKPOINT_DIR -Recurse -Filter *.ckpt |
  ForEach-Object { "  " + $_.FullName.Replace("$env:MIREX_CHECKPOINT_DIR\","") }
$failed = @(Get-ChildItem $LOGS -Filter "train_*.log*" | Where-Object { Select-String -Path $_ -Pattern "Traceback" -Quiet })
if ($failed.Count -gt 0) { Warn "$($failed.Count) job log(s) contain tracebacks - check $LOGS" }
Ok "next: scripts\win\08_fusion.ps1"
```

### `scripts/win/08_fusion.ps1`

```powershell
# Step 08 - stacked fusion + isotonic calibration on LOGO out-of-fold scores.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot
$oof = if ($args.Count -gt 0) { $args[0] } else { "oof_scores.jsonl" }

if (-not (Test-Path $oof)) {
  Warn "MISSING PIECE: $oof does not exist."
  Warn "StackedFusion.fit_from_logo() needs rows shaped"
  Warn '  {"scores": {"a":0.9,...,"e":0.3}, "label": 1, "fold": "suno"}'
  Warn "but nothing in the repo generates them yet. You need a script that,"
  Warn "for each fold, loads that fold's held-out tracks and scores them with"
  Warn "that fold's five checkpoints. This is the one remaining gap."
  Die "see RUNBOOK.md step 09"
}
Say "fitting fusion on $oof"
& $PY scripts\lib\fit_fusion.py $oof | Tee-Object "$LOGS\fusion.log"
Ok "fusion fitted - select on worst-stratum and macro AUROC, never pooled accuracy"
```

### `scripts/win/09_container.ps1`

```powershell
# Step 09 - build the offline submission container and rehearse the runtime.
# Needs Docker Desktop with the WSL2 backend and GPU support enabled.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

Say "pre-populating hf_cache (the Dockerfile COPYs it; build fails if absent)"
New-Item -ItemType Directory -Force -Path "$MirexRoot\hf_cache" | Out-Null
$env:HF_HOME = "$MirexRoot\hf_cache"
& $PY -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m'); AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True); print('  hf cache populated')"

Say "building image"
docker build -t mirex2026-detector .
if ($LASTEXITCODE -ne 0) { Die "docker build failed" }
Ok "built mirex2026-detector"

$rehearsal = if ($env:REHEARSAL_DIR) { $env:REHEARSAL_DIR } else { "$env:MIREX_DATA_DIR\rehearsal" }
if (Test-Path $rehearsal) {
  $count = @(Get-ChildItem $rehearsal -Recurse -Filter *.wav).Count
  Say "runtime rehearsal: $count tracks, ONE gpu, --network none (must beat 24 h)"
  New-Item -ItemType Directory -Force -Path "$env:TEMP\mirex_out" | Out-Null
  $t0 = Get-Date
  docker run --gpus '"device=0"' --network none `
      -v "${rehearsal}:/data/input" -v "$env:TEMP\mirex_out:/data/output" `
      mirex2026-detector
  $el = ((Get-Date) - $t0).TotalSeconds
  if ($count -gt 0) {
    Ok "scored $count tracks in $([int]($el/60)) min - extrapolated 10k: $([math]::Round($el*10000/$count/3600,1)) h"
  }
  Get-Content "$env:TEMP\mirex_out\scores.csv" -TotalCount 3
} else {
  Warn "no rehearsal dir at $rehearsal - set `$env:REHEARSAL_DIR to a folder of WAVs"
  Warn "the 10k-track rehearsal on one GPU is a submission requirement"
}
```

