# Scripts

**`python3 run.py` in the repo root supersedes everything here** — one
cross-platform driver, same commands on Linux, macOS and Windows. See
RUNBOOK.md. These shell scripts remain for reference and do the same work.

The Python helpers in `lib/` are shared by `run.py` and both shell pipelines.

---

# Shell scripts — run these in order

Linux/macOS: `scripts/*.sh`. Windows: `scripts/win/*.ps1` (PowerShell 7+), or —
better — run the bash scripts unchanged under WSL2. See RUNBOOK.md § Windows.

Every script is idempotent and resumable: re-run after any interruption.
Each sources `env.sh`, logs to `logs/`, and tells you what comes next.

## First, point the data dir at your array

```bash
export MIREX_DATA_DIR=/raid/mirex/data
export MIREX_CHECKPOINT_DIR=/raid/mirex/checkpoints
export HF_HOME=/raid/mirex/hf_cache
```

The defaults put 500 GB inside the repo. `00_preflight.sh` warns if you forget.

## Then, in order

| # | Script | What it does | Roughly |
|---|---|---|---|
| 00 | `00_preflight.sh` | GPU, compute capability, disk, ffmpeg | seconds |
| 01 | `01_setup.sh` | venv, deps, 107 tests | 5 min |
| 02 | `02_fetch_data.sh` | the 501 GB sample + registration | hours–days |
| 03 | `03_generate.sh` | ACE-Step / YuE / Mureka / MiniMax, then FLAC | days |
| 04 | `04_quarantine_gate.sh` | **HARD GATE** — zero SDD overlap | minutes |
| 05 | `05_confound_gate.sh` | **HARD GATE** — probe AUROC < 0.60 | ~20 min |
| 06 | `06_harness.sh` | freeze dev set, materialize strata | ~1 h |
| 07 | `07_train_all.sh` | 35 jobs across your GPUs | days–weeks |
| 08 | `08_fusion.sh` | stacked fusion + calibration | minutes |
| 09 | `09_container.sh` | offline Docker + runtime rehearsal | ~1 h |

```bash
./scripts/00_preflight.sh
./scripts/01_setup.sh
./scripts/02_fetch_data.sh
export MUREKA_API_KEY=... MINIMAX_API_KEY=...
./scripts/03_generate.sh
./scripts/04_quarantine_gate.sh
./scripts/05_confound_gate.sh
./scripts/06_harness.sh
./scripts/07_train_all.sh
./scripts/08_fusion.sh
./scripts/09_container.sh
```

## The two gates are not advisory

`04` fails if SDD tracks (a subset of MTG-Jamendo split-0 test, and the parent
of the hidden real set) reached your training pool. `05` fails if labels are
predictable from bitrate, sample rate or loudness rather than content. Both
exit non-zero and both stop the run. If `05` fails, suspect the **subset
composition before the model** — the MTG audio tier and FMA clip length are the
usual causes at 500 GB.

## Useful overrides

```bash
BRANCHES=ab EPOCHS=20 ./scripts/07_train_all.sh   # subset of branches
NGPU=4 ./scripts/07_train_all.sh                  # cap GPUs used
MTG_CAP_GB=120 ./scripts/02_fetch_data.sh         # smaller MTG slice
AUDIT_N=1500 ./scripts/05_confound_gate.sh        # faster gate check
MIREX_SMALL=1 ./scripts/07_train_all.sh           # 95M MERT for branch b
```

## Known gap

`08_fusion.sh` needs `oof_scores.jsonl`, and **nothing in the repo generates
it**. You need a script that, for each LOGO fold, loads that fold's held-out
tracks and scores them with that fold's five checkpoints. It is the one piece
of pipeline code still missing — the script tells you so rather than failing
obscurely.

## Windows

`scripts/win/*.ps1` mirrors every step. PowerShell 7+ is required
(`ForEach-Object -Parallel` in 03, `??` in 05); `env.ps1` checks and exits
otherwise.

```powershell
$env:MIREX_DATA_DIR = "D:\mirex\data"
.\scripts\win\00_preflight.ps1
```

WSL2 is the better path for anything multi-GPU: CUDA passes through, the bash
scripts run unchanged, and you avoid maintaining a second copy. Keep the data
on the Linux filesystem, not `/mnt/c` — cross-filesystem I/O is roughly 10x
slower and this pipeline is I/O-bound.

The Python helpers in `lib/` are cross-platform and shared by both pipelines.
