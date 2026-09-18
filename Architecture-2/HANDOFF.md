# MIREX 2026 — Remaining Work (handoff)

The codebase is complete and fully tested (107/107 tests passing, end-to-end
verified). What remains is execution work on the DGX-1.

Task page re-verified 2026-08-26; see `IMPLEMENTATION_PLAN.md` §1.1 for the
organizers' verbatim wording. **No deadlines are published yet** — registration
date, submission date, and platform are all TBD, so nothing here is on an
external clock.

## 0. Storage budget — ~4.8 TB as currently specified

Sizes measured against the HuggingFace API and the sources' own docs on
2026-08-26, not estimated from track counts:

| Pool | Item | Size |
|---|---|---|
| AI | `udio` (blanchon/udio_dataset) | 583.2 GB |
| AI | `muse` (bolshyC/Muse) | 575.5 GB |
| AI | `suno_audio` (humair025/suno-audio) | 198.8 GB |
| AI | `aime` (disco-eth/AIME) | 58.0 GB |
| AI | `sonics` (awsaf49/sonics) | 30.4 GB |
| AI | `fakemusiccaps` | 12.9 GB |
| AI | `echoes` (Octavian97/Echoes) | 8.0 GB |
| Real | `fma_full` | 879.0 GB |
| Real | `mtg_jamendo` raw_30s/audio | 508.0 GB |
| Real | `musicnet` | 11.1 GB |
| Real | `fma_metadata` | 0.35 GB |
| **Downloads subtotal** | | **≈ 2.80 TB** |
| Generated | ACE-Step 20k (WAV) | ≈ 740 GB |
| Generated | YuE 5k (WAV) | ≈ 185 GB |
| Generated | Mureka + MiniMax 4k (MP3) | ≈ 27 GB |
| Retained | archives kept after extraction | ≈ 903 GB |
| Derived | harness strata cache | 120–260 GB |
| Derived | checkpoints (5 branches x 7 folds) | 50–150 GB |
| **Total steady state** | | **≈ 4.8 TB** |

A DGX-1's 4 x 1.92 TB NVMe in RAID 0 gives ~7 TB, so 4.8 TB fits — but confirm
the actual config first (`lsblk`, `df -h`): RAID 5 instead of RAID 0 leaves
~5.3 TB and the margin mostly disappears.

**Peak, not steady state, is the binding constraint.** `extract_archive()` in
`data_fetch.py` never deletes the archive it unpacked, so `fma_full` needs
879 GB of zip plus 879 GB of extracted audio simultaneously — a 1.76 TB spike
that lands when the rest of the pool is already on disk.

Four levers, in order of value per unit of pain:

1. **Delete archives after extraction** (−903 GB, no data loss). The only code
   change needed; `extract_archive()` currently has no cleanup path.
2. **`mtg_jamendo` `audio-low` instead of `audio`** (−352 GB). Costs high-
   frequency detail on the real class, which is exactly what branch C reads —
   test on a subset before committing.
3. **Store self-generated audio as FLAC, not WAV** (−370 GB). ACE-Step writes
   `.wav` (`ace_step_runner.py:133`). **Do not use MP3 here** — this detector
   keys on codec artifacts, so baking a lossy codec into the AI class only
   would manufacture the precise confound the delivery-chain simulator exists
   to destroy.
4. **`fma_large` instead of `fma_full`** (−786 GB). Biggest single saving,
   biggest cost: `fma_large` is 30 s clips, and both the task and the harness
   run on full tracks. Treat as a last resort.

Levers 1 + 2 + 3 bring the total to ~3.2 TB and remove the extraction spike.

Two cautions: RAID 0 has no redundancy — one SSD failure destroys a pool that
takes weeks of bandwidth to refetch, so keep the metadata DB and any generated
audio backed up off-array. And `sonics` at 30.4 GB is suspiciously small for
97k songs; the HF repo likely hosts only part of the corpus. Verify the track
count after `register_sonics()` rather than trusting the download to be
complete.

## 1. Data downloads

- Command: `python src/data_fetch.py --dataset all`
- `--subset-gb N` fetches small local subsets for dev work on this laptop
  (134 GB free — enough for smoke tests only).
- `mtg_jamendo` audio is not auto-fetched: use MTG's own
  `scripts/download/download.py --dataset raw_30s --type audio`.

## 2. Generation campaign (self-generated training data)

Four of the six candidate test families — Mureka, MiniMax, YuE, ACE-Step —
have **no public training data**. They exist in the training set only if this
campaign runs, and they are four of six graded strata under macro-AUROC. This
is the highest-leverage remaining task.

- Needs `MUREKA_API_KEY` and `MINIMAX_API_KEY` (~$300–800 for ~2k tracks each).
- Open models run on the DGX-1: ACE-Step (20k tracks), YuE (5k, slower).
- Command: `python -m generation.campaign --backend ace_step --count 20000`
- **Volta blocker:** `ace_step_runner.py:82` defaults to `dtype="bfloat16"`.
  V100/P100 (compute capability 7.0/6.0) have no bf16 support. Set fp16 or
  fp32 before launching, or the campaign fails on the first job.

## 3. Branch training

- 8 GPUs, but `train.py` has no DDP/DataParallel — one job owns one GPU. The
  natural parallelism is one LOGO fold per GPU (6 folds + full model), not
  data-parallel scaling of a single run.
- AMP is already enabled: `train.py` sets Lightning `precision="16-mixed"`
  (fp16 + automatic grad scaler), which Volta supports natively. Batch sizes
  come from `config.BRANCHES` and are tuned per branch (a=16, b=8, c=32, d=4,
  e=32). Confirm the DGX-1 generation anyway — P100 has no tensor cores at
  all, and V100 16 GB vs 32 GB changes headroom.
- No flash-attn dependency, which is correct: FA2 needs Ampere or newer.
- `MIREX_SMALL=1` selects the 95M MERT variant for small dev boxes.
- Command per branch/fold: `python src/train.py --branch a --holdout suno`

## 4. Final pipeline (in order, once data is in place)

1. Quarantine: `python src/quarantine.py build && python src/quarantine.py verify`
   (must report zero overlap — hard gate)
2. Confound gate: run the audit; probe AUROC must be < 0.60 before training
3. Harness: freeze the dev set and materialize the evaluation strata
4. Train all branches per the README pipeline (LOGO folds + full models)
5. Fit fusion + calibration on the LOGO out-of-fold scores
6. Build the Docker submission container and run the full-scale runtime
   rehearsal (10k synthetic tracks must finish well under 24 h on one GPU)

## 5. Watch for

The organizers have announced but not released a training dataset and a
baseline model + checkpoint. Registry stubs are in place (`mirex_provided`,
`mirex_baseline`, kind `pending`, skipped by `--dataset all`) plus
`MirexProvidedBaseline` in `baselines.py`. When the training set drops it must
clear our own quarantine gate before use — the organizers' candidate mix names
SDD as a human-music source.

Repo: `/home/divyansh-rawat/mirex` — see `README.md` for exact commands and
`IMPLEMENTATION_PLAN.md` for the full methodology.
