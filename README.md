MIREX 2026 AI-Generated Music Detection

THE PROBLEM

Build a system for the MIREX 2026 challenge: given a full music track (WAV),
output a score 0-1 saying whether it is fully AI-generated or human-made.
The hidden test set uses six generators (Suno, Udio, Mureka, MiniMax, YuE,
ACE-Step) plus real music, and the ranking metric is macro-averaged AUROC
across hidden strata, so the worst generator family dominates your rank.
The hard part: every published detector works near-perfectly on generators
it trained on and collapses on ones it hasn't seen, and most detectors
secretly learn delivery-pipeline artifacts (codec, bitrate, loudness)
instead of AI-ness. Also the hidden real test music comes from the Song
Describer Dataset, which is a subset of MTG-Jamendo's test split, so naive
training on MTG-Jamendo would mean training on the test set.

Full methodology: IMPLEMENTATION_PLAN.md

WHAT IS DONE

- Complete codebase, 107/107 tests passing, verified end-to-end.
- Quarantine system that blocks all contaminated data (SDD tracks, the
  MTG-Jamendo test split, SDD artists, duplicates) with a verify gate that
  fails loudly on any overlap. Live-validated: exactly 706 SDD tracks blocked.
- Delivery-chain simulator: random codec/resample/loudness/pitch/EQ/noise
  augmentation applied identically to both classes so the model cannot learn
  pipeline shortcuts. Bit-reproducible given a seed.
- Confound audit gate: probes that try to predict the label from non-content
  features; training is blocked unless probe AUROC < 0.60.
- Five detection models: (a) wav2vec2-XLS-R speech model, (b) MERT music
  model, (c) physics branch (spectral fakeprint combs + shift-invariant
  log-frequency CNN), (d) 120-second long-context ConvNeXt, (e) real-only
  anomaly detector (OC-Softmax) as insurance for unseen generators.
- Evaluation harness replicating the hidden eval: frozen dev set, condition
  strata (codecs, resampling, pitch, excerpt lengths), leave-one-generator-
  out folds, full metric suite (macro/min/pooled AUROC, EER, AUPRC, FPR/FNR).
- Training pipeline (PyTorch Lightning), score fusion (stacked logistic
  regression on LOGO folds + rank-average fallback + calibration),
  deterministic test-time augmentation.
- Submission entrypoint (inference.py: directory of WAVs -> CSV of scores,
  per-track timeout with fallback so a corrupt file can never disqualify us)
  and a Dockerfile that runs fully offline.
- Data downloaders for all 12 sources with verified URLs, metadata database
  tracking provenance of every file.
- Generation campaign code: ACE-Step and YuE local runners, Mureka and
  MiniMax API clients (verified endpoints), prompt taxonomy matched to the
  MTG-Jamendo tag distribution, resumable job ledger.

WHAT IS LEFT TO DO

1. Data downloads (~1.5-3 TB). Needs a machine with 3-4 TB storage; this box
   has ~166 GB free, so rent a cloud box or use --subset-gb locally.
   Command: python src/data_fetch.py --dataset all

2. Generation campaign. Needs MUREKA_API_KEY and MINIMAX_API_KEY (budget
   ~$300-800 for ~2k tracks per service) and a GPU box for ACE-Step (20k
   tracks) and YuE (5k tracks).
   Command: python -m generation.campaign --backend ace_step --count 20000

3. Branch training. The local RTX 2050 (4 GB) cannot train the 300M SSL
   models; rent A100/L40S time (roughly 2-3 weeks across 4 GPUs). On small
   dev boxes set MIREX_SMALL=1 for the 95M MERT variant.
   Command per branch/fold: python src/train.py --branch a --holdout suno

4. Final pipeline, in order, once data is in place:
   - python src/quarantine.py build && python src/quarantine.py verify
     (must report zero overlap - hard gate)
   - confound audit (probe AUROC must be < 0.60 before training)
   - freeze the dev set and materialize the evaluation strata (harness.py)
   - train all branches (LOGO folds + full models)
   - fit fusion + calibration on the LOGO out-of-fold scores
   - build the Docker container and do the runtime rehearsal (10k synthetic
     tracks must finish well under 24 h on one GPU)

SETUP

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -q
ffmpeg must be on PATH. Optional: fpcalc (chromaprint) for audio dedup.
