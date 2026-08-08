MIREX 2026 AI-Generated Music Detection

PROBLEM
Given a full music track, score 0-1 whether it is fully AI-generated.
Hidden test uses Suno, Udio, Mureka, MiniMax, YuE, ACE-Step; metric is
macro-AUROC across strata, so the worst generator family decides your rank.
Known pitfalls: detectors collapse on unseen generators, learn codec/pipeline
shortcuts instead of AI-ness, and the hidden real music (Song Describer
Dataset) is a subset of MTG-Jamendo test — naive training = test contamination.
Full methodology: IMPLEMENTATION_PLAN.md

DONE
Complete codebase, 107/107 tests passing, verified end-to-end:
- quarantine + verify gate for contaminated data (live-validated, 706 SDD
  tracks blocked)
- class-symmetric delivery-chain augmentation + confound audit gate (<0.60
  probe AUROC required before training)
- five models: XLS-R, MERT, physics/fakeprint CNN, 120s long-context,
  real-only anomaly (OC-Softmax)
- eval harness (frozen dev set, condition strata, leave-one-generator-out),
  training, fusion + calibration, TTA
- submission entrypoint (WAV dir -> CSV, per-track timeout fallback),
  offline Dockerfile
- downloaders for 12 data sources; ACE-Step/YuE runners and Mureka/MiniMax
  API clients for self-generated training data

LEFT TO DO (needs cloud storage/GPU + API keys)
1. Download data (~1.5-3 TB):  python src/data_fetch.py --dataset all
2. Generate AI tracks:  python -m generation.campaign --backend ace_step
3. Train branches:  python src/train.py --branch a --holdout suno
4. Quarantine verify -> confound audit -> harness -> fusion -> Docker.

SETUP
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -q     # ffmpeg required on PATH
