# Coverage, Confounds, and Complementarity: What It Actually Takes to Detect AI-Generated Music in the Wild

**Authors:** [Team]  ·  **Venues:** MIREX 2026 technical report (2–4 pp, ISMIR LBD format) → ISMIR 2027 full paper (extended)

## Abstract (draft)
In-domain AI-music detection is essentially solved (≥99% F1 on SONICS-style benchmarks), yet published detectors collapse out-of-domain: Suno-trained systems reach F1 0.63–0.78 on Udio, and the best 2025 system scores near-random EER on cross-domain song-deepfake tests. We argue — and empirically show on the MIREX 2026 AI-Generated Music Detection task — that the decisive factors are not architectural novelty but (i) *coverage*: closing the generator-family gap with public dumps, open-model self-generation, and API generation with a taxonomy-matched prompt distribution; (ii) *confound control*: a symmetric delivery-chain simulator plus a probe-based audit gate (label prediction from non-content features must stay below 0.60 AUROC); and (iii) *complementarity*: a five-branch ensemble spanning speech-SSL, music-SSL, interpretable deconvolution-comb physics, long-context structure, and a real-only one-class anomaly detector, fused by leave-one-generator-out (LOGO) stacking so fusion weights encode trust-under-novelty. We report per-stratum results, ablations isolating each pillar's contribution, and release our self-generated Mureka/MiniMax/YuE/ACE-Step corpus with full generation settings.

## 1. Introduction
- Deezer: >50% of daily uploads AI-generated on peak days (July 2026) — deployment stakes.
- The generalization asymmetry: in-domain solved / out-of-domain broken (TISMIR arms race; MIREX 2025 baselines: SpecTTTra 1.75% EER at home, 48.9% cross-domain).
- Thesis: coverage + confound control + complementarity beat architecture novelty; the MIREX 2026 macro-AUROC-across-strata metric rewards exactly this.

## 2. Related work
- Datasets & benchmarks: SONICS, FakeMusicCaps, AIME, Echoes (semantic alignment), MoM/CLAM, ArtifactBench.
- Detection: SpecTTTra; Deezer ICASSP'25 (AE-reconstruction training, "real-as-default" failure), ISMIR'25 Fourier fakeprints (deconvolution combs, best paper), ISMIR'26 shift-invariant robustness.
- Singing-voice anti-spoofing lineage: SVDD 2024 (XLS-R+AASIST winners, RawBoost/vocoder aug), SingGraph (MERT+wav2vec2), one-class learning (OC-Softmax, SAMO).
- Zero-shot: MusicDET (real-only flows), Sofia (music-intrinsic MoE).

## 3. Task and threat model
- MIREX 2026 rules; macro-AUROC over strata → worst-family sensitivity (formal argument: concavity of per-family effort allocation ⇒ maximize the minimum).
- Failure taxonomy F1–F6 (cross-generator collapse, pipeline confounds, cheap-transform fragility, real-as-default bias, contamination, short-context ceiling) with the countermeasure map.

## 4. Method
### 4.1 Coverage: the data engine
- Public pools (Muse/Suno-v5, version-labeled Suno, Udio dump, SONICS, Echoes); quarantine protocol (SDD ⊂ MTG-Jamendo split-0 test — contamination analysis); artist-level splits; fingerprint dedup.
- Self-generation: ACE-Step (20k), YuE (5k), DiffRhythm et al.; API: Mureka, MiniMax (~2k each). Prompt taxonomy matched to the MTG-Jamendo tag distribution; sampler-setting variation as difficulty proxy. Corpus released with full settings.
### 4.2 Confound control
- Delivery-chain simulator (codec/resample/loudness/EQ/pitch/stretch/excerpt), identical distribution across classes; ~20% clean passthrough.
- The audit gate: linear + boosted probes on 11 non-content features, held-out grouped split, gate at 0.60 AUROC; per-feature leak attribution. Evidence section: which public dataset pairings fail the gate untreated (e.g., bitrate/sample-rate shortcuts worth 83% precision in prior work).
### 4.3 Complementary branches
- A: XLS-R + layer-weighted aggregation + attentive-stats pooling (SVDD recipe).
- B: MERT + shallow transformer (production/performance cues).
- C: physics — fakeprint comb-peak GBDT + shift-invariant log-frequency CNN (pitch/speed-robust by design).
- D: 120 s ConvNeXt (window seams, structure, energy uniformity).
- E: real-only OC-Softmax anomaly (unseen-version insurance; the only branch immune to coverage gaps by construction).
### 4.4 LOGO selection and fusion
- All model selection on LOGO worst-case/macro AUROC, never iid accuracy.
- Stacked LR on LOGO out-of-fold scores; ridge toward equal weights; rank-average fallback; isotonic calibration (monotone ⇒ primary metric unaffected).
- Track aggregation: s = (1−λ)·mean + λ·top-k over chunk scores; tuned on the harness.

## 5. Experiments
- Internal harness: strata grid (family × {clean, MP3-128/64, Opus-48, 22.05k resample, ±1st pitch} × {30/60/120 s/full} × vocal/instr).
- Baselines: fakeprint-LR, SONICS SpecTTTra checkpoint, XLS-R+AASIST (MIREX-2025 recipe).
- Headline: macro/min AUROC vs baselines; per-stratum tables.
- Ablations (one per pillar): (a) remove self-generated data → per-family collapse on YuE/ACE-Step/Mureka strata; (b) disable simulator/audit → probe AUROC and hidden-strata drop; (c) branch knock-outs → no-single-point-of-failure; LOGO- vs iid-selected checkpoints gap; (d) stacking vs rank-average vs best single.
- Robustness: TTA effect per condition stratum; version-drift proxy (train ≤v4.5 → test v5).

## 6. Results on MIREX 2026 (when released)
- Slots: stacked ensemble / rank-average / best single / TTA ensemble. Per-slot rationale (risk ladder).

## 7. Discussion
- What the physics branch still catches at v5-era fidelity; where the anomaly branch fires; failure analysis on FPR-critical human genres (lo-fi/EDM); the arms-race outlook (evasion economy), calibration semantics for partial-AI content (future MIREX scopes).

## 8. Reproducibility & disclosure
- Full data/pretrained-model/API disclosure table (MIREX requirement); code + corpus release; compute declaration.

## References (anchors)
SONICS (arXiv:2408.14080) · Deezer ICASSP'25 (2501.10111) · Fourier fakeprints (2506.19108) · shift-invariance (2607.27454) · SVDD (2408.16132) · OC-Softmax (2010.13995) · SingGraph (2406.03111) · TISMIR arms race (10.5334/tismir.254) · Echoes (2603.23667) · MusicDET (2605.18072) · ArtifactBench (2604.16254) · MoM/CLAM (2512.00621) · YuE (2503.08638) · ACE-Step (2506.00045) · MusiCoT/Mureka (2503.19611)
