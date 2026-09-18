# MusicScope-CL — Architecture Deep Dive

This document expands on the design rationale summarized in the README.

## 1. Motivation

Most AI-generated-audio detectors focus on **surface artifacts**: vocoder phase discontinuities, high-frequency smearing, spectral checkerboarding. This works today, but is a losing long-term strategy — generative models (Suno v5, Udio, and successors) are rapidly learning to "clean up" these artifacts, the same way modern LLMs learned to avoid telltale phrasing and em-dashes.

StoryScope showed that in text, the more durable signal is **narrative structure** — plot causality, thematic development, non-linear time — because it reflects something closer to *intent*, which current generative architectures don't optimize for directly. MusicScope applies the same logic to music: the durable signal is **musical narrative logic** — harmonic tension, motivic development, rhythmic groove, and dynamic shaping over the course of a track.

MusicScope-CL extends this further: rather than treating the surface-texture branch as a plain supervised classifier, it uses **Contrastive Learning** in two places — unsupervised (SimCLR) to pretrain the backbone without touching any label, and supervised (SupCon) to shape the final fused decision space once labels are introduced. This directly addresses the risk of training on a large, noisily-labeled dataset: the parts of the pipeline that touch labels are pushed as late and as cheap-to-retrain as possible.

## 2. Two-Branch Design

### Branch B — Structural Branch (interpretable, zero-training)

Track A plays the role that an LLM-based feature extractor played in StoryScope (which pulled 304 narrative features from text). We can't "read" audio semantically the same way, so instead we use classical **Music Information Retrieval (MIR)** signal-processing to pull out structurally meaningful, interpretable features:

| Feature | Extraction Method | What it Measures |
|---|---|---|
| Harmonic Entropy / Tension | `librosa.feature.chroma_cqt`, dissonance curve vs. bass note | Key stability, use of borrowed chords/modal mixture |
| Rhythmic Micro-timing | `librosa.onset.onset_strength`, IOI variance | Deviation from a strict quantized grid ("groove") |
| Motivic Recurrence | Self-similarity matrix (SSM) over chroma/mel | Whether themes are repeated verbatim or developed |
| Spectral Density Variance | Spectral flatness / rolloff / centroid over time | Whether the mix leaves dynamic "space" or is uniformly dense |
| Dynamic Envelope Variance | RMS energy over time | Non-linear loudness shaping vs. flat/linear arcs |

All five reduce to a **128-dimensional Musical Narrative Vector** per audio chunk. Because these are closed-form DSP computations (no neural net), this branch requires **zero training** and runs in milliseconds per chunk — critical for the 24-hour MIREX compute budget.

### Branch A — Surface Branch (contrastive, learned texture)

A **128-band log-Mel spectrogram** fed into a **MobileNetV3-Small** backbone (HardSwish activations, squeeze-excitation attention), producing a **576-dimensional Surface Texture Embedding**. This branch catches vocoder artifacts and other surface-level generation fingerprints — when present.

Unlike a plain supervised CNN classifier, this branch is trained in two stages:

- **Stage 1 (SimCLR, unsupervised):** two augmented views of the same spectrogram (e.g. time-mask + pitch-shift vs. freq-mask + noise-add) are pulled together and pushed apart from other tracks via NT-Xent loss. No labels are used — the backbone just learns the "physics" of real vs. synthesized audio texture.
- **Stage 3 (SupCon, supervised):** once frozen, the backbone's embeddings (fused with the structural vector) are projected through a supervised contrastive head that explicitly clusters Human vs. AI representations, rather than drawing a thin cross-entropy boundary.

MobileNetV3-Small is chosen specifically for its small parameter count (~2.5M), which keeps inference fast and training feasible within the competition's compute constraints.

### Fusion Layer

The two embeddings — 576-dim surface, 128-dim structural — are concatenated into a **704-dimensional vector**. This fused vector is projected through the SupCon head (Stage 3b), then a lightweight classifier head (MLP, or XGBoost if latency allows) produces the final probability (Stage 3c). This mirrors StoryScope's own finding that **structured/interpretable features fed into a simple classifier (XGBoost) outperformed raw-text transformers** in both robustness and interpretability — while adding contrastive clustering to widen the margin between classes.

Output is calibrated via **temperature scaling** after training with **label-smoothed BCE loss** on top of the SupCon-shaped embeddings, all chosen to directly optimize AUROC (the MIREX evaluation metric) rather than raw accuracy.

## 3. Why the Two Branches Are Complementary, Not Redundant

- Branch A degrades if a generated track is passed through **de-artifacting post-processing** (denoisers, spectral repair plugins) — the audio equivalent of an LLM removing telltale phrasing. It can also be fooled by a hypothetical generator with genuinely novel spectral characteristics.
- Branch B is largely **immune to this kind of style laundering**, because harmonic tension, motif development, and groove are structural properties that current generative pipelines don't explicitly model or optimize — patching surface artifacts doesn't fix them. Conversely, Branch B could in principle be fooled by a generator engineered for structural coherence, while Branch A would still catch its underlying spectral physics.
- This gives MusicScope-CL resilience against the "hardest case" in AUROC evaluation: AI tracks that are sonically polished, structurally coherent, or both — since defeating one branch does not defeat the other. See the "Uncanny Valley" scenario table in the README for a walkthrough of each failure mode.

## 4. Training Pipeline

1. **Stage 1 — SimCLR pretraining (Branch A only):** Contrastive self-supervised learning on log-Mel spectrograms, so the CNN learns robust texture representations without being biased by noisy/weak labels. No labels touched.
2. **Stage 2 — Structural extraction (Branch B):** Pure signal processing, run once across the full dataset. No training, no labels.
3. **Interlude — Latent space analysis:** UMAP projection of the 128-dim structural vectors, analogous to StoryScope's LDA plot of narrative features. Expected pattern: human tracks scatter broadly (high structural rarity); AI tracks cluster tightly ("AI convergence"). Still no labels used for training — this is diagnostic only.
4. **Stage 3a — Fusion:** Freeze the SimCLR-pretrained MobileNet, concatenate its 576-dim embedding with the 128-dim structural vector into a 704-dim fused vector.
5. **Stage 3b — SupCon (first stage to use labels):** Train a supervised contrastive projection head on the fused vector, explicitly clustering Human vs. AI representations to maximize inter-class margin.
6. **Stage 3c — Classification head:** Train a small MLP/XGBoost on top of the SupCon-shaped embeddings with label-smoothed BCE, then calibrate with temperature scaling.

### Why labels are introduced this late

Stages 1 and 2 are entirely label-free — the backbone and the structural features are shaped purely by unsupervised learning and deterministic signal processing, so they cannot be corrupted by label noise in a large, weakly-curated 500GB dataset. Labels only enter at Stage 3, touching just the fusion/classification head — the cheapest, fastest-to-retrain part of the pipeline. If label quality issues are discovered later, only Stage 3 needs to be redone.

## 4a. Inference-Time Boost: Multiple Instance Learning (MIL)

Because the full pipeline is lightweight, each test track can be sliced into multiple 30-second chunks (e.g. three), scored independently, and the scores averaged for a more robust final prediction — a form of test-time Multiple Instance Learning. This is computationally cheap enough to fit comfortably inside the MIREX 24-hour compute budget and typically improves AUROC over single-chunk scoring.

## 5. Compute Budget Justification

| Component | Approx. Cost |
|---|---|
| MobileNetV3-Small (SimCLR + SupCon) | ~2.5M params, fast forward pass |
| MIR feature extraction | Milliseconds/chunk via `librosa`/`essentia`, no GPU required |
| SupCon projection head | Small (704 → 256 → 128) |
| Final MLP/XGBoost head | Tiny (704 or 128 → hidden → 1) |
| MIL inference (3 chunks/track, averaged) | 3x forward passes, still lightweight |
| **Total inference on hidden test set** | Hours, not days — comfortably under MIREX's 24-hour single-GPU rule |

## 6. Ablation Plan

See the ablation table in the README for the full comparison (Baseline / Surface-only / Structure-only / Fused-no-SupCon / MusicScope-CL). The intent is to demonstrate, with evidence rather than assertion, that (a) Surface + Structure beats either branch alone, and (b) SupCon fusion beats plain linear/cross-entropy fusion — directly echoing StoryScope's own ablation findings, adapted to audio.

## 7. Open Questions / Future Work

- Whether Branch B features generalize across genres (e.g., ambient/drone music has intentionally low rhythmic variance, which could look "AI-like" by these heuristics — may need genre-conditional normalization).
- Whether motif recurrence thresholds need per-genre calibration (e.g., minimalist/repetitive human genres like techno).
- Whether SupCon temperature and batch composition (class balance per batch) need tuning to avoid collapse on the minority class if the 500GB dataset is imbalanced.
- Exploring learned (rather than hand-crafted) structural features as a v2, while keeping the interpretability advantage of Branch B.
