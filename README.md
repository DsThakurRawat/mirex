# MIREX 2026 — AI-Generated Music Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch CUDA 12.4](https://img.shields.io/badge/PyTorch-CUDA%2012.4-orange.svg)](https://pytorch.org/)
[![Validation AUROC](https://img.shields.io/badge/Validation%20AUROC-0.9897-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning and digital signal processing system submitted for the **MIREX 2026 AI-Generated Music Detection** competition. Given any raw music track, the system outputs a calibrated probability $P(\text{AI}) \in [0.0, 1.0]$ determining whether the music was created by an AI music generator (e.g., Suno, Udio, MiniMax, Mureka, YuE, ACE-Step) or composed and produced by human artists.

---

## 🏛️ Repository Overview

This repository hosts two distinct, complementary approaches for AI music detection:

```
d:\mirex\
├── Architecture-1/         # [FLAGSHIP] MusicScope-CL: Dual-Branch Contrastive Learning (0.9897 AUROC)
│   └── mirex/
│       ├── config.py             # Global constants, audio parameters, directory bindings
│       ├── dataset.py            # Paired-view SimCLR augmentations + labeled loaders
│       ├── track_a_narrative.py  # Branch B: 128-dim Structural Narrative DSP extractor
│       ├── track_b_style.py      # Branch A: MobileNetV3 surface style backbone
│       ├── train_simclr.py       # Stage 1: Unsupervised SimCLR pretraining on 215k tracks
│       ├── train_supcon.py       # Stage 3b: Supervised Contrastive (SupCon) margin projection
│       ├── fusion_model.py       # Stage 3c: Calibrated classification head (98.97% AUROC)
│       ├── inference.py          # Stage 5: MIL end-to-end scoring & MIREX CSV generator
│       └── evaluate_test_set.py  # Automated held-out evaluation & scorecard report
│
├── Architecture-2/         # Multi-Branch Leave-One-Generator-Out (LOGO) Quarantine System
│   ├── src/                      # 5-branch ensemble (XLS-R, MERT, Physics CNN, 120s Long-Context, OC-Softmax)
│   ├── scripts/                  # Automated download, generate, audit, and container scripts
│   └── tests/                    # 107/107 unit & regression verification suite
│
├── data/                   # [Local / Gitignored] Shared raw audio datasets & processed artifacts
└── checkpoints/            # [Local / Gitignored] Trained PyTorch model weights (*.pt)
```

---

## ⚡ Architecture-1: MusicScope-CL (What You Need to Know)

### 1. The Core Philosophy: Two Branches that Fail Independently

State-of-the-art generative audio models (Suno v3/v4, Udio) produce convincing local timbre, but struggle to sustain authentic long-term human musical continuity. Conversely, humans leave natural space, rhythmic micro-timing, and harmonic tension.

MusicScope-CL builds a **dual-layer defense**:

```
                         ┌────────────────────────────────────────┐
                         │           Raw Audio Input              │
                         │          (3 × 30s Chunks)              │
                         └───────────────────┬────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
        ┌───────────────────────┐                         ┌───────────────────────┐
        │  Branch A: Surface    │                         │  Branch B: Structure  │
        │  Log-Mel Spectrogram  │                         │  Musical Narrative    │
        │  MobileNetV3 Backbone │                         │  Domain-Specific DSP  │
        └───────────┬───────────┘                         └───────────┬───────────┘
                    │                                                 │
                    ▼                                                 ▼
        ┌───────────────────────┐                         ┌───────────────────────┐
        │   Surface Embedding   │                         │   Narrative Vector    │
        │       (576-dim)       │                         │       (128-dim)       │
        └───────────┬───────────┘                         └───────────┬───────────┘
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │     Fused Embedding       │
                               │         (704-dim)         │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │   SupCon Projection Head  │
                               │  Margin Clustering (128d) │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │   FusionMLP Classifier    │
                               │ + Temperature Calibration │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  P(AI-Generated) ∈ [0, 1] │
                               └───────────────────────────┘
```

* **Branch A (Surface Acoustics — 576-dim)**: Captures microscopic acoustic textures, vocoder smearing, phase anomalies, and high-frequency cutoff artifacts via MobileNetV3-Small.
* **Branch B (Musical Narrative — 128-dim)**: Captures macroscopic structural continuity using signal processing:
  * Harmonic entropy and pitch stability.
  * Inter-Onset-Interval (IOI) variance and rhythmic groove.
  * Self-Similarity Matrix (SSM) recurrence rate (motif development vs. exact looping).
  * Spectral flatness, roll-off, and dynamic loudness arcs.

---

### 2. Four-Stage Training Pipeline

1. **Stage 1 — Unsupervised SimCLR Pretraining**:
   * Pretrained MobileNetV3 on **215,346 raw audio tracks** with time-masking, frequency-masking, pitch-shifting, and additive noise augmentations.
   * Learns robust timbral representations without touching noisy scraped labels.
   * **SimCLR Checkpoint**: `simclr_epoch2.pt` (loss: `2.2113`).

2. **Stage 2 — Structural Feature Extraction**:
   * Extracts 128-dimensional deterministic DSP vectors for all tracks and serializes them to Parquet format.
   * **Dataset Artifact**: `narrative_vectors.parquet`.

3. **Stage 3b — Supervised Contrastive Fusion (SupCon)**:
   * Combines frozen 576-dim surface features and 128-dim structural features (704-dim).
   * Projects into a 128-dim hypersphere trained with SupCon loss to maximize inter-class separation between Human and AI tracks.
   * **SupCon Checkpoint**: `supcon_best.pt` (loss: `3.7390`).

4. **Stage 3c — Calibrated Fusion Classifier**:
   * Trains a lightweight `FusionMLP` ($128 \to 256 \to 64 \to 1$) on SupCon representations with label smoothing ($\alpha = 0.05$).
   * Fits a post-hoc Temperature Scaler ($T = 1.337$) to guarantee calibrated confidence scores.
   * **Fusion Checkpoint**: `fusion_best.pt` (**Validation AUROC: `0.9897`**).

---

### 3. Multiple Instance Learning (MIL) at Inference

Real-world AI tracks often have deceptive, realistic intros or stylized acoustic segments. During inference, [`inference.py`](Architecture-1/mirex/inference.py) automatically splits each track into **3 evenly spaced 30-second chunks** across the song, computes independent probabilities, and averages the results:

$$\hat{P}(\text{AI}) = \frac{1}{N} \sum_{i=1}^{N} \sigma\left(\frac{f(\mathbf{x}_i)}{T}\right)$$

---

## 📊 Performance Benchmarks

| Metric | Result | Benchmark Significance |
|---|---|---|
| **Validation AUROC** | **`0.9897` (98.97%)** | Primary MIREX competition ranking metric |
| **BCE Classification Loss** | **`0.3042`** | Converged from `0.6724` with label smoothing |
| **Learned Temperature ($T$)** | **`1.337`** | Calibrated probabilities (prevents overconfidence) |
| **Pretraining Dataset Size** | **215,346 tracks** | Scraped and curated audio corpus |
| **Total Model Parameters** | **~2.8 Million** | Ultra-lightweight and fast to deploy |
| **Inference Latency** | **~150–250 ms / chunk** | Sub-second scoring on modern GPUs |

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

```powershell
# Create Python 3.12 virtual environment
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1

# Install PyTorch with CUDA acceleration & dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r Architecture-1/mirex/requirement.txt
```

---

### 2. Run Inference on Any Track

Score a single audio file (supports `.mp3`, `.wav`, `.flac`, `.ogg`):

```powershell
python Architecture-1/mirex/inference.py `
  --input "path/to/song.mp3" `
  --simclr_ckpt "Architecture-1/mirex/checkpoints/simclr_epoch2.pt" `
  --supcon_ckpt "Architecture-1/mirex/checkpoints/supcon_best.pt" `
  --fusion_ckpt "Architecture-1/mirex/checkpoints/fusion_best.pt"
```

**Output:**
```text
[Scorer] Device: cuda
[Scorer] SupCon head loaded from checkpoints/supcon_best.pt

  song.mp3: P(AI-generated) = 0.8942  ==> [AI GENERATED]
```

---

### 3. Batch Scoring (MIREX Submission Mode)

Score an entire folder of evaluation tracks and produce the official MIREX CSV submission:

```powershell
python Architecture-1/mirex/inference.py `
  --input_dir "path/to/evaluation_tracks" `
  --output_csv "mirex_submission_scores.csv" `
  --simclr_ckpt "Architecture-1/mirex/checkpoints/simclr_epoch2.pt" `
  --supcon_ckpt "Architecture-1/mirex/checkpoints/supcon_best.pt" `
  --fusion_ckpt "Architecture-1/mirex/checkpoints/fusion_best.pt"
```

---

### 4. Run Held-Out Test Evaluation

Run an automated validation scorecard across unseen Human (FMA) and AI (Sonics/Echoes) songs:

```powershell
python Architecture-1/mirex/evaluate_test_set.py
```

---

## 🔬 Reproducing the Training Stages

```powershell
cd Architecture-1/mirex

# Stage 1: SimCLR Unsupervised Pretraining (Branch A)
python train_simclr.py --input_dir "D:\mirex\data\raw" --epochs 20 --num_workers 4

# Stage 2: Structural Feature Extraction (Branch B)
python track_a_narrative.py --input_dir "D:\mirex\data\raw" --output "D:\mirex\data\processed\narrative_vectors.parquet" --db "D:\mirex\data\processed\metadata.db" --workers 8

# Stage 3: Supervised Contrastive Fusion (SupCon)
python train_supcon.py --input_dir "D:\mirex\data\raw" --narrative "D:\mirex\data\processed\narrative_vectors.parquet" --simclr_ckpt "checkpoints/simclr_epoch2.pt" --epochs 30

# Stage 4: Fusion Classifier & Temperature Calibration
python fusion_model.py --narrative "D:\mirex\data\processed\narrative_vectors.parquet" --style_ckpt "checkpoints/simclr_epoch2.pt" --supcon_ckpt "checkpoints/supcon_best.pt"
```

---

## 📜 Citation & Credits

* **Competition**: MIREX 2026 — AI-Generated Music Detection
* **Author**: Divyansh Rawat (`divyanshrawatofficial@gmail.com`)
* **Repository**: [https://github.com/DsThakurRawat/mirex](https://github.com/DsThakurRawat/mirex)
* **Inspiration**: MusicScope-CL adapts the narrative-logic framing from *StoryScope* into the audio/music domain, fusing it with self-supervised contrastive learning.
