# Quarantined Results: Buggy Earlier Runs

> [!CAUTION]
> **DO NOT USE THESE METRICS IN PRODUCTION OR PUBLICATIONS.**
> These historical evaluation runs contain data corruption and feature distortion caused by an audio padding bug and an uncalibrated decision threshold.

---

## Quarantined Artifacts

1. **`eval_28k_results.csv`**: Full predictions across 28,134 tracks containing 29 NaN predictions and diluted scores on short audio.
2. **`eval_28k_analysis.png`**: Visual evaluation scorecard generated from the flawed predictions.
3. **`eval_130k_interim.png`**: Early interim snapshot from the initial uncalibrated 130k/1M run.

---

## Detailed Root Cause Analysis (Post-Mortem)

### Bug 1: The Global Audio Zero-Padding Leak
* **Defect:** In early iterations of `evaluate_val_28k.py` and `dataset.py`, `load_audio_chunk(..., pad=True)` was called globally before passing audio to both Branch A (Acoustic Style Encoder) and Branch B (128-D Narrative Statistics).
* **Mechanism:**
  * Datasets like **FakeMusicCaps** (AudioLDM and MusicGen) consist of $10\text{ s}$ audio clips.
  * The global pad padded these clips with $20\text{ s}$ of zero samples ($441,000$ zeros) to reach $30\text{ s}$.
  * In Branch A, zero-padding spectrograms is standard for fixed-size 2D CNNs (`target_t = 1292`).
  * In Branch B, computing time-series statistical moments (mean, std, skew, kurtosis, energy) over $66.7\%$ silence drastically distorted the narrative features:
    $$\text{RMS}_{\text{padded}} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} x[t]^2} \approx \frac{1}{\sqrt{3}} \cdot \text{RMS}_{\text{true}} \approx 0.33 \times \text{RMS}_{\text{true}}$$
* **Symptoms & Impact:**
  1. Catastrophic cancellation in Scipy moment calculations (`RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation`).
  2. 29 tracks produced `NaN` scores and failed inference.
  3. AI detection rate on AudioLDM fell to $79.31\%$ (versus the true $99.88\%$ under unpadded extraction).

### Bug 2: Naive Uncalibrated Threshold ($\tau = 0.50$)
* **Defect:** Model raw sigmoid outputs were thresholded at $\tau = 0.50$ without Platt scaling / temperature calibration.
* **Impact:** The raw logits were naturally centered around negative values during contrastive fusion training, making $\tau = 0.50$ too conservative.

---

## Resolution in the Retrained Production Model

1. **Decoupled Audio Pipeline:**
   * Branch A continues to receive padded audio ($30\text{ s}$) strictly for the 2D CNN spectrogram.
   * Branch B receives pure, unpadded audio (`pad=False`), preserving true dynamic range, harmonic content, and tempo statistics.
2. **Platt Scaling & Calibrated Threshold:**
   * Learned temperature $T = 0.9083$ applied to raw logits.
   * Calibrated operational threshold set to $\tau = 0.1800$.
3. **Fixed Results:** See [`new_model_retrained_200k/eval_28k_retrained_results.csv`](../new_model_retrained_200k/eval_28k_retrained_results.csv) where AudioLDM detection reached **`99.88%`** and zero NaNs exist.
