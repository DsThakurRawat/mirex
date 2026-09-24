# Architecture-1 Evaluation Results Directory

This directory organizes all experimental and benchmark results for **MusicScope-CL (Architecture-1)** into three strictly isolated tiers:

```
Architecture-1/results/
├── README.md                      # Master directory guide & comparison
├── buggy_earlier_runs/            # Quarantined results affected by earlier padding/calibration bugs
├── old_model_5k_baseline/         # Initial 5,000-track prototype baseline results
└── new_model_retrained_200k/      # Certified 200,000-track production model benchmark results
```

---

## High-Level Comparison Across All Iterations

| Milestone | Model Iteration | Audio Pipeline | Calibration | Macro-AUROC | Overall Accuracy | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Old Model Baseline** | 5,000-track Prototype | Zero-padded 30s | Uncalibrated ($\tau = 0.50$) | `0.9782` | `92.62%` | Superseded |
| **Buggy Earlier Run** | 200k Interim Model | Zero-padded 30s (Diluted RMS) | Uncalibrated ($\tau = 0.50$) | `0.9814` | `91.07%` (29 NaNs) | **Quarantined (Buggy)** |
| **New Model (Phase 1)** | 200k Retrained Production | Decoupled Unpadded | Calibrated ($\tau = 0.18$) | **`0.9933`** | **`98.54%`** | **Certified Production** |
| **New Model (Phase 2)** | 200k Retrained Production | Decoupled Unpadded | Calibrated ($\tau = 0.18$) | **`0.9970`** | **`98.48%`** | **Certified Production** |
| **New Model (Phase 3)** | 200k Retrained Production | Decoupled Unpadded | Calibrated ($\tau = 0.18$) | *N/A (AI only)* | **`98.89%` AI Caught** | **Certified Production** |

---

## Directory Index

### 1. [`buggy_earlier_runs/`](./buggy_earlier_runs/README.md)
Contains the historical evaluation runs that suffered from:
1. **The Audio Decoupling Bug:** Zero-padding short audio ($10\text{ s}$) with $20\text{ s}$ of zeros in Branch B (narrative statistics), which diluted RMS energy by $66.7\%$ and caused statistical collapse / 29 NaN scores.
2. **Uncalibrated Threshold:** Evaluated using a naive threshold ($\tau = 0.50$) before Platt scaling.

### 2. [`old_model_5k_baseline/`](./old_model_5k_baseline/README.md)
Contains the benchmark results from the initial small-scale prototype trained on only $5,000$ tracks prior to the $200,000$-track scale-up.

### 3. [`new_model_retrained_200k/`](./new_model_retrained_200k/README.md)
Contains the final, verified, and publication-ready evaluation results across all 3 phases:
- **Phase 1:** Zero-shot generalization on 8 unseen architectures ($2,876$ tracks, `0.9933` AUROC).
- **Phase 2:** Full held-out benchmark ($28,134$ physical tracks, `0.9970` AUROC, `98.48%` Accuracy, `1.55%` EER).
- **Phase 3:** Large-scale out-of-domain streaming ($50,000$ Hugging Face tracks, `98.89%` AI Detection Rate).
