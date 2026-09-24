# Baseline Results: Old Model (5,000-Track Prototype)

This folder contains the evaluation results from the initial **5,000-track prototype model**, which established the original proof-of-concept for the dual-branch MusicScope architecture.

---

## Artifacts

1. **`eval_5000_results.csv`**: Detailed per-track predictions and confidence scores for 5,000 test audio tracks.
2. **`eval_5000_analysis.png`**: 4-panel diagnostic visualization including ROC curve, score distributions, per-family accuracy, and error analysis.

---

## Prototype Model Specifications

* **Training Data:** Small balanced subset of $5,000$ tracks ($2,500$ AI vs. $2,500$ Human).
* **Architecture:** Early ResNet-18 style acoustic encoder (576-D) + early 128-D narrative statistical extractor + direct linear fusion (no Supervised Contrastive alignment).
* **Calibration:** Uncalibrated raw sigmoid outputs evaluated at default decision boundary $\tau = 0.50$.

---

## Performance Summary

| Metric | Baseline Score |
| :--- | :---: |
| **Test Set Size** | 5,000 tracks |
| **Macro-AUROC** | `0.9782` |
| **Accuracy** | `92.62%` |
| **Precision** | `94.10%` |
| **Recall** | `90.94%` |
| **F1-Score** | `0.9249` |

---

## Limitations That Motivated 200k Retraining

1. **Restricted Domain Coverage:** The 5k dataset only covered early versions of Suno and basic pop arrangements; it failed on complex non-vocal soundscapes (e.g. ambient drone, cinematic sweeps).
2. **Feature Disconnect:** The acoustic surface features and narrative statistical moments were fused naively with a shallow MLP without Supervised Contrastive (`SupCon`) latent alignment.
3. **Scale:** 5,000 tracks was insufficient to learn fine-grained anti-aliasing artifacts across modern diffusion-based audio architectures.

*This baseline is superseded by the 200,000-track retrained model in [`../new_model_retrained_200k/`](../new_model_retrained_200k/README.md).*
