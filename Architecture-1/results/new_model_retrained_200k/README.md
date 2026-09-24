# Production Results: New Model (200,000-Track Retrained)

> [!NOTE]
> **CERTIFIED PRODUCTION BENCHMARKS**
> These results represent the official, production-certified evaluation benchmarks of **MusicScope-CL (Architecture-1)** after full $200,000$-track scale-up, Supervised Contrastive alignment, decoupled audio padding, and Platt temperature calibration ($T = 0.9083$, $\tau = 0.1800$).

---

## Directory Artifacts

| Category | Filename | Description |
| :--- | :--- | :--- |
| **Phase 1 Benchmark** | [`eval_unseen_generators_results.csv`](./eval_unseen_generators_results.csv) | Zero-shot evaluation across 8 unseen AI architectures ($2,876$ tracks) |
| **Phase 1 Scorecards** | [`unseen_generators_scorecard.png`](./unseen_generators_scorecard.png) | Visual publication scorecard for unseen generator benchmark |
| **Phase 1 Charts** | [`eval_all_generators_chart.png`](./eval_all_generators_chart.png) | Detection rate bar charts across all generator families |
| **Phase 2 Benchmark** | [`eval_28k_retrained_results.csv`](./eval_28k_retrained_results.csv) | Full 28,134 held-out disjoint validation set predictions |
| **Phase 2 Analysis** | [`eval_28k_retrained_analysis.png`](./eval_28k_retrained_analysis.png) | 4-panel ROC curve, score distribution, and generator breakdown |
| **Phase 2 Verification**| [`benchmark_28k_fixed.png`](./benchmark_28k_fixed.png) | Comparison chart verifying resolution of the short-audio padding bug |
| **Phase 3 Scorecard** | [`phase3_streaming_scorecard.png`](./phase3_streaming_scorecard.png) | 50,000-track out-of-domain streaming evaluation scorecard |
| **Latent Geometry** | [`tsne_latent_space.png`](./tsne_latent_space.png) | 2D t-SNE projection of the 704-D SupCon latent embedding space |
| **Narrative Rarity** | [`narrative_rarity_musicscope.png`](./narrative_rarity_musicscope.png) | Violin distribution of narrative rarity percentiles across generators |
| **Full Report** | [`comprehensive_evaluation_report.md`](./comprehensive_evaluation_report.md) | Comprehensive 3-phase technical scorecard report |

---

## Three-Phase Technical Evaluation Summary

### Phase 1: Zero-Shot Generalization on 8 Unseen AI Architectures
* **Evaluated Dataset:** $2,876$ tracks spanning 8 previously unseen AI generator families + FMA Human controls.
* **Core Metrics:**
  * **Macro-AUROC:** `0.9933`
  * **Overall Accuracy:** `98.54%`
  * **Human Control Accuracy:** `100.00%` ($0$ False Positives)
* **Breakdown by Architecture:**
  * **Diffusion / Flow-Matching:** `DiffRhythm` ($100.0\%$), `AceStep` ($100.0\%$), `ElevenLabs` ($100.0\%$), `Cassette` ($98.67\%$).
  * **Autoregressive Audio Transformers:** `SongGen` ($100.0\%$), `Brev` ($95.00\%$), `MusicGen` ($99.52\%$).
  * **Latent Diffusion / Symbolic:** `StableAudio` ($92.68\%$), `Mubert` ($93.75\%$).

### Phase 2: Complete Held-Out Disjoint Benchmark (28,134 Tracks)
* **Evaluated Dataset:** $28,134$ physical audio tracks ($12,172$ AI vs. $15,962$ Human) strictly disjoint from the $159,423$ training tracks ($0$ track leakage).
* **Core Metrics:**
  * **Macro-AUROC:** **`0.9970`**
  * **Overall Accuracy:** **`98.48%`**
  * **Equal Error Rate (EER):** **`1.55%`**
  * **Balanced Accuracy:** **`98.54%`**
  * **F1-Score:** **`0.9822`**
* **Per-Generator Detection Rates:**
  * `AudioLDM`: **`99.88%`** ($842/843$ detected)
  * `MusicGen`: **`99.52%`** ($837/841$ detected)
  * `Mustango`: **`99.88%`** ($841/842$ detected)
  * `Suno`: **`99.69%`** ($4,813/4,828$ detected)
  * `Udio`: **`99.86%`** ($4,811/4,818$ detected)
  * `Human (FMA)`: **`99.16%`** ($15,828/15,962$ correct)

### Phase 3: Out-of-Domain 50,000-Track Streaming Benchmark
* **Evaluated Dataset:** $50,000$ unseen streaming AI tracks ingested from Hugging Face (`bolshyC/Muse`).
* **Core Metrics:**
  * **AI Detection Rate:** **`98.89%`** ($49,444 / 50,000$ tracks detected)
  * **Mean Prediction Score:** **`0.9390`**
  * **High-Confidence Score Tier ($[0.80, 1.00]$):** **`93.70%`** ($46,852$ tracks)

---

## Production Model Architecture

```
Raw Audio (22.05 kHz)
   ├── Branch A (30s Padded) -> Log-Mel (128x1292) -> StyleEncoder (Conv2D) -> z_surf (576-D)
   └── Branch B (Raw Unpadded) -> Librosa/Scipy Extracts (128-D Moments)   -> x_narr (128-D)
                                                                                  │
Concat [z_surf (576-D) || x_narr (128-D)] = 704-D Fused Latent Representation ────┘
   │
   ▼
SupCon Projection Head (704-D -> 128-D Metric Space)
   │
   ▼
Calibrated Fusion MLP (128-D -> Logits) -> Temperature Scaler (T = 0.9083) -> Sigmoid
   │
   ▼
Calibrated Probability p ∈ [0, 1]  (Decision Threshold: τ = 0.1800)
```
