# Architecture-1 Local Development & Benchmarking Scripts

This directory contains local utilities, data preparation tools, multi-core feature extractors, and benchmark visualization scripts for **Architecture-1 (MusicScope-CL)**.

> [!NOTE]
> These scripts are used for offline dataset preparation, large-scale multi-core training, and local analysis. They are kept separate from `Architecture-1/mirex/`, which contains only the clean, official MIREX submission package.

---

## Script Catalog

| Script / Artifact | Description | Usage Command |
| :--- | :--- | :--- |
| **[`split_dataset_200k.py`](file:///d:/mirex/Architecture-1/scripts/split_dataset_200k.py)** | Queries `metadata.db`, filters out macOS sidecar files and corrupt audio, and generates a clean 85% Train (159k) / 15% Val (28k) stratified split. | `python split_dataset_200k.py` |
| **[`extract_narrative_200k.py`](file:///d:/mirex/Architecture-1/scripts/extract_narrative_200k.py)** | High-throughput multi-process (18–20 workers) 128-D narrative feature extractor with auto-resume and 5k-track milestone saving. | `python extract_narrative_200k.py --workers 18 --chunk_size 5000` |
| **[`eval_1000.py`](file:///d:/mirex/Architecture-1/scripts/eval_1000.py)** | Evaluates unseen audio tracks across multiple generator families (FMA, Suno, Echoes, FakeMusicCaps) and outputs detailed CSV predictions. | `python eval_1000.py --n_human 2500 --n_ai 2500` |
| **[`visualize_eval_5000.py`](file:///d:/mirex/Architecture-1/scripts/visualize_eval_5000.py)** | Computes AUROC, EER, Youden's J threshold, confusion matrix, and generates a 6-panel publication-quality analysis chart. | `python visualize_eval_5000.py` |
| **[`evaluate_test_set.py`](file:///d:/mirex/Architecture-1/scripts/evaluate_test_set.py)** | Runs inference and scoring over held-out evaluation sets. | `python evaluate_test_set.py` |
| **[`test_pipeline.py`](file:///d:/mirex/Architecture-1/scripts/test_pipeline.py)** | Automated unit and integration tests covering the 3-stage contrastive pipeline. | `python test_pipeline.py` |
| **[`umap_analysis.py`](file:///d:/mirex/Architecture-1/scripts/umap_analysis.py)** | Dimensionality reduction analysis visualizing the separation between human and AI representations. | `python umap_analysis.py` |
| **[`eval_5000_results.csv`](file:///d:/mirex/Architecture-1/scripts/eval_5000_results.csv)** | Raw predictions, ground-truth labels, and probabilities for the 5,000-track benchmark. | Output data |

---

## Environment

All scripts in this directory are configured to run with the project's Python 3.12 virtual environment:
```powershell
d:\mirex\.venv312\Scripts\python.exe <script_name>.py
```
