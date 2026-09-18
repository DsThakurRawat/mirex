# Data

This directory is not tracked in git (see `.gitignore`).

Expected structure for the ~500GB dataset:

```
data/
├── raw/
│   ├── human/
│   │   ├── track_0001.wav
│   │   └── ...
│   └── ai_generated/
│       ├── suno_v5/
│       ├── udio/
│       └── ...
└── processed/
    ├── narrative_vectors.parquet   # Track A output
    └── style_embeddings.parquet    # Track B output
```

Labeling convention: `0 = human`, `1 = AI-generated`, matching the MIREX 2026 task spec.
