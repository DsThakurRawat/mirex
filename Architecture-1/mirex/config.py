"""MusicScope-CL — Central configuration.

Paths, constants, and shared references used across all modules.
Environment overrides: MIREX_DATA_DIR, MIREX_CHECKPOINT_DIR.
"""
import os
from pathlib import Path

# --- Directory structure --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# Shared data lives at the mirex root (two levels up from Architecture-1/mirex/)
_MIREX_ROOT = PROJECT_ROOT.parent.parent
DATA_DIR = Path(os.environ.get("MIREX_DATA_DIR", _MIREX_ROOT / "data"))

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = Path(os.environ.get(
    "MIREX_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))

# --- Audio constants ------------------------------------------------------
SAMPLE_RATE = 22050          # librosa default; both branches use this
CHUNK_SECONDS = 30           # seconds per analysis chunk
N_MELS = 128                 # mel-spectrogram bands for Branch A

# --- Structural branch (Branch B / track_a_narrative) ---------------------
NARRATIVE_DIM = 128          # target dimensionality of the Musical Narrative Vector

# --- Surface branch (Branch A / track_b_style) ----------------------------
SURFACE_DIM = 576            # MobileNetV3-Small last-conv output channels
SIMCLR_PROJ_DIM = 128        # SimCLR projection head output
SIMCLR_HIDDEN_DIM = 256      # SimCLR projection head hidden layer

# --- Fusion ---------------------------------------------------------------
FUSED_DIM = SURFACE_DIM + NARRATIVE_DIM  # 704
SUPCON_PROJ_DIM = 128        # SupCon projection head output

# --- Training defaults ----------------------------------------------------
SEED = 42
SIMCLR_EPOCHS = 20
SIMCLR_BATCH_SIZE = 32
SIMCLR_LR = 3e-4
SIMCLR_TEMPERATURE = 0.5

SUPCON_EPOCHS = 30
SUPCON_BATCH_SIZE = 64
SUPCON_LR = 1e-3
SUPCON_TEMPERATURE = 0.1

FUSION_EPOCHS = 30
FUSION_LR = 1e-3
LABEL_SMOOTHING = 0.05

# --- Inference / MIL / Decision Boundaries --------------------------------
MIL_CHUNKS = 3               # number of 30s chunks per track at test time
FALLBACK_SCORE = 0.5          # if scoring fails
PER_TRACK_TIMEOUT_S = 60

# Calibrated Decision Thresholds (Empirical from 28k Held-Out Benchmark):
# - Balanced (Default): 99.1% Human Accuracy, catches subtle AI generators
# - Strict: Zero tolerance for AI music (catches 95%+ AI, optimal Youden's J)
# - Studio: High specificity baseline
THRESHOLD_BALANCED = 0.18
THRESHOLD_STRICT = 0.05
THRESHOLD_CONSERVATIVE = 0.50
DEFAULT_DECISION_THRESHOLD = THRESHOLD_BALANCED

# --- Labels ---------------------------------------------------------------
REAL_LABEL = 0
AI_LABEL = 1

# --- Audio file extensions ------------------------------------------------
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}


def ensure_dirs():
    """Create the directory tree. Call explicitly — not at import time."""
    for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR]:
        p.mkdir(parents=True, exist_ok=True)
