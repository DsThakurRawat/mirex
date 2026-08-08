"""Central configuration: paths, dataset registry, strata grid, branch settings.

Everything that multiple modules must agree on lives here. Environment
overrides: MIREX_DATA_DIR, MIREX_CHECKPOINT_DIR, MUREKA_API_KEY, MINIMAX_API_KEY.
"""
import os
from pathlib import Path

# --- Directory structure -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MIREX_DATA_DIR", PROJECT_ROOT / "data"))

RAW_DATA_DIR = DATA_DIR / "raw"
GENERATED_DATA_DIR = DATA_DIR / "generated"      # self-generated AI tracks
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HARNESS_CACHE_DIR = DATA_DIR / "harness_cache"    # materialized eval strata audio

SDD_DIR = RAW_DATA_DIR / "sdd"                    # metadata only — audio is QUARANTINED
MTG_JAMENDO_DIR = RAW_DATA_DIR / "mtg_jamendo"

QUARANTINE_FILE = PROCESSED_DATA_DIR / "quarantine_blocklist.json"
QUARANTINE_REPORT = PROCESSED_DATA_DIR / "quarantine_report.json"
METADATA_DB = PROCESSED_DATA_DIR / "metadata.db"
CENSUS_REPORT = PROCESSED_DATA_DIR / "dataset_census.json"

CHECKPOINT_DIR = Path(os.environ.get("MIREX_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
FUSION_DIR = CHECKPOINT_DIR / "fusion"

# --- Task constants ------------------------------------------------------
SEED = 42
SAMPLE_RATE = 44100          # canonical internal rate; branches resample as needed
EVAL_SAMPLE_RATES = (44100, 48000)   # what MIREX may feed us

# The six named hidden-test generator families (canonical lowercase keys).
TEST_FAMILIES = ["suno", "udio", "mureka", "minimax", "yue", "ace-step"]
# Additional open-generator families we train/hold out for decoder-lineage diversity.
FILLER_FAMILIES = ["diffrhythm", "musicgen", "stable-audio", "riffusion",
                   "songgen", "mubert", "audioldm", "mustango"]
REAL_LABEL, AI_LABEL = 0, 1

# --- Internal harness strata grid (§5 of the plan) -----------------------
# Each stratum = generator_family x condition. Conditions are deterministic
# perturbations applied by harness.py via the simulator's deterministic API.
HARNESS_CONDITIONS = {
    "clean":     {},
    "mp3_128":   {"codec": ("mp3", 128)},
    "mp3_64":    {"codec": ("mp3", 64)},
    "opus_48":   {"codec": ("opus", 48)},
    "rs_22050":  {"resample": 22050},
    "pitch_up1": {"pitch_semitones": 1.0},
}
HARNESS_EXCERPT_SECONDS = [30, 60, 120, None]     # None = full track

# --- Delivery-chain simulator defaults (§4.4) ----------------------------
SIM_CLEAN_PROB = 0.20        # fraction of items passing through untouched
SIM_CODECS = [("mp3", (32, 320)), ("aac", (32, 256)),
              ("vorbis", (45, 256)), ("opus", (24, 192))]
SIM_RESAMPLE_RATES = [22050, 32000, 44100, 48000]
SIM_LUFS_RANGE = (-20.0, -7.0)
SIM_MONO_FOLD_PROB = 0.15
SIM_EXCERPT_MIN_S = 10.0

# --- Branch configs ------------------------------------------------------
BRANCHES = {
    # SVDD-winning recipe: speech SSL front-end, 16 kHz mono, short chunks.
    "a": {"model_name": "facebook/wav2vec2-xls-r-300m", "input_sr": 16000,
          "chunk_s": 6.0, "lr": 1e-5, "head_lr": 1e-4, "batch_size": 16},
    # Music SSL. 24 kHz mono. Set MIREX_SMALL=1 to use the 95M model (dev boxes).
    "b": {"model_name": ("m-a-p/MERT-v1-95M" if os.environ.get("MIREX_SMALL")
                         else "m-a-p/MERT-v1-330M"),
          "input_sr": 24000, "chunk_s": 10.0, "lr": 1e-5, "head_lr": 1e-4,
          "batch_size": 8},
    # Physics: shift-invariant log-frequency CNN + fakeprint GBDT. 44.1 kHz.
    "c": {"input_sr": 44100, "chunk_s": 10.0, "n_fft": 4096, "hop": 1024,
          "lr": 3e-4, "batch_size": 32},
    # Long-context ConvNeXt-T on 120 s mels. 44.1 kHz mono.
    "d": {"input_sr": 44100, "chunk_s": 120.0, "n_mels": 128, "n_fft": 2048,
          "hop": 1024, "lr": 1e-4, "batch_size": 4},
    # Real-only anomaly: OC-Softmax embedding model. 24 kHz mono.
    "e": {"input_sr": 24000, "chunk_s": 10.0, "lr": 1e-4, "batch_size": 32,
          "emb_dim": 256, "oc_m_real": 0.9, "oc_m_fake": 0.2, "oc_alpha": 20.0},
}

# Per-epoch cap on any single source dataset's share within its class (§4.3).
SOURCE_QUOTA_CAP = 0.35

# --- Track-level aggregation defaults (§8; tuned on harness) -------------
AGG_LAMBDA = 0.3
AGG_TOPK_FRAC = 0.25

# --- Inference / submission ----------------------------------------------
PER_TRACK_TIMEOUT_S = 60          # hard cap; fallback score on breach
FALLBACK_SCORE = 0.5
CONFOUND_GATE_AUROC = 0.60

# --- API keys (never hardcode) -------------------------------------------
MUREKA_API_KEY = os.environ.get("MUREKA_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

def ensure_dirs():
    """Create the directory tree. Call explicitly — not at import time."""
    for p in [RAW_DATA_DIR, GENERATED_DATA_DIR, PROCESSED_DATA_DIR,
              HARNESS_CACHE_DIR, SDD_DIR, MTG_JAMENDO_DIR, CHECKPOINT_DIR,
              FUSION_DIR]:
        p.mkdir(parents=True, exist_ok=True)
