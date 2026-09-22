"""
Narrative feature group latency profiler
"""
import sys
import time
from pathlib import Path

_CUR_DIR = Path("Architecture-1/mirex").resolve()
sys.path.insert(0, str(_CUR_DIR))

import config
import librosa
from dataset import load_audio_chunk
from track_a_narrative import (
    harmonic_features,
    rhythmic_features,
    motivic_features,
    spectral_features,
    dynamic_features,
)

p = r"D:\mirex\data\raw\fma\fma_large\000\000002.mp3"
t0 = time.perf_counter()
y = load_audio_chunk(p)
print(f"Audio Load: {(time.perf_counter() - t0)*1000:.1f} ms")

for name, fn in [
    ("harmonic (CQT)", harmonic_features),
    ("rhythmic (onsets)", rhythmic_features),
    ("motivic (recurrence)", motivic_features),
    ("spectral (flatness/cent)", spectral_features),
    ("dynamic (RMS/crest)", dynamic_features),
]:
    t0 = time.perf_counter()
    res = fn(y, config.SAMPLE_RATE)
    print(f"  {name:25s}: {(time.perf_counter() - t0)*1000:6.1f} ms")
