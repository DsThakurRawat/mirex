"""
Audio I/O speed benchmark: SoundFile + SOXR vs Librosa
"""
import time
import soundfile as sf
import librosa
import numpy as np

p = r"D:\mirex\data\raw\fma\fma_large\000\000002.mp3"

# 1. Librosa load
t0 = time.perf_counter()
y1, sr = librosa.load(p, sr=22050, mono=True, duration=30.0)
t_lib = (time.perf_counter() - t0) * 1000

# 2. Soundfile read
t0 = time.perf_counter()
with sf.SoundFile(p) as f:
    orig_sr = f.samplerate
    frames_to_read = int(orig_sr * 30.0)
    data = f.read(frames=frames_to_read, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if orig_sr != 22050:
        import soxr
        y2 = soxr.resample(data, orig_sr, 22050)
    else:
        y2 = data
t_sf = (time.perf_counter() - t0) * 1000

print(f"Librosa load   : {t_lib:.1f} ms")
print(f"SoundFile+soxr : {t_sf:.1f} ms ({t_lib/t_sf:.1f}x faster!)")
print(f"Waveform shape match: {y1.shape} vs {y2.shape}")
