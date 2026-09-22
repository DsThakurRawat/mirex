"""
Environment and hardware acceleration library inspector
"""
for pkg in ["onnx", "onnxruntime", "onnxruntime_gpu", "tensorrt", "soundfile", "soxr", "torchaudio"]:
    try:
        m = __import__(pkg)
        v = getattr(m, "__version__", "installed")
        print(f"{pkg:16s}: {v}")
    except ImportError:
        print(f"{pkg:16s}: NOT installed")
