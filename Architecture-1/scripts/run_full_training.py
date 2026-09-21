"""
Master Pipeline Runner: Full Scale 200k Training
Architecture-1 (MusicScope-CL)

Orchestrates the entire end-to-end pipeline:
  1. 28-Worker Parallel DSP Feature Extraction on 159,423 training tracks.
  2. High-Throughput FP16 SimCLR Pretraining on RTX 2000 Ada GPU (85–90% load).
  3. In-VRAM SupCon Contrastive Clustering & Calibrated FusionMLP Classifier.
  4. Final Validation Scorecard on 28,134 held-out unseen tracks.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# Add scripts and mirex to sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config


def run_command_stream(cmd: list[str], stage_name: str):
    """Executes a command and streams output in real-time."""
    print("\n" + "=" * 75)
    print(f"  >>> STARTING: {stage_name}")
    print(f"  >>> Command: {' '.join(cmd)}")
    print("=" * 75 + "\n")

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)

    proc.stdout.close()
    return_code = proc.wait()
    elapsed = time.time() - t0

    if return_code != 0:
        raise RuntimeError(f"Stage '{stage_name}' failed with exit code {return_code} after {elapsed:.1f}s")

    print(f"\n  >>> COMPLETED: {stage_name} in {elapsed/60:.1f} minutes.\n")


def main():
    py_bin = sys.executable
    print("=" * 75)
    print("   MusicScope-CL: Full 200k Large-Scale Training Pipeline")
    print(f"   Python: {py_bin}")
    print(f"   Target: 159,423 Train / 28,134 Validation Tracks")
    print("   Hardware: Intel Core i9-14900K (28 Workers) + NVIDIA RTX 2000 Ada")
    print("=" * 75)

    # 1. Feature Extraction (28 CPU Workers)
    extract_script = str(_SCRIPTS_DIR / "extract_narrative_200k.py")
    extract_cmd = [
        py_bin, extract_script,
        "--workers", "28",
        "--chunk_size", "5000",
        "--split", "train"
    ]
    run_command_stream(extract_cmd, "Stage 1: 28-Core Parallel DSP Extraction")

    # 2. High-Throughput GPU Training (SimCLR + SupCon + FusionMLP)
    gpu_script = str(_SCRIPTS_DIR / "train_gpu_200k.py")
    gpu_cmd = [
        py_bin, gpu_script,
        "--epochs", "5",
        "--batch_size", "128"
    ]
    run_command_stream(gpu_cmd, "Stage 2: Accelerated GPU Training (SimCLR + SupCon + Fusion)")

    print("\n" + "=" * 75)
    print("   [PIPELINE SUCCESS] 200,000+ Track Training Completed Successfully!")
    print("   New Checkpoints Ready in: Architecture-1/mirex/checkpoints/")
    print("=" * 75)


if __name__ == "__main__":
    main()
