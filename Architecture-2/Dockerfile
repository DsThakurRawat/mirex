# MIREX 2026 submission container (plan §10).
# Build AFTER training: checkpoints/ and any HF model caches must exist.
#   docker build -t mirex2026-detector .
#   docker run --gpus all -v /path/test:/data/input -v /path/out:/data/output \
#       mirex2026-detector --input_dir /data/input --output_csv /data/output/scores.csv
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source, trained weights, fitted fusion, and the pre-downloaded HF cache
# (SSL trunks) — the container must run FULLY OFFLINE (plan §1: external API
# calls prohibited).
COPY src/ /app/src/
COPY checkpoints/ /app/checkpoints/
# Populate with: HF_HOME=./hf_cache python -c "see README packaging step"
COPY hf_cache/ /root/.cache/huggingface/

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    MIREX_CHECKPOINT_DIR=/app/checkpoints PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/src/inference.py"]
CMD ["--input_dir", "/data/input", "--output_csv", "/data/output/scores.csv", "--mode", "ensemble"]
