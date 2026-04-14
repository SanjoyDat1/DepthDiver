# ── DepthDiver Backend ─────────────────────────────────────────────────────────
# Engine: Depth Anything V2 Small (24 MB model weights)
# Peak RAM: ~350 MB   → compatible with Render FREE tier (512 MB)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for Pillow (JPEG / PNG / WebP support)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev \
        libpng-dev \
        libwebp-dev \
        zlib1g-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only FIRST (smaller wheel, no CUDA bloat)
# This avoids pulling in the 2 GB CUDA variant.
RUN pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Install transformers + HuggingFace Hub (for model download)
RUN pip install --no-cache-dir \
        transformers>=4.40.0 \
        accelerate>=0.28.0 \
        huggingface_hub>=0.22.0

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py .

# Pre-download the depth model during build so the first request isn't slow.
# If the build cache is cold this adds ~30 s but avoids a cold-start penalty.
RUN python - <<'EOF'
from transformers import pipeline
import torch
p = pipeline("depth-estimation",
             model="depth-anything/Depth-Anything-V2-Small-hf",
             device="cpu", torch_dtype=torch.float32)
print("Model pre-loaded successfully.")
EOF

# Environment — tune these via Render dashboard if needed
ENV MAX_IMAGE_PX=512 \
    MAX_CONCURRENT_JOBS=1 \
    JOB_TTL=1800 \
    WEB_CONCURRENCY=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
