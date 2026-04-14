# DepthDiver — Backend Dockerfile (memory-optimised)
#
# Minimum recommended Render plan: Standard (2 GB RAM)
# Set environment variable SHARP_LOW_MEM=1 on Render (already default in code).
#
# Build: docker build -t depthdiver-backend .
# Run:   docker run -p 8000:8000 -e SHARP_LOW_MEM=1 depthdiver-backend

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl build-essential libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU-only (lightest possible — saves ~1 GB vs full torch)
RUN pip install --no-cache-dir \
      torch==2.2.2 torchvision==0.17.2 \
      --index-url https://download.pytorch.org/whl/cpu

# SHARP from source
RUN pip install --no-cache-dir \
      "git+https://github.com/apple/ml-sharp.git"

# App deps (Pillow for pre-resize, psutil for memory reporting)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Non-root user
RUN useradd -m appuser && chown -R appuser /app /tmp
USER appuser

# Memory-saving defaults (overridable via Render env vars)
ENV SHARP_DEVICE=cpu
ENV SHARP_LOW_MEM=1
ENV PORT=8000
# Single-process — do NOT increase workers; SHARP needs all available RAM for one job
ENV WEB_CONCURRENCY=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
