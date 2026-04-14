# DepthDiver — Backend Dockerfile
# Installs PyTorch (CPU), SHARP, and the FastAPI server.
# Build: docker build -t depthdiver-backend .
# Run:   docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-site.netlify.app depthdiver-backend

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU-only — lightweight, works on Render free tier)
RUN pip install --no-cache-dir \
      torch==2.2.2 torchvision==0.17.2 \
      --index-url https://download.pytorch.org/whl/cpu

# Install SHARP from source
RUN pip install --no-cache-dir \
      "git+https://github.com/apple/ml-sharp.git"

# Install FastAPI stack
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# App code
COPY main.py .

# Non-root user (security best practice)
RUN useradd -m appuser && chown -R appuser /app /tmp
USER appuser

ENV SHARP_DEVICE=cpu
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
