"""
DepthDiver — FastAPI Backend
Deploy on Render (render.yaml included) or any Python-capable host.

Endpoints:
  GET  /health          → liveness check
  POST /generate        → receive image, run SHARP, return .ply bytes
"""

from __future__ import annotations
import os, shutil, subprocess, tempfile, uuid, logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("depthdiver")

# ── Config ───────────────────────────────────────────────────────────────────
SHARP_CMD        = os.environ.get("SHARP_CMD", "sharp")
SHARP_DEVICE     = os.environ.get("SHARP_DEVICE", "cpu")
SHARP_TIMEOUT    = int(os.environ.get("SHARP_TIMEOUT", "300"))   # seconds
SHARP_CHECKPOINT = os.environ.get("SHARP_CHECKPOINT_PATH", "")   # optional
WORK_ROOT        = Path(tempfile.gettempdir()) / "depthdiver_runs"
ALLOWED_ORIGINS  = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://your-site.netlify.app,http://localhost:3000"
).split(",")

WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DepthDiver API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict via ALLOWED_ORIGINS env in production
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["content-disposition"],
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    sharp_ok = shutil.which(SHARP_CMD) is not None
    return {
        "status": "ok",
        "sharp_available": sharp_ok,
        "sharp_cmd": SHARP_CMD,
        "device": SHARP_DEVICE,
    }


@app.post("/generate")
async def generate(file: UploadFile = File(..., description="Input photograph")):
    """
    Accept an image upload, run SHARP to produce a Gaussian Splat .ply,
    and return the raw .ply bytes as a binary download.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (JPEG, PNG, etc.)")

    session_id = uuid.uuid4().hex[:10]
    run_dir    = WORK_ROOT / session_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Save upload
        suffix   = Path(file.filename or "photo.jpg").suffix or ".jpg"
        img_path = run_dir / f"input{suffix}"
        content  = await file.read()
        if len(content) == 0:
            raise HTTPException(400, detail="Uploaded file is empty.")
        img_path.write_bytes(content)
        log.info("Saved upload: %s (%d bytes)", img_path.name, len(content))

        # 2. Build SHARP command
        out_dir = run_dir / "out"
        out_dir.mkdir(exist_ok=True)

        cmd = [
            SHARP_CMD, "predict",
            "-i", str(img_path),
            "-o", str(out_dir),
            "--device", SHARP_DEVICE,
        ]
        if SHARP_CHECKPOINT:
            cmd += ["--checkpoint", SHARP_CHECKPOINT]

        log.info("Running: %s", " ".join(cmd))

        # 3. Run SHARP
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SHARP_TIMEOUT,
        )

        if proc.returncode != 0:
            log.error("SHARP stdout:\n%s\nstderr:\n%s", proc.stdout[-1000:], proc.stderr[-1000:])
            raise HTTPException(
                500,
                detail=(
                    f"SHARP exited with code {proc.returncode}. "
                    f"Tail: {(proc.stderr or proc.stdout)[-300:]}"
                ),
            )

        # 4. Find output .ply
        ply_path = _find_ply(out_dir, img_path.stem)
        if ply_path is None:
            log.error("No .ply found under %s", out_dir)
            log.error("SHARP stdout:\n%s", proc.stdout[-1000:])
            raise HTTPException(
                500,
                detail=(
                    "SHARP ran but produced no .ply file. "
                    "Check that SHARP is correctly installed and the model weights are present."
                ),
            )

        log.info("Returning .ply: %s (%.1f MB)", ply_path.name, ply_path.stat().st_size / 1e6)

        return FileResponse(
            str(ply_path),
            media_type="application/octet-stream",
            filename="scene.ply",
            background=_cleanup_task(run_dir),
        )

    except subprocess.TimeoutExpired:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(504, detail=f"SHARP timed out after {SHARP_TIMEOUT}s.")
    except HTTPException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        log.exception("Unexpected error")
        raise HTTPException(500, detail=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_ply(out_dir: Path, stem: str) -> Path | None:
    """Return the first .ply we can find under out_dir."""
    # SHARP usually names it <input-stem>.ply
    candidates = [
        out_dir / f"{stem}.ply",
        out_dir / "input.ply",
        out_dir / "output.ply",
        out_dir / "scene.ply",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    # Fallback: any .ply
    for p in sorted(out_dir.rglob("*.ply")):
        if p.stat().st_size > 0:
            return p
    return None


class _cleanup_task:
    """BackgroundTask-compatible callable that removes a directory."""
    def __init__(self, path: Path):
        self.path = path

    def __call__(self):
        shutil.rmtree(self.path, ignore_errors=True)
        log.info("Cleaned up %s", self.path)


# ── Run (local dev) ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
