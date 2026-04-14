"""
DepthDiver — FastAPI Backend (async job queue, memory-optimised)

Memory reality check
--------------------
SHARP loads a full PyTorch neural network. Minimum RAM required:
  - Standard quality (default): ~2.0 GB
  - Low quality (SHARP_LOW_MEM=1): ~1.2 GB  ← recommended for Render Standard
  - Absolute minimum (SHARP_TINY=1): ~0.8 GB ← may degrade quality visibly

Render plan requirements:
  - Free  (512 MB)  → WILL crash, not viable
  - Starter (512 MB)→ WILL crash, not viable
  - Standard (2 GB) → works with SHARP_LOW_MEM=1
  - Pro   (4 GB)    → works at any quality setting

Endpoints:
  GET  /health                   → liveness + memory stats
  POST /generate                 → submit image, returns {job_id} immediately
  GET  /jobs/{job_id}/status     → poll: {status, progress, error}
  GET  /jobs/{job_id}/result     → download .ply (cleanup after serving)
  GET  /jobs/{job_id}/scene.ply  → stream .ply to viewer (no cleanup)
"""

from __future__ import annotations
import gc, os, shutil, subprocess, sys, tempfile, threading, time, uuid, logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("depthdiver")

# ── Config ───────────────────────────────────────────────────────────────────
SHARP_CMD        = os.environ.get("SHARP_CMD", "sharp")
SHARP_DEVICE     = os.environ.get("SHARP_DEVICE", "cpu")
SHARP_TIMEOUT    = int(os.environ.get("SHARP_TIMEOUT", "600"))
SHARP_CHECKPOINT = os.environ.get("SHARP_CHECKPOINT_PATH", "")

# Memory-saving modes — set via Render environment variables
# SHARP_LOW_MEM=1   → resize images to 640 px max + limit threads
# SHARP_TINY=1      → resize images to 512 px max + minimal threads (most savings)
SHARP_LOW_MEM    = os.environ.get("SHARP_LOW_MEM", "1") == "1"   # on by default
SHARP_TINY       = os.environ.get("SHARP_TINY", "0") == "1"

# Max image dimension sent to SHARP (smaller = less RAM, faster, smaller PLY)
if SHARP_TINY:
    MAX_IMAGE_PX = int(os.environ.get("MAX_IMAGE_PX", "512"))
elif SHARP_LOW_MEM:
    MAX_IMAGE_PX = int(os.environ.get("MAX_IMAGE_PX", "768"))
else:
    MAX_IMAGE_PX = int(os.environ.get("MAX_IMAGE_PX", "1024"))

# Block new jobs while one is still running (single worker can't run two in parallel)
MAX_CONCURRENT   = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))

WORK_ROOT        = Path(tempfile.gettempdir()) / "depthdiver_runs"
JOB_TTL          = int(os.environ.get("JOB_TTL", "1800"))   # 30 min

WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock  = threading.Lock()
_active_sem = threading.Semaphore(MAX_CONCURRENT)


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _active_count() -> int:
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j.get("status") == "processing")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DepthDiver API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["content-disposition"],
)


@app.middleware("http")
async def permissive_cors(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_info = {"used_mb": mem.used // 1_000_000, "total_mb": mem.total // 1_000_000,
                    "percent": mem.percent}
    except ImportError:
        mem_info = {}

    return {
        "status":        "ok",
        "sharp_available": shutil.which(SHARP_CMD) is not None,
        "device":        SHARP_DEVICE,
        "low_mem_mode":  SHARP_LOW_MEM,
        "tiny_mode":     SHARP_TINY,
        "max_image_px":  MAX_IMAGE_PX,
        "active_jobs":   _active_count(),
        "memory":        mem_info,
    }


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image.")

    # Reject if already at capacity (prevents OOM from parallel jobs)
    if _active_count() >= MAX_CONCURRENT:
        raise HTTPException(503, detail="Server busy — one job at a time. Please try again in a minute.")

    job_id  = uuid.uuid4().hex[:12]
    run_dir = WORK_ROOT / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if not content:
        raise HTTPException(400, detail="Uploaded file is empty.")

    suffix   = Path(file.filename or "photo.jpg").suffix or ".jpg"
    img_path = run_dir / f"input{suffix}"
    img_path.write_bytes(content)
    log.info("[%s] Saved %s (%d KB)", job_id, img_path.name, len(content) // 1024)

    with _jobs_lock:
        _jobs[job_id] = {
            "status":   "pending",
            "progress": 0.0,
            "run_dir":  str(run_dir),
            "img_path": str(img_path),
            "ply_path": None,
            "error":    None,
            "created":  time.time(),
        }

    threading.Thread(target=_run_sharp, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, detail="Job not found.")
    return {"job_id": job_id, "status": job["status"],
            "progress": job["progress"], "error": job.get("error")}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """One-time download — cleans up temp files after serving."""
    return _serve_ply(job_id, cleanup=True)


@app.get("/jobs/{job_id}/scene.ply")
def job_scene_ply(job_id: str):
    """
    Viewer URL — ends in '.ply' so GaussianSplats3D auto-detects format.
    No immediate cleanup; TTL daemon handles it.
    """
    return _serve_ply(job_id, cleanup=False)


def _serve_ply(job_id: str, *, cleanup: bool) -> FileResponse:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, detail="Job not found (may have expired).")
    if job["status"] == "failed":
        raise HTTPException(422, detail=job.get("error", "Generation failed."))
    if job["status"] != "done":
        raise HTTPException(409, detail=f"Job is {job['status']} — poll /status first.")

    ply_path = Path(job["ply_path"])
    if not ply_path.exists():
        raise HTTPException(500, detail="Result file missing on server.")

    log.info("[%s] Serving PLY (%.1f MB, cleanup=%s)", job_id, ply_path.stat().st_size / 1e6, cleanup)
    return FileResponse(
        str(ply_path),
        media_type="application/octet-stream",
        filename="scene.ply",
        background=_CleanupTask(job_id, Path(job["run_dir"])) if cleanup else None,
    )


# ── Image pre-processing ─────────────────────────────────────────────────────
def _resize_image(img_path: Path, max_px: int) -> Path:
    """
    Resize the image so its longest edge ≤ max_px.
    Smaller input → SHARP uses less RAM + produces a smaller .ply file.
    Returns the path to the (possibly new) resized image.
    """
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as img:
            w, h = img.size
            if max(w, h) <= max_px:
                return img_path  # already small enough
            if w >= h:
                new_w, new_h = max_px, max(1, int(h * max_px / w))
            else:
                new_w, new_h = max(1, int(w * max_px / h)), max_px
            log.info("Resizing %dx%d → %dx%d (max_px=%d)", w, h, new_w, new_h, max_px)
            img = img.convert("RGB")
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            out = img_path.with_name(f"resized_{img_path.name}")
            img.save(out, "JPEG", quality=88, optimize=True)
            return out
    except Exception as exc:
        log.warning("Could not resize image: %s — using original", exc)
        return img_path


# ── SHARP worker ─────────────────────────────────────────────────────────────
def _run_sharp(job_id: str):
    # One job at a time — block until the semaphore is free
    _active_sem.acquire()
    try:
        __run_sharp_inner(job_id)
    finally:
        _active_sem.release()
        # Force Python GC after SHARP exits to reclaim any lingering objects
        gc.collect()


def __run_sharp_inner(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return

    img_path = Path(job["img_path"])
    run_dir  = Path(job["run_dir"])
    out_dir  = run_dir / "out"
    out_dir.mkdir(exist_ok=True)

    _set_job(job_id, status="processing", progress=0.05)

    # ── Step 1: resize image to reduce SHARP memory footprint ────────────────
    img_path = _resize_image(img_path, MAX_IMAGE_PX)
    _set_job(job_id, img_path=str(img_path), progress=0.10)

    log.info("[%s] SHARP starting | device=%s | max_px=%d | low_mem=%s",
             job_id, SHARP_DEVICE, MAX_IMAGE_PX, SHARP_LOW_MEM)

    cmd = [
        SHARP_CMD, "predict",
        "-i", str(img_path),
        "-o", str(out_dir),
        "--device", SHARP_DEVICE,
    ]
    if SHARP_CHECKPOINT:
        cmd += ["--checkpoint", SHARP_CHECKPOINT]

    # ── Step 2: set environment variables that limit PyTorch memory usage ─────
    env = dict(os.environ)
    # Single-threaded CPU ops → lower peak RAM
    env.update({
        "OMP_NUM_THREADS":        "1",
        "MKL_NUM_THREADS":        "1",
        "OPENBLAS_NUM_THREADS":   "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS":    "1",
        # Disable PyTorch CUDA caching allocator (not relevant on CPU but harmless)
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        # Tell PyTorch not to use shared memory for data loading workers
        "PYTHONHASHSEED": "0",
    })
    if SHARP_TINY:
        # Extra-aggressive: disable JIT compilation cache (saves ~100 MB)
        env["PYTORCH_JIT"] = "0"

    # ── Step 3: run SHARP ─────────────────────────────────────────────────────
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SHARP_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        log.error("[%s] SHARP timed out after %ds", job_id, SHARP_TIMEOUT)
        _set_job(job_id, status="failed",
                 error=f"3D generation timed out after {SHARP_TIMEOUT // 60} minutes. "
                       "Try a simpler photo or upgrade your Render plan.")
        return
    except Exception as exc:
        log.exception("[%s] Subprocess error", job_id)
        _set_job(job_id, status="failed", error=str(exc))
        return

    _set_job(job_id, progress=0.90)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        log.error("[%s] SHARP exit %d:\n%s", job_id, proc.returncode, tail)
        # Friendly message for OOM
        friendly = tail
        if "out of memory" in tail.lower() or "killed" in tail.lower():
            friendly = (
                "The server ran out of memory while generating the 3D scene. "
                "On Render: upgrade to the Standard plan (2 GB RAM) and set "
                "SHARP_LOW_MEM=1 in environment variables."
            )
        _set_job(job_id, status="failed", error=friendly)
        return

    # ── Step 4: locate the output .ply ───────────────────────────────────────
    ply = _find_ply(out_dir, img_path.stem)
    if ply is None:
        log.error("[%s] No .ply found. stdout:\n%s", job_id, proc.stdout[-400:])
        _set_job(job_id, status="failed",
                 error="SHARP ran but produced no .ply file. Check model weights are installed.")
        return

    log.info("[%s] Done — PLY %.1f MB", job_id, ply.stat().st_size / 1e6)
    _set_job(job_id, status="done", progress=1.0, ply_path=str(ply))


# ── Cleanup helpers ───────────────────────────────────────────────────────────
class _CleanupTask:
    def __init__(self, job_id: str, path: Path):
        self.job_id = job_id
        self.path   = path

    def __call__(self):
        shutil.rmtree(self.path, ignore_errors=True)
        with _jobs_lock:
            _jobs.pop(self.job_id, None)
        log.info("[%s] Cleaned up", self.job_id)


def _ttl_cleanup():
    """Daemon: purge jobs older than JOB_TTL seconds."""
    while True:
        time.sleep(600)
        now     = time.time()
        expired = []
        with _jobs_lock:
            for jid, job in list(_jobs.items()):
                if now - job.get("created", now) > JOB_TTL:
                    expired.append((jid, job.get("run_dir")))
        for jid, run_dir in expired:
            if run_dir:
                shutil.rmtree(run_dir, ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(jid, None)
            log.info("[%s] TTL expired", jid)


threading.Thread(target=_ttl_cleanup, daemon=True).start()


def _find_ply(out_dir: Path, stem: str) -> Optional[Path]:
    for p in [out_dir / f"{stem}.ply", out_dir / "input.ply",
              out_dir / "output.ply", out_dir / "scene.ply"]:
        if p.exists() and p.stat().st_size > 0:
            return p
    for p in sorted(out_dir.rglob("*.ply")):
        if p.stat().st_size > 0:
            return p
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
