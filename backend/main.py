"""
DepthDiver — FastAPI Backend (async job queue)

Endpoints:
  GET  /health                   → liveness check
  POST /generate                 → submit image, returns {job_id} immediately
  GET  /jobs/{job_id}/status     → poll: {status, progress, error}
  GET  /jobs/{job_id}/result     → download .ply when status == "done"

Using an async job pattern avoids Cloudflare's ~100s upstream timeout that
kills long-lived HTTP connections while SHARP processes the image on CPU.
"""

from __future__ import annotations
import os, shutil, subprocess, tempfile, threading, time, uuid, logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("depthdiver")

# ── Config ───────────────────────────────────────────────────────────────────
SHARP_CMD        = os.environ.get("SHARP_CMD", "sharp")
SHARP_DEVICE     = os.environ.get("SHARP_DEVICE", "cpu")
SHARP_TIMEOUT    = int(os.environ.get("SHARP_TIMEOUT", "600"))   # 10 min max
SHARP_CHECKPOINT = os.environ.get("SHARP_CHECKPOINT_PATH", "")
WORK_ROOT        = Path(tempfile.gettempdir()) / "depthdiver_runs"
JOB_TTL          = int(os.environ.get("JOB_TTL", "3600"))        # 1 h before cleanup

WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ── In-memory job store (single worker only — see Dockerfile CMD) ─────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DepthDiver API", version="2.0.0")

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
    """Ensure CORP header so pages with COEP:require-corp can fetch us."""
    response = await call_next(request)
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    sharp_ok = shutil.which(SHARP_CMD) is not None
    active = sum(1 for j in _jobs.values() if j.get("status") == "processing")
    return {
        "status": "ok",
        "sharp_available": sharp_ok,
        "sharp_cmd": SHARP_CMD,
        "device": SHARP_DEVICE,
        "active_jobs": active,
    }


@app.post("/generate")
async def generate(file: UploadFile = File(..., description="Input photograph")):
    """
    Accept an image, save it, kick off SHARP in a background thread,
    and return {job_id} immediately so the client can poll.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (JPEG, PNG, etc.)")

    job_id  = uuid.uuid4().hex[:12]
    run_dir = WORK_ROOT / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if not content:
        raise HTTPException(400, detail="Uploaded file is empty.")

    suffix   = Path(file.filename or "photo.jpg").suffix or ".jpg"
    img_path = run_dir / f"input{suffix}"
    img_path.write_bytes(content)
    log.info("[%s] Saved %s (%d bytes)", job_id, img_path.name, len(content))

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

    # Launch SHARP in background — returns to client immediately
    t = threading.Thread(target=_run_sharp, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, detail="Job not found (may have expired).")
    return {
        "job_id":   job_id,
        "status":   job["status"],    # pending | processing | done | failed
        "progress": job["progress"],  # 0.0 – 1.0
        "error":    job.get("error"),
    }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, detail="Job not found (may have expired).")
    if job["status"] == "failed":
        raise HTTPException(422, detail=job.get("error", "Generation failed."))
    if job["status"] != "done":
        raise HTTPException(409, detail=f"Job is still {job['status']}. Poll /status first.")

    ply_path = Path(job["ply_path"])
    if not ply_path.exists():
        raise HTTPException(500, detail="Result file not found on server.")

    log.info("[%s] Serving result (%.1f MB)", job_id, ply_path.stat().st_size / 1e6)

    return FileResponse(
        str(ply_path),
        media_type="application/octet-stream",
        filename="scene.ply",
        background=_CleanupTask(job_id, Path(job["run_dir"])),
    )


# ── Background SHARP worker ───────────────────────────────────────────────────
def _run_sharp(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return

    img_path = Path(job["img_path"])
    run_dir  = Path(job["run_dir"])
    out_dir  = run_dir / "out"
    out_dir.mkdir(exist_ok=True)

    _set_job(job_id, status="processing", progress=0.05)
    log.info("[%s] SHARP starting on %s", job_id, img_path.name)

    cmd = [SHARP_CMD, "predict", "-i", str(img_path), "-o", str(out_dir), "--device", SHARP_DEVICE]
    if SHARP_CHECKPOINT:
        cmd += ["--checkpoint", SHARP_CHECKPOINT]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SHARP_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.error("[%s] SHARP timed out after %ds", job_id, SHARP_TIMEOUT)
        _set_job(job_id, status="failed", error=f"SHARP timed out after {SHARP_TIMEOUT}s.")
        return
    except Exception as exc:
        log.exception("[%s] SHARP subprocess error", job_id)
        _set_job(job_id, status="failed", error=str(exc))
        return

    _set_job(job_id, progress=0.9)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-400:]
        log.error("[%s] SHARP exit %d:\n%s", job_id, proc.returncode, tail)
        _set_job(job_id, status="failed",
                 error=f"SHARP exited {proc.returncode}: {tail}")
        return

    ply = _find_ply(out_dir, img_path.stem)
    if ply is None:
        log.error("[%s] No .ply produced. stdout:\n%s", job_id, proc.stdout[-400:])
        _set_job(job_id, status="failed",
                 error="SHARP ran but produced no .ply file. Check model weights.")
        return

    log.info("[%s] Done — %.1f MB", job_id, ply.stat().st_size / 1e6)
    _set_job(job_id, status="done", progress=1.0, ply_path=str(ply))


# ── Background cleanup task ───────────────────────────────────────────────────
class _CleanupTask:
    def __init__(self, job_id: str, path: Path):
        self.job_id = job_id
        self.path   = path

    def __call__(self):
        shutil.rmtree(self.path, ignore_errors=True)
        with _jobs_lock:
            _jobs.pop(self.job_id, None)
        log.info("[%s] Cleaned up", self.job_id)


# ── Periodic TTL cleanup (runs every 10 min in a daemon thread) ───────────────
def _ttl_cleanup():
    while True:
        time.sleep(600)
        now = time.time()
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
            log.info("[%s] TTL-expired", jid)


threading.Thread(target=_ttl_cleanup, daemon=True).start()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_ply(out_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        out_dir / f"{stem}.ply",
        out_dir / "input.ply",
        out_dir / "output.ply",
        out_dir / "scene.ply",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    for p in sorted(out_dir.rglob("*.ply")):
        if p.stat().st_size > 0:
            return p
    return None


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
