"""
DepthDiver — FastAPI Backend  (depth-based, free-tier compatible)

Architecture
------------
Instead of Apple SHARP (needs 2+ GB RAM), we use Depth Anything V2 Small
(~24 MB weights, ~350 MB peak RAM with PyTorch). This fits comfortably in
Render's free tier (512 MB).

How it works
------------
1. Resize input image to ≤ 512 px
2. Run Depth Anything V2 Small → per-pixel depth map
3. Back-project every pixel to a 3D point using the depth value
4. Write a Gaussian Splat .ply file (INRIA format) readable by GaussianSplats3D
5. Return the .ply to the viewer

Quality vs SHARP
----------------
- Pros: instant (5–30 s), free, no OOM
- Cons: "3D photo" effect (great for near-frontal, holes at extreme angles)
  This is exactly what iPhone Cinematic Mode / Facebook 3D Photos use.

Endpoints
---------
  GET  /health
  POST /generate              → {job_id}
  GET  /jobs/{id}/status      → {status, progress, error}
  GET  /jobs/{id}/result      → PLY download (cleanup)
  GET  /jobs/{id}/scene.ply   → PLY for viewer (no cleanup)
"""

from __future__ import annotations
import gc, logging, os, shutil, struct, tempfile, threading, time, uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("depthdiver")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_IMAGE_PX     = int(os.environ.get("MAX_IMAGE_PX", "512"))   # px (longest edge)
DEPTH_MODEL      = os.environ.get("DEPTH_MODEL",
                                  "depth-anything/Depth-Anything-V2-Small-hf")
JOB_TTL          = int(os.environ.get("JOB_TTL", "1800"))       # 30 min
MAX_CONCURRENT   = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))

WORK_ROOT = Path(tempfile.gettempdir()) / "depthdiver_runs"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ── Model (loaded lazily on first job, then cached) ───────────────────────────
_pipeline       = None
_pipeline_lock  = threading.Lock()


def _get_pipeline():
    """Load the depth pipeline once and cache it for the lifetime of the process."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        log.info("Loading depth model: %s", DEPTH_MODEL)
        from transformers import pipeline as hf_pipeline
        import torch
        _pipeline = hf_pipeline(
            "depth-estimation",
            model=DEPTH_MODEL,
            device="cpu",
            torch_dtype=torch.float32,
        )
        log.info("Depth model loaded.")
        return _pipeline


# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock  = threading.Lock()
_active_sem = threading.Semaphore(MAX_CONCURRENT)


def _set_job(jid: str, **kw):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def _get_job(jid: str) -> Optional[dict]:
    with _jobs_lock:
        return dict(_jobs[jid]) if jid in _jobs else None


def _active_count() -> int:
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j["status"] == "processing")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="DepthDiver API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["content-disposition"],
)


@app.middleware("http")
async def cors_headers(request: Request, call_next):
    r = await call_next(request)
    r.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    r.headers["Access-Control-Allow-Origin"]  = "*"
    return r


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    mem_info = {}
    try:
        import psutil
        m = psutil.virtual_memory()
        mem_info = {"used_mb": m.used // 1_000_000,
                    "total_mb": m.total // 1_000_000,
                    "percent": m.percent}
    except ImportError:
        pass
    return {
        "status":       "ok",
        "engine":       "depth-anything-v2-small",
        "max_image_px": MAX_IMAGE_PX,
        "active_jobs":  _active_count(),
        "model_loaded": _pipeline is not None,
        "memory":       mem_info,
    }


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, detail="File must be an image.")
    if _active_count() >= MAX_CONCURRENT:
        raise HTTPException(503, detail="Server busy — try again in a moment.")

    content = await file.read()
    if not content:
        raise HTTPException(400, detail="Empty file.")

    jid     = uuid.uuid4().hex[:12]
    run_dir = WORK_ROOT / jid
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix   = Path(file.filename or "photo.jpg").suffix or ".jpg"
    img_path = run_dir / f"input{suffix}"
    img_path.write_bytes(content)
    log.info("[%s] Saved %s (%d KB)", jid, img_path.name, len(content) // 1024)

    with _jobs_lock:
        _jobs[jid] = {
            "status":   "pending",
            "progress": 0.0,
            "run_dir":  str(run_dir),
            "img_path": str(img_path),
            "ply_path": None,
            "error":    None,
            "created":  time.time(),
        }

    threading.Thread(target=_worker, args=(jid,), daemon=True).start()
    return {"job_id": jid}


@app.get("/jobs/{jid}/status")
def job_status(jid: str):
    job = _get_job(jid)
    if not job:
        raise HTTPException(404, detail="Job not found.")
    return {"job_id": jid, "status": job["status"],
            "progress": job["progress"], "error": job.get("error")}


@app.get("/jobs/{jid}/result")
def job_result(jid: str):
    return _serve(jid, cleanup=True)


@app.get("/jobs/{jid}/scene.ply")
def job_scene_ply(jid: str):
    return _serve(jid, cleanup=False)


def _serve(jid: str, *, cleanup: bool) -> FileResponse:
    job = _get_job(jid)
    if not job:
        raise HTTPException(404, detail="Job not found (may have expired).")
    if job["status"] == "failed":
        raise HTTPException(422, detail=job.get("error", "Generation failed."))
    if job["status"] != "done":
        raise HTTPException(409, detail=f"Job is {job['status']} — poll /status first.")
    ply = Path(job["ply_path"])
    if not ply.exists():
        raise HTTPException(500, detail="Result file missing.")
    log.info("[%s] Serving PLY %.1f MB (cleanup=%s)", jid, ply.stat().st_size / 1e6, cleanup)
    return FileResponse(str(ply), media_type="application/octet-stream",
                        filename="scene.ply",
                        background=_Cleanup(jid, Path(job["run_dir"])) if cleanup else None)


# ── Worker ────────────────────────────────────────────────────────────────────
def _worker(jid: str):
    _active_sem.acquire()
    try:
        _do_generate(jid)
    finally:
        _active_sem.release()
        gc.collect()


def _do_generate(jid: str):
    job = _get_job(jid)
    if not job:
        return

    img_path = Path(job["img_path"])
    run_dir  = Path(job["run_dir"])
    ply_path = run_dir / "scene.ply"

    _set_job(jid, status="processing", progress=0.05)
    log.info("[%s] Starting depth pipeline on %s", jid, img_path.name)

    try:
        # ── 1. Load + resize image ─────────────────────────────────────────
        _set_job(jid, progress=0.10)
        img = Image.open(img_path).convert("RGB")
        img = _resize(img, MAX_IMAGE_PX)
        log.info("[%s] Image size: %dx%d", jid, img.width, img.height)

        # ── 2. Depth estimation ───────────────────────────────────────────
        _set_job(jid, progress=0.20)
        pipe   = _get_pipeline()
        result = pipe(img)
        depth  = np.array(result["depth"], dtype=np.float32)   # (H, W), higher=closer in some models
        _set_job(jid, progress=0.60)

        # ── 3. Build Gaussian Splat PLY ───────────────────────────────────
        _set_job(jid, progress=0.70)
        _depth_to_gaussian_ply(img, depth, ply_path)
        _set_job(jid, progress=0.95)

        log.info("[%s] Done — PLY %.1f MB", jid, ply_path.stat().st_size / 1e6)
        _set_job(jid, status="done", progress=1.0, ply_path=str(ply_path))

    except MemoryError:
        _set_job(jid, status="failed",
                 error="Server ran out of memory. Try a smaller image.")
    except Exception as exc:
        log.exception("[%s] Error", jid)
        _set_job(jid, status="failed", error=str(exc))


# ── Depth → Gaussian PLY ──────────────────────────────────────────────────────

# Spherical harmonics C0 constant
SH_C0 = 0.28209479177387814


def _resize(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _depth_to_gaussian_ply(img: Image.Image, depth: np.ndarray, out: Path):
    """
    Back-project an RGB image + depth map into a Gaussian Splat .ply file.

    Each pixel becomes one Gaussian blob at the 3D position derived from depth.
    Output is in INRIA v1 format, readable by GaussianSplats3D.
    """
    h, w   = depth.shape
    colors = np.array(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    # Normalise depth to [0, 1] — larger value = closer to camera
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.ones_like(depth) * 0.5

    # Pinhole camera back-projection
    # z in scene units: map depth [0,1] → z in [0.5, 4.5] (closer = smaller z)
    z_scene = (1.0 - depth) * 4.0 + 0.5

    fx = fy = float(w) * 1.1      # rough focal length
    cx, cy  = w / 2.0, h / 2.0

    yi, xi = np.mgrid[0:h, 0:w]
    x_scene = (xi.astype(np.float32) - cx) * z_scene / fx
    y_scene = -(yi.astype(np.float32) - cy) * z_scene / fy  # flip Y

    # Stack positions
    pts = np.stack([x_scene, y_scene, -z_scene], axis=-1)  # (H, W, 3)

    # Remove sky / background (top 5% farthest depth = likely sky or far BG)
    foreground = depth.flatten() > np.percentile(depth.flatten(), 5)

    pts_f  = pts.reshape(-1, 3)[foreground].astype(np.float32)
    clrs_f = colors.reshape(-1, 3)[foreground].astype(np.float32)
    n      = len(pts_f)

    log.info("PLY: %d points", n)

    # Gaussian parameters
    # SH DC coefficient: color = SH_C0 * f_dc + 0.5  →  f_dc = (color−0.5)/SH_C0
    f_dc     = ((clrs_f - 0.5) / SH_C0).astype(np.float32)      # (n, 3)
    opacity  = np.full(n, 4.6,  dtype=np.float32)               # sigmoid(4.6)≈0.99
    # Scale: smaller blobs look cleaner; exp(-5.3)≈0.005
    scale    = np.full((n, 3), -5.3, dtype=np.float32)
    # Identity quaternion rotation
    rot      = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0   # w=1, x=y=z=0
    normals  = np.zeros((n, 3), dtype=np.float32)

    # Build structured array for fast binary write
    dtype = np.dtype([
        ("x",      np.float32), ("y",      np.float32), ("z",      np.float32),
        ("nx",     np.float32), ("ny",     np.float32), ("nz",     np.float32),
        ("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32),
        ("opacity",np.float32),
        ("scale_0",np.float32), ("scale_1",np.float32), ("scale_2",np.float32),
        ("rot_0",  np.float32), ("rot_1",  np.float32),
        ("rot_2",  np.float32), ("rot_3",  np.float32),
    ])

    verts         = np.empty(n, dtype=dtype)
    verts["x"],  verts["y"],  verts["z"]  = pts_f[:,0],  pts_f[:,1],  pts_f[:,2]
    verts["nx"], verts["ny"], verts["nz"] = 0.0, 0.0, 0.0
    verts["f_dc_0"] = f_dc[:, 0]
    verts["f_dc_1"] = f_dc[:, 1]
    verts["f_dc_2"] = f_dc[:, 2]
    verts["opacity"]= opacity
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = scale[:,0], scale[:,1], scale[:,2]
    verts["rot_0"],   verts["rot_1"]  = rot[:, 0], rot[:, 1]
    verts["rot_2"],   verts["rot_3"]  = rot[:, 2], rot[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with open(out, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


# ── Cleanup helpers ───────────────────────────────────────────────────────────
class _Cleanup:
    def __init__(self, jid: str, path: Path):
        self.jid, self.path = jid, path

    def __call__(self):
        shutil.rmtree(self.path, ignore_errors=True)
        with _jobs_lock:
            _jobs.pop(self.jid, None)
        log.info("[%s] Cleaned up", self.jid)


def _ttl_daemon():
    while True:
        time.sleep(300)
        now = time.time()
        with _jobs_lock:
            expired = [(j, d["run_dir"])
                       for j, d in list(_jobs.items())
                       if now - d.get("created", now) > JOB_TTL]
        for jid, rdir in expired:
            shutil.rmtree(rdir, ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(jid, None)
            log.info("[%s] TTL expired", jid)


threading.Thread(target=_ttl_daemon, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
