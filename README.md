# DepthDiver

**Turn any photo into a 3D world you can walk through.**

Single-image 3D Gaussian Splatting powered by Apple's [SHARP](https://github.com/apple/ml-sharp) model, with optional Claude AI scene analysis. Deployed as a static mobile-first web app on **Netlify** with a FastAPI backend on **Render**.

---

## How it works

1. You upload a photo  
2. The backend runs Apple's SHARP model — 30–90 s  
3. A Gaussian Splat `.ply` streams back to your browser  
4. WebGL renders it at 100+ FPS — you drag, pinch, and tilt to explore  
5. (Optional) Claude analyses the scene and rates the 3D quality  

---

## Architecture

```
Browser (Netlify static)
  │
  ├─ POST /generate ──────────────────────→ Render backend (FastAPI + SHARP)
  │                                         returns .ply bytes
  │
  ├─ POST /.netlify/functions/analyze ───→ Netlify Function → Anthropic API
  └─ POST /.netlify/functions/quality ───→ Netlify Function → Anthropic API
```

| Layer | Platform | Cost |
|-------|----------|------|
| Frontend + Claude proxy | Netlify | Free |
| SHARP 3D generation | Render (Docker) | Free tier available |
| AI (Claude) | Anthropic API | Pay-per-use |

---

## Local development

```bash
# Clone
git clone https://github.com/your-username/DepthDiver.git
cd DepthDiver

# Run original Gradio app (all-in-one, no deployment needed)
pip install -r requirements.txt
python app.py
```

Or run the production architecture locally:

```bash
# Terminal 1 — backend
cd backend
pip install fastapi uvicorn python-multipart
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/apple/ml-sharp.git
uvicorn main:app --reload

# Terminal 2 — frontend (any static server)
npx serve public
# or: python -m http.server 3000 --directory public
```

---

## Deploy to Netlify + Render

### Step 1 — Deploy the backend to Render

1. Push this repository to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render detects `render.yaml` automatically
4. Click **Apply** — the Docker build installs PyTorch + SHARP (~5–10 min first build)
5. Wait for the health check at `/health` to go green
6. **Copy your Render service URL**, e.g. `https://depthdiver-api.onrender.com`

> **Render free tier** spins down after 15 min of inactivity. The first request after spin-down takes ~30 s extra. Upgrade to Starter ($7/mo) for always-on.

### Step 2 — Deploy the frontend to Netlify

#### Option A — Netlify UI (easiest)

1. Go to [Netlify](https://app.netlify.com) → **Add new site** → **Import an existing project**
2. Connect your GitHub repo
3. Set build settings:
   - **Base directory**: *(leave empty)*
   - **Build command**: *(from netlify.toml — auto-detected)*
   - **Publish directory**: `public`
4. Add environment variables (Site settings → Environment variables):
   | Key | Value |
   |-----|-------|
   | `BACKEND_URL` | `https://depthdiver-api.onrender.com` |
   | `ANTHROPIC_API_KEY` | `sk-ant-api03-…` *(optional — lets you cover costs for all users)* |
5. **Deploy site** → done!

#### Option B — Netlify CLI

```bash
npm install -g netlify-cli

# Set your backend URL
export BACKEND_URL="https://depthdiver-api.onrender.com"

netlify deploy --prod --dir public
```

### Step 3 — Test it

1. Open your Netlify URL  
2. Upload a photo  
3. Click **Create 3D** and wait 30–90 s  
4. Drag to explore your scene!

---

## Environment variables

### Netlify (frontend / functions)

| Variable | Required | Description |
|----------|----------|-------------|
| `BACKEND_URL` | **Yes** | Your Render backend URL |
| `ANTHROPIC_API_KEY` | No | If set, users don't need to provide their own key |

### Render (backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARP_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SHARP_TIMEOUT` | `300` | Max seconds for SHARP to run |
| `SHARP_CHECKPOINT_PATH` | — | Path to custom model weights |
| `ALLOWED_ORIGINS` | `*` | Comma-separated allowed origins (e.g. your Netlify URL) |

---

## Project structure

```
DepthDiver/
├── app.py                    # Original Gradio app (local dev)
├── requirements.txt          # Gradio app deps
├── .env.example
│
├── public/                   # ← Netlify serves this
│   ├── index.html            # Mobile-first SPA
│   ├── app.css               # All styles
│   ├── app.js                # All client-side JS
│   └── config.js             # backendUrl config (patched at build time)
│
├── netlify/
│   └── functions/
│       ├── analyze.mjs       # Claude pre-analysis proxy
│       └── quality.mjs       # Claude quality-check proxy
│
├── netlify.toml              # Netlify build + headers config
│
└── backend/                  # ← Deploy to Render
    ├── main.py               # FastAPI + SHARP API
    ├── requirements.txt
    ├── Dockerfile
    └── render.yaml           # One-click Render deploy
```

---

## Frequently asked questions

**Q: Why can't everything run on Netlify?**  
Netlify functions time out at 10 seconds. Apple's SHARP model takes 30–90 seconds to generate a 3D scene. It must run on a server that supports long-running processes.

**Q: Is my photo uploaded anywhere besides my own server?**  
No. Your photo goes directly to the Render backend you deployed — your own server. The Netlify functions only see a resized 1024-px JPEG for Claude, and only if you enable AI mode.

**Q: How large are the output .ply files?**  
Typically 30–150 MB depending on scene complexity. They download directly to your browser from the backend, then render entirely in WebGL.

**Q: Can I use a GPU on Render?**  
Render offers GPU instances (A10G). Set `SHARP_DEVICE=cuda` and choose a GPU plan. This cuts processing time to ~5–15 s.

**Q: Can I open the .ply in Blender or other software?**  
Yes — download via the "Save 3D file" button, or open directly in [SuperSplat](https://supersplat.playcanvas.com).

---

## Acknowledgements

- [Apple ml-sharp](https://github.com/apple/ml-sharp) — single-image Gaussian Splatting
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) — WebGL renderer
- [SuperSplat](https://supersplat.playcanvas.com) — browser-based splat editor
- [Anthropic Claude](https://anthropic.com) — scene analysis and quality rating
