/**
 * DepthDiver — main application logic
 * Pure ES module, loaded with type="module"
 */

// ─── Config ─────────────────────────────────────────────────────────────────
const CFG = window.__DEPTHDIVER__ || {};
const BACKEND_URL = (CFG.backendUrl || "http://localhost:8000").replace(/\/$/, "");
const GSP_CDN =
  "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4/build/gaussian-splats-3d.module.js";

// Is the backend URL still pointing at localhost (= not yet deployed)?
const BACKEND_IS_LOCAL =
  BACKEND_URL.includes("localhost") || BACKEND_URL.includes("127.0.0.1");

// ─── Content ─────────────────────────────────────────────────────────────────
const FUN_FACTS = [
  "SHARP generates over 1 million tiny 3D blobs from a single photo.",
  "Each blob remembers its colour from many different viewing angles.",
  "3D Gaussian Splatting renders at 100+ FPS in your browser.",
  "The model internally works at 1536 × 1536 resolution.",
  "Your image never leaves your own server — complete privacy.",
  "Apple's ML research team published SHARP as open-source software.",
  "The result is a real 3D scene, not a flat image or video loop.",
];

const PROC_STEPS = [
  { id: "read",    label: "Reading your photo" },
  { id: "build",   label: "Building the 3D world" },
  { id: "finish",  label: "Polishing the details" },
  { id: "ai",      label: "AI analysing scene" },
  { id: "viewer",  label: "Spinning up the viewer" },
];

const MODELS = [
  { value: "claude-opus-4-5",    label: "Claude Opus (best quality)" },
  { value: "claude-sonnet-4-5",  label: "Claude Sonnet (balanced)" },
  { value: "claude-haiku-4-5",   label: "Claude Haiku (fastest)" },
];

// ─── State ───────────────────────────────────────────────────────────────────
const S = {
  screen:    "upload",
  file:      null,
  mode:      "free",
  apiKey:    "",
  model:     "claude-sonnet-4-5",
  plyUrl:    null,       // blob URL for the PLY
  plyBytes:  null,       // ArrayBuffer (kept for download)
  sceneData: null,
  qaData:    null,
  viewer:    null,
  spinning:  false,
  gyroOn:    false,
  elapsed:   0,
  elapsedId: null,
  factId:    null,
  steps:     {},         // step status: "pending"|"active"|"done"
};

// ─── DOM helpers ─────────────────────────────────────────────────────────────
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function show(el) { if (el) el.hidden = false; }
function hide(el) { if (el) el.hidden = true; }

// ─── Screen transitions ───────────────────────────────────────────────────────
function goScreen(name) {
  $$(".screen").forEach(s => s.classList.remove("active"));
  const next = $(`#screen-${name}`);
  if (next) next.classList.add("active");
  S.screen = name;
}

// ─── Processing steps ─────────────────────────────────────────────────────────
function initSteps() {
  const list = $("#proc-steps");
  list.innerHTML = "";
  PROC_STEPS.forEach(({ id, label }) => {
    S.steps[id] = "pending";
    const el = document.createElement("div");
    el.className = "proc-step";
    el.id = `step-${id}`;
    el.innerHTML = `
      <div class="proc-dot pending" id="dot-${id}">—</div>
      <span id="lbl-${id}">${label}</span>`;
    list.appendChild(el);
  });
}

function setStep(id, status, label) {
  S.steps[id] = status;
  const row = $(`#step-${id}`);
  const dot = $(`#dot-${id}`);
  if (!row || !dot) return;
  row.className = `proc-step ${status}`;
  dot.className = `proc-dot ${status}`;
  dot.textContent = status === "done" ? "✓" : status === "active" ? "●" : "—";
  if (label) $(`#lbl-${id}`).textContent = label;
  if (status === "active") {
    $("#proc-title").textContent = label || $(`#lbl-${id}`).textContent;
  }
}

// ─── Upload zone ─────────────────────────────────────────────────────────────
function initDropZone() {
  const zone    = $("#drop-zone");
  const input   = $("#file-input");
  const preview = $("#preview-img");

  const load = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    S.file = file;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    zone.classList.add("has-image");
    $("#create-btn").disabled = false;
    checkConfigWarning();
  };

  input.addEventListener("change", () => load(input.files[0]));

  zone.addEventListener("click", (e) => {
    if (e.target !== input) input.click();
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    load(e.dataTransfer.files[0]);
  });
}

// ─── Mode selection ───────────────────────────────────────────────────────────
function initModeCards() {
  $$(".mode-card").forEach(card => {
    card.addEventListener("click", () => {
      $$(".mode-card").forEach(c => c.classList.remove("checked"));
      card.classList.add("checked");
      const radio = card.querySelector("input[type=radio]");
      radio.checked = true;
      S.mode = radio.value;
      const keySection = $("#ai-key-section");
      if (S.mode === "ai") { show(keySection); }
      else { hide(keySection); }
    });
  });
  // Default: free mode checked
  const defaultCard = $('.mode-card input[value="free"]')?.closest(".mode-card");
  if (defaultCard) defaultCard.classList.add("checked");
}

// ─── Config warning ───────────────────────────────────────────────────────────
function checkConfigWarning() {
  const warn = $("#config-warning");
  if (!warn) return;
  // Show warning whenever backend is localhost — regardless of whether a file
  // is chosen — so the user knows before they even try.
  if (BACKEND_IS_LOCAL) { show(warn); }
  else { hide(warn); }
}

// ─── Elapsed timer + fun facts ────────────────────────────────────────────────
function startElapsed() {
  S.elapsed = 0;
  S.elapsedId = setInterval(() => {
    S.elapsed++;
    const el = $("#proc-elapsed");
    if (el) {
      const m = Math.floor(S.elapsed / 60);
      const s = S.elapsed % 60;
      el.textContent = m > 0 ? `${m}m ${s}s elapsed` : `${s}s elapsed`;
    }
  }, 1000);

  let fi = 0;
  const factEl = $("#proc-fact");
  const showFact = () => {
    if (factEl) {
      factEl.style.opacity = "0";
      setTimeout(() => {
        factEl.textContent = FUN_FACTS[fi % FUN_FACTS.length];
        factEl.style.opacity = "1";
        fi++;
      }, 300);
    }
  };
  showFact();
  S.factId = setInterval(showFact, 5000);
}

function stopElapsed() {
  clearInterval(S.elapsedId);
  clearInterval(S.factId);
}

// ─── Image utilities ──────────────────────────────────────────────────────────
async function resizeForApi(file, maxPx = 1024) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      let w = img.width, h = img.height;
      if (w > maxPx || h > maxPx) {
        if (w > h) { h = Math.round(h * maxPx / w); w = maxPx; }
        else       { w = Math.round(w * maxPx / h); h = maxPx; }
      }
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      canvas.toBlob(b => b ? resolve(b) : reject(new Error("toBlob failed")), "image/jpeg", 0.85);
    };
    img.onerror = reject;
    img.src = url;
  });
}

async function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function captureViewerFrame() {
  const canvas = document.querySelector("#viewer-wrap canvas");
  if (!canvas) return null;
  return new Promise(r => canvas.toBlob(b => r(b), "image/jpeg", 0.82));
}

// ─── Sample image ─────────────────────────────────────────────────────────────
async function loadSampleImage() {
  const canvas = document.createElement("canvas");
  canvas.width = 640; canvas.height = 480;
  const ctx = canvas.getContext("2d");

  // Sky
  const sky = ctx.createLinearGradient(0, 0, 0, 300);
  sky.addColorStop(0, "#1a6cc8"); sky.addColorStop(1, "#7eb8f7");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, 640, 300);
  // Ground
  ctx.fillStyle = "#4a7c3f"; ctx.fillRect(0, 300, 640, 180);
  // Sun
  ctx.fillStyle = "#ffe066";
  ctx.beginPath(); ctx.arc(90, 80, 42, 0, Math.PI * 2); ctx.fill();
  // House body
  ctx.fillStyle = "#c8956b"; ctx.fillRect(210, 230, 220, 130);
  // Roof
  ctx.fillStyle = "#6d3b0a";
  ctx.beginPath(); ctx.moveTo(190, 230); ctx.lineTo(320, 140); ctx.lineTo(450, 230); ctx.fill();
  // Door
  ctx.fillStyle = "#4a2508"; ctx.fillRect(292, 298, 56, 62);
  ctx.fillStyle = "#8b5a2b"; ctx.beginPath(); ctx.arc(316, 332, 5, 0, Math.PI * 2); ctx.fill();
  // Window
  ctx.fillStyle = "#bfdbfe"; ctx.fillRect(380, 258, 36, 32);
  ctx.strokeStyle = "#6d3b0a"; ctx.lineWidth = 3;
  ctx.strokeRect(380, 258, 36, 32);
  // Tree
  ctx.fillStyle = "#2e7d32";
  ctx.beginPath(); ctx.arc(520, 255, 58, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#5d4037"; ctx.fillRect(510, 298, 20, 62);
  // Path
  ctx.fillStyle = "#b5a07a";
  ctx.beginPath();
  ctx.moveTo(290, 360); ctx.lineTo(350, 360); ctx.lineTo(370, 480); ctx.lineTo(260, 480);
  ctx.fill();

  return new Promise(r => canvas.toBlob(blob => {
    const file = new File([blob], "sample_house.jpg", { type: "image/jpeg" });
    r(file);
  }, "image/jpeg", 0.9));
}

// ─── API calls ────────────────────────────────────────────────────────────────
async function callAnalyze(imageBase64, mediaType, apiKey, model) {
  const res = await fetch("/.netlify/functions/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ imageBase64, mediaType, apiKey, model }),
  });
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`Analysis failed (${res.status}): ${t.slice(0, 200)}`);
  }
  return res.json();
}

async function callQuality(originalBase64, previewBase64, sceneData, apiKey, model) {
  const res = await fetch("/.netlify/functions/quality", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ originalBase64, previewBase64, sceneData, apiKey, model }),
  });
  if (!res.ok) return null;
  return res.json().catch(() => null);
}

// ─── Polling helpers ──────────────────────────────────────────────────────────

/** Poll /jobs/{id}/status every 4 s until status is "done" or "failed". */
async function pollUntilDone(jobId) {
  const url      = `${BACKEND_URL}/jobs/${jobId}/status`;
  const maxWait  = 15 * 60 * 1000; // 15 min absolute cap
  const interval = 4000;
  const started  = Date.now();

  while (Date.now() - started < maxWait) {
    await sleep(interval);

    let data;
    try {
      const r = await fetch(url, { cache: "no-store" });
      data = await r.json();
    } catch {
      // Transient network hiccup — keep polling
      continue;
    }

    if (data.status === "done")   return;
    if (data.status === "failed") throw new Error(data.error || "3D generation failed on the server.");

    // Update step label with a hint
    const lbl = data.status === "processing"
      ? "Building your 3D world…"
      : "Starting up…";
    setStep("build", "active", lbl);
  }
  throw new Error("Generation timed out after 15 minutes. Try a simpler photo.");
}

/** Simple sleep helper. */
const sleep = ms => new Promise(r => setTimeout(r, ms));

/** Fetch with one automatic retry on transient network failure. */
async function fetchWithRetry(url, options = {}, retries = 1) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetch(url, options);
    } catch (err) {
      if (attempt === retries) throw err;
      await sleep(3000); // wait before retry
    }
  }
}

/** Convert a raw network TypeError into a readable message. */
function networkFriendlyError(err) {
  const msg = String(err?.message || err).toLowerCase();
  if (msg.includes("load failed") || msg.includes("failed to fetch") || msg.includes("networkerror")) {
    return new Error(
      BACKEND_IS_LOCAL
        ? "Backend not connected — deploy backend/ to Render and set BACKEND_URL in Netlify."
        : `Cannot reach the backend at ${BACKEND_URL}. ` +
          "Make sure your Render service is running (/health should return 200)."
    );
  }
  return err;
}

// ─── Main pipeline ────────────────────────────────────────────────────────────
async function createScene() {
  const errEl = $("#proc-error");
  errEl.classList.remove("visible");
  goScreen("processing");
  initSteps();
  startElapsed();

  try {
    // Step 1: optional AI pre-analysis (non-blocking — fires in parallel)
    let analysisPromise = null;
    if (S.mode === "ai" && S.apiKey) {
      setStep("ai", "active", "AI reading your photo…");
      const compressed = await resizeForApi(S.file, 1024);
      const b64        = await blobToBase64(compressed);
      analysisPromise  = callAnalyze(b64, "image/jpeg", S.apiKey, S.model);
    } else {
      setStep("ai", "done", "AI reading your photo");
    }

    // Step 2: Submit image to SHARP backend (returns job_id immediately — no timeout risk)
    setStep("read",  "done");
    setStep("build", "active", "Submitting your photo…");

    // Guard: backend not configured yet
    if (BACKEND_IS_LOCAL && location.protocol === "https:") {
      throw new Error(
        "Backend not connected. Deploy the backend/ folder to Render.com, " +
        "then add BACKEND_URL in Netlify environment variables. See README."
      );
    }

    const formData = new FormData();
    formData.append("file", S.file, S.file.name || "photo.jpg");

    let submitRes;
    try {
      submitRes = await fetchWithRetry(`${BACKEND_URL}/generate`, {
        method: "POST", body: formData,
      });
    } catch (networkErr) {
      throw networkFriendlyError(networkErr);
    }

    if (!submitRes.ok) {
      const detail = await submitRes.json().then(j => j.detail).catch(() => `Error ${submitRes.status}`);
      if (submitRes.status === 503) {
        throw new Error("The 3D server is waking up (Render free tier cold start). " +
          "Wait 30 s then tap Try again.");
      }
      throw new Error(detail);
    }

    const { job_id } = await submitRes.json();
    setStep("build", "active", "Building the 3D world…");

    // Step 3: Poll until SHARP finishes (no long-lived connection = no Cloudflare timeout)
    await pollUntilDone(job_id);
    setStep("build",  "done");
    setStep("finish", "active", "Downloading your 3D scene…");

    // Step 4: Download the finished .ply
    let plyRes;
    try {
      plyRes = await fetchWithRetry(`${BACKEND_URL}/jobs/${job_id}/result`);
    } catch (networkErr) {
      throw networkFriendlyError(networkErr);
    }
    if (!plyRes.ok) {
      const d = await plyRes.json().then(j => j.detail).catch(() => `Error ${plyRes.status}`);
      throw new Error(d);
    }

    let plyBytes;
    try {
      plyBytes = await plyRes.arrayBuffer();
    } catch {
      throw new Error("Download dropped — check your signal and try again.");
    }

    S.plyBytes = plyBytes;
    const plyBlob = new Blob([S.plyBytes], { type: "application/octet-stream" });
    if (S.plyUrl) URL.revokeObjectURL(S.plyUrl);
    S.plyUrl = URL.createObjectURL(plyBlob);

    setStep("finish", "done");

    // Step 4: wait for analysis if it's still running
    if (analysisPromise) {
      S.sceneData = await analysisPromise.catch(e => {
        console.warn("Analysis failed:", e); return null;
      });
      setStep("ai", "done");
    }

    // Step 5: init viewer
    setStep("viewer", "active", "Spinning up the viewer…");
    await initViewer(S.plyUrl);
    setStep("viewer", "done");

    // Step 6: quality check (after viewer renders first frame)
    if (S.mode === "ai" && S.apiKey && S.sceneData) {
      await new Promise(r => setTimeout(r, 1200)); // let viewer render
      const frame = await captureViewerFrame();
      if (frame) {
        const prevB64 = await blobToBase64(frame);
        const origCompressed = await resizeForApi(S.file, 1024);
        const origB64 = await blobToBase64(origCompressed);
        S.qaData = await callQuality(origB64, prevB64, S.sceneData, S.apiKey, S.model)
          .catch(() => null);
      }
    }

    stopElapsed();
    renderSheet();
    goScreen("ready");
    showHint("Drag to look around  ·  Pinch to zoom");

  } catch (err) {
    stopElapsed();
    console.error("Pipeline error:", err);

    const msg = String(err.message || err);
    const isSetup = msg.toLowerCase().includes("backend not connected") ||
                    msg.toLowerCase().includes("backend_url");

    errEl.classList.add("visible");
    errEl.innerHTML = isSetup
      ? `<strong>Backend not connected</strong>
         The 3D generation service isn't set up yet. Here's what to do:
         <ol style="margin:10px 0 10px 18px;line-height:1.9;font-size:0.85rem">
           <li>Deploy the <code>backend/</code> folder to
               <a href="https://render.com" target="_blank" style="color:#93c5fd">Render.com</a>
               (free tier, takes ~10 min first build)</li>
           <li>Copy your Render service URL</li>
           <li>In Netlify → Site settings → Environment variables, add
               <code>BACKEND_URL = https://your-service.onrender.com</code></li>
           <li>Trigger a Netlify redeploy</li>
         </ol>
         <a href="https://github.com/SanjoyDat1/DepthDiver#deploy-in-3-steps"
            target="_blank" style="color:#93c5fd;font-weight:600">
           Full guide in the README →
         </a>
         <br><button class="retry-btn" onclick="location.reload()" style="margin-top:14px">← Back</button>`
      : `<strong>Something went wrong</strong>
         ${escHtml(msg)}
         <br><button class="retry-btn" onclick="location.reload()">← Try again</button>`;
  }
}

// ─── 3D Viewer ────────────────────────────────────────────────────────────────
async function initViewer(plyUrl) {
  const wrap = $("#viewer-wrap");
  wrap.innerHTML = ""; // clear any old canvas

  const loadWrap = $("#viewer-load-wrap");
  const loadBar  = $("#viewer-load-bar");
  const loadPct  = $("#viewer-load-pct");
  loadWrap.classList.add("visible");
  loadBar.style.width = "0%";

  // Import both Viewer and SceneFormat so we can tell the viewer that our
  // blob URL contains a PLY file — blob: URLs have no extension, so the
  // viewer can't auto-detect the format and throws "File format not supported".
  const { Viewer, SceneFormat } = await import(GSP_CDN);

  const viewer = new Viewer({
    selfDrivenMode:         true,
    sharedMemoryForWorkers: false,
    cameraUp:               [0, -1, 0],
    initialCameraPosition:  [-0.5, -3, 6],
    initialCameraLookAt:    [0, 0, 0],
    rootElement:            wrap,
  });

  S.viewer = viewer;

  // SceneFormat.Ply = 0  (must be explicit because blob URLs have no extension)
  const plyFormat = SceneFormat?.Ply ?? 0;

  await viewer.addSplatScene(plyUrl, {
    format:                     plyFormat,
    splatAlphaRemovalThreshold: 5,
    showLoadingUI:              false,
    onProgress: (p) => {
      const pct = Math.round(p * 100);
      loadBar.style.width = `${pct}%`;
      loadPct.textContent  = `${pct}%`;
    },
  });

  viewer.start();
  loadWrap.classList.remove("visible");
}

// ─── Viewer control buttons ───────────────────────────────────────────────────
function initViewerControls() {
  // Reset
  $("#vbtn-reset").addEventListener("click", () => {
    if (S.viewer) {
      S.viewer.camera.position.set(-0.5, -3, 6);
      S.viewer.camera.lookAt(0, 0, 0);
    }
  });

  // Spin toggle
  const spinBtn = $("#vbtn-spin");
  let spinRaf;
  const doSpin = () => {
    if (!S.viewer || !S.spinning) return;
    const cam = S.viewer.camera;
    const r = Math.sqrt(cam.position.x ** 2 + cam.position.z ** 2);
    const angle = Math.atan2(cam.position.z, cam.position.x) + 0.008;
    cam.position.x = r * Math.cos(angle);
    cam.position.z = r * Math.sin(angle);
    cam.lookAt(0, 0, 0);
    spinRaf = requestAnimationFrame(doSpin);
  };
  spinBtn.addEventListener("click", () => {
    S.spinning = !S.spinning;
    spinBtn.classList.toggle("on", S.spinning);
    spinBtn.textContent = S.spinning ? "⏹ Stop" : "↺ Spin";
    if (S.spinning) doSpin();
    else cancelAnimationFrame(spinRaf);
  });

  // Gyro
  const gyroBtn = $("#vbtn-gyro");
  gyroBtn.addEventListener("click", toggleGyro);
  if (!/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) hide(gyroBtn);
}

// ─── Gyroscope ────────────────────────────────────────────────────────────────
let gyroHandler = null;

function toggleGyro() {
  if (!S.gyroOn) enableGyro(); else disableGyro();
}

function enableGyro() {
  const run = () => {
    let baseAlpha = null, baseBeta = null;
    gyroHandler = (e) => {
      if (!S.viewer) return;
      const a = e.alpha || 0;
      const b = e.beta  || 0;
      if (baseAlpha === null) { baseAlpha = a; baseBeta = b; return; }
      const da = ((a - baseAlpha + 540) % 360) - 180;
      const db = b - baseBeta;
      const cam = S.viewer.camera;
      const r   = cam.position.distanceTo({ x: 0, y: 0, z: 0 } );
      const theta = Math.atan2(cam.position.z, cam.position.x) - da * 0.012;
      const phi   = Math.acos(Math.max(-1, Math.min(1, cam.position.y / r))) + db * 0.012;
      cam.position.set(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi),
        r * Math.sin(phi) * Math.sin(theta)
      );
      cam.lookAt(0, 0, 0);
    };
    window.addEventListener("deviceorientation", gyroHandler, true);
    S.gyroOn = true;
    $("#vbtn-gyro").classList.add("on");
    $("#vbtn-gyro").textContent = "📡 Gyro On";
  };

  if (typeof DeviceOrientationEvent !== "undefined" &&
      typeof DeviceOrientationEvent.requestPermission === "function") {
    DeviceOrientationEvent.requestPermission()
      .then(r => { if (r === "granted") run(); })
      .catch(console.error);
  } else {
    run();
  }
}

function disableGyro() {
  if (gyroHandler) window.removeEventListener("deviceorientation", gyroHandler, true);
  gyroHandler = null;
  S.gyroOn = false;
  const b = $("#vbtn-gyro");
  b.classList.remove("on");
  b.textContent = "📡 Motion";
}

// ─── Bottom sheet ──────────────────────────────────────────────────────────────
function initBottomSheet() {
  const sheet     = $("#bottom-sheet");
  const handle    = $(".sheet-handle-row");
  const peekHeight = 100;
  let startY = 0, startTranslate = 0, isDragging = false;

  const getTranslate = () => {
    const style = window.getComputedStyle(sheet);
    const matrix = new WebKitCSSMatrix(style.transform);
    return matrix.m42;
  };

  const setTranslate = (y) => {
    const max = sheet.offsetHeight - peekHeight;
    const clamped = Math.max(0, Math.min(max, y));
    sheet.style.transition = "none";
    sheet.style.transform = `translateY(${clamped}px)`;
  };

  const snapSheet = (velocity) => {
    const current  = getTranslate();
    const max      = sheet.offsetHeight - peekHeight;
    const midpoint = max / 2;
    sheet.style.transition = "";
    sheet.style.transform  = "";

    if (velocity > 0.5 || current > midpoint) {
      sheet.classList.remove("expanded");
    } else {
      sheet.classList.add("expanded");
    }
  };

  handle.addEventListener("touchstart", (e) => {
    isDragging = true;
    startY = e.touches[0].clientY;
    startTranslate = getTranslate();
  }, { passive: true });

  handle.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    const dy = e.touches[0].clientY - startY;
    setTranslate(startTranslate + dy);
  }, { passive: true });

  handle.addEventListener("touchend", (e) => {
    if (!isDragging) return;
    isDragging = false;
    const endY    = e.changedTouches[0].clientY;
    const vel     = (endY - startY) / 200; // rough velocity
    snapSheet(vel);
  });

  handle.addEventListener("click", () => {
    sheet.classList.toggle("expanded");
  });
}

// ─── Render sheet contents ────────────────────────────────────────────────────
function renderSheet() {
  const scroll = $("#sheet-scroll");

  // Download button — uses blob URL
  const dlBtn = $("#sheet-dl");
  dlBtn.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = S.plyUrl;
    a.download = "depthdiver_scene.ply";
    a.click();
  });

  // SuperSplat link
  const ssLink = $("#sheet-supersplat");
  ssLink.href = "https://supersplat.playcanvas.com";

  // AI section
  const aiSection = $("#sheet-ai");
  if (S.sceneData) {
    show(aiSection);
    renderSceneCard(S.sceneData);
  } else {
    hide(aiSection);
  }

  // Quality section
  const qaSection = $("#sheet-qa");
  if (S.qaData) {
    show(qaSection);
    renderQaCard(S.qaData);
  } else {
    hide(qaSection);
  }

  // Expanders
  $$(".expander-header").forEach(btn => {
    btn.addEventListener("click", () => {
      btn.closest(".expander").classList.toggle("open");
    });
  });
}

function renderSceneCard(data) {
  const conf = data.depth_confidence || "medium";
  const badgeClass = conf === "high" ? "badge-great" : conf === "medium" ? "badge-good" : "badge-tricky";
  const badgeLabel = conf === "high" ? "Great for 3D" : conf === "medium" ? "Good" : "Challenging";

  const el = $("#scene-card");
  el.innerHTML = `
    <div class="result-label">AI scene analysis</div>
    <div class="ai-section">
      <span class="ai-badge ${badgeClass}">${badgeLabel}</span>
      <div class="ai-section-title">${escHtml(data.scene_summary || "")}</div>
      <p class="ai-text">${escHtml(data.preprocessing_advice || "")}</p>
      ${data.main_objects?.length ? `<ul class="ai-list" style="margin-top:8px">${
        data.main_objects.slice(0,6).map(o => `<li>${escHtml(o)}</li>`).join("")
      }</ul>` : ""}
    </div>
    ${data.tour_suggestions?.length ? `
    <div class="result-label" style="margin-top:12px">Camera tour ideas</div>
    <div class="ai-section">
      <ul class="ai-list">${
        data.tour_suggestions.slice(0,4).map(t => `<li>${escHtml(t)}</li>`).join("")
      }</ul>
    </div>` : ""}
    <div class="expander" style="margin-top:10px">
      <button class="expander-header">Raw analysis JSON</button>
      <div class="expander-body">
        <pre class="json-pre">${escHtml(JSON.stringify(data, null, 2))}</pre>
      </div>
    </div>`;

  $$(".expander-header", el).forEach(btn => {
    btn.addEventListener("click", () => btn.closest(".expander").classList.toggle("open"));
  });
}

function renderQaCard(data) {
  const score = data.quality_score ?? 0;
  const color = score >= 7 ? "var(--green)" : score >= 4 ? "#d97706" : "var(--red)";
  const bg    = score >= 7 ? "var(--green-lt)" : score >= 4 ? "#fefce8" : "var(--red-lt)";
  const bd    = score >= 7 ? "var(--green-bd)" : score >= 4 ? "#fef08a" : "var(--red-bd)";

  const el = $("#qa-card");
  el.innerHTML = `
    <div class="result-label">AI quality check</div>
    <div style="background:${bg};border:1px solid ${bd};border-radius:12px;padding:14px 16px">
      <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:4px">
        <span style="font-size:1.6rem;font-weight:800;color:${color}">${score}</span>
        <span style="color:var(--sub);font-size:0.85rem">/ 10</span>
      </div>
      <p style="font-size:0.85rem;color:var(--sub);line-height:1.5">${escHtml(data.quality_summary || "")}</p>
      ${data.visible_artifacts?.length ? `
        <div style="margin-top:8px;font-size:0.82rem;font-weight:600;color:var(--text)">Things to watch for:</div>
        <ul class="ai-list">${
          data.visible_artifacts.slice(0,4).map(a => `<li>${escHtml(a)}</li>`).join("")
        }</ul>` : ""}
    </div>`;
}

// ─── Hint overlay ─────────────────────────────────────────────────────────────
function showHint(text) {
  const h = $("#viewer-hint");
  h.textContent = text;
  h.classList.add("show");
  setTimeout(() => h.classList.remove("show"), 3500);
}

// ─── Utility ──────────────────────────────────────────────────────────────────
function escHtml(s) {
  if (!s) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
function boot() {
  initDropZone();
  initModeCards();
  initBottomSheet();

  // API key field
  $("#ai-key-input").addEventListener("input", e => { S.apiKey = e.target.value.trim(); });

  // Create button
  $("#create-btn").addEventListener("click", createScene);

  // Sample button
  $("#example-btn").addEventListener("click", async () => {
    const file  = await loadSampleImage();
    const input = $("#file-input");
    const dt    = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    // Trigger our handler
    const zone  = $("#drop-zone");
    const preview = $("#preview-img");
    S.file = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    zone.classList.add("has-image");
    $("#create-btn").disabled = false;
    checkConfigWarning();
  });

  // Create another
  $("#create-another-btn").addEventListener("click", () => {
    if (S.plyUrl) URL.revokeObjectURL(S.plyUrl);
    if (S.viewer) { try { S.viewer.dispose(); } catch {} S.viewer = null; }
    S.file = S.plyUrl = S.sceneData = S.qaData = S.plyBytes = null;
    S.mode = "free"; S.apiKey = "";
    S.spinning = false; S.gyroOn = false;
    // Reset drop zone
    const zone    = $("#drop-zone");
    const preview = $("#preview-img");
    zone.classList.remove("has-image");
    preview.src = ""; preview.hidden = true;
    $("#file-input").value = "";
    $("#create-btn").disabled = true;
    $$(".mode-card").forEach(c => c.classList.remove("checked"));
    $('.mode-card input[value="free"]')?.closest(".mode-card")?.classList.add("checked");
    hide($("#ai-key-section"));
    $("#bottom-sheet").classList.remove("expanded");
    goScreen("upload");
  });

  // Viewer controls (only initialised when needed)
  // We attach them now so they're ready when screen-ready becomes active
  initViewerControls();

  // Fullscreen
  $("#vbtn-fs").addEventListener("click", () => {
    if (!document.fullscreenElement) {
      $("#screen-ready").requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.();
    }
  });

  // Show config warning immediately on load if backend is still localhost
  checkConfigWarning();

  // Hide AI section + QA card by default
  hide($("#sheet-ai"));
  hide($("#sheet-qa"));
}

document.addEventListener("DOMContentLoaded", boot);
