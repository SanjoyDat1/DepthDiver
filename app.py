#!/usr/bin/env python3
"""
LensWander Pro — Single-image 3D with Claude intelligence.

Fast: Apple ml-sharp only (zero cost) -> .ply + SuperSplat.
Pro:  Claude Pre-Analysis -> SHARP -> Claude Post-QA -> .ply + scene_intelligence.json + quality_report.json.
"""

from __future__ import annotations

import base64
import io
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

import gradio as gr
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

# =============================================================================
# CSS — clean minimal theme
# =============================================================================

_APP_CSS = """
:root {
  --bg: #fafafa;
  --card: #ffffff;
  --border: #e5e7eb;
  --text: #111111;
  --text-secondary: #666666;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --accent-light: #eff6ff;
  --radius: 16px;
  --radius-sm: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
}

footer { display: none !important; }

.gradio-container {
  background: var(--bg) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  color: var(--text) !important;
  max-width: 920px !important;
  margin: 0 auto !important;
  padding: 24px 16px !important;
}

.gradio-container * { color: var(--text) !important; }
.gradio-container .prose { color: var(--text) !important; }
.gradio-container label { color: var(--text) !important; font-weight: 500 !important; }

h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.01em; margin-bottom: 4px !important; }

.block, .panel, [data-testid="block"],
textarea, input:not([type="hidden"]), select {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
  transition: border-color 150ms ease, box-shadow 150ms ease !important;
}
textarea:focus, input:focus, select:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
  outline: none !important;
}

button, .gr-button {
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  transition: all 150ms ease !important;
  cursor: pointer !important;
}
button:hover, .gr-button:hover {
  filter: brightness(0.95);
}
button:active, .gr-button:active {
  transform: scale(0.98);
}
button.primary, .gr-button.primary {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
}
button.primary:hover, .gr-button.primary:hover {
  background: var(--accent-hover) !important;
}
button.secondary, .gr-button.secondary {
  background: #f3f4f6 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

.lw-subtitle { color: var(--text-secondary) !important; font-size: 0.92rem; margin-top: 0 !important; }

.lw-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 20px 24px;
  animation: lw-fadeUp 0.4s ease both;
}

@keyframes lw-fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.lw-results-row { gap: 20px !important; }
.lw-results-row > .column { animation: lw-fadeUp 0.5s ease both; }
.lw-results-row > .column:nth-child(2) { animation-delay: 0.1s; }

/* Upload zone */
.image-container {
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  transition: border-color 200ms ease, background 200ms ease !important;
}
.image-container:hover {
  border-color: var(--accent) !important;
  background: var(--accent-light) !important;
}

/* Loading card */
@keyframes lw-pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.5; }
}
@keyframes lw-shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.lw-loading {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  padding: 32px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.lw-loading::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 30%, var(--accent-light) 50%, transparent 70%);
  background-size: 200% 100%;
  animation: lw-shimmer 3s ease infinite;
  opacity: 0.4;
  pointer-events: none;
}
.lw-loading h3 { margin: 0 0 16px 0; font-size: 1.1rem; position: relative; z-index: 1; }
.lw-step { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 0.88rem; position: relative; z-index: 1; }
.lw-step .dot {
  width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; flex-shrink: 0; transition: all 300ms ease;
}
.lw-step .dot.pending  { background: #f3f4f6; color: #9ca3af; }
.lw-step .dot.active   { background: var(--accent); color: #fff; animation: lw-pulse 1.5s ease infinite; }
.lw-step .dot.done     { background: #22c55e; color: #fff; }
.lw-step .label { color: var(--text-secondary); }
.lw-step .label.active { color: var(--text); font-weight: 600; }
.lw-step .label.done   { color: #22c55e; }
.lw-elapsed { margin-top: 16px; font-size: 0.82rem; color: var(--text-secondary); position: relative; z-index: 1; }
.lw-fun-fact {
  margin-top: 12px; font-size: 0.78rem; color: var(--text-secondary); font-style: italic;
  position: relative; z-index: 1; min-height: 1.4em;
}

/* Confidence badge */
.lw-badge { display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.04em; padding: 3px 10px; border-radius: 999px; }
.lw-badge.high   { background: #dcfce7; color: #166534 !important; }
.lw-badge.medium { background: #fef9c3; color: #854d0e !important; }
.lw-badge.low    { background: #fee2e2; color: #991b1b !important; }

/* Quality report callout */
.lw-qa-card {
  background: var(--accent-light);
  border: 1px solid #bfdbfe;
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-top: 16px;
  animation: lw-fadeUp 0.6s ease both;
}
.lw-qa-card h4 { margin: 0 0 8px 0; font-size: 0.95rem; }
.lw-qa-card p, .lw-qa-card li { font-size: 0.85rem; color: var(--text-secondary) !important; line-height: 1.55; }

/* SuperSplat CTA */
a.lw-cta {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 12px 20px; font-weight: 700; border-radius: var(--radius-sm);
  text-decoration: none; width: 100%;
  background: var(--accent); color: #fff !important;
  transition: background 150ms ease;
}
a.lw-cta:hover { background: var(--accent-hover); }

/* Expandable details */
details { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0; margin: 8px 0; overflow: hidden; }
details summary {
  padding: 10px 14px; font-weight: 600; font-size: 0.88rem; cursor: pointer;
  list-style: none; display: flex; align-items: center; gap: 8px;
}
details summary::before { content: '\\25B6'; font-size: 0.65rem; transition: transform 200ms ease; }
details[open] summary::before { transform: rotate(90deg); }
details > :not(summary) { padding: 0 14px 12px 14px; }

/* Progress bar override */
.progress-bar-wrap { border-radius: 999px !important; }

/* Code block */
.code-wrap, .cm-editor { border-radius: var(--radius-sm) !important; }

/* GPU info at bottom */
.lw-info-footer { margin-top: 24px; }
.lw-info-footer summary { font-size: 0.85rem; color: var(--text-secondary) !important; cursor: pointer; }
.lw-info-footer p { font-size: 0.82rem; color: var(--text-secondary) !important; }

/* ── Hero ── */
.lw-hero { text-align: center; padding: 20px 0 8px; }
.lw-hero h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important; margin-bottom: 6px !important; }
.lw-tagline { font-size: 1.05rem; color: var(--text-secondary) !important; margin: 0 !important; }

/* ── 3-step guide ── */
.lw-steps {
  display: flex; align-items: center; justify-content: center;
  gap: 0; margin: 20px 0 24px; flex-wrap: wrap; gap: 4px;
}
.lw-step-item {
  display: flex; align-items: center; gap: 8px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 9px 16px;
  font-size: 0.85rem; font-weight: 600; color: var(--text);
  white-space: nowrap;
}
.lw-step-num {
  width: 22px; height: 22px; background: var(--accent); color: #fff !important;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700; flex-shrink: 0;
}
.lw-step-arrow { font-size: 1rem; color: #d1d5db; padding: 0 4px; }
@media (max-width: 540px) { .lw-step-arrow { display: none; } }

/* ── Upload tip ── */
.lw-upload-tip {
  text-align: center; font-size: 0.82rem; color: var(--text-secondary) !important;
  margin: 6px 0 18px !important;
}

/* ── Mode selector ── */
.lw-mode-card input[type="radio"] + label {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 12px 16px !important;
}
.lw-mode-card input[type="radio"]:checked + label {
  border-color: var(--accent) !important;
  background: var(--accent-light) !important;
}

/* ── AI key help link ── */
.lw-key-help { display: flex; align-items: flex-end; padding-bottom: 6px; }
.lw-key-help a {
  font-size: 0.85rem; font-weight: 600; color: var(--accent) !important;
  text-decoration: none; border: 1px solid #bfdbfe;
  padding: 8px 14px; border-radius: 10px; white-space: nowrap;
  transition: background 150ms ease;
}
.lw-key-help a:hover { background: var(--accent-light); }

/* ── Ready banner ── */
.lw-ready {
  background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: var(--radius);
  padding: 14px 20px; display: flex; align-items: center; gap: 10px;
  font-weight: 600; font-size: 0.95rem; color: #166534 !important;
  animation: lw-fadeUp 0.4s ease both; margin-bottom: 4px;
}
.lw-ready * { color: #166534 !important; }

/* ── Section divider label ── */
.lw-section-label {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--text-secondary) !important; margin: 16px 0 8px;
}

/* Immersive 3D viewer section */
.lw-viewer-wrap {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  background: #0a0a0a !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.14) !important;
  border: 1px solid var(--border) !important;
  animation: lw-fadeUp 0.5s ease both;
  animation-delay: 0.15s;
  margin-bottom: 20px;
}
.lw-viewer-wrap iframe {
  display: block !important;
  width: 100% !important;
  height: 600px !important;
  border: none !important;
  background: #0a0a0a !important;
}
@media (max-width: 640px) {
  .lw-viewer-wrap iframe { height: 420px !important; }
}
.lw-viewer-label {
  padding: 10px 16px 6px;
  font-size: 0.78rem;
  color: var(--text-secondary) !important;
  text-align: center;
  background: #0a0a0a;
}
.lw-viewer-label a { color: var(--accent) !important; }
"""

# =============================================================================
# Constants
# =============================================================================

SUPERSPLAT_URL = "https://playcanvas.com/supersplat/editor/"
SHARP_REPO = "https://github.com/apple/ml-sharp"
WORK_ROOT = Path(tempfile.gettempdir()) / "lenswander_pro"
INPUT_BASENAME = "input"
PREVIEW_MAX_POINTS = 52_000

DEFAULT_CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
CLAUDE_MODEL_CHOICES: list[tuple[str, str]] = [
    ("Claude 3.5 Sonnet (recommended)", "claude-3-5-sonnet-20241022"),
    ("Claude Sonnet 4", "claude-sonnet-4-20250514"),
    ("Claude 3 Opus", "claude-3-opus-20240229"),
]

# =============================================================================
# Immersive 3D viewer template
# =============================================================================
# Served as viewer.html via Gradio's file endpoint (allowed_paths).
# PLY_URL placeholder is replaced by _build_viewer_page().
# Uses @mkkellogg/gaussian-splats-3d (bundles Three.js internally).
# Camera math uses direct position.set / lookAt to avoid a second Three.js import.

_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>LensWander 3D</title>
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 100%; height: 100%; overflow: hidden; background: #0a0a0a; touch-action: none; }

#loading {
  position: fixed; inset: 0; z-index: 200;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: #0a0a0a; color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
  transition: opacity 0.6s ease;
}
#loading.out { opacity: 0; pointer-events: none; }
.ring {
  width: 48px; height: 48px;
  border: 3px solid rgba(255,255,255,0.1);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.85s linear infinite;
  margin-bottom: 20px;
}
@keyframes spin { to { transform: rotate(360deg); } }
#load-title { font-size: 17px; font-weight: 600; letter-spacing: -0.01em; margin-bottom: 6px; }
#load-sub   { font-size: 12px; color: rgba(255,255,255,0.4); margin-bottom: 24px; }
#load-track { width: 180px; height: 2px; background: rgba(255,255,255,0.08); border-radius: 1px; }
#load-fill  { height: 100%; width: 0%; background: #3b82f6; border-radius: 1px; transition: width 0.2s ease; }

#hint {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  background: rgba(0,0,0,0.65); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.1); border-radius: 100px;
  color: rgba(255,255,255,0.9); font-family: -apple-system, sans-serif;
  font-size: 13px; font-weight: 500; padding: 10px 22px;
  white-space: nowrap; pointer-events: none; z-index: 50;
  opacity: 0; transition: opacity 0.4s ease;
}
#hint.show { opacity: 1; }

#badge {
  position: fixed; top: 14px; left: 16px; z-index: 30;
  font-family: -apple-system, sans-serif; font-size: 13px; font-weight: 700;
  color: rgba(255,255,255,0.5); letter-spacing: 0.02em; pointer-events: none;
}
#top-right { position: fixed; top: 11px; right: 13px; z-index: 30; display: flex; gap: 8px; }

.btn {
  height: 35px; padding: 0 14px;
  background: rgba(255,255,255,0.07); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.12); border-radius: 100px;
  color: rgba(255,255,255,0.85); font-family: -apple-system, sans-serif;
  font-size: 13px; font-weight: 600; cursor: pointer;
  transition: all 0.15s ease; user-select: none; white-space: nowrap;
  display: inline-flex; align-items: center; gap: 5px;
}
.btn:hover  { background: rgba(255,255,255,0.13); border-color: rgba(255,255,255,0.22); }
.btn:active { transform: scale(0.93); }
.btn.on     { background: rgba(59,130,246,0.55); border-color: rgba(59,130,246,0.8); color: #fff; }

#hud {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 30;
  display: flex; justify-content: center; align-items: center; gap: 8px;
  padding: 14px 16px 18px;
  background: linear-gradient(to top, rgba(0,0,0,0.55) 0%, transparent 100%);
  pointer-events: none;
}
#hud .btn { pointer-events: auto; }
</style>
</head>
<body>

<div id="loading">
  <div class="ring"></div>
  <div id="load-title">Building 3D World</div>
  <div id="load-sub">Loading Gaussians…</div>
  <div id="load-track"><div id="load-fill"></div></div>
</div>

<div id="hint"></div>
<div id="badge">LensWander Pro</div>

<div id="top-right">
  <button class="btn" id="gyro-btn" style="display:none" onclick="toggleGyro()">&#128241; Motion</button>
  <button class="btn" onclick="doFullscreen()">&#x26F6;</button>
</div>

<div id="hud">
  <button class="btn" onclick="goHome()">&#8635; Reset</button>
  <button class="btn" id="spin-btn" onclick="toggleSpin()">&#9654; Spin</button>
  <button class="btn" onclick="gotoTop()">&#11014; Top</button>
  <button class="btn" onclick="gotoFront()">&#9711; Front</button>
</div>

<script type="module">
import * as GaussianSplats3D from 'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4/build/gaussian-splats-3d.module.js';

const PLY_URL  = '__PLY_URL__';
const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

// ── DOM refs ──
const fillEl  = document.getElementById('load-fill');
const subEl   = document.getElementById('load-sub');
const loadEl  = document.getElementById('loading');
const hintEl  = document.getElementById('hint');

// ── State ──
let viewer, controls, camera;
let spinning  = false;
let gyroOn    = false;
let gyroState = null;   // {beta0, gamma0, phi0, theta0, r}

// ── Hint ──
function showHint(msg, ms) {
  hintEl.textContent = msg;
  hintEl.classList.add('show');
  clearTimeout(hintEl._t);
  hintEl._t = setTimeout(() => hintEl.classList.remove('show'), ms || 3500);
}

// ── Smooth lerp camera ──
function lerpCam(tp, tt, ms) {
  ms = ms || 700;
  const p0 = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
  const t0 = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
  const ts = performance.now();
  function eio(t) { return t < 0.5 ? 2*t*t : 1-((-2*t+2)*((-2*t+2)))/2; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function step() {
    const prog = Math.min((performance.now() - ts) / ms, 1);
    const e = eio(prog);
    camera.position.set(lerp(p0.x, tp.x, e), lerp(p0.y, tp.y, e), lerp(p0.z, tp.z, e));
    controls.target.set(lerp(t0.x, tt.x, e), lerp(t0.y, tt.y, e), lerp(t0.z, tt.z, e));
    controls.update();
    if (prog < 1) requestAnimationFrame(step);
  }
  step();
}

// ── View presets (SHARP: Y-down, Z-forward) ──
// Home: slightly above and in front of the scene
const H_POS  = { x: -0.5, y: -3, z: 6 };
const H_TARG = { x: 0, y: 0, z: 0 };

window.goHome    = () => lerpCam(H_POS, H_TARG);
window.gotoTop   = () => lerpCam({ x: 0, y: -9, z: 0.1 }, H_TARG);
window.gotoFront = () => lerpCam({ x: 0, y: -0.5, z: 8  }, H_TARG);

// ── Auto-spin ──
window.toggleSpin = function() {
  spinning = !spinning;
  const btn = document.getElementById('spin-btn');
  btn.classList.toggle('on', spinning);
  btn.textContent = spinning ? '\u23F8 Stop' : '\u25B6 Spin';
  controls.autoRotate      = spinning;
  controls.autoRotateSpeed = 0.9;
};

// ── Fullscreen ──
window.doFullscreen = function() {
  const el = document.documentElement;
  if (el.requestFullscreen)            el.requestFullscreen();
  else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
};

// ── Gyroscope ──
window.toggleGyro = function() {
  if (gyroOn) { stopGyro(); return; }
  if (typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function') {
    DeviceOrientationEvent.requestPermission()
      .then(s => { if (s === 'granted') startGyro(); else showHint('\uD83D\uDEAB Permission denied'); })
      .catch(() => showHint('\uD83D\uDEAB Sensor error'));
  } else {
    startGyro();
  }
};

function startGyro() {
  gyroOn = true; gyroState = null;
  controls.enabled = false;
  window.addEventListener('deviceorientation', onOrient, true);
  document.getElementById('gyro-btn').classList.add('on');
  document.getElementById('gyro-btn').textContent = '\uD83D\uDCF1 ON';
  showHint('\uD83D\uDCF1 Tilt your phone to explore', 5000);
}

function stopGyro() {
  gyroOn = false; gyroState = null;
  controls.enabled = true;
  window.removeEventListener('deviceorientation', onOrient, true);
  const btn = document.getElementById('gyro-btn');
  btn.classList.remove('on');
  btn.textContent = '\uD83D\uDCF1 Motion';
  goHome();
}

function onOrient(e) {
  if (!gyroOn || e.beta == null) return;
  // β = front/back tilt (degrees), γ = left/right tilt
  const beta  = e.beta  || 0;
  const gamma = e.gamma || 0;

  if (!gyroState) {
    // Capture initial spherical position of camera
    const dx = camera.position.x - controls.target.x;
    const dy = camera.position.y - controls.target.y;
    const dz = camera.position.z - controls.target.z;
    const r  = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const phi0   = Math.acos(Math.max(-1, Math.min(1, dy / r)));
    const theta0 = Math.atan2(dx, dz);
    gyroState = { beta0: beta, gamma0: gamma, phi0, theta0, r };
    return;
  }

  const DEG = Math.PI / 180;
  // Map deltas: front-tilt moves polar angle, side-tilt moves azimuth
  let phi   = gyroState.phi0   + (beta  - gyroState.beta0)  * DEG * 0.7;
  let theta = gyroState.theta0 - (gamma - gyroState.gamma0) * DEG * 0.7;
  phi = Math.max(0.04, Math.min(Math.PI - 0.04, phi));

  const r = gyroState.r;
  const t = controls.target;
  camera.position.set(
    t.x + r * Math.sin(phi) * Math.sin(theta),
    t.y + r * Math.cos(phi),
    t.z + r * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(t.x, t.y, t.z);
}

// ── Init viewer ──
viewer = new GaussianSplats3D.Viewer({
  selfDrivenMode:         false,
  sharedMemoryForWorkers: false,
  useWorkers:             true,
  workerConfig:           { crossOriginIsolated: false },
  cameraUp:               [0, -1, 0],          // SHARP: y-down
  initialCameraPosition:  [-0.5, -3, 6],
  initialCameraLookAt:    [0, 0, 0],
  gpuAcceleratedSort:     true,
  dynamicScene:           false,
});
camera   = viewer.camera;
controls = viewer.controls;
controls.enableDamping      = true;
controls.dampingFactor      = 0.07;
controls.screenSpacePanning = false;
controls.minDistance        = 0.25;
controls.maxDistance        = 35;
controls.rotateSpeed        = isMobile ? 0.5 : 0.65;
controls.zoomSpeed          = isMobile ? 0.75 : 1.0;
controls.panSpeed           = 0.8;

// ── Load scene ──
viewer.addSplatScene(PLY_URL, {
  splatAlphaRemovalThreshold: 5,
  showLoadingUI:  false,
  onProgress: (p) => {
    const pct = Math.round(p * 100);
    fillEl.style.width = pct + '%';
    subEl.textContent  = pct + '% loaded';
  },
}).then(() => {
  loadEl.classList.add('out');
  setTimeout(() => { loadEl.style.display = 'none'; }, 650);

  // Show mobile features
  if (isMobile && typeof DeviceOrientationEvent !== 'undefined') {
    document.getElementById('gyro-btn').style.display = 'inline-flex';
  }
  showHint(
    isMobile
      ? '\uD83D\uDC46 Drag to orbit \u00B7 Pinch to zoom'
      : '\uD83D\uDDB1 Drag to orbit \u00B7 Scroll to zoom \u00B7 Right-drag to pan',
    4500
  );
  animate();
}).catch(err => {
  document.querySelector('.ring').style.display = 'none';
  document.getElementById('load-title').textContent = '\u26A0 Load failed';
  subEl.textContent = err && err.message ? err.message : 'Could not load scene';
});

// ── Render loop ──
function animate() {
  requestAnimationFrame(animate);
  if (!gyroOn) controls.update();
  viewer.update();
  viewer.render();
}
</script>
</body>
</html>"""


FUN_FACTS = [
    "SHARP generates ~1.2 million Gaussians per image in a single forward pass.",
    "The neural network runs internally at 1536 x 1536 resolution.",
    "3D Gaussian Splatting was introduced in the 2023 SIGGRAPH paper by Kerbl et al.",
    "SHARP produces metric-scale output — real-world distances are preserved.",
    "SuperSplat renders Gaussians at 100+ FPS in your browser.",
    "Each Gaussian stores position, color (spherical harmonics), opacity, scale, and rotation.",
    "SHARP uses OpenCV conventions: x right, y down, z forward.",
]

# =============================================================================
# Claude prompts
# =============================================================================

SCENE_INTELLIGENCE_SYSTEM = """You are a world-class 3D scene reconstruction specialist with deep expertise in photogrammetry, Gaussian Splatting, and novel view synthesis.

Analyze the provided photograph for single-image 3D reconstruction. Output ONLY valid JSON matching this schema:

{
  "scene_summary": "one-sentence overview of the scene",
  "detailed_caption": "rich paragraph: scene content, lighting, time of day, mood, materials",
  "preprocessing_advice": "honest assessment of image quality for 3D reconstruction: mention blur, reflections, flat textures, extreme wide-angle, low light, or other issues that will hurt depth estimation. If the image looks good, say so briefly.",
  "depth_confidence": "high | medium | low",
  "objects": [{"name": "...", "estimated_3d_position": "front-center / left-midground / etc.", "properties": "color, material, size"}],
  "lighting_analysis": "direction, type, shadows, reflections",
  "estimated_geometry": "layout description (room, outdoor, object on table, etc.)",
  "potential_artifacts": ["specific areas where single-image 3D will struggle and why"],
  "optimal_viewing_angles": ["best angle 1 to inspect the splat", "best angle 2", "best angle 3"],
  "recommended_camera_paths": ["4-6 orbits/fly-throughs"],
  "novel_view_descriptions": [
    {"view": "30 deg left", "caption": "detailed description preserving lighting and style"},
    {"view": "30 deg right", "caption": "..."},
    {"view": "20 deg up", "caption": "..."},
    {"view": "20 deg down", "caption": "..."},
    {"view": "behind", "caption": "..."},
    {"view": "top-down", "caption": "..."},
    {"view": "close-up", "caption": "..."},
    {"view": "wide pull-back", "caption": "..."}
  ]
}

depth_confidence rules:
- "high": well-lit, textured scene, clear depth cues (perspective lines, size gradients, occlusion)
- "medium": decent but some problem areas (flat walls, uniform textures, moderate blur)
- "low": significant issues (mirrors, glass, extreme blur, very flat subject, pure sky/fog)

In potential_artifacts, always mention depth ambiguity, side-profile flattening, and thin structures.
In recommended_camera_paths, include at least one side-on / cross-depth orbit.
Keep novel_view_descriptions consistent with the original photo's lighting and color grading."""

SCENE_INTELLIGENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_summary": {"type": "string"},
        "detailed_caption": {"type": "string"},
        "preprocessing_advice": {"type": "string"},
        "depth_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "estimated_3d_position": {"type": "string"},
                    "properties": {"type": "string"},
                },
                "required": ["name", "estimated_3d_position", "properties"],
                "additionalProperties": True,
            },
        },
        "lighting_analysis": {"type": "string"},
        "estimated_geometry": {"type": "string"},
        "potential_artifacts": {"type": "array", "items": {"type": "string"}},
        "optimal_viewing_angles": {"type": "array", "items": {"type": "string"}},
        "recommended_camera_paths": {"type": "array", "items": {"type": "string"}},
        "novel_view_descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"view": {"type": "string"}, "caption": {"type": "string"}},
                "required": ["view", "caption"],
                "additionalProperties": True,
            },
        },
    },
    "required": [
        "scene_summary", "detailed_caption", "preprocessing_advice", "depth_confidence",
        "objects", "lighting_analysis", "estimated_geometry", "potential_artifacts",
        "optimal_viewing_angles", "recommended_camera_paths", "novel_view_descriptions",
    ],
    "additionalProperties": True,
}

QUALITY_REPORT_SYSTEM = """You are a 3D reconstruction quality inspector. You will receive two images:
1. The original photograph
2. A scatter-plot preview of the 3D Gaussian Splat (XY image plane + XZ and YZ side profiles)

Compare them and output ONLY valid JSON:
{
  "quality_score": 7,
  "quality_summary": "one sentence overall assessment",
  "visible_artifacts": ["specific artifact 1 with location", "artifact 2"],
  "inspection_checklist": [
    "What to check when orbiting in SuperSplat — item 1",
    "item 2",
    "item 3"
  ]
}

quality_score: integer 1-10 (10 = perfect reconstruction, 1 = completely broken).
visible_artifacts: compare the scatter preview to the original — note smearing, missing regions, depth collapse, color shifts.
inspection_checklist: 3-5 actionable things the user should look for when loading the .ply in a 3D viewer."""

QUALITY_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "quality_score": {"type": "integer"},
        "quality_summary": {"type": "string"},
        "visible_artifacts": {"type": "array", "items": {"type": "string"}},
        "inspection_checklist": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["quality_score", "quality_summary", "visible_artifacts", "inspection_checklist"],
    "additionalProperties": True,
}

# =============================================================================
# Device helpers
# =============================================================================


def _resolve_sharp_device() -> str:
    raw = os.environ.get("SHARP_DEVICE", "").strip()
    if raw and raw != "default":
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_info_html() -> str:
    dev = _resolve_sharp_device()
    parts = [f"SHARP device: <code>{dev}</code>"]
    if torch.cuda.is_available():
        parts.append(f"CUDA: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        parts.append("Apple Metal (MPS): available")
    else:
        parts.append("GPU: none detected (CPU mode — slower)")
    ckpt = os.environ.get("SHARP_CHECKPOINT_PATH", "").strip()
    if ckpt:
        parts.append(f"Custom checkpoint: <code>{ckpt}</code>")
    return " &middot; ".join(parts)


# =============================================================================
# SHARP runner
# =============================================================================


def _find_sharp_command() -> list[str]:
    exe = shutil.which("sharp")
    if exe:
        return [exe]
    return [
        sys.executable, "-c",
        "import sys; from sharp.cli import main_cli; sys.argv = ['sharp'] + sys.argv[1:]; main_cli()",
    ]


def _run_sharp_predict(
    image_path: Path, out_dir: Path, device: str,
    progress: gr.Progress, frac_lo: float = 0.12, frac_hi: float = 0.82,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _find_sharp_command() + [
        "predict", "-i", str(image_path), "-o", str(out_dir), "--device", device,
    ]
    ckpt = os.environ.get("SHARP_CHECKPOINT_PATH", "").strip()
    if ckpt:
        cmd.extend(["-c", ckpt])
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if device == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    holder: dict[str, Any] = {}

    def _worker() -> None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, env=env)
        holder["returncode"] = proc.returncode
        holder["stdout"] = proc.stdout or ""

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    t0 = time.monotonic()
    backend = device.upper()
    while thread.is_alive():
        elapsed = time.monotonic() - t0
        frac = frac_lo + (frac_hi - frac_lo) * (elapsed / (elapsed + 18.0))
        frac = min(frac_hi - 0.01, frac)
        progress(frac, desc=f"Building 3D Gaussians (SHARP {backend})… {int(elapsed)}s")
        time.sleep(0.35)
    thread.join(timeout=5.0)

    rc = holder.get("returncode", -1)
    out = holder.get("stdout", "")
    if rc != 0:
        tail = "\n".join(out.splitlines()[-40:]) if out else "(no output)"
        raise RuntimeError(f"SHARP exited with code {rc}.\n\n{tail}")


# =============================================================================
# PLY preview (matplotlib)
# =============================================================================


def _ply_preview_image(ply_path: Path) -> Image.Image | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from plyfile import PlyData
        import numpy as np
    except Exception:
        return None
    try:
        def _sigmoid(t: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(t.astype(np.float64), -30, 30)))

        ply = PlyData.read(str(ply_path))
        vtx = next(e for e in ply.elements if e.name == "vertex")
        n = len(vtx["x"])
        if n == 0:
            return None
        props = {p.name for p in vtx.properties}
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(n, PREVIEW_MAX_POINTS), replace=False)
        x = np.asarray(vtx["x"], dtype=np.float64)[idx]
        y = np.asarray(vtx["y"], dtype=np.float64)[idx]
        z = np.asarray(vtx["z"], dtype=np.float64)[idx] if "z" in props else np.zeros_like(x)

        if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(props):
            c0 = np.sqrt(1.0 / (4.0 * np.pi))
            rgb = np.stack([
                np.clip(np.asarray(vtx["f_dc_0"])[idx].astype(np.float64) * c0 + 0.5, 0, 1),
                np.clip(np.asarray(vtx["f_dc_1"])[idx].astype(np.float64) * c0 + 0.5, 0, 1),
                np.clip(np.asarray(vtx["f_dc_2"])[idx].astype(np.float64) * c0 + 0.5, 0, 1),
            ], axis=1)
            alpha = np.clip(0.08 + 0.72 * _sigmoid(np.asarray(vtx["opacity"], dtype=np.float64)[idx]), 0.04, 0.95) if "opacity" in props else np.full(len(x), 0.38)
            sizes = np.full(len(x), 2.1)
            if {"scale_0", "scale_1", "scale_2"}.issubset(props):
                gm = np.exp((np.asarray(vtx["scale_0"], dtype=np.float64)[idx] + np.asarray(vtx["scale_1"], dtype=np.float64)[idx] + np.asarray(vtx["scale_2"], dtype=np.float64)[idx]) / 3.0)
                sizes = np.clip(1.2 + 5.5 * (gm / (float(np.median(gm)) + 1e-9)), 0.9, 7.0)
        else:
            z0 = z - float(np.median(z))
            za = float(np.percentile(np.abs(z0), 95) or 1.0)
            t = np.clip((z0 / za + 1) * 0.5, 0, 1)
            rgb = np.stack([t, 0.55 * (1 - t), 1.0 - 0.35 * t], axis=1)
            alpha = np.full(len(x), 0.4)
            sizes = np.full(len(x), 1.8)

        fig = plt.figure(figsize=(10.2, 5.35), dpi=150)
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.22, 1.0], wspace=0.28, hspace=0.36)
        ax_xy = fig.add_subplot(gs[:, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[1, 1])

        for ax in (ax_xy, ax_xz, ax_yz):
            ax.set_facecolor("#fafafa")
            ax.tick_params(colors="#111", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#e5e7eb")
        fig.patch.set_facecolor("#ffffff")

        ax_xy.scatter(x, y, c=rgb, s=sizes, alpha=alpha, linewidths=0, rasterized=True)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.set_title("Image plane (XY)", color="#111", fontsize=10, fontweight="600")
        ax_xy.set_xlabel("x", color="#111", fontsize=8)
        ax_xy.set_ylabel("y", color="#111", fontsize=8)

        ax_xz.scatter(x, z, c=rgb, s=sizes * 0.9, alpha=alpha, linewidths=0, rasterized=True)
        ax_xz.set_aspect("equal", adjustable="box")
        ax_xz.set_title("Side — X vs depth", color="#111", fontsize=10, fontweight="600")
        ax_xz.set_xlabel("x", color="#111", fontsize=8)
        ax_xz.set_ylabel("z (depth)", color="#111", fontsize=8)

        ax_yz.scatter(y, z, c=rgb, s=sizes * 0.9, alpha=alpha, linewidths=0, rasterized=True)
        ax_yz.set_aspect("equal", adjustable="box")
        ax_yz.set_title("Side — Y vs depth", color="#111", fontsize=10, fontweight="600")
        ax_yz.set_xlabel("y", color="#111", fontsize=8)
        ax_yz.set_ylabel("z (depth)", color="#111", fontsize=8)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception:
        return None


# =============================================================================
# Viewer page builder
# =============================================================================


def _build_viewer_page(ply_path: Path) -> str:
    """Write viewer.html next to the PLY. Returns iframe HTML for gr.HTML.

    The viewer HTML is served via Gradio's /gradio_api/file= endpoint
    (requires WORK_ROOT in allowed_paths). The PLY is fetched same-origin
    so there are no CORS issues.

    No blank pixels: 3DGS renders every pixel as a Gaussian blend —
    the dark background shows through only for angles far outside the
    camera's original frustum, which SHARP fills via learned priors.
    """
    ply_url = f"/gradio_api/file={ply_path}"
    page = _VIEWER_TEMPLATE.replace("__PLY_URL__", ply_url)
    viewer_path = ply_path.parent.parent / "viewer.html"   # run_dir/viewer.html
    viewer_path.write_text(page, encoding="utf-8")
    iframe_src = f"/gradio_api/file={viewer_path}"
    return (
        f'<div class="lw-viewer-wrap">'
        f'<iframe src="{iframe_src}"'
        f' allow="gyroscope; accelerometer; fullscreen"'
        f' allowfullscreen loading="lazy" title="3D Scene Viewer"></iframe>'
        f'<p class="lw-viewer-label">No 3D? &nbsp;'
        f'<a href="{SUPERSPLAT_URL}" target="_blank" rel="noopener">Open in SuperSplat ↗</a>'
        f' &nbsp;·&nbsp; Drag to orbit · Scroll to zoom · Pinch on mobile</p>'
        f'</div>'
    )


# =============================================================================
# Claude API calls
# =============================================================================


def _encode_image_b64(image: Image.Image, quality: int = 90) -> tuple[str, str]:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _anthropic_api_key(pasted: str) -> str:
    return (pasted or "").strip() or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()


def _strip_json_fences(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _call_claude(*, api_key: str, model: str, system: str, schema: dict, images: list[Image.Image], user_text: str) -> dict[str, Any]:
    import anthropic

    img_blocks: list[dict[str, Any]] = []
    for img in images:
        mt, b64 = _encode_image_b64(img)
        img_blocks.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
    img_blocks.append({"type": "text", "text": user_text})

    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": 16384, "system": system,
        "messages": [{"role": "user", "content": img_blocks}],
    }
    kw_struct = {**kwargs, "output_config": {"format": {"type": "json_schema", "schema": schema}}}
    try:
        msg = client.messages.create(**kw_struct)
    except Exception:
        msg = client.messages.create(**kwargs)

    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text" and hasattr(b, "text")]
    raw = "\n".join(parts).strip()
    if not raw:
        raise RuntimeError("Claude returned an empty response.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_json_fences(raw))


# =============================================================================
# Loading state HTML builder
# =============================================================================

STEP_LABELS = [
    "Reading your photo",
    "Building your 3D world",
    "Rendering depth views",
    "AI quality check",
    "Finishing up",
]


def _loading_html(active_idx: int, elapsed_s: int = 0, fun_fact_idx: int = 0) -> str:
    steps_html = ""
    for i, label in enumerate(STEP_LABELS):
        if i < active_idx:
            dot_cls, lbl_cls, icon = "done", "done", "&#10003;"
        elif i == active_idx:
            dot_cls, lbl_cls, icon = "active", "active", str(i + 1)
        else:
            dot_cls, lbl_cls, icon = "pending", "", str(i + 1)
        steps_html += f'<div class="lw-step"><span class="dot {dot_cls}">{icon}</span><span class="label {lbl_cls}">{html.escape(label)}</span></div>\n'
    fact = html.escape(FUN_FACTS[fun_fact_idx % len(FUN_FACTS)])
    return f"""<div class="lw-loading">
<h3>Creating your 3D scene...</h3>
{steps_html}
<div class="lw-elapsed">{elapsed_s}s elapsed</div>
<div class="lw-fun-fact">{fact}</div>
</div>"""


# =============================================================================
# Result formatters
# =============================================================================


def _scene_intelligence_md(data: dict[str, Any]) -> str:
    lines: list[str] = []
    conf = data.get("depth_confidence", "medium")
    badge_cls = conf if conf in ("high", "medium", "low") else "medium"
    badge_label = {"high": "Great for 3D", "medium": "Good for 3D", "low": "Tricky for 3D"}.get(conf, conf)
    lines.append(f'<span class="lw-badge {badge_cls}">{badge_label}</span>\n')
    lines.append(f"**{data.get('scene_summary', '')}**\n")
    advice = data.get("preprocessing_advice", "")
    if advice:
        lines.append(f"> {advice}\n")

    lines.append("<details><summary>Detailed caption</summary>\n\n" + str(data.get("detailed_caption", "")) + "\n\n</details>\n")

    objs = data.get("objects") or []
    if objs:
        obj_lines = "\n".join(f"- **{o.get('name','?')}** — {o.get('estimated_3d_position','')} — {o.get('properties','')}" for o in objs if isinstance(o, dict))
        lines.append(f"<details><summary>Objects & layout</summary>\n\n{obj_lines}\n\n</details>\n")

    lines.append(f"<details><summary>Lighting & geometry</summary>\n\n**Lighting:** {data.get('lighting_analysis','')}\n\n**Geometry:** {data.get('estimated_geometry','')}\n\n</details>\n")

    arts = data.get("potential_artifacts") or []
    if arts:
        art_lines = "\n".join(f"- {a}" for a in arts)
        lines.append(f"<details><summary>Potential artifacts</summary>\n\n{art_lines}\n\n</details>\n")

    angles = data.get("optimal_viewing_angles") or []
    if angles:
        ang_lines = "\n".join(f"{i}. {a}" for i, a in enumerate(angles, 1))
        lines.append(f"<details><summary>Best viewing angles</summary>\n\n{ang_lines}\n\n</details>\n")

    cams = data.get("recommended_camera_paths") or []
    if cams:
        cam_lines = "\n".join(f"{i}. {c}" for i, c in enumerate(cams, 1))
        lines.append(f"<details><summary>Camera paths</summary>\n\n{cam_lines}\n\n</details>\n")

    nvs = data.get("novel_view_descriptions") or []
    if nvs:
        nv_lines = "\n".join(f"**{nv.get('view','')}:** {nv.get('caption','')}\n" for nv in nvs if isinstance(nv, dict))
        lines.append(f"<details><summary>Novel view descriptions</summary>\n\n{nv_lines}\n\n</details>\n")

    return "\n".join(lines)


def _quality_report_html(data: dict[str, Any]) -> str:
    score = data.get("quality_score", "?")
    summary = html.escape(str(data.get("quality_summary", "")))
    artifacts = data.get("visible_artifacts") or []
    checklist = data.get("inspection_checklist") or []
    art_html = "".join(f"<li>{html.escape(str(a))}</li>" for a in artifacts)
    chk_html = "".join(f"<li>{html.escape(str(c))}</li>" for c in checklist)
    return f"""<div class="lw-qa-card">
<h4>&#129302; AI Quality Check &mdash; {score}/10</h4>
<p>{summary}</p>
{"<p><strong>Things to watch for:</strong></p><ul>" + art_html + "</ul>" if art_html else ""}
{"<p><strong>While exploring in 3D:</strong></p><ul>" + chk_html + "</ul>" if chk_html else ""}
</div>"""


def _tour_notes_html(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    paths = data.get("recommended_camera_paths") or []
    nvs = data.get("novel_view_descriptions") or []
    lines = ["=== DepthDiver — Camera Tour Suggestions ===", ""]
    lines.append("--- Best paths to fly through ---")
    for p in paths:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("--- Novel views ---")
    for nv in nvs:
        if isinstance(nv, dict):
            lines.append(f"[{nv.get('view','')}] {nv.get('caption','')}")
    body = "\n".join(lines)
    pre_safe = html.escape(body)
    js_lit = json.dumps(body)
    return f"""<div>
<p style="font-size:0.82rem;font-weight:600;color:#666;margin-bottom:6px;">Camera tour suggestions — paste into your notes while flying in 3D</p>
<pre style="background:#f8f9fa;border:1px solid #e5e7eb;border-radius:12px;padding:12px;max-height:180px;overflow:auto;font-size:0.78rem;white-space:pre-wrap;">{pre_safe}</pre>
<script>function lwCopy(){{navigator.clipboard.writeText({js_lit})}}</script>
<button type="button" onclick="lwCopy()" style="margin-top:8px;padding:6px 14px;border-radius:8px;border:1px solid #e5e7eb;background:#f3f4f6;cursor:pointer;font-weight:600;font-size:0.82rem;">Copy suggestions</button>
</div>"""


# =============================================================================
# Example image
# =============================================================================


def _make_example_image() -> Image.Image:
    w, h = 768, 512
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / h
        r, g, b = int(40 + t * 30), int(120 - t * 60), int(200 - t * 150)
        for x in range(w):
            px[x, y] = (max(0, r), max(0, g), max(0, b))
    dr = ImageDraw.Draw(img)
    dr.ellipse((w // 2 - 40, 60, w // 2 + 40, 140), fill=(255, 245, 200))
    dr.rectangle((0, h * 2 // 3, w, h), fill=(60, 110, 70))
    dr.rectangle((w // 4, h // 2, w * 3 // 4, h * 2 // 3), fill=(90, 70, 55))
    dr.polygon([(w // 2, h // 2 - 40), (w // 2 - 70, h // 2 + 30), (w // 2 + 70, h // 2 + 30)], fill=(70, 55, 45))
    return img


# =============================================================================
# Pipeline
# =============================================================================

_last_run_dir: Path | None = None


def generate_pipeline(
    image: Image.Image | None,
    api_key_field: str,
    model_id: str,
    mode: Literal["fast", "pro"],
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, ...]:
    """
    Returns 14 updates:
      status_md, loading_html, viewer_section (HTML), results_row,
      orig_out, analysis_md, json_code, right_title, preview,
      download_files, supersplat_html, tour_html, qa_html, ai_accordion
    """
    global _last_run_dir
    prog = progress
    hidden = gr.update(visible=False)
    empty_dl = gr.update(value=None, visible=False)
    no_qa = gr.update(value="")
    no_viewer = gr.update(value="", visible=False)
    no_ai = gr.update(visible=False)

    if image is None:
        return (
            gr.update(value=""),
            gr.update(value=""), no_viewer, hidden,
            gr.update(value=None),
            gr.update(value=""), gr.update(value='{}'),
            gr.update(value=""), gr.update(value=None),
            empty_dl, gr.update(value=""), gr.update(value=""), no_qa, no_ai,
        )

    rgb_input = image.convert("RGB")
    device = _resolve_sharp_device()
    api_key = _anthropic_api_key(api_key_field)
    used_pro = False
    scene_data: dict[str, Any] | None = None
    quality_data: dict[str, Any] | None = None
    json_path: Path | None = None
    qa_path: Path | None = None
    run_mode: Literal["fast", "pro"] = mode

    try:
        if _last_run_dir and _last_run_dir.exists():
            shutil.rmtree(_last_run_dir, ignore_errors=True)
    except OSError:
        pass

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    session = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_dir = WORK_ROOT / session
    run_dir.mkdir(parents=True, exist_ok=True)
    _last_run_dir = run_dir

    in_path = run_dir / f"{INPUT_BASENAME}.png"
    out_dir = run_dir / "gaussians"
    ply_path = out_dir / f"{INPUT_BASENAME}.ply"
    fact_idx = 0
    t_start = time.monotonic()

    def _elapsed() -> int:
        return int(time.monotonic() - t_start)

    try:
        # Show loading state
        prog(0.02, desc="Starting…")

        # --- Step 0: Pre-analysis (Pro only) ---
        if run_mode == "pro":
            if not api_key:
                run_mode = "fast"
            else:
                try:
                    prog(0.05, desc="Claude analyzing scene…")
                    scene_data = _call_claude(
                        api_key=api_key,
                        model=model_id or DEFAULT_CLAUDE_MODEL,
                        system=SCENE_INTELLIGENCE_SYSTEM,
                        schema=SCENE_INTELLIGENCE_SCHEMA,
                        images=[rgb_input],
                        user_text="Analyze this photograph for 3D Gaussian splatting. Follow the system instructions exactly.",
                    )
                    json_path = run_dir / "scene_intelligence.json"
                    json_path.write_text(json.dumps(scene_data, indent=2), encoding="utf-8")
                    used_pro = True
                except Exception as ai_exc:
                    scene_data = None
                    json_path = None
                    run_mode = "fast"

        # --- Step 1: Save & run SHARP ---
        prog(0.10, desc="Saving image…")
        rgb_input.save(in_path, format="PNG")

        _run_sharp_predict(in_path, out_dir, device=device, progress=prog, frac_lo=0.12, frac_hi=0.82)

        if not ply_path.is_file():
            raise FileNotFoundError(f"SHARP did not produce: {ply_path}")

        # --- Step 2: Preview + viewer page ---
        prog(0.84, desc="Generating preview…")
        preview_img = _ply_preview_image(ply_path)
        viewer_iframe = _build_viewer_page(ply_path)

        # --- Step 3: Post-QA (Pro only, if we have preview) ---
        if used_pro and preview_img is not None and api_key:
            try:
                prog(0.88, desc="Claude inspecting reconstruction…")
                quality_data = _call_claude(
                    api_key=api_key,
                    model=model_id or DEFAULT_CLAUDE_MODEL,
                    system=QUALITY_REPORT_SYSTEM,
                    schema=QUALITY_REPORT_SCHEMA,
                    images=[rgb_input, preview_img],
                    user_text="Compare the original photo (first image) with the 3D reconstruction preview (second image). Rate the quality.",
                )
                qa_path = run_dir / "quality_report.json"
                qa_path.write_text(json.dumps(quality_data, indent=2), encoding="utf-8")
            except Exception:
                quality_data = None

        # --- Step 4: Assemble results ---
        prog(0.95, desc="Preparing results…")

        downloads: list[str] = [str(ply_path)]
        if json_path and json_path.is_file():
            downloads.append(str(json_path))
        if qa_path and qa_path.is_file():
            downloads.append(str(qa_path))

        mode_label = "AI-Enhanced" if used_pro else "Free"
        status = f'<div class="lw-ready">&#x2713;&nbsp; Your 3D world is ready! &nbsp;<span style="font-weight:400;font-size:0.85rem;">({mode_label} mode &middot; {device.upper()})</span></div>'

        if used_pro and scene_data:
            analysis_md = _scene_intelligence_md(scene_data)
            json_text = json.dumps(scene_data, indent=2)
        else:
            analysis_md = ""
            json_text = json.dumps({"note": "AI Insights are only available in With AI mode."}, indent=2)

        right_title = "**Depth map views** — how your scene looks from the top, front, and side"

        supersplat_html = (
            f'<a class="lw-cta" href="{SUPERSPLAT_URL}" target="_blank" rel="noopener noreferrer">'
            f'Open in full browser viewer &rarr;</a>'
            f'<p style="font-size:0.78rem;color:#666;margin-top:6px;text-align:center;">'
            f'Drag your 3D file into SuperSplat to share or explore on any device</p>'
        )

        qa_html = _quality_report_html(quality_data) if quality_data else ""

        try:
            progress(1.0, desc="Complete")
        except Exception:
            pass

        return (
            gr.update(value=status),
            gr.update(value=""),
            gr.update(value=viewer_iframe, visible=True),
            gr.update(visible=True),
            gr.update(value=rgb_input),
            gr.update(value=analysis_md, visible=True),
            gr.update(value=json_text, visible=True),
            gr.update(value=right_title),
            gr.update(value=preview_img) if preview_img else gr.update(value=None),
            gr.update(value=downloads, visible=True),
            gr.update(value=supersplat_html),
            gr.update(value=_tour_notes_html(scene_data if used_pro else None)),
            gr.update(value=qa_html),
            gr.update(visible=used_pro, open=True),
        )
    except Exception as exc:
        try:
            progress(1.0, desc="Error")
        except Exception:
            pass
        tb = traceback.format_exc()
        short_err = str(exc).split("\n")[0][:200]
        err = (
            f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:12px;'
            f'padding:16px 20px;font-family:system-ui;">'
            f'<strong style="color:#991b1b;">Something went wrong</strong>'
            f'<p style="color:#7f1d1d;margin-top:6px;font-size:0.88rem;">{html.escape(short_err)}</p>'
            f'<details style="margin-top:8px;"><summary style="cursor:pointer;font-size:0.8rem;color:#666;">Show technical details</summary>'
            f'<pre style="font-size:0.72rem;white-space:pre-wrap;margin-top:8px;color:#666;">{html.escape(tb)}</pre>'
            f'</details></div>'
        )
        return (
            gr.update(value=err),
            gr.update(value=""), no_viewer, hidden,
            gr.update(value=rgb_input),
            gr.update(value=""), gr.update(value=json.dumps({"error": str(exc)}, indent=2)),
            gr.update(value=""), gr.update(value=None),
            empty_dl, gr.update(value=""), gr.update(value=""), no_qa, no_ai,
        )


# =============================================================================
# Gradio app
# =============================================================================


def build_app() -> gr.Blocks:
    theme = gr.themes.Default(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    sample = _make_example_image()
    model_values = [m for _, m in CLAUDE_MODEL_CHOICES]
    model_default = DEFAULT_CLAUDE_MODEL if DEFAULT_CLAUDE_MODEL in model_values else CLAUDE_MODEL_CHOICES[0][1]

    with gr.Blocks(title="DepthDiver") as demo:

        # ── Hero ──────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="lw-hero">
          <h1>DepthDiver</h1>
          <p class="lw-tagline">Turn any photo into a 3D world you can walk through</p>
        </div>
        """)

        # ── 3-step guide ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="lw-steps">
          <div class="lw-step-item"><span class="lw-step-num">1</span> Upload a photo</div>
          <span class="lw-step-arrow">&#8594;</span>
          <div class="lw-step-item"><span class="lw-step-num">2</span> Create 3D</div>
          <span class="lw-step-arrow">&#8594;</span>
          <div class="lw-step-item"><span class="lw-step-num">3</span> Walk around it</div>
        </div>
        """)

        # ── Upload ────────────────────────────────────────────────────────────
        in_image = gr.Image(
            label="Drop your photo here — or click to browse",
            type="pil", image_mode="RGB", height=320,
            sources=["upload", "clipboard"],
        )
        gr.HTML(
            "<p class='lw-upload-tip'>"
            "&#128247; Works great with: objects, rooms, buildings, outdoor scenes, people, food — any photo!"
            "</p>"
        )

        # ── Mode selection ────────────────────────────────────────────────────
        mode_radio = gr.Radio(
            choices=[
                ("Free  —  instant 3D, no account needed", "fast"),
                ("With AI  —  adds scene analysis & quality tips  (needs Anthropic API key, ~$0.05/photo)", "pro"),
            ],
            value="fast",
            label="How would you like to create?",
        )

        # AI key row — only shown when "With AI" is selected
        with gr.Row(visible=False) as ai_key_row:
            with gr.Column(scale=4):
                api_key_in = gr.Textbox(
                    label="Your Anthropic API key",
                    type="password",
                    placeholder="sk-ant-api03-…  (or set ANTHROPIC_API_KEY in .env)",
                    lines=1,
                    info="Your key is used only for this request — never stored.",
                )
            with gr.Column(scale=1, elem_classes=["lw-key-help"]):
                gr.HTML(
                    '<a href="https://console.anthropic.com" target="_blank" rel="noopener">'
                    "Get a free key &#8599;</a>"
                )

        # model_in defined later inside the accordion (still reachable by create_btn.click)

        # ── Create button ─────────────────────────────────────────────────────
        create_btn = gr.Button("Create 3D  \u2192", variant="primary", size="lg")
        ex_btn = gr.Button("Try a sample photo first", variant="secondary", size="sm")

        # ── Status + loading ──────────────────────────────────────────────────
        status_md = gr.HTML(value="")
        loading_html = gr.HTML(value="")

        # ── 3D Viewer — full-width hero ───────────────────────────────────────
        viewer_section = gr.HTML(value="", visible=False)

        # ── Main results row ──────────────────────────────────────────────────
        results = gr.Row(visible=False, elem_classes=["lw-results-row"])
        with results:
            with gr.Column(scale=1):
                gr.HTML("<p class='lw-section-label'>Download</p>")
                download_files = gr.File(
                    label="Your 3D file  (.ply — drag into SuperSplat or any 3D viewer)",
                    file_count="multiple",
                    interactive=False,
                    visible=False,
                )
                supersplat_html = gr.HTML()
            with gr.Column(scale=1):
                gr.HTML("<p class='lw-section-label'>Depth views</p>")
                right_title = gr.Markdown("")
                preview = gr.Image(
                    label="Top, front & side",
                    type="pil", interactive=False,
                )
                orig_out = gr.Image(
                    label="Your original photo",
                    type="pil", interactive=False, height=160,
                )

        # ── AI Insights accordion (only shown in With AI mode) ────────────────
        with gr.Accordion(
            "&#x2728; AI Insights — what the AI noticed about your scene",
            open=True, visible=False,
        ) as ai_accordion:
            analysis_md = gr.Markdown("", visible=True)
            tour_html = gr.HTML()
            with gr.Accordion("Technical data (JSON)", open=False):
                json_code = gr.Code(
                    label="", language="json", lines=12, interactive=False,
                )

        # ── Quality report ────────────────────────────────────────────────────
        qa_html = gr.HTML(value="")

        # ── System / about ────────────────────────────────────────────────────
        with gr.Accordion("About & system info", open=False, elem_classes=["lw-info-footer"]):
            gr.HTML(f"<p>{_device_info_html()}</p>")
            gr.Markdown(
                f"[Apple ml-sharp]({SHARP_REPO}) &middot; "
                f"[SuperSplat]({SUPERSPLAT_URL}) &middot; "
                "[Anthropic Claude](https://docs.anthropic.com/)"
            )
            # Power-user model selector — defined here so it renders once
            model_in = gr.Dropdown(
                label="AI model (optional — leave as-is for best results)",
                choices=[(a, b) for a, b in CLAUDE_MODEL_CHOICES],
                value=model_default,
            )

        # ── Wire up events ────────────────────────────────────────────────────
        outs = [
            status_md, loading_html,
            viewer_section, results,
            orig_out, analysis_md, json_code,
            right_title, preview,
            download_files, supersplat_html, tour_html, qa_html,
            ai_accordion,
        ]

        # Show / hide the API key row when mode changes
        mode_radio.change(
            fn=lambda m: gr.update(visible=(m == "pro")),
            inputs=[mode_radio],
            outputs=[ai_key_row],
        )

        def _run(im, key, mdl, mode, prog: gr.Progress = gr.Progress()):
            return generate_pipeline(im, key, mdl, mode, prog)

        create_btn.click(
            fn=_run,
            inputs=[in_image, api_key_in, model_in, mode_radio],
            outputs=outs,
            show_progress="minimal",
        )
        ex_btn.click(fn=lambda: sample.copy(), inputs=[], outputs=[in_image])

    demo._launch_kw = {"theme": theme, "css": _APP_CSS}
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        # WORK_ROOT must be in allowed_paths so Gradio's /gradio_api/file=
        # endpoint serves viewer.html and the .ply to the embedded iframe.
        allowed_paths=[str(WORK_ROOT)],
        **getattr(demo, "_launch_kw", {}),
    )
