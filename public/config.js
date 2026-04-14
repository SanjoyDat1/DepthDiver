/**
 * DepthDiver — Deployment Configuration
 *
 * Before deploying to Netlify, set BACKEND_URL to the URL of your deployed
 * backend (see backend/ folder — deploy to Render, Railway, or Fly.io).
 *
 * On Netlify, set this as an environment variable: BACKEND_URL
 * OR edit this file directly before deploying.
 */
window.__DEPTHDIVER__ = {
  // Your Render/Railway backend URL — e.g. "https://depthdiver-api.onrender.com"
  // Leave as-is for local development (assumes backend on localhost:8000)
  backendUrl: typeof process !== "undefined" && process.env
    ? (process.env.BACKEND_URL || "http://localhost:8000")
    : (window.__BACKEND_URL__ || "http://localhost:8000"),
};
