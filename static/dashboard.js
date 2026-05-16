/**
 * dashboard.js  —  Real-time emotion updates via Server-Sent Events
 */

"use strict";

// ── Emotion metadata ──────────────────────────────────────────────────────────
const EMOTIONS = ["angry", "contempt", "happy", "sad", "surprise", "neutral", "calm"];

const META = {
  angry:    { emoji: "😠", color: "#DC2626" },
  contempt: { emoji: "😒", color: "#71717A" },
  happy:    { emoji: "😊", color: "#F59E0B" },
  sad:      { emoji: "😢", color: "#3B82F6" },
  surprise: { emoji: "😲", color: "#06B6D4" },
  neutral:  { emoji: "😐", color: "#6B7280" },
  calm:     { emoji: "😌", color: "#C8C8C8" },
};

// ── State ─────────────────────────────────────────────────────────────────────
const timeline   = [];       // last 20 dominant emotions
const MAX_TL     = 20;
let   lastUpdate = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusDot    = document.getElementById("statusDot");
const statusText   = document.getElementById("statusText");
const domEmoji     = document.getElementById("domEmoji");
const domLabel     = document.getElementById("domLabel");
const domConf      = document.getElementById("domConf");
const topCenter    = document.querySelector(".topbar-center");
const bgGlow       = document.getElementById("bgGlow");
const updateBadge  = document.getElementById("updateBadge");
const frameTime    = document.getElementById("frameTime");
const tlContainer  = document.getElementById("timeline");
const statFacesEl  = document.getElementById("statFaces");
const statDomVal   = document.getElementById("statDomVal");
const statConfVal  = document.getElementById("statConfVal");
const statTimeEl   = document.getElementById("statTime");
const toastEl      = document.getElementById("toast");


// ── SSE connection ────────────────────────────────────────────────────────────
function connectSSE() {
  const sse = new EventSource("/emotion_stream");

  sse.onopen = () => {
    statusDot.classList.add("live");
    statusText.textContent = "Live";
  };

  sse.onmessage = (e) => {
    try {
      const faces = JSON.parse(e.data);
      handleData(faces);
    } catch (_) {}
  };

  sse.onerror = () => {
    statusDot.classList.remove("live");
    statusText.textContent = "Reconnecting…";
    sse.close();
    setTimeout(connectSSE, 2000);
  };
}


// ── Data handler ──────────────────────────────────────────────────────────────
function handleData(faces) {
  lastUpdate = new Date();
  statTimeEl.textContent = lastUpdate.toLocaleTimeString();
  updateBadge.textContent = "● Live";

  const facesEl = document.querySelector("#statFaces .stat-val");
  if (facesEl) facesEl.textContent = faces.length;

  if (faces.length === 0) {
    // No face: zero out bars, reset banner
    EMOTIONS.forEach(e => setBar(e, 0));
    setDominant("", 0);
    return;
  }

  const face = faces[0];   // use first face
  const scores   = face.scores   || {};
  const dominant = face.dominant || "";

  // Update bars
  EMOTIONS.forEach(e => setBar(e, scores[e] ?? 0));

  // Highlight dominant bar row
  document.querySelectorAll(".bar-row").forEach(row => {
    row.classList.toggle("dominant", row.dataset.emotion === dominant);
  });

  // Top banner
  const conf = scores[dominant] ?? 0;
  setDominant(dominant, conf);

  // Stats
  statDomVal.textContent  = dominant ? dominant.charAt(0).toUpperCase() + dominant.slice(1) : "—";
  statConfVal.textContent = dominant ? `${conf.toFixed(1)}%` : "—";

  // Timeline push
  if (dominant) {
    timeline.push({ emotion: dominant, conf });
    if (timeline.length > MAX_TL) timeline.shift();
    renderTimeline();
  }

  // Frame time counter
  frameTime.textContent = `Updated: ${lastUpdate.toLocaleTimeString()}`;
}


// ── Set an emotion bar ────────────────────────────────────────────────────────
function setBar(emotion, value) {
  const bar = document.getElementById(`bar-${emotion}`);
  const pct = document.getElementById(`pct-${emotion}`);
  if (bar) bar.style.width = `${Math.min(value, 100)}%`;
  if (pct) pct.textContent = `${value.toFixed(1)}%`;
}


// ── Update dominant emotion display ──────────────────────────────────────────
function setDominant(emotion, conf) {
  const meta  = META[emotion] || { emoji: "😐", color: "#6B7280" };
  const label = emotion ? emotion.toUpperCase() : "NO FACE";
  const confStr = emotion ? `${conf.toFixed(1)}% confidence` : "Position face in camera";

  domEmoji.textContent = meta.emoji;
  domLabel.textContent  = label;
  domConf.textContent   = confStr;
  domLabel.style.color  = meta.color;

  // Bounce emoji on change
  domEmoji.style.transform = "scale(1.4) rotate(-8deg)";
  setTimeout(() => { domEmoji.style.transform = ""; }, 350);

  // Update top banner border glow
  topCenter.style.borderColor = meta.color + "55";
  topCenter.style.boxShadow   = `0 0 24px ${meta.color}30`;

  // Shift background glow colour
  bgGlow.style.background = `radial-gradient(ellipse 60% 50% at 50% 50%,
    ${meta.color}15 0%, transparent 70%)`;
}


// ── Timeline renderer ─────────────────────────────────────────────────────────
function renderTimeline() {
  tlContainer.innerHTML = "";
  timeline.forEach((entry, i) => {
    const meta   = META[entry.emotion] || {};
    const height = Math.max(8, (entry.conf / 100) * 52);
    const bar    = document.createElement("div");
    bar.className = "tl-bar";
    bar.style.height     = `${height}px`;
    bar.style.background = meta.color || "#6B7280";
    bar.style.boxShadow  = `0 0 6px ${meta.color || "#6B7280"}80`;
    bar.setAttribute("data-tip", `${entry.emotion} ${entry.conf.toFixed(0)}%`);
    tlContainer.appendChild(bar);
  });
}


// ── Snapshot ──────────────────────────────────────────────────────────────────
async function takeSnapshot() {
  try {
    showToast("📸 Saving snapshot…");
    const res = await fetch("/snapshot");
    if (!res.ok) throw new Error("Failed");
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `emotion_snap_${Date.now()}.jpg`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("✅ Snapshot saved!");
  } catch (err) {
    showToast("❌ Snapshot failed");
  }
}


// ── Toast helper ──────────────────────────────────────────────────────────────
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toastEl.classList.remove("show"), 3000);
}


// ── Kick off ──────────────────────────────────────────────────────────────────
connectSSE();
