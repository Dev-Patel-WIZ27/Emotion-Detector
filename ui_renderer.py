"""
ui_renderer.py
All OpenCV drawing helpers for the emotion dashboard HUD.
"""

import cv2
import numpy as np
import time
from emotion_engine import EMOTIONS, EMOTION_COLORS, EMOTION_EMOJI

# ── palette ──────────────────────────────────────────────────────────────────
BG_PANEL   = (18,  18,  28)      # dark navy
ACCENT     = (0,  220, 255)      # cyan
WHITE      = (255, 255, 255)
GREY       = (130, 130, 130)
GREEN_GOOD = (0,  210,  90)
BAR_BG     = (50,  50,  65)

PANEL_W    = 310                 # sidebar width in pixels
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
def draw_panel(canvas: np.ndarray, faces: list, fps: float, elapsed: float):
    """Draw the full sidebar panel onto canvas (in-place)."""
    h, w = canvas.shape[:2]
    # Sidebar background
    panel = canvas[:, w - PANEL_W:].copy()
    overlay = np.full_like(panel, BG_PANEL, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.88, panel, 0.12, 0, panel)
    canvas[:, w - PANEL_W:] = panel

    px = w - PANEL_W + 12  # left x inside panel

    # ── header ───────────────────────────────────────────────────────────────
    _draw_text(canvas, "EMOTION  DETECTOR", (px, 32), FONT, 0.62, ACCENT, 2)
    _draw_text(canvas, "Real-time  AI  Analysis", (px, 54), FONT_SMALL, 0.42, GREY, 1)
    cv2.line(canvas, (w - PANEL_W, 64), (w, 64), ACCENT, 1)

    # ── stats row ────────────────────────────────────────────────────────────
    faces_detected = len([f for f in faces if not f.get("error")])
    _draw_text(canvas, f"Faces: {faces_detected}",
               (px, 86), FONT_SMALL, 0.48, WHITE, 1)
    _draw_text(canvas, f"Interval: {elapsed:.1f}s",
               (px + 110, 86), FONT_SMALL, 0.48, WHITE, 1)
    _draw_text(canvas, f"Analyse FPS: {fps}",
               (px, 104), FONT_SMALL, 0.42, GREY, 1)

    cv2.line(canvas, (w - PANEL_W, 112), (w, 112), (50, 50, 70), 1)

    # ── per-face emotion bars ─────────────────────────────────────────────────
    y_cursor = 122
    if not faces:
        _draw_text(canvas, "No face detected",
                   (px, y_cursor + 20), FONT_SMALL, 0.5, GREY, 1)
        _draw_text(canvas, "Please face the camera",
                   (px, y_cursor + 42), FONT_SMALL, 0.42, GREY, 1)
    else:
        for fi, face in enumerate(faces[:3]):          # max 3 faces shown
            if face.get("error"):
                _draw_text(canvas, face["error"][:35],
                           (px, y_cursor + 16), FONT_SMALL, 0.38, (0, 80, 200), 1)
                y_cursor += 30
                continue

            dominant = face.get("dominant", "")
            scores   = face.get("scores", {})

            # face index label
            label = f"Face {fi + 1}  —  {dominant.upper()}"
            col   = EMOTION_COLORS.get(dominant, WHITE)
            _draw_text(canvas, label, (px, y_cursor + 16), FONT_SMALL, 0.52, col, 1)
            y_cursor += 24

            # confidence bar for each emotion
            for emo in EMOTIONS:
                val  = scores.get(emo, 0.0)
                frac = val / 100.0
                bar_x = px
                bar_y = y_cursor + 4
                bar_w = PANEL_W - 24
                bar_h = 14
                emoji = EMOTION_EMOJI.get(emo, "")
                ecol  = EMOTION_COLORS.get(emo, WHITE)

                # background bar
                cv2.rectangle(canvas,
                               (bar_x, bar_y),
                               (bar_x + bar_w, bar_y + bar_h),
                               BAR_BG, -1)
                # filled portion
                fill = int(bar_w * frac)
                if fill > 0:
                    cv2.rectangle(canvas,
                                   (bar_x, bar_y),
                                   (bar_x + fill, bar_y + bar_h),
                                   ecol, -1)

                # label left
                _draw_text(canvas, f"{emo[:4]}",
                           (bar_x, bar_y - 1), FONT_SMALL, 0.34, WHITE, 1)
                # percent right
                _draw_text(canvas, f"{val:5.1f}%",
                           (bar_x + bar_w - 48, bar_y + 11),
                           FONT_SMALL, 0.36, WHITE, 1)
                y_cursor += 20

            y_cursor += 10
            cv2.line(canvas,
                     (w - PANEL_W, y_cursor),
                     (w, y_cursor), (50, 50, 70), 1)
            y_cursor += 8

    # ── bottom timestamp ──────────────────────────────────────────────────────
    ts = time.strftime("%H:%M:%S")
    _draw_text(canvas, ts, (px, h - 12), FONT_SMALL, 0.46, GREY, 1)
    _draw_text(canvas, "Press Q to quit | S to snapshot",
               (px, h - 28), FONT_SMALL, 0.38, GREY, 1)


# ─────────────────────────────────────────────────────────────────────────────
def draw_face_boxes(canvas: np.ndarray, faces: list):
    """Draw bounding boxes + dominant emotion label on main video area."""
    for face in faces:
        if face.get("error"):
            continue
        r = face.get("region", {})
        x  = r.get("x", 0);  y  = r.get("y", 0)
        fw = r.get("w", 0);  fh = r.get("h", 0)
        if fw < 10 or fh < 10:
            continue

        dominant = face.get("dominant", "")
        col = EMOTION_COLORS.get(dominant, WHITE)

        # glow box (wider border drawn first in dim colour)
        cv2.rectangle(canvas, (x - 2, y - 2), (x + fw + 2, y + fh + 2),
                       tuple(c // 3 for c in col), 2)
        cv2.rectangle(canvas, (x, y), (x + fw, y + fh), col, 2)

        # dominant emotion banner above box
        banner = f" {dominant.upper()} {EMOTION_EMOJI.get(dominant, '')} "
        (tw, th), _ = cv2.getTextSize(banner, FONT_SMALL, 0.6, 1)
        cv2.rectangle(canvas, (x, y - th - 10), (x + tw + 4, y), col, -1)
        cv2.putText(canvas, banner, (x + 2, y - 6),
                    FONT_SMALL, 0.6, (10, 10, 10), 1, cv2.LINE_AA)

        # confidence %
        conf = face.get("scores", {}).get(dominant, 0)
        _draw_text(canvas, f"{conf:.1f}%", (x + fw + 6, y + 20),
                   FONT_SMALL, 0.55, col, 1)


# ─────────────────────────────────────────────────────────────────────────────
def draw_top_banner(canvas: np.ndarray, faces: list):
    """Full-width top banner showing the dominant emotion of face #1."""
    if not faces or faces[0].get("error"):
        return
    dominant = faces[0].get("dominant", "")
    conf = faces[0].get("scores", {}).get(dominant, 0)
    col = EMOTION_COLORS.get(dominant, WHITE)
    emoji = EMOTION_EMOJI.get(dominant, "")
    text = f"{emoji}  {dominant.upper()}  |  {conf:.1f}%  confidence"

    h, w = canvas.shape[:2]
    # semi-transparent strip at the very top
    strip = canvas[:38, :w - PANEL_W].copy()
    overlay = np.zeros_like(strip)
    overlay[:] = (*col[::-1], 0)[:3]          # same colour as emotion (RGB→BGR skip)
    overlay = np.full_like(strip, (col[0]//5, col[1]//5, col[2]//5))
    cv2.addWeighted(overlay, 0.75, strip, 0.25, 0, strip)
    canvas[:38, :w - PANEL_W] = strip

    cv2.putText(canvas, text, (12, 26),
                FONT, 0.65, col, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
def draw_scanning_ring(canvas: np.ndarray, face: dict, t: float):
    """Animated ring around the first detected face (pulse effect)."""
    r = face.get("region", {})
    cx = r.get("x", 0) + r.get("w", 0) // 2
    cy = r.get("y", 0) + r.get("h", 0) // 2
    radius = max(r.get("w", 60), r.get("h", 60)) // 2 + 20
    pulse  = int(8 * abs(np.sin(t * 2)))
    dominant = face.get("dominant", "")
    col = EMOTION_COLORS.get(dominant, WHITE)
    cv2.circle(canvas, (cx, cy), radius + pulse, col, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius + pulse + 4, tuple(c // 3 for c in col), 1)


# ─────────────────────────────────────────────────────────────────────────────
def _draw_text(img, text, org, font, scale, color, thickness):
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)
