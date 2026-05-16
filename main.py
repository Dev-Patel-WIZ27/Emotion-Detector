"""
main.py
Entry point – opens webcam, drives the render loop, handles key presses.

Usage:
    python main.py                          # webcam (default cam 0)
    python main.py --cam 1                  # secondary camera
    python main.py --interval 0.5           # analyse every 0.5 s (faster)
    python main.py --detector retinaface    # more accurate detector
    python main.py --image photo.jpg        # analyse single image
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

from emotion_engine import EmotionEngine, DEEPFACE_OK
from ui_renderer   import (
    PANEL_W, draw_panel, draw_face_boxes,
    draw_top_banner, draw_scanning_ring,
)

SNAPSHOT_DIR = "snapshots"


# ─────────────────────────────────────────────────────────────────────────────
def run_webcam(cam_index: int, interval: float, detector: str):
    """Main webcam loop."""
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}")
        sys.exit(1)

    # Prefer HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)

    engine = EmotionEngine(analyze_every=interval, detector=detector)
    engine.start()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    t0 = time.time()
    last_analyse_ts = t0

    print("\n" + "=" * 60)
    print("  🎭  Emotion Detector  —  LIVE WEBCAM")
    print("=" * 60)
    print(f"  Camera : {cam_index}  |  Interval : {interval}s  |  Detector : {detector}")
    print("  Q = quit     S = snapshot     +/- = change interval")
    print("=" * 60 + "\n")

    if not DEEPFACE_OK:
        print("[WARNING] DeepFace not installed – showing camera only.")
        print("          Run:  pip install deepface\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Submit frame for async analysis
        engine.submit_frame(frame)

        # Fetch latest result
        faces = engine.get_result()
        now   = time.time()
        elapsed = now - last_analyse_ts
        if faces:
            last_analyse_ts = now        # reset on fresh data arrival

        # ── Build canvas ──────────────────────────────────────────────────
        h, w = frame.shape[:2]
        canvas_w = w + PANEL_W
        canvas   = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        # ── Draw HUD ──────────────────────────────────────────────────────
        draw_face_boxes(canvas, faces)
        draw_top_banner(canvas, faces)

        # Animated ring on first face
        if faces and not faces[0].get("error"):
            draw_scanning_ring(canvas, faces[0], now - t0)

        draw_panel(canvas, faces, engine.fps_display, engine.analyze_every)

        # Print emotion scores to console every analysis cycle
        _console_print(faces, now, t0)

        cv2.imshow("🎭 Emotion Detector", canvas)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts  = time.strftime("%Y%m%d_%H%M%S")
            fp  = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.jpg")
            cv2.imwrite(fp, canvas)
            print(f"[SNAPSHOT] Saved → {fp}")
        elif key == ord("+") or key == ord("="):
            engine.analyze_every = max(0.2, engine.analyze_every - 0.1)
            print(f"[INFO] Interval → {engine.analyze_every:.1f}s")
        elif key == ord("-"):
            engine.analyze_every = min(5.0, engine.analyze_every + 0.1)
            print(f"[INFO] Interval → {engine.analyze_every:.1f}s")

    engine.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Session ended.")


# ─────────────────────────────────────────────────────────────────────────────
def run_image(path: str, detector: str):
    """Analyse a single image and show result."""
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {path}")
        sys.exit(1)

    engine = EmotionEngine(analyze_every=0.0, detector=detector)
    print(f"[INFO] Analysing {path} …")
    faces = engine._analyze(frame)

    h, w  = frame.shape[:2]
    canvas_w = w + PANEL_W
    canvas   = np.zeros((h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    draw_face_boxes(canvas, faces)
    draw_top_banner(canvas, faces)
    draw_panel(canvas, faces, 0.0, 0.0)

    _console_print(faces, time.time(), time.time())

    out = path.replace(".", "_emotion.")
    cv2.imwrite(out, canvas)
    print(f"[SAVED] {out}")

    cv2.imshow("🎭 Emotion Detector — Image", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
_last_print: float = 0.0

def _console_print(faces: list, now: float, t0: float):
    """Print emotion table to console at most once per second."""
    global _last_print
    if now - _last_print < 0.95:
        return
    _last_print = now

    if not faces:
        print(f"\r[{now - t0:6.1f}s]  No face detected …", end="", flush=True)
        return

    lines = []
    for i, face in enumerate(faces[:3]):
        if face.get("error"):
            lines.append(f"  Face {i+1}: {face['error']}")
            continue
        dom    = face.get("dominant", "?")
        scores = face.get("scores", {})
        bar    = "  ".join(
            f"{e[:3].upper()}:{scores.get(e, 0):5.1f}%" for e in
            ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        )
        lines.append(f"  Face {i+1} [{dom.upper():8s}]  {bar}")

    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}]")
    for l in lines:
        print(l)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🎭 Real-time Facial Emotion Detector"
    )
    parser.add_argument("--cam",      type=int,   default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Analysis interval in seconds (default: 1.0)")
    parser.add_argument("--detector", type=str,   default="opencv",
                        choices=["opencv", "retinaface", "mtcnn", "ssd"],
                        help="DeepFace detector backend")
    parser.add_argument("--image",    type=str,   default=None,
                        help="Path to an image file (skips webcam)")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.detector)
    else:
        run_webcam(args.cam, args.interval, args.detector)


if __name__ == "__main__":
    main()
