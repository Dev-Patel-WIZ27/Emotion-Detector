"""
app.py  -  Flask web server
Routes:
  GET /               -> dashboard HTML page
  GET /video_feed     -> MJPEG stream (annotated webcam frames)
  GET /emotion_stream -> Server-Sent Events with JSON emotion data
  GET /snapshot       -> saves + returns a JPEG snapshot
"""

import json
import time
import threading
import io
import os

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_file

from emotion_engine import EmotionEngine, EMOTIONS, EMOTION_COLORS

# ── Config ────────────────────────────────────────────────────────────────────
CAM_INDEX    = 0
INTERVAL     = 0.1   # Super fast updates (10 times per second)
FRAME_W      = 960
FRAME_H      = 540
SNAPSHOT_DIR = "snapshots"

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_lock         = threading.Lock()
_latest_frame = None
_engine       = EmotionEngine(analyze_every=INTERVAL)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ── Camera thread ─────────────────────────────────────────────────────────────
def _camera_thread():
    global _latest_frame
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[CAMERA] Standard open failed, trying DSHOW...")
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        print("[CAMERA] CRITICAL: Could not open any webcam!")
        return

    print("[CAMERA] Successfully opened webcam.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.05)
            continue
        _engine.submit_frame(frame)
        with _lock:
            _latest_frame = frame.copy()


# ── Annotate helper ───────────────────────────────────────────────────────────
_ui_smoothed_boxes = {}

def _annotate(frame: np.ndarray, faces: list, fps: float) -> np.ndarray:
    global _ui_smoothed_boxes
    out = frame.copy()
    
    # Clean up stale trackers
    if len(faces) < len(_ui_smoothed_boxes):
        _ui_smoothed_boxes = {k: v for k, v in _ui_smoothed_boxes.items() if k < len(faces)}
    
    for i, face in enumerate(faces):
        if face.get("error"):
            continue
        r  = face.get("region", {})
        x  = r.get("x", 0);  y  = r.get("y", 0)
        fw = r.get("w", 0);  fh = r.get("h", 0)
        if fw < 10 or fh < 10:
            continue
            
        # 30fps Smooth UI Interpolation
        target_box = np.array([x, y, fw, fh], dtype=np.float32)
        if i not in _ui_smoothed_boxes:
            _ui_smoothed_boxes[i] = target_box
        else:
            # 0.15 alpha means it glides smoothly across the high framerate video
            _ui_smoothed_boxes[i] = 0.15 * target_box + 0.85 * _ui_smoothed_boxes[i]
            
        sx, sy, sfw, sfh = _ui_smoothed_boxes[i].astype(int)

            
        dom = face.get("dominant", "")
        conf = face.get("scores", {}).get(dom, 0)
        
        # Solid Blue Bounding Box
        cv2.rectangle(out, (sx, sy), (sx + sfw, sy + sfh), (255, 0, 0), 2)
        
        # Bright Green Text above the box
        label = f"{dom} ({conf:.1f}%)"
        cv2.putText(out, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
    return out


# ── Generators ────────────────────────────────────────────────────────────────
def _gen_video():
    while True:
        with _lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        faces = _engine.get_result()
        annotated = _annotate(frame, faces, _engine.fps_display)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


def _gen_emotions():
    while True:
        faces = _engine.get_result()
        payload = [
            {"dominant": f.get("dominant", ""), "scores": f.get("scores", {})}
            for f in faces if not f.get("error")
        ]
        yield f"data: {json.dumps(payload)}\n\n"
        time.sleep(0.1)  # Faster push to the client


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_gen_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/emotion_stream")
def emotion_stream():
    return Response(_gen_emotions(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/snapshot")
def snapshot():
    with _lock:
        frame = _latest_frame
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    faces = _engine.get_result()
    annotated = _annotate(frame, faces, _engine.fps_display)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    fp  = os.path.join(SNAPSHOT_DIR, f"snap_{ts}.jpg")
    cv2.imwrite(fp, annotated)
    _, buf = cv2.imencode(".jpg", annotated)
    return send_file(io.BytesIO(buf.tobytes()),
                     mimetype="image/jpeg",
                     download_name=f"snap_{ts}.jpg",
                     as_attachment=True)


# ── Boot ──────────────────────────────────────────────────────────────────────
def _start_background():
    _engine.start()
    threading.Thread(target=_camera_thread, daemon=True).start()


if __name__ == "__main__":
    _start_background()
    print("\n" + "=" * 55)
    print("  [EMOTION DETECTOR]  Web Dashboard")
    print("=" * 55)
    print("  Open in browser ->  http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
