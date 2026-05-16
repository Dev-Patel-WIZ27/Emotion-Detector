"""
emotion_engine.py
─────────────────────────────────────────────────────────────────────────────
100% TensorFlow-free emotion engine.

Face detection  : OpenCV Haar Cascade (built-in, zero download)
Emotion model   : FER+ ONNX model (auto-downloaded ~35 MB on first run)
                  github.com/onnx/models  →  emotion_ferplus

Works on Python 3.14, no CUDA needed.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import threading
import time
import queue
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_EMOTIONS = ["angry", "contempt", "happy", "sad", "surprise", "neutral"]
DERIVED_EMOTIONS = ["calm"]
EMOTIONS = BASE_EMOTIONS + DERIVED_EMOTIONS

EMOTION_COLORS = {
    "angry":    (0,   40, 220),
    "contempt": (100, 100, 100),
    "happy":    (0,  210, 255),
    "sad":      (220, 100,  0),
    "surprise": (0,  200, 200),
    "neutral":  (170, 170, 170),
    "calm":     (200, 200, 200),
}

EMOTION_EMOJI = {
    "angry":    "😠",
    "contempt": "😒",
    "happy":    "😊",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
    "calm":     "😌",
}

# FER+ model output order → our label mapping
_FERPLUS_MAP = [
    "neutral",   # 0
    "happy",     # 1
    "surprise",  # 2
    "sad",       # 3
    "angry",     # 4
    "disgust",   # 5
    "fear",      # 6
    "contempt",  # 7
]

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_ferplus.onnx")
MODEL_URL  = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)


# ── Model download ────────────────────────────────────────────────────────────
FALLBACK_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/"
    "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

def _download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(MODEL_PATH):
        return True
    print("[EMOTION ENGINE] Downloading emotion model (~35 MB) ...")
    for url in [MODEL_URL, FALLBACK_URL]:
        try:
            print(f"[EMOTION ENGINE] Trying: {url[:60]}...")
            urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"[EMOTION ENGINE] Model saved -> {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[EMOTION ENGINE] URL failed: {e}")
            if os.path.isfile(MODEL_PATH):
                os.remove(MODEL_PATH)  # remove partial download
    print("[EMOTION ENGINE] All downloads failed.")
    return False


# ── Preprocessing helpers ─────────────────────────────────────────────────────
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize, grayscale, normalize → (1,1,64,64) float32."""
    gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    arr   = sized.astype(np.float32)
    return arr.reshape(1, 1, 64, 64)


def _merge_ferplus(scores_raw: np.ndarray) -> dict:
    """Map 8 FER+ scores → our base emotions."""
    out = {e: 0.0 for e in BASE_EMOTIONS}
    for i, val in enumerate(scores_raw):
        label = _FERPLUS_MAP[i]
        if label in out:
            out[label] += float(val) * 100.0   # percentage
    return out

def _derive_composite_emotions(scores: dict) -> dict:
    """Compute derived emotions, highly tuned for sensitivity."""
    s = scores.copy()
    
    # 1. Dampen neutral (which FER+ aggressively over-predicts) and boost negatives
    s["neutral"]  *= 0.5
    if "angry" in s: s["angry"]    *= 2.5
    if "sad" in s: s["sad"]      *= 1.8
    if "surprise" in s: s["surprise"] *= 1.5
    if "contempt" in s: s["contempt"] *= 1.8

    # 2. Derive composite emotions additively
    s["calm"]     = (s.get("neutral", 0) * 0.6) + (s.get("happy", 0) * 0.2)
    
    # 3. Re-normalize to 100%
    total = sum(s.values())
    if total > 0:
        s = {k: (v / total) * 100.0 for k, v in s.items()}
        
    return s


# ── Emotion Engine ────────────────────────────────────────────────────────────
class EmotionEngine:
    def __init__(self, analyze_every: float = 1.0, detector: str = "opencv"):
        self.analyze_every = analyze_every
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self.latest_result: list   = []
        self.last_analyzed_at      = 0.0
        self.fps_display           = 0.0
        self._lock                 = threading.Lock()
        self._running              = False
        self._session              = None
        self._face_cascade         = None
        self.detector_type         = detector
        
        # State for windowed smoothing
        self._history              = {}
        self._smoothed_boxes       = {}
        self.window_size           = 6
        self.alpha                 = 0.35
        self.frame_count           = 0

    # ── Public ────────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def submit_frame(self, frame: np.ndarray):
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        self._frame_q.put(frame.copy())

    def get_result(self) -> list:
        with self._lock:
            return list(self.latest_result)

    # ── Worker thread ─────────────────────────────────────────────────────────
    def _worker(self):
        # Load models once in worker thread
        self._face_cascade = self._load_face_cascade()
        self._session      = self._load_onnx()

        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self.frame_count += 1
            if self.frame_count % 5 != 0:
                continue

            t0     = time.time()
            result = self._analyze(frame)
            with self._lock:
                self.latest_result = result
                elapsed = time.time() - t0
                self.fps_display = round(1.0 / max(elapsed, 0.001), 1)

    # ── Model loaders ─────────────────────────────────────────────────────────
    def _load_face_cascade(self):
        if self.detector_type in ["retinaface", "mtcnn"]:
            # Use OpenCV DNN SSD Face Detector as a highly accurate backend
            proto = os.path.join(MODEL_DIR, "deploy.prototxt")
            model = os.path.join(MODEL_DIR, "res10.caffemodel")
            if os.path.isfile(proto) and os.path.isfile(model):
                net = cv2.dnn.readNetFromCaffe(proto, model)
                print("[EMOTION ENGINE] OpenCV DNN High-Accuracy Face Detector loaded.")
                return net
            else:
                print("[EMOTION ENGINE] DNN models missing. Falling back to Haar.")

        cascades = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt_tree.xml",
        ]
        loaded = []
        for name in cascades:
            path = cv2.data.haarcascades + name
            if os.path.isfile(path):
                cc = cv2.CascadeClassifier(path)
                if not cc.empty():
                    loaded.append(cc)
                    print(f"[EMOTION ENGINE] Cascade loaded: {name}")
        return loaded

    def _load_onnx(self):
        if not _download_model():
            return None
        try:
            sess = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            print("[EMOTION ENGINE] ONNX emotion model loaded OK")
            return sess
        except Exception as e:
            print(f"[EMOTION ENGINE] ONNX load error: {e}")
            return None

    # ── Analysis ──────────────────────────────────────────────────────────────
    def _detect_faces(self, frame: np.ndarray):
        """Returns list of (x, y, w, h) tuples."""
        h_img, w_img = frame.shape[:2]
        
        # If using DNN
        if not isinstance(self._face_cascade, list):
            net = self._face_cascade
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            rects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                    (startX, startY, endX, endY) = box.astype("int")
                    rects.append((startX, startY, endX - startX, endY - startY))
            return rects

        # If using Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_cl = clahe.apply(gray)

        min_face = max(30, min(h_img, w_img) // 8)
        settings = [
            (1.05, 4, (min_face, min_face)),
            (1.1,  3, (min_face, min_face)),
            (1.15, 2, (min_face // 2, min_face // 2)),
            (1.2,  1, (min_face // 2, min_face // 2)),
        ]

        for cc in self._face_cascade:
            if cc is None or (hasattr(cc, 'empty') and cc.empty()):
                continue
            for src in [gray_cl, gray]:
                for sf, mn, ms in settings:
                    try:
                        rects = cc.detectMultiScale(
                            src, scaleFactor=sf, minNeighbors=mn, minSize=ms, flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        if len(rects) > 0:
                            return rects
                    except Exception:
                        pass
        return []

    def _analyze(self, frame: np.ndarray) -> list:
        if self._face_cascade is None or self._session is None:
            return []

        h, w  = frame.shape[:2]
        faces_rect = self._detect_faces(frame)

        if len(faces_rect) == 0:
            return []

        results = []
        for i, (x, y, fw, fh) in enumerate(faces_rect):
            x1 = max(0, x);    y1 = max(0, y)
            x2 = min(w, x+fw); y2 = min(h, y+fh)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            inp    = _preprocess_face(face_roi)
            out    = self._session.run(None, {"Input3": inp})[0][0]
            probs  = _softmax(out)
            base_scores = _merge_ferplus(probs)
            all_scores = _derive_composite_emotions(base_scores)
            
            # Apply EMA + Window Smoothing
            if i not in self._history:
                self._history[i] = [all_scores]
            else:
                self._history[i].append(all_scores)
                if len(self._history[i]) > self.window_size:
                    self._history[i].pop(0)
            
            # Weighted average over the window
            smoothed = {e: 0.0 for e in EMOTIONS}
            total_weight = 0
            for w_idx, hist_scores in enumerate(self._history[i]):
                weight = self.alpha ** (len(self._history[i]) - 1 - w_idx)
                total_weight += weight
                for k in EMOTIONS:
                    smoothed[k] += hist_scores[k] * weight
            
            smoothed = {k: v / total_weight for k, v in smoothed.items()}
            dominant = max(smoothed, key=smoothed.get)
            
            # Smooth the bounding box (EMA)
            box = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
            if i not in self._smoothed_boxes:
                self._smoothed_boxes[i] = box
            else:
                self._smoothed_boxes[i] = self.alpha * box + (1 - self.alpha) * self._smoothed_boxes[i]
            
            sx, sy, sw, sh = self._smoothed_boxes[i].astype(int)
            
            results.append({
                "dominant": dominant,
                "scores":   {k: round(v, 1) for k, v in smoothed.items()},
                "region":   {"x": sx, "y": sy, "w": sw, "h": sh},
            })

        # Clear stale trackers if faces disappeared
        if len(faces_rect) < len(self._history):
            self._history = {k: v for k, v in self._history.items() if k < len(faces_rect)}
            self._smoothed_boxes = {k: v for k, v in self._smoothed_boxes.items() if k < len(faces_rect)}

        return results

