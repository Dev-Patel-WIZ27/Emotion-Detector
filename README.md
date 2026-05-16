# Emotion-Detector
Detects what type of emotion you are having

A **real-time**, software-model facial emotion detection app powered by **DeepFace** + **OpenCV**.  
Detects 7 emotions with a live dashboard HUD.

---

## 📦 Project Structure

```
emotion_detector/
├── app.py              ← Flask backend & Video streaming
├── emotion_engine.py   ← Analysis engine (threaded)
├── main.py             ← CLI runner
├── run.py              ← V2 Entry point
├── requirements.txt    ← Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Navigate to project folder
cd emotion_detector

# 2. (Recommended) create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶ Live webcam 
```bash
python run.py --webcam --backend retinaface
```

### ▶ Open in Browser
Navigate to `http://localhost:5000` to see the live dashboard.

---

## 📊 Emotion Dashboard

The app shows a live **browser dashboard** with:
- **Video feed** — Minimalist high-performance UI matching OpenCV standards
- **Emotion bars** — All 7 emotions with percentage bars
- **Smooth tracking** — Exponential Moving Average smoothing on face tracking

**7 detected emotions:**
Angry 😠 | Contempt 😒 | Happy 😊 | Sad 😢 | Surprise 😲 | Neutral 😐 | Calm 😌

---

## 🛠 Troubleshooting

**"No face detected"**: Ensure good lighting; face the camera directly.  
**Slow first run**: The app will automatically download the ONNX model files to the `models/` directory on first launch.

