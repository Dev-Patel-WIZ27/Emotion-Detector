import cv2, numpy as np, time
from emotion_engine import EmotionEngine

e = EmotionEngine()
e._face_cascade = e._load_face_cascade()
e._session = e._load_onnx()
print("Cascades loaded:", len(e._face_cascade))
print("ONNX OK:", e._session is not None)

# Grab several frames from webcam (camera needs a few frames to warm up)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1)
frame = None
for _ in range(10):
    ret, f = cap.read()
    if ret:
        frame = f
    time.sleep(0.05)
cap.release()

if frame is not None:
    print("Frame shape:", frame.shape)
    result = e._analyze(frame)
    print("Faces detected:", len(result))
    for face in result:
        dom = face["dominant"]
        scores = face["scores"]
        print("  Dominant:", dom)
        for k, v in scores.items():
            print(f"    {k}: {v:.1f}%")
    if not result:
        cv2.imwrite("test_frame.jpg", frame)
        print("No face found - saved test_frame.jpg for inspection")
else:
    print("ERROR: Could not read webcam frame")
