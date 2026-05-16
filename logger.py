"""
logger.py
Optional CSV logger – records per-face emotion scores to a CSV file
so you can review them later or plot them.

Usage (add to main.py if desired):
    from logger import EmotionLogger
    log = EmotionLogger("session.csv")
    log.write(faces)
"""

import csv
import os
import time


HEADER = ["timestamp", "face_index", "dominant",
          "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionLogger:
    def __init__(self, filepath: str = "emotion_log.csv"):
        self.filepath = filepath
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.isfile(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(HEADER)

    def write(self, faces: list):
        """Append one row per face to the CSV."""
        if not faces:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for i, face in enumerate(faces):
            if face.get("error"):
                continue
            scores = face.get("scores", {})
            row = [ts, i, face.get("dominant", "")]
            row += [round(scores.get(e, 0.0), 2) for e in EMOTIONS]
            rows.append(row)
        if rows:
            with open(self.filepath, "a", newline="") as f:
                csv.writer(f).writerows(rows)
