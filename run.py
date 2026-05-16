import argparse
from app import app, _start_background, _engine

def main():
    parser = argparse.ArgumentParser(description="🎭 Emotion Detector v2")
    parser.add_argument("--webcam", action="store_true", help="Launch webcam dashboard")
    parser.add_argument("--backend", type=str, default="opencv", choices=["opencv", "retinaface", "mtcnn"], help="Face detector backend")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  [EMOTION DETECTOR v2]  Web Dashboard")
    print("=" * 55)
    print(f"  Backend : {args.backend}")
    print("  Open in browser ->  http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")

    _engine.detector_type = args.backend
    _start_background()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
