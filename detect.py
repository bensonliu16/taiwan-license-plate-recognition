"""Lightweight YOLO-only license-plate detector."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Video path or webcam index.")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("models/license_plate_yolo.pt"),
        help="Path to YOLO plate weights.",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.detector.is_file():
        raise FileNotFoundError(f"Detector weights not found: {args.detector}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(args.detector)).to(device)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            started = time.perf_counter()
            result = detector.predict(
                frame,
                conf=args.confidence,
                iou=args.iou,
                verbose=False,
                device=device,
            )[0]
            fps = 1.0 / max(time.perf_counter() - started, 1e-9)

            for box in result.boxes:
                confidence = float(box.conf[0].cpu())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"plate {confidence:.2f}",
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Plates: {len(result.boxes)} | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

