"""Detect and recognize Taiwanese license plates in video streams."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from ultralytics import YOLO


DEFAULT_DETECTOR = Path("models/license_plate_yolo.pt")
DEFAULT_OCR_MODEL = "OpenGVLab/InternVL3-2B-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect license plates with YOLO and recognize their text."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video path, RTSP URL, or webcam index (default: 0).",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=DEFAULT_DETECTOR,
        help=f"Path to YOLO plate weights (default: {DEFAULT_DETECTOR}).",
    )
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help="Hugging Face OCR/vision-language model identifier.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Minimum plate-detection confidence (default: 0.70).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run inference on every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Maximum display rate; use 0 for no limit (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for an annotated MP4 output.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the preview window (useful for remote runs).",
    )
    return parser.parse_args()


def resolve_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def load_ocr_model(model_name: str, device: str):
    processor = AutoProcessor.from_pretrained(model_name)
    load_kwargs: dict[str, object] = {"low_cpu_mem_usage": True}

    if device == "cuda":
        load_kwargs.update(device_map="cuda", torch_dtype=torch.bfloat16)
    else:
        load_kwargs.update(torch_dtype=torch.float32)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **load_kwargs,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, processor


def recognize_plate(plate_image, model, processor) -> str:
    image = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Read the Taiwanese license plate number in this image. "
                        "Return only the plate number, or UNCLEAR if it cannot be read."
                    ),
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = {
        key: value.to(
            device=model_device,
            dtype=model_dtype if torch.is_floating_point(value) else value.dtype,
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    prompt_length = inputs["input_ids"].shape[1]
    text = processor.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    return text or "UNCLEAR"


def create_writer(path: Path, capture: cv2.VideoCapture) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if not args.detector.is_file():
        raise FileNotFoundError(
            f"Detector weights not found: {args.detector}. "
            "See models/README.md for setup instructions."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    detector = YOLO(str(args.detector)).to(device)
    ocr_model, processor = load_ocr_model(args.ocr_model, device)

    capture = cv2.VideoCapture(resolve_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = create_writer(args.output, capture) if args.output else None
    frame_number = 0
    last_annotations: list[tuple[int, int, int, int, str, float]] = []

    try:
        while True:
            loop_started = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            if frame_number % args.frame_stride == 0:
                last_annotations = []
                result = detector.predict(
                    frame,
                    conf=args.confidence,
                    verbose=False,
                    device=device,
                )[0]

                for box in result.boxes:
                    confidence = float(box.conf[0].cpu())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    plate_text = recognize_plate(cropped, ocr_model, processor)
                    last_annotations.append(
                        (x1, y1, x2, y2, plate_text, confidence)
                    )

            for x1, y1, x2, y2, plate_text, confidence in last_annotations:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{plate_text} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Taiwan License Plate Recognition", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.target_fps > 0:
                remaining = (1.0 / args.target_fps) - (
                    time.perf_counter() - loop_started
                )
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

