# Taiwan License Plate Recognition

A real-time computer-vision pipeline that detects Taiwanese license plates with
a custom YOLO model and recognizes the plate text with a vision-language model.
It supports video files, webcams, and RTSP streams, with optional annotated
video output.

## Highlights

- Custom YOLO license-plate detection
- Taiwanese plate-text recognition with InternVL3
- Automatic CUDA acceleration with a CPU fallback
- Configurable confidence, frame stride, source, and output path
- Lightweight detection-only mode for faster testing
- No plate images, videos, or model weights stored in the repository

## Pipeline

```text
Video / webcam / RTSP stream
          │
          ▼
  YOLO plate detector
          │
          ▼
    Crop each plate
          │
          ▼
 InternVL3 text recognition
          │
          ▼
Annotated preview or MP4 output
```

## Project structure

```text
.
├── main.py             # Detection + plate-text recognition
├── detect.py           # Faster YOLO-only detection baseline
├── models/
│   └── README.md       # Private/local model setup
├── requirements.txt
└── .gitignore
```

## Setup

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Follow [`models/README.md`](models/README.md) to place the custom detector at
   `models/license_plate_yolo.pt`.

The OCR model is downloaded from Hugging Face on the first recognition run and
requires several gigabytes of memory. A CUDA-capable GPU is strongly
recommended.

## Usage

Run detection and recognition on a video:

```bash
python main.py --source sample.mp4
```

Use a webcam:

```bash
python main.py --source 0
```

Save annotated output and reduce OCR workload by processing every fifth frame:

```bash
python main.py --source sample.mp4 --frame-stride 5 --output outputs/result.mp4
```

Run only the lightweight YOLO detector:

```bash
python detect.py --source sample.mp4
```

Press `Q` or `Esc` to stop a preview.

## Privacy and responsible use

License plates can identify or track real people. Use this project only on
footage you are authorized to process. Avoid publishing readable plates,
location information, or personal data in datasets and demonstrations. Blur or
replace plate numbers before sharing screenshots or videos.

## Model and data note

Model weights, training images, downloaded processor files, and test videos are
excluded from Git. Before distributing any model or dataset, verify ownership,
privacy requirements, and the relevant licenses.

## Built with

Python · YOLO · OpenCV · PyTorch · Transformers · InternVL3

