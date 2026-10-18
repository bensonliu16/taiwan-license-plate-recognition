import argparse
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeTensor:
    def __init__(self, shape=(1, 4), dtype="int64"):
        self.shape = shape
        self.dtype = dtype

    def to(self, **_kwargs):
        return self

    def __getitem__(self, _key):
        return self


class InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def load_main_module():
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 1
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
    fake_cv2.CAP_PROP_FPS = 5
    fake_cv2.VideoCapture = object
    fake_cv2.VideoWriter = object
    fake_cv2.VideoWriter_fourcc = lambda *_args: 1234
    fake_cv2.cvtColor = lambda image, _mode: image

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.is_floating_point = lambda _value: False
    fake_torch.inference_mode = InferenceMode
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = types.SimpleNamespace(fromarray=lambda image: image)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = object
    fake_transformers.AutoProcessor = object

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object

    module_path = Path(__file__).parents[1] / "main.py"
    with patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "torch": fake_torch,
            "PIL": fake_pil,
            "transformers": fake_transformers,
            "ultralytics": fake_ultralytics,
        },
    ):
        spec = importlib.util.spec_from_file_location("plate_main_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeParameter:
    device = "cpu"
    dtype = "float32"


class FakeModel:
    def parameters(self):
        yield FakeParameter()

    def generate(self, **_inputs):
        return FakeTensor()


class FakeProcessor:
    def apply_chat_template(self, *_args, **_kwargs):
        return {
            "input_ids": FakeTensor(shape=(1, 4)),
            "pixel_values": FakeTensor(shape=(1, 3, 32, 32)),
        }

    def decode(self, _tokens, skip_special_tokens):
        self.skip_special_tokens = skip_special_tokens
        return "ABC-1234"


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_main_module()

    def test_webcam_source_becomes_integer(self):
        self.assertEqual(self.module.resolve_source("0"), 0)

    def test_video_path_is_preserved(self):
        self.assertEqual(self.module.resolve_source("samples/demo.mp4"), "samples/demo.mp4")

    def test_rtsp_url_is_preserved(self):
        url = "rtsp://camera.example.invalid/stream"
        self.assertEqual(self.module.resolve_source(url), url)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_main_module()

    def test_default_arguments_match_documentation(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = self.module.parse_args()
        self.assertEqual(args.source, "0")
        self.assertEqual(args.confidence, 0.70)
        self.assertEqual(args.frame_stride, 1)
        self.assertFalse(args.no_display)

    def test_custom_arguments_are_parsed(self):
        argv = [
            "main.py",
            "--source",
            "sample.mp4",
            "--confidence",
            "0.55",
            "--frame-stride",
            "5",
            "--no-display",
        ]
        with patch.object(sys, "argv", argv):
            args = self.module.parse_args()
        self.assertEqual(args.source, "sample.mp4")
        self.assertEqual(args.confidence, 0.55)
        self.assertEqual(args.frame_stride, 5)
        self.assertTrue(args.no_display)


class OcrTests(unittest.TestCase):
    def test_recognize_plate_returns_decoded_text(self):
        module = load_main_module()
        processor = FakeProcessor()
        result = module.recognize_plate(object(), FakeModel(), processor)
        self.assertEqual(result, "ABC-1234")
        self.assertTrue(processor.skip_special_tokens)


class ValidationTests(unittest.TestCase):
    def test_invalid_frame_stride_fails_before_loading_models(self):
        module = load_main_module()
        args = argparse.Namespace(
            source="0",
            detector=Path("missing.pt"),
            ocr_model="unused",
            confidence=0.7,
            frame_stride=0,
            target_fps=0,
            output=None,
            no_display=True,
        )
        with patch.object(module, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "frame-stride"):
                module.main()


if __name__ == "__main__":
    unittest.main()

