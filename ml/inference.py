"""Face detection and closed-set classification; scores are not calibrated probabilities."""

import json
import math
import os
import threading
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from app.model_status import check_models, model_paths

from .dataset import default_transforms
from .models import FaceClassifier

TRANSFORM = default_transforms(train=False)
_LOCK = threading.RLock()
_CACHE = {"key": None, "value": None}


def select_device():
    selected = os.environ.get("FACE_LAB_DEVICE", "auto")
    if selected != "auto":
        return torch.device(selected)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(base_dir: Path | None = None):
    paths = model_paths(base_dir)
    missing = check_models(base_dir)
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    device = select_device()
    key = (str(device), tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths))
    with _LOCK:
        if _CACHE["key"] == key:
            return _CACHE["value"]
        raw_classes = json.loads(paths[2].read_text(encoding="utf-8"))
        if not isinstance(raw_classes, dict):
            raise ValueError("classes.json must be an object")
        classes = {int(k): v for k, v in raw_classes.items()}
        if len(classes) < 2 or set(classes) != set(range(len(classes))):
            raise ValueError(
                "At least two classes with contiguous IDs starting at zero are required"
            )
        if any(not isinstance(v, str) or not v.strip() for v in classes.values()):
            raise ValueError("Class names must be non-empty strings")
        model = FaceClassifier(num_classes=len(classes), pretrained=False)
        model.load_state_dict(torch.load(paths[1], map_location="cpu", weights_only=True))
        model.to(device).eval()
        # YOLO .pt files can contain executable pickle objects. Load trusted weights only.
        from ultralytics import YOLO

        detector = YOLO(str(paths[0]))
        _CACHE.update(key=key, value=(detector, model, classes, device))
        return _CACHE["value"]


def draw_box(draw: ImageDraw.ImageDraw, box, label, score):
    x1, y1, x2, y2 = box
    color = "#a8783e" if label == "unknown" else "#225943"
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text = f"{label} ({score:.2f})"
    font = ImageFont.load_default(size=14)
    try:
        draw.text((x1 + 3, max(0, y1 - 20)), text, fill=color, font=font, stroke_width=0)
    except UnicodeEncodeError:
        draw.text((x1 + 3, max(0, y1 - 20)), f"ID ({score:.2f})", fill=color, font=font)


def annotate_image(
    image: Image.Image, conf: float, unknown_threshold: float, base_dir: Path | None = None
):
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (conf, unknown_threshold)):
        raise ValueError("Thresholds must be finite numbers between 0 and 1")
    # Crop from the unmodified source; draw only on the returned copy.
    source = image.convert("RGB")
    annotated = source.copy()
    predictions, detections = [], []
    with _LOCK, torch.inference_mode():
        detector, model, classes, device = load_models(base_dir)
        results = detector(source, conf=conf, verbose=False, max_det=32)
        boxes = results[0].boxes if results and results[0].boxes is not None else None
        if boxes is None:
            return annotated, predictions, detections
        coordinates = boxes.xyxy.cpu().tolist()
        detector_scores = boxes.conf.cpu().tolist()
        for box, detector_conf in zip(coordinates, detector_scores):
            if not all(math.isfinite(value) for value in box):
                continue
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(source.width, x2), min(source.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = source.crop(
                (
                    max(0, x1 - 10),
                    max(0, y1 - 10),
                    min(source.width, x2 + 10),
                    min(source.height, y2 + 10),
                )
            )
            tensor = TRANSFORM(crop).unsqueeze(0).to(device)
            logits = model.classify(tensor)
            score, index = torch.softmax(logits, dim=1).max(dim=1)
            value = score.item()
            label = classes[index.item()] if value >= unknown_threshold else "unknown"
            if not math.isfinite(value):
                raise ValueError("Model returned a non-finite score")
            clean_box = (x1, y1, x2, y2)
            draw_box(ImageDraw.Draw(annotated), clean_box, label, value)
            predictions.append(f"{label} ({value:.2f})")
            detections.append(
                dict(zip(("x1", "y1", "x2", "y2"), clean_box))
                | {
                    "label": label,
                    "conf": round(value, 4),
                    "detector_conf": round(detector_conf, 4),
                }
            )
    return annotated, predictions, detections
