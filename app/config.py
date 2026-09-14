"""Project-relative configuration; environment variables may use absolute paths."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def configured_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default)).expanduser()
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


DATASET_DIR = configured_path("DATASET_DIR", "datasets")
MODELS_DIR = configured_path("MODELS_DIR", "models")
REPORTS_DIR = configured_path("REPORTS_DIR", "reports")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
VERSION = "1.0.0"
