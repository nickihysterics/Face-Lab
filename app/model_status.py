"""Inspect model assets without importing or downloading the ML stack."""

from pathlib import Path

from app.config import MODELS_DIR

MODEL_FILES = ("yolov8n-face.pt", "arcface_model.pth", "classes.json")


def model_paths(base_dir: Path | None = None):
    base = Path(base_dir or MODELS_DIR).resolve()
    return tuple(base / name for name in MODEL_FILES)


def check_models(base_dir: Path | None = None) -> list[str]:
    return [
        path.name
        for path in model_paths(base_dir)
        if not path.is_file() or path.stat().st_size == 0
    ]
