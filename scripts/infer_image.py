"""Инференс на одном изображении с сохранением результата."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import MODELS_DIR, REPORTS_DIR
from app.images import decode_image
from ml.inference import annotate_image, check_models
from ml.utils import StepLogger, ensure_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--unknown-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    missing = check_models(MODELS_DIR)
    if missing:
        raise SystemExit("Missing model files: " + ", ".join(missing))

    out_dir = REPORTS_DIR / f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(out_dir)
    logger = StepLogger(out_dir)

    try:
        image_path = Path(args.image)
        with logger.step("Load image"):
            img = decode_image(image_path.read_bytes())
            logger.check("input_image", "OK", str(image_path))

        with logger.step("Run inference"):
            annotated, predictions, _ = annotate_image(
                img, args.conf, args.unknown_threshold, base_dir=MODELS_DIR
            )
            logger.check("predictions", "OK", str(len(predictions)))

        out_path = Path(args.out) if args.out else out_dir / "result.jpg"
        with logger.step("Save result"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(out_path)
            logger.check("output_saved", "OK", str(out_path))

    finally:
        logger.close()

    print(f"Result saved to {out_path}")


if __name__ == "__main__":
    main()
