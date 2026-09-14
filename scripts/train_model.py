"""Запуск обучения модели распознавания лиц."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import DATASET_DIR, MODELS_DIR, REPORTS_DIR
from ml.train import train


def main():
    parser = argparse.ArgumentParser(description="Train ArcFace classifier")
    parser.add_argument("--base", type=str, default=None, help="Dataset base path")
    parser.add_argument("--splits", type=str, default=None, help="Splits CSV path")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--export-dir", type=str, default=None)
    parser.add_argument("--auto-prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error("epochs >= 1, batch-size >= 2 and workers >= 0 are required")

    base = Path(args.base) if args.base else DATASET_DIR / "faces"
    base_dir = base.resolve()
    splits = Path(args.splits) if args.splits else None

    if splits is None and args.auto_prepare:
        prep_out = REPORTS_DIR / f"dataset_prepare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "prepare_dataset.py"),
            "--out",
            str(prep_out),
            "--base",
            str(base_dir),
            "--seed",
            str(args.seed),
        ]
        subprocess.run(cmd, check=True)
        splits = prep_out / "splits.csv"

    if splits is None:
        raise SystemExit("Splits CSV is required. Pass --splits or use --auto-prepare.")

    out_dir = Path(args.out) if args.out else None
    export_dir = Path(args.export_dir) if args.export_dir else MODELS_DIR

    train(
        base=base_dir,
        splits_csv=splits,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        pretrained=args.pretrained,
        device=args.device,
        export_dir=export_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
