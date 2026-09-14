"""Prepare one explicit identity dataset instead of mixing unrelated source collections."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import DATASET_DIR, REPORTS_DIR
from ml.prepare import (
    collect_from_class_dirs,
    collect_from_csv,
    deduplicate_rows,
    stratified_split,
    write_splits,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate images and create identity-stratified splits"
    )
    parser.add_argument("--base", type=Path, default=DATASET_DIR / "faces")
    parser.add_argument(
        "--labels", type=Path, help="Optional CSV with path,label relative to --base"
    )
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()
    ratios = (args.train, args.val, args.test)
    if any(not math.isfinite(r) or not 0 < r < 1 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        parser.error("train, val and test must be positive and sum to one")
    try:
        rows = (
            collect_from_csv(args.labels, args.base)
            if args.labels
            else collect_from_class_dirs(args.base)
        )
        rows, removed = deduplicate_rows(rows, args.base)
        splits, small = stratified_split(rows, args.seed, args.train, args.val)
        if len({label for _, label in rows}) < 2 or small:
            raise ValueError(
                "Provide at least two classes with at least three distinct images each"
            )
        args.out.mkdir(parents=True, exist_ok=True)
        write_splits(args.out / "splits.csv", splits)
        report = {
            "seed": args.seed,
            "classes": len({label for _, label in rows}),
            "images": len(rows),
            "duplicates_removed": removed,
            "splits": {k: len(v) for k, v in splits.items()},
        }
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Splits: {args.out / 'splits.csv'}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
