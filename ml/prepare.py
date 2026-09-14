"""Deterministic class-stratified splits, image validation and exact duplicate removal."""

import csv
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_from_class_dirs(root: Path, base: Path | None = None):
    root = Path(root).resolve()
    base = Path(base or root).resolve()
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist")
    rows = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or directory.name.startswith("."):
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTS:
                resolved = path.resolve()
                if not resolved.is_relative_to(base):
                    raise ValueError("Image path escapes dataset directory")
                rows.append((resolved.relative_to(base).as_posix(), directory.name))
    return rows


def collect_from_csv(csv_path: Path, base: Path):
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"path", "label"} <= set(reader.fieldnames or []):
            raise ValueError("CSV must contain path,label columns")
        for row in reader:
            path, label = row["path"], row["label"].strip()
            if not label or not path:
                raise ValueError("Empty label or image path")
            resolved = (base / path).resolve()
            if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
                raise ValueError("Image is missing or outside dataset directory")
            rows.append((resolved.relative_to(base.resolve()).as_posix(), label))
    return rows


def deduplicate_rows(rows, base: Path):
    seen, unique, removed = {}, [], 0
    for path, label in sorted(rows):
        with Image.open(base / path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in seen:
            if seen[digest] != label:
                raise ValueError("Identical image has conflicting class labels")
            removed += 1
            continue
        seen[digest] = label
        unique.append((path, label))
    return unique, removed


def split_counts(n, train_ratio, val_ratio):
    if (
        any(not math.isfinite(x) or x <= 0 or x >= 1 for x in (train_ratio, val_ratio))
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError("All three split ratios must be positive and sum to one")
    if n < 3:
        return n, 0, 0, True
    train_count = max(1, min(n - 2, int(n * train_ratio)))
    val_count = max(1, min(n - train_count - 1, int(n * val_ratio)))
    return train_count, val_count, n - train_count - val_count, False


def stratified_split(rows, seed=42, train_ratio=0.8, val_ratio=0.1):
    split_counts(0, train_ratio, val_ratio)
    grouped = defaultdict(list)
    paths = set()
    for path, label in sorted(rows):
        if path in paths:
            raise ValueError("Duplicate image path")
        paths.add(path)
        grouped[label].append((path, label))
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    small = []
    for label, group in sorted(grouped.items()):
        rng.shuffle(group)
        train_count, val_count, _, is_small = split_counts(len(group), train_ratio, val_ratio)
        if is_small:
            small.append((label, len(group)))
        splits["train"].extend(group[:train_count])
        splits["val"].extend(group[train_count : train_count + val_count])
        splits["test"].extend(group[train_count + val_count :])
    return splits, small


def write_splits(path: Path, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["path", "label", "split"])
        for split, rows in splits.items():
            writer.writerows((path, label, split) for path, label in rows)


def load_splits(csv_path: Path):
    splits = {"train": [], "val": [], "test": []}
    paths = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"path", "label", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Split CSV must contain path,label,split")
        for row in reader:
            path, label, split = row["path"], row["label"], row["split"]
            if split not in splits or not label.strip() or not path or path in paths:
                raise ValueError("Invalid split, empty label or repeated path across splits")
            paths.add(path)
            splits[split].append((path, label))
    train_labels = {label for _, label in splits["train"]}
    if len(train_labels) < 2 or not splits["val"]:
        raise ValueError(
            "At least two training classes and a non-empty validation split are required"
        )
    if any(label not in train_labels for split in ("val", "test") for _, label in splits[split]):
        raise ValueError("Validation and test labels must be present in training data")
    return splits
