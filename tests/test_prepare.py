import pytest
from PIL import Image

from ml.prepare import (
    collect_from_csv,
    deduplicate_rows,
    load_splits,
    split_counts,
    stratified_split,
    write_splits,
)
from scripts.create_demo_dataset import create_dataset


def test_reproducible_and_disjoint():
    rows = [(f"{label}/{n}.png", label) for label in ("a", "b") for n in range(10)]
    forward, _ = stratified_split(rows, seed=42)
    reverse, _ = stratified_split(list(reversed(rows)), seed=42)
    assert forward == reverse
    assert [len(forward[s]) for s in ("train", "val", "test")] == [16, 2, 2]
    paths = [p for group in forward.values() for p, _ in group]
    assert len(paths) == len(set(paths)) == 20


@pytest.mark.parametrize("ratios", [(0.9, 0.2), (-0.1, 0.1), (float("nan"), 0.1), (0.8, 0)])
def test_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        split_counts(5, *ratios)


def test_duplicate_images_and_labels(tmp_path):
    image = Image.new("RGB", (16, 16), "#447755")
    image.save(tmp_path / "a.png")
    image.save(tmp_path / "b.png")
    rows, removed = deduplicate_rows([("a.png", "person"), ("b.png", "person")], tmp_path)
    assert len(rows) == 1 and removed == 1
    with pytest.raises(ValueError, match="conflicting"):
        deduplicate_rows([("a.png", "one"), ("b.png", "two")], tmp_path)


def test_csv_paths_and_split_validation(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\n../outside.jpg,a\n")
    with pytest.raises(ValueError):
        collect_from_csv(labels, tmp_path)
    splits = {"train": [("a.png", "a"), ("b.png", "b")], "val": [("c.png", "a")], "test": []}
    path = tmp_path / "splits.csv"
    write_splits(path, splits)
    assert load_splits(path) == splits
    splits["val"] = [("a.png", "a")]
    write_splits(path, splits)
    with pytest.raises(ValueError, match="repeated"):
        load_splits(path)
    splits["val"] = [("c.png", "unknown")]
    write_splits(path, splits)
    with pytest.raises(ValueError, match="present in training"):
        load_splits(path)


def test_synthetic_generation_does_not_overwrite(tmp_path):
    root = tmp_path / "synthetic"
    create_dataset(root, count=3)
    assert len(list(root.rglob("*.png"))) == 6
    with pytest.raises(ValueError, match="empty"):
        create_dataset(root)
