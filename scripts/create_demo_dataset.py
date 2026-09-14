"""Create synthetic patterns for pipeline smoke tests, never an accuracy benchmark."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def create_dataset(destination: Path, count: int = 6):
    if count < 3:
        raise ValueError("At least three images per class are required")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty")
    for class_index, label in enumerate(("pattern_a", "pattern_b")):
        directory = destination / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Image.new(
                "RGB", (112, 112), (220, 235, 220) if class_index == 0 else (232, 217, 192)
            )
            draw = ImageDraw.Draw(image)
            box = (15 + index, 15, 85, 85 - index)
            if class_index == 0:
                draw.ellipse(box, fill=(45, 85 + index, 65))
            else:
                draw.rectangle(box, fill=(120 + index, 80, 45))
            image.save(directory / f"{index:03}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    try:
        create_dataset(args.out, args.count)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Synthetic pipeline fixtures: {args.out}. Not face-recognition training data.")


if __name__ == "__main__":
    main()
