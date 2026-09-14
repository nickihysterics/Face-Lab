"""Download a detector only from an explicitly chosen HTTPS URL, checking SHA-256."""

import argparse
import hashlib
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import MODELS_DIR


def download(url: str, expected: str, destination: Path):
    if not url.startswith("https://") or not re.fullmatch("[a-fA-F0-9]{64}", expected):
        raise ValueError("Provide an HTTPS URL and the expected SHA-256 from a trusted source")
    if destination.exists():
        raise FileExistsError(
            "Destination already exists; move it before downloading another version"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".weights-", dir=destination.parent)
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            if not source.geturl().startswith("https://"):
                raise ValueError("Refusing a redirect to an insecure URL")
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > 200 * 1024 * 1024:
                    raise ValueError("Detector download exceeds 200 MB")
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError("SHA-256 mismatch: downloaded file was rejected")
        # Hard-link publication refuses to overwrite a concurrently created destination.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download trusted YOLO face weights with checksum verification"
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "yolov8n-face.pt")
    args = parser.parse_args()
    try:
        download(args.url, args.sha256, args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Download failed: {error}\n")
    print(f"Verified weights: {args.out}")


if __name__ == "__main__":
    main()
