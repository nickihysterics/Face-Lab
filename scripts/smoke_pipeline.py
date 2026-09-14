"""Exercise preparation, one CPU training epoch and export on synthetic patterns."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ml.train import train
from scripts.create_demo_dataset import create_dataset


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="face-lab-smoke-") as temporary:
        root = Path(temporary)
        base = root / "patterns"
        create_dataset(base, count=4)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_dataset.py"),
                "--base",
                str(base),
                "--out",
                str(root / "prepared"),
            ],
            check=True,
        )
        out, _ = train(
            base,
            root / "prepared/splits.csv",
            out_dir=root / "training",
            epochs=1,
            batch_size=2,
            pretrained=False,
            device="cpu",
            export_dir=root / "export",
        )
        summary = json.loads((out / "summary.json").read_text())
        assert summary["test"]["samples"] == 2
        assert summary["pretrained"] is False
        classes = json.loads((root / "export/classes.json").read_text())
        assert set(classes.values()) == {"pattern_a", "pattern_b"}
        state = torch.load(root / "export/arcface_model.pth", weights_only=True)
        assert state["arc.weight"].shape == (2, 512)
    print("PASS: synthetic preparation → training → validation → test → export. No accuracy claim.")


if __name__ == "__main__":
    main()
