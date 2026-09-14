"""Train with ArcFace margin; validate and test with label-free inference logits."""

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import FaceDataset, build_label_map
from .models import FaceClassifier
from .prepare import load_splits
from .utils import StepLogger, seed_everything


def accuracy(logits, target):
    return int((logits.argmax(dim=1) == target).sum()), target.numel()


def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model.classify(images)  # Never pass the answer to the predictor.
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            hits, count = accuracy(logits, labels)
            correct += hits
            total += count
    return (
        {"loss": loss_sum / total, "accuracy": correct / total, "samples": total} if total else None
    )


def train(
    base: Path,
    splits_csv: Path,
    out_dir: Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
    device: str = "auto",
    export_dir: Path | None = None,
    workers: int = 0,
):
    if (
        epochs < 1
        or batch_size < 2
        or workers < 0
        or not math.isfinite(lr)
        or lr <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError(
            "Require epochs >= 1, batch_size >= 2, workers >= 0, lr > 0 and weight_decay >= 0"
        )
    splits = load_splits(splits_csv)
    if len(splits["train"]) < 2:
        raise ValueError("Training requires at least two images for BatchNorm")
    label_map = build_label_map(splits["train"])
    seed_everything(seed)
    torch_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    out_dir = Path(out_dir or Path("reports") / f"train_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(out_dir)
    history, best_acc = [], -1.0
    model_path = out_dir / "arcface_model.pth"
    try:
        with logger.step("Prepare data"):
            train_ds = FaceDataset(base, splits["train"], label_map, train=True)
            drop_last = len(train_ds) % batch_size == 1
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                drop_last=drop_last,
            )
            val_loader = DataLoader(
                FaceDataset(base, splits["val"], label_map, train=False),
                batch_size=batch_size,
                num_workers=workers,
            )
            logger.check("singleton_batch_dropped", "WARN" if drop_last else "OK", str(drop_last))
        with logger.step("Init model"):
            model = FaceClassifier(
                num_classes=len(label_map.label_to_idx), pretrained=pretrained
            ).to(torch_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1, epochs + 1):
            with logger.step(f"Epoch {epoch}"):
                model.train()
                loss_sum = seen = 0
                for images, labels in train_loader:
                    images, labels = images.to(torch_device), labels.to(torch_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(images, labels), labels)
                    if not torch.isfinite(loss):
                        raise ValueError("Training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * labels.numel()
                    seen += labels.numel()
                validation = evaluate(model, val_loader, torch_device)
                if not math.isfinite(validation["loss"]):
                    raise ValueError("Validation produced a non-finite loss")
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": loss_sum / seen,
                        "train_samples": seen,
                        "val_loss": validation["loss"],
                        "val_acc": validation["accuracy"],
                    }
                )
                logger.log(json.dumps(history[-1]))
                if validation["accuracy"] > best_acc:
                    best_acc = validation["accuracy"]
                    torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch_device, weights_only=True))
        test_loader = DataLoader(
            FaceDataset(base, splits["test"], label_map, train=False),
            batch_size=batch_size,
            num_workers=workers,
        )
        test_metrics = evaluate(model, test_loader, torch_device)
        (out_dir / "classes.json").write_text(
            json.dumps(label_map.idx_to_label, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        summary = {
            "best_val_accuracy": best_acc,
            "test": test_metrics,
            "seed": seed,
            "epochs": epochs,
            "device": str(torch_device),
            "pretrained": pretrained,
            "torch_version": torch.__version__,
            "evaluation": "closed-set classification; no claim of open-set accuracy",
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if export_dir:
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            if export_dir.resolve() != out_dir.resolve():
                for filename in ("arcface_model.pth", "classes.json", "summary.json"):
                    staged = export_dir / (filename + ".tmp")
                    shutil.copy2(out_dir / filename, staged)
                    staged.replace(export_dir / filename)
        logger.log(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    finally:
        logger.close()
    return out_dir, best_acc
