"""Датасет и вспомогательные функции для обучения классификатора лиц."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class LabelMap:
    """Двунаправленное отображение метка ↔ индекс."""

    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]


def load_rows(csv_path: Path) -> List[Tuple[str, str]]:
    """Читает CSV со столбцами path/label и возвращает список пар."""
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["path"], r["label"]))
    return rows


def build_label_map(rows: List[Tuple[str, str]]) -> LabelMap:
    """Строит отображение классов из списка (path, label)."""
    labels = sorted({label for _, label in rows})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return LabelMap(label_to_idx, idx_to_label)


def default_transforms(train: bool = True):
    """Создает набор трансформаций для train/val режимов."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class FaceDataset(Dataset):
    """PyTorch-датасет для изображений лиц."""

    def __init__(
        self, base_dir: Path, rows: List[Tuple[str, str]], label_map: LabelMap, train: bool = True
    ):
        """Инициализирует датасет с базовой директорией и метками."""
        self.base_dir = base_dir
        self.rows = rows
        self.label_map = label_map
        # Для обучения включаем аугментации, для валидации — только нормализация.
        self.transform = default_transforms(train=train)

    def __len__(self):
        """Возвращает количество примеров."""
        return len(self.rows)

    def __getitem__(self, idx):
        """Читает изображение и возвращает (тензор, индекс класса)."""
        rel_path, label = self.rows[idx]
        img_path = (self.base_dir / rel_path).resolve()
        if not img_path.is_relative_to(self.base_dir.resolve()):
            raise ValueError("Image path must stay inside the dataset directory")
        try:
            with Image.open(img_path) as source:
                img = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {img_path}") from exc
        img = self.transform(img)
        label_idx = self.label_map.label_to_idx[label]
        return img, label_idx
