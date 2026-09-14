"""Утилиты для логирования шагов, работы с путями и воспроизводимости."""

import csv
import random
import time
from datetime import datetime
from pathlib import Path

from PIL import Image


def now_ts():
    """Возвращает текущий таймстамп в человекочитаемом формате."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class StepLogger:
    """Логгер шагов и проверок с записью в файлы log.txt и checks.csv."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.log_path = out_dir / "log.txt"
        self.checks_path = out_dir / "checks.csv"
        self._log = self.log_path.open("w", encoding="utf-8")
        self._checks = self.checks_path.open("w", encoding="utf-8", newline="")
        self._checks_writer = csv.DictWriter(
            self._checks, fieldnames=["time", "check", "status", "detail"]
        )
        self._checks_writer.writeheader()

    def log(self, msg: str):
        """Записывает строку в лог и печатает её в stdout."""
        line = f"[{now_ts()}] {msg}"
        print(line)
        self._log.write(line + "\n")
        self._log.flush()

    def check(self, name: str, status: str, detail: str = ""):
        """Фиксирует результат проверки с произвольной детализацией."""
        self._checks_writer.writerow(
            {
                "time": now_ts(),
                "check": name,
                "status": status,
                "detail": detail,
            }
        )
        self._checks.flush()

    def step(self, title: str):
        """Контекстный менеджер для измерения времени шага."""
        return _Step(self, title)

    def close(self):
        """Закрывает файловые дескрипторы логов."""
        self._log.close()
        self._checks.close()


class _Step:
    """Внутренний контекстный менеджер шага логирования."""

    def __init__(self, logger: StepLogger, title: str):
        self.logger = logger
        self.title = title
        self.start = None

    def __enter__(self):
        """Фиксирует старт шага и пишет лог."""
        self.start = time.time()
        self.logger.log(f"START: {self.title}")
        return self

    def __exit__(self, exc_type, exc, tb):
        """Логирует завершение шага и длительность."""
        dur = time.time() - self.start
        if exc:
            self.logger.log(f"END: {self.title} (ERROR after {dur:.2f}s)")
        else:
            self.logger.log(f"END: {self.title} ({dur:.2f}s)")


def seed_everything(seed: int):
    """Фиксирует случайность для воспроизводимости в Python/NumPy/PyTorch."""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: Path):
    """Создаёт директорию, если она не существует."""
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    """Загружает изображение и приводит к RGB."""
    with Image.open(path) as source:
        return source.convert("RGB")


def save_image(img: Image.Image, path: Path):
    """Сохраняет изображение, создавая родительские папки при необходимости."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
