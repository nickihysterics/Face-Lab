"""Bounded image decoding shared by HTTP and CLI entry points."""

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES


class InvalidImage(ValueError):
    pass


def decode_image(content: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> Image.Image:
    if not content or len(content) > max_bytes:
        raise InvalidImage("Изображение должно быть непустым и не больше 8 МБ.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Поддерживаются JPEG, PNG и WebP.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("Максимальный размер — 16 мегапикселей.")
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise InvalidImage("Файл повреждён или не является поддерживаемым изображением.") from error
