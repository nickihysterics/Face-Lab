"""A deterministic illustration, explicitly separate from model inference."""

import base64

from app.config import BASE_DIR


def demo_result():
    import json

    sample = json.loads((BASE_DIR / "examples/demo-response.json").read_text(encoding="utf-8"))
    sample["image_base64"] = base64.b64encode(
        (BASE_DIR / "examples/demo-scene.svg").read_bytes()
    ).decode("ascii")
    sample["image_format"] = "svg+xml"
    return sample
