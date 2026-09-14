import json
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from ml.models import ArcMarginProduct, FaceClassifier
from ml.train import evaluate

pytestmark = pytest.mark.ml


def test_arcface_backward_and_inference():
    torch.set_num_threads(2)
    head = ArcMarginProduct(8, 3)
    features = torch.randn(4, 8, requires_grad=True)
    logits = head(features, torch.tensor([0, 1, 2, 0]))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert torch.isfinite(logits).all() and torch.isfinite(features.grad).all()
    model = FaceClassifier(2, pretrained=False).eval()
    with torch.inference_mode():
        output = model.classify(torch.rand(2, 3, 112, 112))
    assert output.shape == (2, 2) and torch.isfinite(output).all()


def test_evaluation_does_not_request_labels():
    class Predictor:
        def eval(self):
            pass

        def classify(self, images):
            return images

        def __call__(self, *args):
            raise AssertionError("Validation must not call training forward")

    result = evaluate(
        Predictor(), [(torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1]))], "cpu"
    )
    assert result["accuracy"] == 1.0 and result["samples"] == 2


def test_inference_preserves_source_and_crops(monkeypatch):
    import ml.inference as inference

    source = Image.new("RGB", (80, 80), "white")
    expected = source.tobytes()
    boxes = SimpleNamespace(
        xyxy=torch.tensor([[5, 5, 50, 50], [10, 10, 60, 60], [90, 90, 100, 100]]),
        conf=torch.tensor([0.9, 0.8, 0.7]),
    )

    class Predictor:
        def classify(self, crop):
            assert torch.all(crop == 1), "Earlier annotations must not contaminate later crops"
            return torch.tensor([[2.0, 0.0]])

    monkeypatch.setattr(
        inference,
        "load_models",
        lambda base: (
            lambda *a, **k: [SimpleNamespace(boxes=boxes)],
            Predictor(),
            {0: "a", 1: "b"},
            "cpu",
        ),
    )
    annotated, predictions, detections = inference.annotate_image(source, 0.25, 0.95)
    assert source.tobytes() == expected and annotated.tobytes() != expected
    assert len(detections) == 2 and all(d["label"] == "unknown" for d in detections)


def test_model_cache_is_scoped_to_paths(tmp_path, monkeypatch):
    import ml.inference as inference

    for folder in ("first", "second"):
        root = tmp_path / folder
        root.mkdir()
        (root / "classes.json").write_text(json.dumps({"0": folder, "1": "other"}))
        (root / "arcface_model.pth").write_bytes(b"fixture")
        (root / "yolov8n-face.pt").write_bytes(b"fixture")

    class Model:
        def __init__(self, **kwargs):
            pass

        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(inference, "FaceClassifier", Model)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})
    import sys

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: path))
    monkeypatch.setattr(inference, "_CACHE", {"key": None, "value": None})
    first = inference.load_models(tmp_path / "first")
    assert inference.load_models(tmp_path / "first") is first
    second = inference.load_models(tmp_path / "second")
    assert second[2][0] == "second" and second is not first
