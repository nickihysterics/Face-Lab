import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import create_app
from app.model_status import MODEL_FILES


def image_bytes(format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "#557766").save(stream, format=format)
    return stream.getvalue()


@pytest.fixture
def api(tmp_path):
    return create_app(tmp_path)


@pytest.fixture
def client(api):
    with TestClient(api) as client:
        yield client


def test_demo_without_models(client, tmp_path):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    status = client.get("/api/status/").json()
    assert status["missing_models"] == list(MODEL_FILES)
    response = client.get("/api/demo/")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "demo" and len(data["detections"]) == 2
    assert "Нейросеть не запускалась" in data["notice"]
    assert b"<svg" in base64.b64decode(data["image_base64"])
    assert response.headers["cache-control"] == "no-store"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("path", ["/api/predict/", "/api/predict_json/"])
def test_missing_models(client, path):
    response = client.post(path, files={"file": ("face.png", image_bytes(), "image/png")})
    assert response.status_code == 503
    assert "yolov8n-face.pt" in response.json()["error"]


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", "hello"])
def test_thresholds(client, value):
    response = client.post(
        "/api/predict/", files={"file": ("x.png", image_bytes())}, data={"conf": value}
    )
    assert response.status_code == 422
    assert isinstance(response.json()["error"], str)


@pytest.mark.parametrize("content", [b"", b"not an image", image_bytes("GIF")])
def test_bad_upload(client, content):
    response = client.post("/api/predict/", files={"file": ("wrong.jpg", content, "image/jpeg")})
    assert response.status_code == 400


def test_request_limit(tmp_path):
    with TestClient(create_app(tmp_path, max_upload_bytes=32)) as client:
        response = client.post("/api/predict/", files={"file": ("x.png", image_bytes())})
        assert response.status_code == 413
        response = client.post("/api/predict/", files={"file": ("x.png", b"x" * 100000)})
        assert response.status_code == 413


def test_prediction_and_lock_release(api, monkeypatch):
    for name in MODEL_FILES:
        (api.state.models_dir / name).write_bytes(b"fixture")
    calls = []

    def predictor(image, conf, unknown_threshold, models_dir):
        calls.append((image.size, conf, unknown_threshold, models_dir))
        return image.copy(), [], []

    monkeypatch.setattr("app.api.run_inference", predictor)
    with TestClient(api) as client:
        for route in ("predict", "predict_json"):
            response = client.post(f"/api/{route}/", files={"file": ("x.png", image_bytes())})
            assert response.status_code == 200
            assert response.json()["mode"] == "inference"
            assert ("image_base64" in response.json()) == (route == "predict")
        assert calls[0][0] == (64, 48)
        api.state.inference_lock.acquire()
        assert (
            client.post("/api/predict/", files={"file": ("x.png", image_bytes())}).status_code
            == 429
        )
        api.state.inference_lock.release()

        def broken(*args):
            raise ValueError("private model details")

        monkeypatch.setattr("app.api.run_inference", broken)
        response = client.post("/api/predict/", files={"file": ("x.png", image_bytes())})
        assert response.status_code == 500 and "private model details" not in response.text
        assert api.state.inference_lock.acquire(blocking=False)
        api.state.inference_lock.release()
