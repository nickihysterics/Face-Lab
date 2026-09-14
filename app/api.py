"""Local inference API with a lightweight, clearly labelled demonstration."""

import base64
import io
import logging
import threading
from importlib.util import find_spec
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import BASE_DIR, MAX_UPLOAD_BYTES, MODELS_DIR, VERSION
from app.demo import demo_result
from app.images import InvalidImage, decode_image
from app.model_status import check_models

logger = logging.getLogger(__name__)


class UploadLimitMiddleware:
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise StarletteHTTPException(413, "Запрос превышает допустимый размер.")
            return message

        await self.app(scope, limited_receive, send)


def run_inference(image, conf, unknown_threshold, models_dir):
    # Import only when needed: the application and demo start without PyTorch.
    from ml.inference import annotate_image

    return annotate_image(image, conf, unknown_threshold, base_dir=models_dir)


def create_app(models_dir: Path = MODELS_DIR, max_upload_bytes: int = MAX_UPLOAD_BYTES):
    application = FastAPI(
        title="Face Lab API",
        version=VERSION,
        description="Учебный локальный сервис. /api/demo/ возвращает заданный пример, не результат нейросети.",
    )
    application.state.models_dir = Path(models_dir)
    application.state.inference_lock = threading.Lock()
    application.add_middleware(UploadLimitMiddleware, limit=max_upload_bytes + 65536)
    application.mount("/static", StaticFiles(directory=BASE_DIR / "web/static"), name="static")
    application.mount("/examples", StaticFiles(directory=BASE_DIR / "examples"), name="examples")

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request, error):
        return JSONResponse(
            {"error": error.detail}, status_code=error.status_code, headers=error.headers
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Avoid echoing uploaded bytes or non-finite JSON values from invalid input.
        return JSONResponse(
            {"error": "Проверьте файл и пороги: допустимы конечные числа от 0 до 1."},
            status_code=422,
        )

    @application.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(BASE_DIR / "web/index.html")

    @application.get("/health")
    def health():
        return {"status": "ok", "version": VERSION}

    @application.get("/api/status/")
    def status():
        missing = check_models(application.state.models_dir)
        dependencies = all(
            find_spec(name) is not None for name in ("torch", "torchvision", "ultralytics")
        )
        return {
            "version": VERSION,
            "demo_available": True,
            "missing_models": missing,
            "ml_installed": dependencies,
            "model_files_present": not missing,
            "inference_configured": not missing and dependencies,
            "note": "Проверяется наличие файлов и пакетов; совместимость весов проверяется при загрузке.",
        }

    @application.get("/api/demo/")
    def demo():
        return demo_result()

    def predict(file, conf, unknown_threshold, include_image):
        content = file.file.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            raise HTTPException(413, "Файл превышает допустимый размер.")
        try:
            image = decode_image(content, max_bytes=max_upload_bytes)
        except InvalidImage as error:
            raise HTTPException(400, str(error)) from error
        missing = check_models(application.state.models_dir)
        if missing:
            raise HTTPException(
                503, "Добавьте файлы моделей: " + ", ".join(missing) + ". Демо доступно без весов."
            )
        if not application.state.inference_lock.acquire(blocking=False):
            raise HTTPException(
                429, "Модель обрабатывает другое изображение. Повторите запрос позже."
            )
        try:
            annotated, predictions, detections = run_inference(
                image, conf, unknown_threshold, application.state.models_dir
            )
        except ImportError as error:
            raise HTTPException(503, "Установите зависимости из requirements-ml.txt.") from error
        except Exception as error:
            logger.exception("Inference failed")
            raise HTTPException(
                500,
                "Не удалось выполнить инференс. Проверьте совместимость весов и журнал сервера.",
            ) from error
        finally:
            application.state.inference_lock.release()
        result = {"mode": "inference", "predictions": predictions, "detections": detections}
        if include_image:
            buffer = io.BytesIO()
            annotated.save(buffer, format="JPEG", quality=90)
            result.update(
                image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                image_format="jpeg",
            )
        return result

    # Sync endpoints are executed by FastAPI in a worker thread, not on the event loop.
    @application.post("/api/predict/")
    def api_predict(
        file: UploadFile = File(...),
        conf: float = Form(0.25, ge=0, le=1, allow_inf_nan=False),
        unknown_threshold: float = Form(0.5, ge=0, le=1, allow_inf_nan=False),
    ):
        return predict(file, conf, unknown_threshold, True)

    @application.post("/api/predict_json/")
    def api_predict_json(
        file: UploadFile = File(...),
        conf: float = Form(0.25, ge=0, le=1, allow_inf_nan=False),
        unknown_threshold: float = Form(0.5, ge=0, le=1, allow_inf_nan=False),
    ):
        return predict(file, conf, unknown_threshold, False)

    return application


app = create_app()
