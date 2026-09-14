FROM python:3.12.13-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && groupadd --gid 1000 lab && useradd --uid 1000 --gid lab --create-home lab
COPY app ./app
COPY ml ./ml
COPY scripts ./scripts
COPY web ./web
COPY examples ./examples
RUN mkdir -p models reports datasets && chown -R lab:lab /app
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ml
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-ml.txt
USER lab

FROM base AS web
USER lab
