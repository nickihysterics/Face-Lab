# Проверки и выпуск

## Локально

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app ml scripts tests
python -m ruff format --check app ml scripts tests
```

Тесты проверяют API без моделей, ограничения файла и multipart-запроса, неверные пороги, освобождение блокировки после ошибки, воспроизводимость разбиения, дубликаты и выход путей за базовую папку. Загрузчик тестируется с локально подставленным ответом: проверяются checksum, отказ от перезаписи и очистка собственного временного файла.

```bash
python -m pip install -r requirements-ml.txt
python -m pytest -q
python scripts/smoke_pipeline.py
```

ML-тесты проверяют конечные градиенты ArcFace, настоящий forward ResNet-50, валидацию без целевых меток в предикторе, независимость исходных кропов от нарисованных рамок и разделение кэша по каталогам. YOLO в регрессионных тестах заменяется контролируемым двойником. Отдельный smoke-тест выполняет реальное обучение одну эпоху и экспорт на синтетических фигурах.

```bash
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
python scripts/check_browser.py
# Для обновления изображений в README:
python scripts/check_browser.py --screenshots docs/screenshots
```

На Linux для Chromium могут потребоваться системные пакеты: `python -m playwright install --with-deps chromium`. Скрипт сам запускает временный локальный сервер. Проверяются демо, JSON, ошибка при отсутствии весов, восстановление интерфейса, отсутствие горизонтального переполнения на 390 и 320 px и ошибок JavaScript. Запускайте его без реальных весов в `models/`.

## CI

- Лёгкие тесты и Ruff: Ubuntu, Windows и macOS, Python 3.12.
- ML: Ubuntu, CPU PyTorch, тесты и полный синтетический цикл.
- Браузер: Ubuntu, Chromium.
- Docker: сборка target `web`, `/health` и `/api/demo/`, непривилегированный пользователь.

Точные прямые версии находятся в `requirements*.txt`. Для изолированной проверки лёгкой установки не устанавливайте ML-зависимости: `tests/test_ml.py` должен быть пропущен. CUDA, MPS и настоящий внешний YOLO-checkpoint не входят в матрицу CI.

## Релиз

Тег вида `v*` запускает тот же набор проверок. Только после их успеха CI создаёт `Face-Lab-<версия>-source.zip` из Git-дерева и файл `SHA256SUMS`, затем публикует GitHub Release. Датасеты, веса, окружение и локальные резервные копии в архив не входят. Это исходный релиз, а не готовая обученная система.
