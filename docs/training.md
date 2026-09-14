# Обучение и инференс

## Окружение

Создайте окружение Python 3.12 по [README](../README.md#быстрый-старт), затем установите `requirements-ml.txt`. Обучение на CPU поддерживается; оно может быть медленным на больших данных. Для Linux с CPU можно сначала установить CPU-колёса, затем остальные зависимости:

```bash
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ml.txt
```

На macOS используйте обычный `requirements-ml.txt`. Для CUDA подберите сборку по [официальной инструкции PyTorch](https://pytorch.org/get-started/locally/); GPU-конфигурация не является обязательной частью демонстрации.

## Синтетический эксперимент

Для проверки всех стадий достаточно `python scripts/smoke_pipeline.py`. Он создаёт временные фигуры, делает одну эпоху реального обучения на CPU, проверяет экспорт и удаляет временные файлы. Веса ImageNet не скачиваются.

Для пошагового разбора с сохранением артефактов:

```bash
python scripts/create_demo_dataset.py --out datasets/synthetic --count 6
python scripts/prepare_dataset.py --base datasets/synthetic --out reports/synthetic-data
python scripts/train_model.py --base datasets/synthetic --splits reports/synthetic-data/splits.csv --out reports/synthetic-train --epochs 1 --batch-size 2 --device cpu --export-dir reports/synthetic-models
```

Экспорт намеренно помещён в `reports/`, чтобы тестовые классы фигур не подменили модель распознавания лиц. Повторное создание синтетических данных требует новой или пустой папки. Пошаговая версия есть в [ноутбуке](../pipeline_notebook.ipynb).

## Подготовка своего набора

Вход классификатора — заранее подготовленные кропы лиц. Детектор при обучении классификатора не запускается. Укажите один источник: `datasets/faces/<метка>/<изображение>` либо CSV с колонками `path,label`. В CSV пути относительны к `--base` и не могут выходить за эту папку:

```csv
path,label
participant_01/001.jpg,participant_01
participant_01/002.jpg,participant_01
participant_02/001.jpg,participant_02
```

Это иллюстрация формата, а не достаточная обучающая выборка. Для автоматического разбиения нужны минимум два класса и три разных изображения на каждый класс после удаления дубликатов.

```bash
python scripts/prepare_dataset.py --base datasets/faces --out reports/dataset --seed 42
# Альтернатива с CSV:
python scripts/prepare_dataset.py --base datasets/faces --labels datasets/labels.csv --out reports/dataset
```

Подготовка проверяет декодирование, удаляет точные дубликаты по RGB-пикселям и отвергает одинаковые изображения с противоречивыми метками. При долях `0.8/0.1/0.1` каждый класс распределяется между тремя частями; малые выборки округляются с сохранением хотя бы одного изображения в каждой части. В `report.json` записываются фактические количества.

Для оценки на независимых сессиях создайте собственный CSV `path,label,split`, где `split` равен `train`, `val` или `test`. Путь не должен повторяться, val обязателен, метки val/test должны существовать в train. Проверка готового CSV не заменяет проверку содержимого на похожие кадры и дубликаты.

## Обучение

```bash
python scripts/train_model.py --base datasets/faces --splits reports/dataset/splits.csv --epochs 5 --batch-size 16 --seed 42 --export-dir models
```

| Аргумент | Назначение |
| --- | --- |
| `--auto-prepare` | Создать разбиение, если `--splits` не передан |
| `--pretrained` | Разрешить загрузку ImageNet-весов ResNet-50; по умолчанию выключено |
| `--device cpu` | Явно использовать CPU; `auto` выбирает CUDA при наличии |
| `--workers 0` | Число процессов загрузки; 0 удобно для переносимого старта |
| `--batch-size 16` | Размер батча, минимум 2 из-за BatchNorm |
| `--lr 0.001` | Скорость обучения Adam |
| `--weight-decay 0.0001` | Регуляризация оптимизатора |
| `--out reports/experiment` | Явная папка результатов; выбирайте новую для каждого эксперимента |
| `--export-dir models` | Куда скопировать лучший state_dict, метки и summary |

Для воспроизводимости `--seed` применяется к подготовке при `--auto-prepare` и к обучению. Неполный последний батч из одного изображения пропускается и отмечается в журнале. Метрики потерь усредняются по реально обработанным примерам.

В папке запуска появляются `arcface_model.pth`, `classes.json`, `history.json`, `summary.json`, `log.txt`, `checks.csv`. Лучшая модель выбирается по accuracy на validation. Test оценивается один раз после загрузки лучшего checkpoint; если в пользовательском CSV test отсутствует, поле `test` в summary равно `null`. Эти метрики относятся к известным классам, а не к качеству порога `unknown`.

При повторном использовании `--out` отчёты перезаписываются. Перед экспортом в папку работающего сервера остановите его: файлы модели и меток заменяются последовательно.

## Детектор лиц

Классификатор не создаёт `yolov8n-face.pt`. Нужен отдельный YOLO-детектор, обученный именно на лицах. Пример стороннего проекта — [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face); его веса не включены и не скачиваются автоматически. Проверьте источник, условия использования и совместимость выбранного checkpoint с установленной Ultralytics. Любой выбранный файл нужно проверить в своей среде; встроенной гарантированно протестированной версии детектора нет.

Загрузчик принимает только явно заданные HTTPS-адрес и ожидаемый SHA-256:

```bash
python scripts/download_weights.py --url "https://trusted.example/yolov8n-face.pt" --sha256 "EXPECTED_64_HEX_SHA256"
```

Адрес и checksum выше — **заполнители**, замените их данными конкретного доверенного артефакта. Нужна контрольная сумма, полученная независимо от загружаемого файла. Неверная сумма, переход на HTTP или объём больше 200 МиБ приводят к отказу. Существующий файл назначения не перезаписывается.

Проверьте [три файла моделей](../models/README.md), перезапустите сервер и загрузите изображение. Успешный `/api/status/` подтверждает наличие файлов, но не качество или совместимость весов.

## CLI-инференс

```bash
python scripts/infer_image.py --image example.jpg --conf 0.25 --unknown-threshold 0.5 --out reports/result.jpg
```

По умолчанию вывод и журнал попадают в отдельную папку `reports/infer_<дата>/`. Используются те же модели, преобразования и пороги, что у API.

## Docker

По умолчанию `docker compose up --build web` собирает лёгкий target `web`. Для CPU-инференса:

```bash
# macOS / Linux
FACE_LAB_TARGET=ml docker compose up --build web
```

```powershell
# Windows PowerShell
$env:FACE_LAB_TARGET = "ml"
docker compose up --build web
```

Папка `models` подключена к веб-сервису только для чтения. Датасеты ему не монтируются. Обучение запускается отдельным сервисом:

```bash
docker compose --profile train run --rm train
```

Он читает `datasets/faces`, пишет `reports` и `models`, запускает 5 эпох без предобученных весов. На Linux обеспечьте запись в эти две папки для UID/GID 1000 либо передайте свой UID/GID через `docker compose run --user`. Образ `ml` использует CPU PyTorch и заметно тяжелее `web`. CI собирает и проверяет только лёгкий Docker-образ; ML-пайплайн отдельно проверяется без контейнера на Linux CPU.
