# Astana Tree Detection — что уже реализовано

Дайджест текущего функционала (последнее обновление: 2026-05-16).
Эта страница для других контрибьюторов и AI-ассистентов чтобы быстро вкатиться.

Точку входа смотри в `README.md`. Конкретные эксперименты по YOLO — в `yolov train dataset/annotations_merged/README.md`.

---

## Команда и ответственность

| Кто | Зона |
|---|---|
| **Rasul Aidarkhanov** | YOLOv8-seg pipeline, backend wiring, frontend, общая product-логика |
| **Anuar Totin** | SAM integration (in progress). DeepForest — его существующая ветка `Added-deepforest_fine_tuned`, веса .pl у него локально |
| **Berik Sharipov** | Mask R-CNN (in progress, ветка `feat/maskrcnn`) — instance segmentation as a two-stage baseline vs YOLO |
| **Supervisor**: Syndar Satbayev | |

Свежая база для всех: ветка **`yolov8-work`** на `origin` — отбраничивайтесь оттуда, не от `main`.

---

## Backend (FastAPI, Python 3.12)

### Архитектура
- **Adapter pattern**: `backend/models/*_adapter.py` имплементируют общий интерфейс из `base.py`.
- **Registry** (`backend/models/__init__.py`): регистрация адаптеров при старте.
- **Lazy loading**: веса грузятся в память при первом `/api/predict`, не при старте.
- **Persistent storage**: SQLite, файл `storage/app.db`. Три таблицы — `snapshots`, `runs`, `detections` со `ON DELETE CASCADE`. Слой в `backend/db.py`. Никаких in-memory словарей — рестарт бэка ничего не теряет.

### Реализованные модели
| Adapter | Имя | Веса | Замечания |
|---|---|---|---|
| `YOLOAdapter` | `yolo` | `weights/yolo_satellite.pt` (v2-finetune, Box mAP50=0.372) | Sliding-window tiled inference (`tile_size=640`, `overlap=128`, `single_shot_limit=768`) + global NMS поверх overlap-зон. Совпадает с тренировочной геометрией. |
| `MaskRCNNAdapter` | `maskrcnn` | `weights/maskrcnn_astana.pt` (опц., torchvision pretrained как fallback) | Тот же sliding-window tiled inference что у YOLO + global NMS. Berik's pipeline, ветка `feat/maskrcnn`. |
| `DeepForestAdapter` | `deepforest` | `weights/deepforest_astana.pl` (опц.) + pretrained `weecology/deepforest-tree` (fallback) | graceful: если .pl нет, falls back на pretrained. Tiled inference через DF native (`predict_tile`, 400px patch). |
| `EnsembleAdapter` | `ensemble` | оба выше | WBF (Weighted Box Fusion). |

### Geo conversion (`backend/geo.py`)
- 4 `GeoMode`: `NONE`, `CORNERS_2` (axis-aligned, NW+SE), `CORNERS_4` (bilinear, четыре угла — handles rotation), `GEOTIFF_AFFINE` (читает аффинную трансформацию из tif EXIF).
- `annotate_detections` заполняет в каждой `Detection`:
  - `lat`, `lng` (центр bbox)
  - `crown_diameter_m` (если есть `pixel_size_m`)
  - `mask_polygon_geo` — каждая вершина mask polygon в lat/lng
  - `box_geo` — 4 угла bbox (NW, NE, SE, SW) в lat/lng → рисуется как полигон на Leaflet, корректен в `CORNERS_4` режиме тоже

### Map capture (`backend/map_capture.py`)
- `POST /api/capture_from_map` — пользователь рисует прямоугольник на Leaflet, бэк скачивает спутниковые тайлы для bbox, склеивает (ThreadPoolExecutor, 8 потоков), обрезает до точного pixel bbox, сохраняет как обычный upload.
- Ограничение: 144 тайла на запрос (12×12 ≈ 3072×3072 px) — защита от DoS.
- Серый-placeholder тайл при сетевой ошибке, без обвала всей склейки.

### Switchable tile providers (`backend/map_capture.py::TILE_PROVIDERS` + `GET /api/providers`)
**Зачем:** YOLO/Mask R-CNN тренировались на Google Earth Pro скриншотах. Esri даёт другую цветопередачу + иногда другую дату съёмки → domain shift → recall просаживается. Решение — переключаемый источник тайлов; для лучших результатов берём Google Satellite tiles (та же image base что использует Google Earth Pro).

| Provider key | Endpoint | Max zoom | Замечание |
|---|---|---|---|
| `esri` | `server.arcgisonline.com/.../World_Imagery/...` | 19 | Бесплатный, без ключей. Стабильный. Default для backward compat. |
| `google` | `mt{0-3}.google.com/vt/lyrs=s` | 20 | Та же image base что Google Earth Pro. Unofficial endpoint, для академического прототипа OK. |

Endpoint `GET /api/providers` отдаёт список с URL-шаблонами и `max_zoom`; фронт строит из него провайдер-dropdown **и** Leaflet base layer (один источник правды — Leaflet на экране показывает ровно то же, что backend качает для capture).

Параметр `provider` (default `esri`) принимается в `POST /api/capture_from_map` и `POST /api/scan_region`. Имя файла-снимка включает provider (`map_capture_google_z18.png`, `scan_google_z19_r0c0.png`) — видно в city-aggregate listings.

### Auto-Zoom Region Scan (`backend/region_scan.py` + `POST /api/scan_region`)
**Зачем:** пользователь рисует большой прямоугольник на Leaflet (например 1.5×1.5 км целого района). На том зуме, на котором рисует, кроны деревьев = 4–6 пикселей, и YOLO/Mask R-CNN ничего не находят — обучались на GSD ~0.5 м/px (zoom 18–19). Если же запросить весь bbox на zoom 19 одним кадром — упрётесь в MAX_TILES=144 (≈3000×3000 px).

**Решение — три уровня тайлинга (см. `region_scan.py` docstring):**
- **Layer 0** — `region_scan.split_bbox_to_subregions`: bbox + target_zoom → сетка NxN под-bbox-ов, каждый ≤ ~100 тайлов. Корень `n = ceil(sqrt(total_tiles / max_per_sub))`.
- **Layer 1** — существующий `map_capture.capture_bbox` для каждого под-bbox: ESRI-тайлы → склейка → точный crop.
- **Layer 2** — sliding-window 640+128 внутри адаптера (YOLO и Mask R-CNN), global NMS поверх overlap-зон.

**Endpoint `POST /api/scan_region`:**
```jsonc
// Request
{
  "nw": {"lat": 51.165, "lng": 71.465},
  "se": {"lat": 51.155, "lng": 71.480},
  "zoom": 19,                  // optional, default 19, range 14..19
  "model": "yolo",             // или maskrcnn / deepforest / ensemble / deepforest_sam2
  "confidence": 0.25,
  "max_subregions": 9          // optional, default 9 — защита от 30-минутных скан-сессий
}
// Response
{
  "sub_count": 4, "ok_count": 4, "total_trees": 287, "duration_ms": 42130,
  "zoom": 19, "model": "yolo",
  "bbox": {...},
  "sub_regions": [
    {"row":0,"col":0,"snapshot_id":"abc","job_id":"def","tree_count":71,"duration_ms":9800,"sub_bbox":{...}},
    ...
  ]
}
```

Каждый успешный под-регион сохраняется как обычный `snapshot` + `run`, поэтому **в city-aggregate view (`/api/snapshots`, `/api/detections`) результаты появляются автоматически** — никакой отдельной таблицы scan-сессий нет. Если под-регион упал на capture или predict — endpoint не прерывается, просто в `sub_regions[i]` появится `"error"`. Frontend после успеха переключается в city view и обновляет агрегат.

**UI:** кнопка `Auto-Zoom Scan` в Upload-секции (зелёная, рядом с `Capture from map`). Зум зафиксирован на 19, скан-режим рисует прямоугольник зелёным цветом (capture-mode остался оранжевым). Лимит 9 под-регионов ≈ 3×3 = ~1.5×1.5 км @ z19 за один запрос.

### REST endpoints
| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/status` | Состояние сервера + агрегаты (snapshots / runs / total trees) |
| `GET` | `/api/providers` | Список tile-провайдеров с URL/max_zoom для синхронизации Leaflet base layer. |
| `POST` | `/api/upload` | Загрузка PNG/JPG/TIFF/GeoTIFF. Возвращает `ImageMeta`. |
| `POST` | `/api/capture_from_map` | `{nw, se, zoom}` → ImageMeta с bounds. |
| `POST` | `/api/scan_region` | **Auto-Zoom Region Scan** — большой bbox любого размера → сетка под-регионов на zoom 19 → capture+predict каждого → snapshots в БД. |
| `GET` | `/api/image/{id}` | Сам PNG для отображения. |
| `GET` | `/api/image/{id}/meta` | Meta снимка. |
| `POST` | `/api/predict` | Inference (model, confidence, geo). Сохраняет run+detections. |
| `GET` | `/api/result/{job_id}` | Поднять прошлый result. |
| `GET` | `/api/snapshots` | Список всех snapshots с агрегатами (run_count, total_trees, last_model). |
| `GET` | `/api/detections` | **Aggregate**: все детекции с опц. bbox / model / min_confidence фильтрами. По умолчанию только last run per snapshot (без дублей). |
| `GET` | `/api/aggregate/stats` | Сводка: snapshots / runs / trees / avg conf / avg crown. |
| `DELETE` | `/api/snapshots/{id}` | Каскадом: snapshot + runs + detections + PNG с диска. |
| `DELETE` | `/api/runs/{job_id}` | Один прогон. |
| `POST` | `/api/export/{job_id}/{geojson\|csv\|html}` | Экспорт. |
| `GET` | `/api/history` | Последние N прогонов. |

---

## Frontend (React 18 UMD + Babel-standalone + Leaflet, без build-step)

Файлы: `frontend/{index.html, app.jsx, api.js, styles.css}`. Сервируется напрямую FastAPI'ем по `/`.

### Двa view modes (переключаются в сайдбаре)

**1. Single image view** (default — workflow одного снимка)
- Upload zone: drag-drop, file picker, либо `Capture from map` (Leaflet rectangle).
- Detection controls: выбор модели (YOLO/DF/Ensemble), `Run detection` (с прогрессом+ETA).
- Geo panel: выбрать `GeoMode` + ввести координаты или перетаскивать NW/SE маркеры прямо на карте.
- Stats panel: trees / coverage / avg conf / area.
- Confidence filter (high/med/low).
- Export panel: GeoJSON / CSV / HTML.
- History panel.

**2. City map view** (aggregate всего что в БД)
- AggregateStatsPanel: карточки trees / snapshots / runs / avg conf / avg crown.
- SnapshotsList: список snapshots с метой (count runs, total trees, last model, GPS coords) + кнопка delete (каскад на БД и файл).
- Layers / Display panels те же.
- На карте — **все деревья всех сохранённых прогонов одновременно** (limit 50k для безопасности). Главный money shot для defense.

### Detection display switch (segmented control)
Три варианта рендера каждой детекции, **взаимоисключающие**:
- **Point** — circleMarker по центру, мелкий dot.
- **BBox** — `L.polygon` с 4 углами в lat/lng (корректен даже в режиме `corners_4` rotation).
- **Polygon** — мaска YOLO как `L.polygon` с прозрачной заливкой.

Дефолт — Polygon (главное визуальное преимущество YOLO). Fallback: если для детекции нет данных для выбранного режима (например polygon у DeepForest), рендерится точкой — детекция не пропадает с карты.

### Map layers
- Base layer toggle: Esri World Imagery (sat) / CartoDB (clean).
- Image overlay (с opacity slider) — только в single view.
- Draggable NW/SE rectangle для georef'а вручную.

### Connection drift fix
В CORNERS_2 mode `imageBounds` следует за `geo.corners_2` (не за исходным `image.bounds`), так что когда пользователь двигает NW/SE маркеры — картинка-overlay двигается вместе с ними. Без этого фикса деревья после rerun смещались относительно картинки.

---

## ML scripts (`ml/`)

| Скрипт | Назначение |
|---|---|
| `coco_to_yolo_seg.py` | CVAT COCO 1.0 → YOLOv8-seg формат. Handles Cyrillic class names, sanitizes filenames to ASCII. |
| `tile_dataset.py` | Sliding-window 640+overlap, clip polygons via shapely, drop fragments < min_area. |
| `train_yolo.py` | Wrapper над Ultralytics с тюненной aug под satellite (умеренный hsv, агрессивный flip, mosaic+mixup+copy_paste). |
| `prelabel_coco.py` | Tiled inference готовой YOLO модели → COCO 1.0 для CVAT pre-labels. На v1 best дал recall ~50%, не использовался для v2 разметки. |
| `merge_coco.py` | Объединить N COCO JSON в один (renumber id, dedupe categories). |
| `split_coco.py` | Детерминистично разделить COCO на train/val (seed=42 default). Val выбирается ПЕРВЫМ, чтобы не cherry-pick'ать «лёгкие» в val. |
| `evaluate.py` | Метрики. |
| `convert_cvat_to_yolo.py` | Legacy. Использовать `coco_to_yolo_seg.py`. |
| `merge_datasets.py` | Legacy. |

---

## Production-state YOLO модели

Все в `runs/segment/<name>/weights/best.pt`. Гитнорятся (большие), но папка `runs/` локально остаётся.

| Run | Старт | Epochs | Wall time | Box mAP50 (merged val) | Mask mAP50 (merged val) |
|---|---|---|---|---|---|
| `astana_tiled_x_max` (v1) | yolov8x-seg.pt | 397 | ~60 min | 0.265 | 0.240 |
| `astana_tiled_x_v2_fromscratch` | yolov8x-seg.pt | 204 | ~62 min | 0.319 | 0.288 |
| **`astana_tiled_x_v2_finetune` (production)** | v1 best.pt | 173 | ~56 min | **0.372** | **0.331** |

`weights/yolo_satellite.pt` = v2-finetune best.pt (md5 `f88d0d3dc6d1609e17c7670639e38b24`).

**Важная methodology note**: v1 на своём оригинальном маленьком 4-tile val давал 0.681 Box mAP50 — этот номер **не сравним** с числами выше. Сравнения только на merged val.

---

## Что НЕ реализовано

- **DeepForest fine-tuned .pl** — отсутствует в репо (~600 MB, у Anuar локально).
- **SAM** — Anuar пишет.
- **Berik's model** — Mask R-CNN (torchvision `maskrcnn_resnet50_fpn_v2`), in progress on `feat/maskrcnn`. Trains on `annotations_merged` COCO. See `docs/maskrcnn.md`.
- **Bulk upload папки** GeoTIFF — UX заплатка, сейчас можно по одной.
- **Compare mode** — две даты, diff деревьев.
- **Click на snapshot в city list → load в single view** — навигация-удобство.
- **Async/streaming прогресс для Auto-Zoom Scan** — сейчас endpoint синхронный, фронт ждёт ~30–60 сек блок-spinner'ом. Нормально для 9 под-регионов; для 25+ нужен WebSocket / SSE.

## Запуск локально

```bash
# venv (Python 3.12)
python -m venv venv
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Веса YOLO кладутся вручную — pipeline не auto-downloads:
# weights/yolo_satellite.pt   ← из runs/segment/astana_tiled_x_v2_finetune/weights/best.pt
# weights/deepforest_astana.pl ← попросить у Anuar (опц.)

# Поднять сервер
uvicorn backend.main:app --host 127.0.0.1 --port 8000
# или start.bat (Windows)
```

Открыть http://127.0.0.1:8000. UI и API docs (`/docs`) сразу доступны.
