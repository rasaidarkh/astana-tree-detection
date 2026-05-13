# Merged v1+v2 COCO annotations

Готовый объединённый датасет для тренировки **любой** модели сегментации/детекции
деревьев на спутниковых снимках Астаны.

## Содержимое

| Файл | Изображений | Polygons | Источник |
|---|---|---|---|
| `instances_Train.json` | 44 | 3270 | 16 от v1 (CVAT, 2026-04) + 28 от v2 (2026-05) |
| `instances_Validation.json` | 10 | 275 | 5 от v1 + 5 от v2 |

- **Категория** одна: `Дерево` (id=1). Это Cyrillic — учитывайте UTF-8 при чтении JSON.
- **Координаты** полигонов и bbox — в **пиксельных** координатах относительно соответствующего PNG.
- **Format**: стандартный COCO 1.0 (можно ингестить в Ultralytics, MMDetection, Detectron2 и т.д.).

## Где сами PNG-снимки

Изображения лежат в двух папках:
- `../фотографии/Снимок экрана 2026-04-01 *.png` — v1 батч (20 файлов, в git).
- `../новые фотографии/Снимок экрана 2026-05-10 *.png` — v2 raw (33 размеченных + 24 неразмеченных оригинала, **все в git**).

⚠️ **Один файл обрезан вручную**: `Снимок экрана 2026-05-10 102326.png` — оригинал из CVAT был 1613×1138,
обрезан до 1613×862 потому что нижняя часть осталась без аннотаций. В соответствующих COCO-файлах
этой картинки `height=862`, полигоны лежат «как есть» (crop был снизу, все Y ≤ 749).

## Как собрать в YOLO формат для тренировки

```bash
# 0. Активировать venv (CPU достаточно для подготовки данных)
.\venv\Scripts\activate    # Windows
# source venv/bin/activate # Linux/Mac

# 1. Скопировать v2 фотографии в общую папку (v1 уже там, v2 дубли gitignored)
cp "yolov train dataset/новые фотографии/Снимок экрана 2026-05-10"*.png \
   "yolov train dataset/фотографии/"

# 2. COCO → YOLOv8-seg формат
python ml/coco_to_yolo_seg.py \
    --train-coco "yolov train dataset/annotations_merged/instances_Train.json" \
    --val-coco   "yolov train dataset/annotations_merged/instances_Validation.json" \
    --images-dir "yolov train dataset/фотографии" \
    --output     "yolov train dataset/yolo"

# 3. Tiled 640×640 (важно — кроны деревьев ~20-40 px, без тайлинга при imgsz=640
#    оригинальные 1700×1100 будут ужаты, кроны уменьшатся до ~7 px)
python ml/tile_dataset.py \
    --input  "yolov train dataset/yolo" \
    --output "yolov train dataset/yolo_tiled" \
    --tile-size 640 --overlap 128 --min-area 25

# Готовый dataset.yaml: yolov train dataset/yolo_tiled/dataset.yaml
```

## Тренировка (для контекста, если кто-то идёт по YOLO пути)

```bash
# С нуля от COCO-pretrained
python ml/train_yolo.py \
    --data "yolov train dataset/yolo_tiled/dataset.yaml" \
    --weights yolov8x-seg.pt \
    --imgsz 640 --batch 2 \
    --epochs 500 --patience 100 --device 0 \
    --name astana_tiled_x_v3

# ИЛИ fine-tune от существующего лучшего чекпоинта (production)
python ml/train_yolo.py \
    --data "yolov train dataset/yolo_tiled/dataset.yaml" \
    --weights runs/segment/astana_tiled_x_v2_finetune/weights/best.pt \
    --imgsz 640 --batch 2 \
    --epochs 200 --patience 50 --device 0 \
    --name astana_tiled_x_v3_finetune
```

## Baseline / точки отсчёта

Все три YOLO-чекпоинта валидированы на **этом самом** val (10 tiles, 258 polygons):

| Модель | Box P | Box R | **Box mAP50** | Mask mAP50 | Weights |
|---|---|---|---|---|---|
| v1 (yolov8x, 397 ep) | 0.336 | 0.310 | 0.265 | 0.240 | `runs/segment/astana_tiled_x_max/` |
| v2 from-scratch (204 ep) | 0.345 | 0.333 | 0.319 | 0.288 | `runs/segment/astana_tiled_x_v2_fromscratch/` |
| **v2 fine-tune (173 ep) — production** | **0.425** | **0.391** | **0.372** | **0.331** | `runs/segment/astana_tiled_x_v2_finetune/` |

Если будешь тренировать **другую модель** (DeepForest, SAM, Mask R-CNN, etc.) — сравнивай против
этих чисел на **той же** валидации. mAP сильно зависит от выбора val (v1 на своём старом маленьком val
давал 0.681, на merged → 0.265 — разница в 2.6× от одной и той же модели). Поэтому валидируем на одном.

## Истории

- 24 неразмеченных PNG из v2 batch'а намеренно оставлены в `новые фотографии/` без COCO-записей.
  Это не «trees отсутствуют» а «команда не успела дойти до них» — для тренировки они не используются
  (модель училась бы что в этих кадрах деревьев нет, а они есть).
- v2 fine-tune обходит v2 from-scratch по всем метрикам (+5 пп Box mAP50) — старт от v1 best.pt
  даёт реальный буст. Имейте в виду при выборе стартовой точки.
