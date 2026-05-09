# Datasets

## Структура

```
data/
├── raw/             исходники (CVAT экспорты, оригинальные снимки)
│   ├── cvat_15.xml         — твоя свежая разметка из CVAT
│   ├── images_15/          — 15 новых снимков
│   └── astana_yolov8_134/  — старый LabelMe датасет, скопирован с pipeline/yolov8seg/
├── annotated/       промежуточные файлы (не обязательно)
└── processed/       готовые YOLOv8 датасеты, на которых обучаемся
    ├── cvat_15/             только новые
    ├── astana_134/          только старые
    └── combined/            объединённый (134 + 15)
```

## Workflow

### 1. Конвертация CVAT-разметки

```bash
python ml/convert_cvat_to_yolo.py \
    --annotations data/raw/cvat_15.xml \
    --images data/raw/images_15/ \
    --output data/processed/cvat_15/ \
    --train-ratio 0.85
```

Поддерживаемые форматы экспорта из CVAT:
- **CVAT for Images 1.1** (XML) — рекомендуется
- **COCO 1.0** (JSON)

### 2. Объединение со старым датасетом

```bash
python ml/merge_datasets.py \
    --old C:/Users/Rasul/DeepLearning/pipeline/yolov8seg/dataset \
    --new data/processed/cvat_15 \
    --output data/processed/combined \
    --train-ratio 0.85
```

### 3. Тренировка

```bash
python ml/train_yolo.py \
    --data data/processed/combined/dataset.yaml \
    --weights weights/yolo_satellite.pt \
    --epochs 50 \
    --imgsz 1024 \
    --batch 4 \
    --name astana_v2
```

### 4. Оценка

```bash
python ml/evaluate.py \
    --data data/processed/combined/dataset.yaml \
    --models yolo deepforest ensemble \
    --conf 0.25 \
    --radius-px 30 \
    --output docs/ablation.csv
```

## Замечания по аннотированию

При разметке плотного satellite-изображения важны:

- **Согласованная политика "одно дерево vs группа"**. Реши заранее: каждое дерево = 1 полигон, или сильно overlapping кроны = 1 полигон.
- **Грубые овалы лучше точных полигонов.** Для satellite-разрешения детальный контур не нужен — он только увеличит шум разметки и тренировочное время.
- **Bounding boxes тоже годятся.** YOLOv8-seg может тренироваться на боксах (тогда маски будет генерировать YOLO сама из боксов в predict-время).
- **15 кадров → агрессивная аугментация.** Mosaic + flipud=0.5 + scale=0.4 даёт x10-x20 эффективных образцов.
