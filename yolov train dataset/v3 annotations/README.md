# v3 dataset — Earth Pro batch (May 2026)

Третий батч размеченных снимков — **больше данных в том же распределении**,
что v1/v2: Google Earth Pro скриншоты Астаны на zoom 17-19. Цель — улучшить
метрики на текущем val (recall + precision) и собрать более robust модель
за счёт большего разнообразия районов.

> **NB про domain shift** — production-сайт качает тайлы из Google Maps API
> (`mt0.google.com/vt/lyrs=s`), а **все** наши training data (v1, v2, v3)
> сняты с Google Earth Pro. Это другой rendering pipeline и иногда другая
> дата съёмки. Train ≠ deploy distribution. Чтобы закрыть этот gap — нужен
> отдельный v4 батч из самого сервиса (можно прямо в UI наш через Auto-Zoom
> Scan и сохранение PNG-ов). Не делаем сейчас, помним.

## Версии датасета — что куда

| Папка | Файлы | Train / Val | Источник | Кто использует |
|---|---|---|---|---|
| `annotations/` | `instances_Train.json`, `instances_Validation.json` | 16 + 5 | v1 CVAT export (2026-04, Earth Pro) | legacy, не трогаем |
| `annotations_merged/` | `instances_Train.json`, `instances_Validation.json` | 44 + 10 | v1 + v2 merge (2026-05, Earth Pro) | production v2 train target |
| `v3 annotations/annotations/` | `instances_default.json` | 24 images, ~1900 polygons (idle, разметка обновляется) | v3 (2026-05-17+, Earth Pro) | **этот батч** |

PNG-файлы:
- v1: `../фотографии/Снимок экрана 2026-04-01 *.png` (в git)
- v2: `../новые фотографии/Снимок экрана 2026-05-10 *.png` (в git)
- v3: `../v3 фотографии для finetune/Снимок экрана 2026-05-17 *.png` (в git)

Категория всегда одна: `Дерево` (id=1), Cyrillic в JSON — читай UTF-8.

> **Update history** — разметка v3 обновляется по мере того как команда
> добавляет polygons. Файл `instances_default.json` перетирается, git
> diff покажет рост числа annotations. Перед тренировкой обнови ветку
> (`git pull`) чтобы взять последнюю версию.

---

## Идея для всех трёх веток

Все три модели тренируются на **одном и том же merged-датасете**: v1+v2+v3.
Валидация — на одном merged-val. Это даёт честное сравнение mAP в diploma
ablation table (Chapter 4).

**Pipeline для подготовки данных одинаков для всех:**

1. `ml/split_coco.py` — разделить v3 COCO 80/20 (val ≈ 5 кадров, seed=42)
2. `ml/merge_coco.py` — слить v2-merged + v3 → объединённый train/val
3. Дальше формат расходится:
   - **YOLO** → `ml/coco_to_yolo_seg.py` → `ml/tile_dataset.py` (640/128)
   - **Mask R-CNN** → читает COCO напрямую через `CocoMaskRCNNDataset`
   - **DeepForest** → `ml/coco_to_deepforest_csv.py` (polygon ignored, берём bbox)

Команды для каждого шага — в docstring соответствующего скрипта (`--help`).

---

## По командам — что и откуда стартовать

### Rasul · YOLOv8-seg

**Стартовая точка:** `weights/yolo_satellite.pt` это **v2-finetune** (md5
`f88d0d3dc6d1609e17c7670639e38b24`, runs/segment/astana_tiled_x_v2_finetune/).
Не v2-fromscratch, не v1, не yolov8x-seg.pt pretrained. Это та модель которая
сейчас в production.

**Что делать:** fine-tune от v2-finetune весов на merged v1+v2+v3. Тот же
`ml/train_yolo.py --weights weights/yolo_satellite.pt`. Параметры обучения
(epochs/patience/imgsz) подбирать ad-hoc — обычно finetune сходится быстрее
чем fromscratch, начни с 100 эпох / patience 30 и смотри на curves.

После тренировки производственные веса лежат в `runs/segment/<name>/weights/best.pt`
— скопировать в `weights/yolo_satellite.pt` чтобы backend подхватил.

### Berik · Mask R-CNN

**Стартовая точка на выбор:**
- `weights/maskrcnn_astana.pt` если уже есть (это твоя предыдущая v1+v2 fine-tune
  если она была), → **fine-tune от неё**.
- Или torchvision COCO V1 backbone (default в `MaskRCNNAdapter.build_model`) →
  **train from scratch на v1+v2+v3**.

Обе опции валидные. Тренируешь обе — получаешь дополнительную строку в
ablation-таблице (fine-tune vs scratch), это сильная diploma-методология.
Тренируешь одну — это тоже ок, просто выбери ту которая логичнее с твоей
позиции (если уже есть рабочий v1+v2 checkpoint = fine-tune быстрее).

Скрипт твой: `ml/train_maskrcnn.py`. Параметры по умолчанию (epochs=50,
batch=2, lr=0.005, SGD+StepLR) проверены, подходят под 8 GB VRAM.

`CocoMaskRCNNDataset` принимает COCO JSON напрямую — ничего не конвертируй,
просто укажи `--train-json` и `--val-json` от merge'нутого датасета и
`--images-roots` через все три photo папки.

### Anuar · DeepForest (+ SAM 2)

**Стартовая точка:** твой собственный `~150-image fine-tuned DF checkpoint`
(`weights/deepforest_astana.pl` если он там, иначе там где у тебя локально).
**Не pretrained NEON** — у тебя уже есть гораздо более релевантный baseline.

**Что делать:** fine-tune от твоего существующего DF на merged v1+v2+v3.
Не от pretrained — у тебя 150 кадров уже scoped на Астану, prior сильно
ближе чем NEON-овский американский лес.

**Конверсия данных:** DeepForest это detection-only (без сегментации), у
него свой CSV-формат (`image_path,xmin,ymin,xmax,ymax,label`). Polygon
из COCO игнорируется, берём только bbox-поле. Конвертер сделан —
`ml/coco_to_deepforest_csv.py --help` для деталей.

**Тренировка:** в repo нет готового `ml/train_deepforest.py` (мы его не
делали, у тебя пайплайн был свой). Канонический способ — через DF API:
```python
m = df_main.deepforest()
# Загрузить твой существующий .pl, не pretrained
m.load_from_checkpoint("weights/deepforest_astana.pl")  # или твой path
m.config["train"]["csv_file"] = "yolov train dataset/v3_deepforest/train.csv"
m.config["train"]["root_dir"] = "yolov train dataset/v3_merged/images"
m.config["validation"]["csv_file"] = "yolov train dataset/v3_deepforest/val.csv"
m.create_trainer()
m.trainer.fit(m)
m.save_model("weights/deepforest_astana.pl")
```

Если хочешь я могу сделать work-ready `ml/train_deepforest.py` — попроси,
сделаю по твоим параметрам.

**SAM 2 не трогать** — это inference-time refiner, не train-time компонент.
После fine-tune DF он продолжает работать с новыми DF-bbox-ами тем же
способом (см. `backend/models/deepforest_sam2_adapter.py`).

---

## Eval baseline (общая точка отсчёта)

Все три модели сравниваются на **одном merged-val** (v1+v2+v3, ≈ 15 image
/ ~700+ polygons после финализации v3 разметки).

Текущий v1+v2 baseline (для отсчёта):

| Модель | Box mAP@50 | Mask mAP@50 | Weights |
|---|---|---|---|
| YOLO v1 (yolov8x, 397 ep) | 0.265 | 0.240 | `runs/segment/astana_tiled_x_max/` |
| YOLO v2 from-scratch (204 ep) | 0.319 | 0.288 | `runs/segment/astana_tiled_x_v2_fromscratch/` |
| **YOLO v2 fine-tune (173 ep) — production** | **0.372** | **0.331** | `runs/segment/astana_tiled_x_v2_finetune/` |
| Mask R-CNN (v1+v2, 50 ep) | TBD | target ≥ 0.45 | `weights/maskrcnn_astana.pt` |
| DeepForest (Anuar's 150-img fine-tune) | TBD | — | `weights/deepforest_astana.pl` |

После v3 тренировок таблица расширяется до v3-строк в той же структуре.
Mask без значения у DF — он detection-only.

---

## .gitignore (для справки)

```
# Generated artifacts (regenerate via scripts):
v3_merged/
v3_yolo/
v3_yolo_tiled/
v3_deepforest/

# Splits inside v3 annotations/ (regenerated by split_coco.py):
v3 annotations/annotations/instances_Train.json
v3 annotations/annotations/instances_Validation.json
```

Оригинальные PNG + `instances_default.json` — committed. См. корневой `.gitignore`.
