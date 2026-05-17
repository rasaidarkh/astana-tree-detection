# v3 dataset — Earth Pro batch (May 2026)

Третий батч размеченных снимков — **больше данных в том же распределении**,
что v1/v2: Google Earth Pro скриншоты Астаны на zoom 17-19. Цель — улучшить
метрики на текущем val (recall + precision) и собрать более robust модель
за счёт большего разнообразия районов.

> **NB про domain shift** — production-сайт качает тайлы из Google Maps API
> (`mt0.google.com/vt/lyrs=s`), а **все** наши training data (v1, v2, v3)
> сняты с Google Earth Pro. Это другой rendering pipeline и иногда другая
> дата съёмки. Train ≠ deploy distribution. Чтобы закрыть этот gap — нужен
> отдельный v4 батч из сервиса (можно прямо в UI наш через Auto-Zoom Scan
> и сохранение PNG-ов). Не делаем сейчас, помним.

## Версии датасета — что куда

| Папка | Файлы | Train / Val | Источник | Кто использует |
|---|---|---|---|---|
| `annotations/` | `instances_Train.json`, `instances_Validation.json` | 16 + 5 | v1 CVAT export (2026-04, Earth Pro) | legacy, не трогаем |
| `annotations_merged/` | `instances_Train.json`, `instances_Validation.json` | 44 + 10 | v1 + v2 merge (2026-05, Earth Pro) | **production v2 train target** |
| `v3 annotations/annotations/` | `instances_default.json` | 24 (нужно сплитнуть) | **v3** (2026-05-17, Earth Pro) | **этот батч** |

PNG-файлы:
- v1: `../фотографии/Снимок экрана 2026-04-01 *.png` (в git, 20 файлов)
- v2: `../новые фотографии/Снимок экрана 2026-05-10 *.png` (в git, 33 размеченных + 24 unlabeled)
- **v3: `../v3 фотографии для finetune/Снимок экрана 2026-05-17 *.png`** (в git, **24 кадра, идёт разметка**)

Категория всегда одна: `Дерево` (id=1), Cyrillic в JSON — читай UTF-8.

## Статус v3 разметки

⚠️ **Разметка ещё не закончена.** На момент 2026-05-17 размечено 24 PNG из ~50
запланированных. Команда добавит ещё кадры в течение недели — следи за этой
папкой. Текущий `instances_default.json` содержит то что готово; финальная
версия будет с full coverage.

Когда придут новые кадры: они появятся в `v3 фотографии для finetune/` +
обновлённый `instances_default.json` в этой папке. Гитом отслеживается всё.

## Что делать (по моделям)

### Rasul — YOLOv8-seg v3-finetune

Fine-tune от v2 production-весов на **merged v2+v3** datasete. Не from-scratch —
v2 best.pt уже учила distribution-у крон Астаны, v3 это smooth domain adapt.

```bash
# 0. venv с GPU torch (см. docs/ml-setup.md если CUDA не работает)
.\venv\Scripts\activate

# 1. Сплит v3 на 80/20 (используем seed=42 для воспроизводимости)
python ml/split_coco.py \
    --input  "yolov train dataset/v3 annotations/annotations/instances_default.json" \
    --train  "yolov train dataset/v3 annotations/annotations/instances_Train.json" \
    --val    "yolov train dataset/v3 annotations/annotations/instances_Validation.json" \
    --val-count 5 --seed 42

# 2. Merge v2 + v3 (train и val отдельно)
python ml/merge_coco.py \
    --inputs "yolov train dataset/annotations_merged/instances_Train.json" \
             "yolov train dataset/v3 annotations/annotations/instances_Train.json" \
    --output "yolov train dataset/v3_merged/instances_Train.json"
python ml/merge_coco.py \
    --inputs "yolov train dataset/annotations_merged/instances_Validation.json" \
             "yolov train dataset/v3 annotations/annotations/instances_Validation.json" \
    --output "yolov train dataset/v3_merged/instances_Validation.json"

# 3. Собрать все PNG в одну папку (coco_to_yolo_seg.py принимает один --images-dir)
mkdir -p "yolov train dataset/v3_merged/images"
cp -n "yolov train dataset/фотографии/"*.png             "yolov train dataset/v3_merged/images/"
cp -n "yolov train dataset/новые фотографии/"*.png       "yolov train dataset/v3_merged/images/"
cp -n "yolov train dataset/v3 фотографии для finetune/"*.png "yolov train dataset/v3_merged/images/"

# 4. COCO → YOLOv8-seg
python ml/coco_to_yolo_seg.py \
    --train-coco "yolov train dataset/v3_merged/instances_Train.json" \
    --val-coco   "yolov train dataset/v3_merged/instances_Validation.json" \
    --images-dir "yolov train dataset/v3_merged/images" \
    --output     "yolov train dataset/v3_yolo"

# 5. Tiled 640+128 overlap (matching v2 геометрию)
python ml/tile_dataset.py \
    --input  "yolov train dataset/v3_yolo" \
    --output "yolov train dataset/v3_yolo_tiled" \
    --tile-size 640 --overlap 128 --min-area 25

# 6. Fine-tune от v2 production-весов (НЕ от yolov8x-seg.pt — это терять v2 опыт)
python ml/train_yolo.py \
    --data    "yolov train dataset/v3_yolo_tiled/dataset.yaml" \
    --weights "weights/yolo_satellite.pt" \
    --imgsz 640 --batch 8 \
    --epochs 200 --patience 50 --device 0 \
    --name astana_tiled_x_v3_finetune
```

Шаги 1-5 идемпотентны (можно перезапускать). Результат → `runs/segment/astana_tiled_x_v3_finetune/`.
После тренировки скопировать `weights/best.pt` → `weights/yolo_satellite.pt` чтобы backend подхватил.

### Berik — Mask R-CNN от-нуля на merged v2+v3

v3 это **просто больше данных в том же распределении** (Earth Pro). Текущий
`weights/maskrcnn_astana.pt` (если есть) тренирован на v1+v2 = той же
distribution. Делать from-scratch retrain не обязательно — fine-tune от
существующего тоже сработает. Но **рекомендуется from-scratch** по простой
причине: ты на v3 ещё не делал ни одного train run, baseline-таблица для
диплома будет чище если все три (v1, v2, v3) пройдены одной рукой за один
заход с pretrained COCO V1 backbone. Меньше шансов что reviewer спросит
"а почему именно эта стартовая точка?"

```bash
# Шаги 1-2 те же что выше (сплит v3 + merge COCO):
python ml/split_coco.py \
    --input  "yolov train dataset/v3 annotations/annotations/instances_default.json" \
    --train  "yolov train dataset/v3 annotations/annotations/instances_Train.json" \
    --val    "yolov train dataset/v3 annotations/annotations/instances_Validation.json" \
    --val-count 5 --seed 42

python ml/merge_coco.py \
    --inputs "yolov train dataset/annotations_merged/instances_Train.json" \
             "yolov train dataset/v3 annotations/annotations/instances_Train.json" \
    --output "yolov train dataset/v3_merged/instances_Train.json"
python ml/merge_coco.py \
    --inputs "yolov train dataset/annotations_merged/instances_Validation.json" \
             "yolov train dataset/v3 annotations/annotations/instances_Validation.json" \
    --output "yolov train dataset/v3_merged/instances_Validation.json"

# Mask R-CNN читает COCO напрямую через CocoMaskRCNNDataset — НЕ нужны YOLO-форматные
# tiling-шаги (это специфика Ultralytics). Просто передай ему путь к JSON-ам и
# несколько --images-roots, dataset.py сам ищет PNG по filename.

python -m ml.train_maskrcnn \
    --train-json "yolov train dataset/v3_merged/instances_Train.json" \
    --val-json   "yolov train dataset/v3_merged/instances_Validation.json" \
    --images-roots \
        "yolov train dataset/фотографии" \
        "yolov train dataset/новые фотографии" \
        "yolov train dataset/v3 фотографии для finetune" \
    --output     "weights/maskrcnn_astana.pt" \
    --epochs 50 --batch-size 2 --lr 0.005 \
    --log-dir lightning_logs/maskrcnn_v3
```

**Почему from-scratch в данном случае:**
1. У тебя ещё не было train run на v3 → чистый baseline без зависимости
   от прошлого checkpoint'а.
2. torchvision COCO V1 backbone уже даёт generic-инициализацию (миллионы
   аннотаций из MS COCO), не нужен ни v1 ни v2 prior.
3. 63 train + 15 val (v2+v3 merge) достаточно чтобы вытянуть от-нуля за
   ~50 эпох — у нас же не миллион кадров чтобы экономить compute через
   fine-tune.

Альтернатива: **fine-tune от существующего `weights/maskrcnn_astana.pt`**
если он уже есть. Тоже валидный путь, обычно сходится быстрее (~30 эпох
вместо 50). Если хочешь, сделай оба ран'а в разные имена
(`maskrcnn_v3_scratch.pt` и `maskrcnn_v3_finetune.pt`) — получишь
дополнительную строчку для ablation-таблицы диплома (fine-tune vs scratch
на одной модели = ML-методология).

### Anuar — DeepForest + SAM 2

DF не тренировался на нашем датасете (использует pretrained NEON). Для v3
у тебя два варианта:

1. **Оставить как есть** — DeepForest как был, fallback на pretrained
   `weecology/deepforest-tree`. Тестируй inference на v3 val вместе с
   остальными для comparison table.
2. **Fine-tune DF на v3** — если хочешь. У DF свой формат, см.
   `deepforest.main.deepforest.create_trainer()`. Не наша часть пайплайна,
   но если соберёшь — добавим в ablation.

## Eval baseline (общая точка отсчёта)

Все три модели сравниваются на **одном merged-val** (`v3_merged/instances_Validation.json`,
15 image / ~726 polygons после полной разметки v3). Хранится в коммитах после
финализации v3 разметки.

Текущий baseline (на чистом v1+v2 merged-val, до v3):

| Модель | Box mAP@50 | Mask mAP@50 | Weights |
|---|---|---|---|
| YOLO v1 (yolov8x, 397 ep) | 0.265 | 0.240 | `runs/segment/astana_tiled_x_max/` |
| YOLO v2 from-scratch (204 ep) | 0.319 | 0.288 | `runs/segment/astana_tiled_x_v2_fromscratch/` |
| **YOLO v2 fine-tune (173 ep) — production** | **0.372** | **0.331** | `runs/segment/astana_tiled_x_v2_finetune/` |
| Mask R-CNN (50 ep, v1+v2) | TBD | target ≥ 0.45 | `weights/maskrcnn_astana.pt` |
| DeepForest (pretrained NEON) | TBD | — | hub: `weecology/deepforest-tree` |

После того как все натренируются на v3 datasete — заполнить таблицу в
`docs/maskrcnn.md` и в дипломе (Chapter 4 / Ablation).

## Гитнор-правила

```
# Generated artifacts (regenerate via scripts above):
v3_merged/
v3_yolo/
v3_yolo_tiled/

# Splits inside v3 annotations/ (regenerated by split_coco.py):
v3 annotations/annotations/instances_Train.json
v3 annotations/annotations/instances_Validation.json

# Original v3 photos + instances_default.json — committed.
```

См. корневой `.gitignore`.
