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

### Anuar — DeepForest fine-tune на merged v1+v2+v3

DeepForest это **детектор** (только bbox-ы, без сегментации). У него
свой формат тренировочного файла — CSV, не COCO. Поэтому нужна
**конверсия polygon → bbox**: берём `bbox` поле из COCO-аннотации (xywh)
и переводим в xyxy → пишем в CSV `image_path,xmin,ymin,xmax,ymax,label`.

Скрипт уже готов в `ml/coco_to_deepforest_csv.py`. polygon-сегментация
просто игнорируется (DF её не использует).

```bash
# Шаги 1-2 — тот же сплит + merge что у Берика:
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

# Собрать все PNG в одну папку — DF резолвит image_path относительно root_dir:
mkdir -p "yolov train dataset/v3_merged/images"
cp -n "yolov train dataset/фотографии/"*.png             "yolov train dataset/v3_merged/images/"
cp -n "yolov train dataset/новые фотографии/"*.png       "yolov train dataset/v3_merged/images/"
cp -n "yolov train dataset/v3 фотографии для finetune/"*.png "yolov train dataset/v3_merged/images/"

# 3. Конверсия COCO → DeepForest CSV (polygon → bbox)
python ml/coco_to_deepforest_csv.py \
    --train-coco "yolov train dataset/v3_merged/instances_Train.json" \
    --val-coco   "yolov train dataset/v3_merged/instances_Validation.json" \
    --root-dir   "yolov train dataset/v3_merged/images" \
    --output-dir "yolov train dataset/v3_deepforest"

# 4. Fine-tune DeepForest от pretrained NEON backbone.
# DeepForest нет готового train-скрипта в нашем repo — пишется ad-hoc,
# minimal-пример внизу. Сохрани его как ml/train_deepforest.py.

python ml/train_deepforest.py \
    --train-csv "yolov train dataset/v3_deepforest/train.csv" \
    --val-csv   "yolov train dataset/v3_deepforest/val.csv" \
    --root-dir  "yolov train dataset/v3_merged/images" \
    --output    "weights/deepforest_astana.pl" \
    --epochs 30 --batch-size 1
```

**Минимальный `ml/train_deepforest.py` (под который параметры выше) —
напиши примерно так:**

```python
# ml/train_deepforest.py
import argparse
from pathlib import Path
from deepforest import main as df_main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--root-dir", required=True)
    p.add_argument("--output", default="weights/deepforest_astana.pl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    a = p.parse_args()

    m = df_main.deepforest()
    m.load_model(model_name="weecology/deepforest-tree", revision="main")

    m.config["train"]["csv_file"] = a.train_csv
    m.config["train"]["root_dir"] = a.root_dir
    m.config["train"]["lr"] = a.lr
    m.config["validation"]["csv_file"] = a.val_csv
    m.config["validation"]["root_dir"] = a.root_dir
    m.config["batch_size"] = a.batch_size
    m.config["train"]["epochs"] = a.epochs

    m.create_trainer()
    m.trainer.fit(m)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.output)
    print(f"saved → {a.output}")

if __name__ == "__main__":
    main()
```

**Что важно для Anuar:**
- DeepForest использует RetinaNet backbone, `score_thresh=0.1` default,
  `patch_size=400` (это всё в `venv/Lib/site-packages/deepforest/conf/config.yaml`).
- Веса сохраняй как `.pl` (PyTorch Lightning checkpoint) — это формат
  который ожидает наш `DeepForestAdapter` через `torch.load → state_dict`.
- Tiled inference (`patch_size=400`, `patch_overlap=0.05`) уже встроено в
  DF — после fine-tune он будет работать на больших Astana-снимках
  правильно через `model.predict_tile()`.
- **SAM 2** это inference-time refiner крон, не train-time компонент.
  После fine-tune DF, SAM 2 продолжает работать с новым DF-bbox-ом тем
  же способом (см. `backend/models/deepforest_sam2_adapter.py`). Не нужно
  трогать SAM 2 для v3.

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
