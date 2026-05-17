"""COCO (1.0) → DeepForest training CSV.

DeepForest ждёт CSV формата `image_path,xmin,ymin,xmax,ymax,label`, по одной
строке на bbox. Polygon-сегментация игнорируется (DF — детектор, не сегментер):
для каждой COCO-аннотации берём её `bbox` поле (xywh) и переводим в xyxy.

`label` для всех берётся одинаковый — `Tree` (английский, чтобы не возиться
с unicode в DeepForest internals). Картинки с нулевым bbox-ом пропускаются.

Пример:
    python ml/coco_to_deepforest_csv.py \
        --train-coco "yolov train dataset/v3_merged/instances_Train.json" \
        --val-coco   "yolov train dataset/v3_merged/instances_Validation.json" \
        --root-dir   "yolov train dataset/v3_merged/images" \
        --output-dir "yolov train dataset/v3_deepforest"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def coco_to_rows(json_path: Path, label: str = "Tree") -> tuple[list[dict], int]:
    """Возвращает (список dict-строк для CSV, число пропущенных без bbox)."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_id = {img["id"]: img["file_name"] for img in data["images"]}
    rows: list[dict] = []
    skipped = 0
    for ann in data["annotations"]:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            skipped += 1
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        filename = by_id.get(ann["image_id"])
        if not filename:
            skipped += 1
            continue
        rows.append({
            "image_path": filename,
            "xmin": round(x, 2),
            "ymin": round(y, 2),
            "xmax": round(x + w, 2),
            "ymax": round(y + h, 2),
            "label": label,
        })
    return rows, skipped


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-coco", required=True, type=Path)
    p.add_argument("--val-coco", required=True, type=Path)
    p.add_argument("--root-dir", required=True, type=Path,
                   help="Папка где лежат PNG. DeepForest резолвит image_path как root_dir/<image_path>")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--label", default="Tree",
                   help="Класс. Default 'Tree' (английский, чтобы не возиться с unicode)")
    args = p.parse_args()

    for json_path in (args.train_coco, args.val_coco):
        if not json_path.exists():
            sys.exit(f"COCO not found: {json_path}")
    if not args.root_dir.exists():
        sys.exit(f"Image root not found: {args.root_dir}")

    train_rows, train_skip = coco_to_rows(args.train_coco, label=args.label)
    val_rows, val_skip = coco_to_rows(args.val_coco, label=args.label)

    train_csv = args.output_dir / "train.csv"
    val_csv = args.output_dir / "val.csv"
    write_csv(train_rows, train_csv)
    write_csv(val_rows, val_csv)

    train_imgs = len(set(r["image_path"] for r in train_rows))
    val_imgs = len(set(r["image_path"] for r in val_rows))

    print(f"Train CSV → {train_csv}")
    print(f"  {len(train_rows)} bboxes across {train_imgs} images (skipped {train_skip})")
    print(f"Val   CSV → {val_csv}")
    print(f"  {len(val_rows)} bboxes across {val_imgs} images (skipped {val_skip})")
    print(f"\nUse with DeepForest:")
    print(f"  model.config['train']['csv_file'] = '{train_csv}'")
    print(f"  model.config['train']['root_dir'] = '{args.root_dir}'")
    print(f"  model.config['validation']['csv_file'] = '{val_csv}'")
    print(f"  model.config['validation']['root_dir'] = '{args.root_dir}'")


if __name__ == "__main__":
    main()
