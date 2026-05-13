"""Разделяет COCO 1.0 JSON на train/val по числу val или доле.

Зачем: после CVAT экспорта получается один instances_default.json без
train/val разделения. ml/merge_coco.py ждёт уже разделённые файлы.

Пример:
    python ml/split_coco.py \
        --input  "yolov train dataset/new annotations/annotations/instances_default.cleaned.json" \
        --train  "yolov train dataset/new annotations/annotations/instances_Train.json" \
        --val    "yolov train dataset/new annotations/annotations/instances_Validation.json" \
        --val-count 5 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _subset(coco: dict, image_ids: set[int]) -> dict:
    images = [im for im in coco["images"] if im["id"] in image_ids]
    annotations = [a for a in coco["annotations"] if a["image_id"] in image_ids]
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--val-count", type=int, help="Сколько изображений в val (фиксированное число)")
    group.add_argument("--val-frac", type=float, help="Доля val, например 0.15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    imgs = list(coco["images"])
    if not imgs:
        sys.exit("Empty images list")

    rng = random.Random(args.seed)
    rng.shuffle(imgs)

    if args.val_count is not None:
        n_val = args.val_count
    else:
        n_val = max(1, int(round(len(imgs) * args.val_frac)))
    if n_val >= len(imgs):
        sys.exit(f"val_count {n_val} >= total images {len(imgs)}")

    val_imgs = imgs[:n_val]
    train_imgs = imgs[n_val:]
    val_ids = {im["id"] for im in val_imgs}
    train_ids = {im["id"] for im in train_imgs}

    train_coco = _subset(coco, train_ids)
    val_coco = _subset(coco, val_ids)

    args.train.parent.mkdir(parents=True, exist_ok=True)
    args.val.parent.mkdir(parents=True, exist_ok=True)
    with args.train.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    with args.val.open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    print(f"Total: {len(imgs)} images, {len(coco['annotations'])} annotations")
    print(f"  train -> {args.train}")
    print(f"    images={len(train_coco['images'])}, annotations={len(train_coco['annotations'])}")
    print(f"  val   -> {args.val}")
    print(f"    images={len(val_coco['images'])}, annotations={len(val_coco['annotations'])}")
    print(f"\nVal files (seed={args.seed}):")
    for im in val_imgs:
        print(f"  {im['file_name']}")


if __name__ == "__main__":
    main()
