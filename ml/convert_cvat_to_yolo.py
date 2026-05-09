"""Конвертер аннотаций CVAT (XML 1.1 или COCO/JSON) → YOLOv8-seg формат.

CVAT экспортирует разметку в нескольких форматах. Скрипт поддерживает оба самых
ходовых:
  1. CVAT for Images 1.1 (XML)         — выбираем при экспорте "CVAT for images 1.1"
  2. COCO 1.0 (JSON)                   — выбираем при экспорте "COCO 1.0"

YOLOv8-seg требует:
  dataset/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt    # одна строка = "class_id x1 y1 x2 y2 ... xn yn" (нормализованные)

Пример:
    python ml/convert_cvat_to_yolo.py \
        --annotations data/raw/cvat_export.xml \
        --images data/raw/images/ \
        --output data/processed/cvat_15/ \
        --train-ratio 0.85
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {"Tree": 0, "tree": 0, "TREE": 0}


def parse_cvat_xml(xml_path: Path) -> list[dict]:
    """Парсит CVAT XML 1.1 → список {filename, width, height, polygons:[{class, points}]}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = []

    for img_node in root.findall("image"):
        name = img_node.get("name")
        width = int(img_node.get("width"))
        height = int(img_node.get("height"))
        polygons = []

        for poly in img_node.findall("polygon"):
            label = poly.get("label", "")
            if label not in CLASS_MAP:
                continue
            points_str = poly.get("points", "")
            # CVAT format: "x1,y1;x2,y2;..."
            points = [tuple(map(float, p.split(","))) for p in points_str.split(";") if p]
            if len(points) < 3:
                continue
            polygons.append({"class": CLASS_MAP[label], "points": points})

        # Также bounding boxes — конвертируем в прямоугольный полигон
        for box in img_node.findall("box"):
            label = box.get("label", "")
            if label not in CLASS_MAP:
                continue
            x1 = float(box.get("xtl"))
            y1 = float(box.get("ytl"))
            x2 = float(box.get("xbr"))
            y2 = float(box.get("ybr"))
            polygons.append(
                {
                    "class": CLASS_MAP[label],
                    "points": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                }
            )

        images.append({"filename": name, "width": width, "height": height, "polygons": polygons})

    return images


def parse_coco_json(json_path: Path) -> list[dict]:
    """COCO 1.0 → список как из parse_cvat_xml."""
    with open(json_path) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_image: dict[int, dict] = {
        img["id"]: {"filename": img["file_name"], "width": img["width"], "height": img["height"], "polygons": []}
        for img in data["images"]
    }

    for ann in data["annotations"]:
        if ann["image_id"] not in by_image:
            continue
        cls_name = cats.get(ann["category_id"])
        if cls_name not in CLASS_MAP:
            continue
        seg = ann.get("segmentation", [])
        # COCO segmentation: list of polygons, each as flat [x1,y1,x2,y2,...]
        if isinstance(seg, list) and seg:
            for poly_flat in seg:
                if len(poly_flat) < 6:
                    continue
                points = [(poly_flat[i], poly_flat[i + 1]) for i in range(0, len(poly_flat), 2)]
                by_image[ann["image_id"]]["polygons"].append(
                    {"class": CLASS_MAP[cls_name], "points": points}
                )
        elif "bbox" in ann:
            x, y, w, h = ann["bbox"]
            by_image[ann["image_id"]]["polygons"].append(
                {"class": CLASS_MAP[cls_name], "points": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]}
            )

    return list(by_image.values())


def write_yolo_label(out_path: Path, polygons: list[dict], width: int, height: int):
    """Записывает YOLOv8-seg файл с нормализованными координатами."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for poly in polygons:
        cls = poly["class"]
        norm = []
        for x, y in poly["points"]:
            nx = max(0.0, min(1.0, x / width))
            ny = max(0.0, min(1.0, y / height))
            norm.extend([nx, ny])
        coords = " ".join(f"{c:.6f}" for c in norm)
        lines.append(f"{cls} {coords}")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="CVAT → YOLOv8-seg converter")
    parser.add_argument("--annotations", required=True, help="CVAT XML или COCO JSON")
    parser.add_argument("--images", required=True, help="Папка с исходными снимками")
    parser.add_argument("--output", required=True, help="Куда писать YOLO датасет")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations = Path(args.annotations)
    images_dir = Path(args.images)
    out = Path(args.output)

    if annotations.suffix.lower() == ".xml":
        records = parse_cvat_xml(annotations)
    elif annotations.suffix.lower() == ".json":
        records = parse_coco_json(annotations)
    else:
        raise ValueError(f"Unsupported annotation format: {annotations.suffix}")

    print(f"Parsed {len(records)} image records, {sum(len(r['polygons']) for r in records)} total polygons")

    random.seed(args.seed)
    random.shuffle(records)
    split_idx = int(len(records) * args.train_ratio)

    for split_name, split_records in [("train", records[:split_idx]), ("val", records[split_idx:])]:
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for rec in split_records:
            src = images_dir / rec["filename"]
            if not src.exists():
                # Иногда CVAT в filename кладёт относительный путь
                candidates = list(images_dir.rglob(Path(rec["filename"]).name))
                if not candidates:
                    print(f"  [skip] {rec['filename']} not found")
                    continue
                src = candidates[0]
            shutil.copy2(src, img_out / src.name)
            write_yolo_label(lbl_out / (Path(src.name).stem + ".txt"), rec["polygons"], rec["width"], rec["height"])

        print(f"  {split_name}: {len(split_records)} images")

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: tree\n"
        f"\n"
        f"nc: 1\n"
    )
    print(f"\nDataset written to {out.resolve()}")
    print(f"YAML config: {yaml_path}")
    print(f"\nNext step: python ml/train_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()
