"""Объединяет несколько COCO 1.0 JSON в один. Перенумеровывает image-id и
annotation-id чтобы не конфликтовали. Категории дедуплицируются по имени.

Использовать когда есть отдельные экспорты CVAT (старый task v1 + новый task v2)
и нужно собрать единый train/val JSON для ml/coco_to_yolo_seg.py.

Пример (по одному merge на каждый сплит):
    python ml/merge_coco.py \
        --inputs "yolov train dataset/annotations/instances_Train.json" \
                 "yolov train dataset/annotations_v2_export/instances_Train.json" \
        --output "yolov train dataset/annotations_merged/instances_Train.json"
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path,
                        help="Два или больше COCO JSON файла")
    parser.add_argument("--output", required=True, type=Path,
                        help="Куда записать слитый COCO JSON")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        sys.exit("Need at least 2 input files to merge")
    for p in args.inputs:
        if not p.exists():
            sys.exit(f"Input not found: {p}")

    merged: dict = {
        "info": {"description": "Merged: " + " + ".join(p.name for p in args.inputs)},
        "licenses": [{"id": 0, "name": "Unknown", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    cat_name_to_id: dict[str, int] = {}
    seen_filenames: dict[str, int] = {}
    next_img_id = 1
    next_ann_id = 1

    for src_idx, path in enumerate(args.inputs):
        print(f"Reading [{src_idx + 1}] {path}")
        with path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_remap: dict[int, int] = {}
        for cat in coco.get("categories", []):
            name = cat["name"]
            if name not in cat_name_to_id:
                new_id = len(merged["categories"]) + 1
                cat_name_to_id[name] = new_id
                merged["categories"].append({
                    "id": new_id,
                    "name": name,
                    "supercategory": cat.get("supercategory", ""),
                })
            cat_remap[cat["id"]] = cat_name_to_id[name]

        img_remap: dict[int, int] = {}
        skipped = 0
        for img in coco.get("images", []):
            fname = img["file_name"]
            if fname in seen_filenames:
                print(f"  [!] duplicate filename '{fname}' "
                      f"(also in input #{seen_filenames[fname] + 1}) — skipping")
                skipped += 1
                continue
            seen_filenames[fname] = src_idx
            new_img = dict(img)
            new_img["id"] = next_img_id
            img_remap[img["id"]] = next_img_id
            merged["images"].append(new_img)
            next_img_id += 1

        kept = 0
        for ann in coco.get("annotations", []):
            if ann["image_id"] not in img_remap:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = img_remap[ann["image_id"]]
            new_ann["category_id"] = cat_remap[ann["category_id"]]
            merged["annotations"].append(new_ann)
            next_ann_id += 1
            kept += 1
        print(f"  -> +{len(img_remap)} images (skipped {skipped} dups), +{kept} annotations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nMerged: {len(merged['images'])} images, "
          f"{len(merged['annotations'])} annotations, "
          f"{len(merged['categories'])} categories")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
