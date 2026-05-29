"""Собрать 14-image COCO val JSON для apples-to-apples eval YOLO/DF/MRCNN.

Источники:
  - annotations_merged/instances_Validation.json (10 imgs: 5 v1 + 5 v2)
  - v3 annotations/annotations/instances_Validation.json (5 v3 imgs)

Исключение: «Снимок экрана 2026-04-01 194422.png» (был в YOLO train corpus per
--dup-policy keep-train), чтобы избежать leakage в этом eval.

Итог: 14 imgs (4 v1 + 5 v2 + 5 v3) с непрерывными image_id / annotation_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC_V1V2 = ROOT / "yolov train dataset" / "annotations_merged" / "instances_Validation.json"
SRC_V3 = ROOT / "yolov train dataset" / "v3 annotations" / "annotations" / "instances_Validation.json"
OUT = ROOT / "yolov train dataset" / "annotations_merged_14img_val.json"
EXCLUDE_NAMES = {"Снимок экрана 2026-04-01 194422.png"}


def load(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    d1 = load(SRC_V1V2)
    d2 = load(SRC_V3)
    print(f"v1+v2 source: {len(d1['images'])} imgs, {len(d1['annotations'])} anns")
    print(f"v3 source:    {len(d2['images'])} imgs, {len(d2['annotations'])} anns")

    # Canonical category (taking first source as authoritative)
    cats = d1["categories"]
    cat_id = cats[0]["id"]

    merged_images = []
    merged_anns = []
    next_img_id = 1
    next_ann_id = 1

    def take(d: dict, label: str) -> int:
        nonlocal next_img_id, next_ann_id
        kept = 0
        old_to_new_img = {}
        for img in d["images"]:
            if img["file_name"] in EXCLUDE_NAMES:
                print(f"  EXCLUDE {label}: {img['file_name']}")
                continue
            old = img["id"]
            new = next_img_id
            next_img_id += 1
            old_to_new_img[old] = new
            merged_images.append({**img, "id": new})
            kept += 1
        for ann in d["annotations"]:
            if ann["image_id"] not in old_to_new_img:
                continue
            new_img = old_to_new_img[ann["image_id"]]
            merged_anns.append({
                **ann,
                "id": next_ann_id,
                "image_id": new_img,
                "category_id": cat_id,
            })
            next_ann_id += 1
        return kept

    n1 = take(d1, "v1+v2")
    n2 = take(d2, "v3")

    out = {
        "info": {"description": "14-image merged val (4 v1 + 5 v2 + 5 v3, excluding 194422.png)"},
        "licenses": d1.get("licenses", []),
        "categories": cats,
        "images": merged_images,
        "annotations": merged_anns,
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\nKept: {n1} from v1+v2, {n2} from v3 → total {len(merged_images)} imgs, {len(merged_anns)} anns")
    print(f"Saved → {OUT.relative_to(ROOT)}")

    # Sanity: per-image polygon count
    from collections import Counter
    cnt = Counter(a["image_id"] for a in merged_anns)
    print("\nPer-image polygon counts:")
    for img in merged_images:
        print(f"  id={img['id']:2}  polys={cnt[img['id']]:3}  {img['file_name']}")


if __name__ == "__main__":
    main()
