"""COCO (pre-split train/val) -> YOLOv8-seg.

Use this when you exported from CVAT as TWO COCO JSONs (one per subset) and want
to keep that split as-is — no random shuffle. Filenames are sanitized to ASCII
because Ultralytics is sometimes flaky with non-Latin paths.

Example:
    python ml/coco_to_yolo_seg.py \
        --train-coco "yolov train dataset/annotations/instances_Train.json" \
        --val-coco   "yolov train dataset/annotations/instances_Validation.json" \
        --images-dir "yolov train dataset/фотографии" \
        --output     "yolov train dataset/yolo"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_coco(json_path: Path) -> tuple[list[dict], dict[int, str]]:
    """Возвращает (records, cat_id_to_name).

    record = {filename, width, height, polygons:[{cat_id, points:[(x,y),...]}]}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_image: dict[int, dict] = {
        img["id"]: {
            "filename": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "polygons": [],
        }
        for img in data["images"]
    }

    for ann in data["annotations"]:
        rec = by_image.get(ann["image_id"])
        if rec is None:
            continue
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and seg:
            for poly_flat in seg:
                if len(poly_flat) < 6:
                    continue
                points = [(poly_flat[i], poly_flat[i + 1]) for i in range(0, len(poly_flat), 2)]
                rec["polygons"].append({"cat_id": ann["category_id"], "points": points})
        elif "bbox" in ann:
            x, y, w, h = ann["bbox"]
            rec["polygons"].append(
                {
                    "cat_id": ann["category_id"],
                    "points": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                }
            )

    return list(by_image.values()), cats


def write_yolo_label(out_path: Path, polygons: list[dict], width: int, height: int, cat_to_idx: dict[int, int]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for poly in polygons:
        cls = cat_to_idx[poly["cat_id"]]
        norm = []
        for x, y in poly["points"]:
            nx = max(0.0, min(1.0, x / width))
            ny = max(0.0, min(1.0, y / height))
            norm.extend([nx, ny])
        coords = " ".join(f"{c:.6f}" for c in norm)
        lines.append(f"{cls} {coords}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-coco", required=True)
    p.add_argument("--val-coco", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--dup-policy",
        choices=["keep-train", "keep-val", "drop-both"],
        default="keep-train",
        help="что делать с файлом, попавшим и в train и в val",
    )
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    out = Path(args.output)

    train_records, train_cats = parse_coco(Path(args.train_coco))
    val_records, val_cats = parse_coco(Path(args.val_coco))

    cats = {**train_cats, **val_cats}
    cat_to_idx = {cid: i for i, cid in enumerate(sorted(cats))}
    print(f"Categories: {cats} -> indices {cat_to_idx}")

    train_files = {r["filename"] for r in train_records}
    val_files = {r["filename"] for r in val_records}
    dup = train_files & val_files
    if dup:
        print(f"\n[!] {len(dup)} duplicate(s) in both splits — applying policy '{args.dup_policy}':")
        for d in dup:
            print(f"    {d}")
        if args.dup_policy == "keep-train":
            val_records = [r for r in val_records if r["filename"] not in dup]
        elif args.dup_policy == "keep-val":
            train_records = [r for r in train_records if r["filename"] not in dup]
        else:
            train_records = [r for r in train_records if r["filename"] not in dup]
            val_records = [r for r in val_records if r["filename"] not in dup]

    name_map: dict[str, str] = {}
    skipped: list[str] = []

    for split_name, records in [("train", train_records), ("val", val_records)]:
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        kept = 0
        for i, rec in enumerate(sorted(records, key=lambda r: r["filename"]), start=1):
            src = images_dir / rec["filename"]
            if not src.exists():
                # try basename match for nested folders
                candidates = list(images_dir.rglob(Path(rec["filename"]).name))
                if not candidates:
                    skipped.append(f"{split_name}/{rec['filename']}")
                    continue
                src = candidates[0]

            ext = src.suffix.lower()
            new_name = f"img_{split_name}_{i:03d}{ext}"
            shutil.copy2(src, img_out / new_name)
            write_yolo_label(
                lbl_out / (Path(new_name).stem + ".txt"),
                rec["polygons"],
                rec["width"],
                rec["height"],
                cat_to_idx,
            )
            name_map[f"{split_name}/{new_name}"] = rec["filename"]
            kept += 1

        print(f"  {split_name}: {kept} images, {sum(len(r['polygons']) for r in records if (images_dir / r['filename']).exists() or list(images_dir.rglob(Path(r['filename']).name)))} polygons")

    if skipped:
        print(f"\n[!] {len(skipped)} image(s) skipped (file not found):")
        for s in skipped:
            print(f"    {s}")

    (out / "filename_map.json").write_text(
        json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    names_block = "\n".join(f"  {idx}: tree" for idx in cat_to_idx.values())
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names_block}\n\n"
        f"nc: {len(cat_to_idx)}\n",
        encoding="utf-8",
    )

    print(f"\nDataset -> {out.resolve()}")
    print(f"YAML    -> {out / 'dataset.yaml'}")
    print(f"Map     -> {out / 'filename_map.json'}")
    print(f"\nNext: python ml/train_yolo.py --data \"{out / 'dataset.yaml'}\"")


if __name__ == "__main__":
    main()
