"""Объединить старый LabelMe-датасет (134 кадра) и новый CVAT-датасет (15 кадров)
в один YOLOv8 датасет для финальной тренировки.

Пример:
    python ml/merge_datasets.py \
        --old C:/Users/Rasul/DeepLearning/pipeline/yolov8seg/dataset \
        --new data/processed/cvat_15 \
        --output data/processed/combined \
        --train-ratio 0.85
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def collect_pairs(src: Path) -> list[tuple[Path, Path]]:
    """src/images/{train,val}/*.jpg + src/labels/{train,val}/*.txt → список пар."""
    pairs = []
    for split in ("train", "val"):
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="Существующий YOLO датасет (134 кадра)")
    parser.add_argument("--new", required=True, help="Новый YOLO датасет из CVAT (15 кадров)")
    parser.add_argument("--output", required=True, help="Куда писать объединённый")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    pairs_old = collect_pairs(Path(args.old))
    pairs_new = collect_pairs(Path(args.new))
    print(f"Old: {len(pairs_old)} pairs · New: {len(pairs_new)} pairs · Total: {len(pairs_old) + len(pairs_new)}")

    # Префикс имён чтобы избежать коллизий
    all_pairs = [("old_" + p[0].stem, p) for p in pairs_old] + [("new_" + p[0].stem, p) for p in pairs_new]

    random.seed(args.seed)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * args.train_ratio)

    for split_name, items in [("train", all_pairs[:split_idx]), ("val", all_pairs[split_idx:])]:
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for prefix, (img, lbl) in items:
            shutil.copy2(img, img_out / f"{prefix}{img.suffix}")
            shutil.copy2(lbl, lbl_out / f"{prefix}.txt")
        print(f"  {split_name}: {len(items)} pairs")

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
    print(f"\nMerged dataset → {out.resolve()}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
