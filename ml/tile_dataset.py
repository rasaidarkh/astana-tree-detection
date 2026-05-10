"""Тайлинг YOLOv8-seg датасета: режет большие снимки на тайлы фиксированного
размера и пере-проецирует полигональные аннотации на тайлы.

Зачем: на спутниковых снимках кроны крошечные (20–40 px). Тренировка на
imgsz=640 жмёт картинку 1711x1135 в 640x424, и кроны становятся ~7 px —
ниже порога детектируемости. Тайлинг позволяет тренировать на 640 без
потери разрешения.

Вход: YOLO-датасет (images/{train,val} + labels/{train,val} + dataset.yaml)
Выход: тот же формат, но с тайлами.

Пример:
    python ml/tile_dataset.py \
        --input  "yolov train dataset/yolo" \
        --output "yolov train dataset/yolo_tiled" \
        --tile-size 640 --overlap 128 --min-area 25
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from shapely.geometry import Polygon, box
from shapely.validation import make_valid


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, Polygon]]:
    """Читает YOLO-seg .txt -> [(class_id, shapely.Polygon в пиксельных коорд)]."""
    if not label_path.exists():
        return []
    polys = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        pts = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
        if len(pts) < 3:
            continue
        try:
            p = Polygon(pts)
            if not p.is_valid:
                p = make_valid(p)
            if p.is_empty or p.area < 1:
                continue
            polys.append((cls, p))
        except Exception:
            continue
    return polys


def write_yolo_label(out_path: Path, polys: list[tuple[int, list[tuple[float, float]]]], tw: int, th: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, pts in polys:
        norm = []
        for x, y in pts:
            nx = max(0.0, min(1.0, x / tw))
            ny = max(0.0, min(1.0, y / th))
            norm.extend([nx, ny])
        coords = " ".join(f"{c:.6f}" for c in norm)
        lines.append(f"{cls} {coords}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def clip_polygons_to_tile(
    polys: list[tuple[int, Polygon]],
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    min_area: float,
) -> list[tuple[int, list[tuple[float, float]]]]:
    """Пересекает полигоны с границами тайла, переносит в локальные координаты."""
    tile_box = box(tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)
    out: list[tuple[int, list[tuple[float, float]]]] = []
    for cls, p in polys:
        if not p.intersects(tile_box):
            continue
        clipped = p.intersection(tile_box)
        if clipped.is_empty:
            continue
        # MultiPolygon -> разбираем по частям
        geoms = []
        if clipped.geom_type == "Polygon":
            geoms = [clipped]
        elif clipped.geom_type == "MultiPolygon":
            geoms = list(clipped.geoms)
        elif clipped.geom_type == "GeometryCollection":
            geoms = [g for g in clipped.geoms if g.geom_type == "Polygon"]
        for g in geoms:
            if g.area < min_area:
                continue
            ext = list(g.exterior.coords)
            # shapely замыкает кольцо — последняя точка == первая, у YOLO лишняя
            if len(ext) > 1 and ext[0] == ext[-1]:
                ext = ext[:-1]
            if len(ext) < 3:
                continue
            local = [(x - tile_x, y - tile_y) for x, y in ext]
            out.append((cls, local))
    return out


def tile_split(
    in_root: Path,
    out_root: Path,
    split: str,
    tile_size: int,
    overlap: int,
    min_area: float,
    keep_empty: bool,
) -> tuple[int, int, int]:
    img_dir = in_root / "images" / split
    lbl_dir = in_root / "labels" / split
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        return 0, 0, 0

    stride = tile_size - overlap
    n_src = 0
    n_tiles = 0
    n_polys = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        n_src += 1
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        polys = load_yolo_labels(lbl_dir / (img_path.stem + ".txt"), w, h)

        # сетка: последний ряд/колонка докатываются вплотную к краю
        ys = list(range(0, max(1, h - tile_size + 1), stride))
        if not ys or ys[-1] + tile_size < h:
            ys.append(max(0, h - tile_size))
        xs = list(range(0, max(1, w - tile_size + 1), stride))
        if not xs or xs[-1] + tile_size < w:
            xs.append(max(0, w - tile_size))

        for tile_idx_y, ty in enumerate(ys):
            for tile_idx_x, tx in enumerate(xs):
                tw = min(tile_size, w - tx)
                th = min(tile_size, h - ty)
                clipped = clip_polygons_to_tile(polys, tx, ty, tw, th, min_area)
                if not clipped and not keep_empty:
                    continue

                tile_img = img.crop((tx, ty, tx + tw, ty + th))
                tile_name = f"{img_path.stem}__y{ty:04d}_x{tx:04d}{img_path.suffix.lower()}"
                tile_img.save(out_img / tile_name)
                write_yolo_label(out_lbl / (Path(tile_name).stem + ".txt"), clipped, tw, th)
                n_tiles += 1
                n_polys += len(clipped)

    return n_src, n_tiles, n_polys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Папка с YOLO-датасетом (images/, labels/, dataset.yaml)")
    p.add_argument("--output", required=True)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--min-area", type=float, default=25.0,
                   help="Минимум пикселей для оставленного фрагмента полигона после клипа")
    p.add_argument("--keep-empty", action="store_true",
                   help="Сохранять тайлы без аннотаций (полезно как hard negatives, но раздувает датасет)")
    args = p.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    print(f"Tiling {in_root} -> {out_root}")
    print(f"  size={args.tile_size}, overlap={args.overlap}, min_area={args.min_area}, keep_empty={args.keep_empty}")
    print()

    for split in ["train", "val"]:
        n_src, n_tiles, n_polys = tile_split(
            in_root, out_root, split, args.tile_size, args.overlap, args.min_area, args.keep_empty
        )
        print(f"  {split}: {n_src} src -> {n_tiles} tiles, {n_polys} polygons")

    # Сгенерим обновлённый dataset.yaml
    src_yaml = in_root / "dataset.yaml"
    names_block = "  0: tree"
    nc = 1
    if src_yaml.exists():
        # очень простой парсинг чтобы вытащить names — не тащим pyyaml
        in_names = False
        names_lines = []
        for line in src_yaml.read_text(encoding="utf-8").splitlines():
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if line.startswith(" ") or line.startswith("\t"):
                    names_lines.append(line)
                else:
                    in_names = False
            if line.startswith("nc:"):
                try:
                    nc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        if names_lines:
            names_block = "\n".join(names_lines)

    (out_root / "dataset.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names_block}\n\n"
        f"nc: {nc}\n",
        encoding="utf-8",
    )

    # перенесём filename_map.json если есть (для дебага)
    fmap = in_root / "filename_map.json"
    if fmap.exists():
        shutil.copy2(fmap, out_root / "filename_map.json")

    print(f"\nYAML -> {out_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
