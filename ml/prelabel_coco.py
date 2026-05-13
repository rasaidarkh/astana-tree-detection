"""Pre-label images with YOLOv8-seg via TILED inference, output COCO 1.0 JSON
that CVAT can ingest as initial annotations.

Зачем тайлить: модель тренировалась на 640-px тайлах (см. ml/tile_dataset.py),
кроны там ~20-40 px. Если предиктить весь снимок 1700x1100 на imgsz=640,
ultralytics ужмёт его до 640x424 — кроны станут ~7 px и большая часть пропадёт.
Тайлим как при тренировке, склеиваем результат с глобальным NMS.

Пример:
    python ml/prelabel_coco.py \
        --images "yolov train dataset/новые фотографии" \
        --weights runs/segment/astana_tiled_x_max/weights/best.pt \
        --output "yolov train dataset/prelabel_v2.json" \
        --conf 0.20

Дальше в CVAT: создаёшь task с этими снимками, Actions -> Upload annotations
-> COCO 1.0 -> prelabel_v2.json. Полигоны лягут пред-расставленными.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> list[int]:
    """Vanilla NMS over xyxy boxes. Возвращает индексы оставленных."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[1:][iou < iou_thresh]
    return keep


def predict_tiled(model, img_path: Path, tile_size: int, overlap: int, conf: float):
    """Slide window over image, predict each tile, deduplicate via NMS."""
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    stride = max(1, tile_size - overlap)

    if W <= tile_size and H <= tile_size:
        x_origins, y_origins = [0], [0]
    else:
        x_origins = list(range(0, max(1, W - overlap), stride))
        y_origins = list(range(0, max(1, H - overlap), stride))
        if x_origins[-1] + tile_size < W:
            x_origins.append(max(0, W - tile_size))
        if y_origins[-1] + tile_size < H:
            y_origins.append(max(0, H - tile_size))

    polys: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    scores: list[float] = []

    for y0 in y_origins:
        for x0 in x_origins:
            x_end = min(x0 + tile_size, W)
            y_end = min(y0 + tile_size, H)
            tile = im.crop((x0, y0, x_end, y_end))
            results = model.predict(tile, imgsz=tile_size, conf=conf, verbose=False)
            r = results[0]
            if r.masks is None or r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for poly, box, score in zip(r.masks.xy, xyxy, confs):
                if poly.shape[0] < 3:
                    continue
                gp = poly.copy().astype(float)
                gp[:, 0] += x0
                gp[:, 1] += y0
                gb = box.copy().astype(float)
                gb[[0, 2]] += x0
                gb[[1, 3]] += y0
                polys.append(gp)
                boxes.append(gb)
                scores.append(float(score))

    if not boxes:
        return [], [], (W, H)

    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    keep = nms(boxes_np, scores_np, iou_thresh=0.5)
    return [polys[i] for i in keep], [boxes_np[i] for i in keep], (W, H)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path,
                        help="Папка с .png/.jpg для пре-разметки")
    parser.add_argument("--weights", required=True, type=Path,
                        help="YOLOv8-seg .pt чекпоинт")
    parser.add_argument("--output", required=True, type=Path,
                        help="Куда записать COCO 1.0 JSON")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Порог уверенности (ниже = больше recall, больше шума). "
                             "Удалить лишнее в CVAT проще чем добавить — берём с запасом.")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Размер тайла, должен совпадать с imgsz при тренировке")
    parser.add_argument("--overlap", type=int, default=128,
                        help="Перекрытие между тайлами для NMS-склейки границ")
    parser.add_argument("--class-id", type=int, default=1,
                        help="COCO category id (CVAT обычно ожидает 1-based)")
    parser.add_argument("--class-name", default="Дерево")
    args = parser.parse_args()

    if not args.images.exists():
        sys.exit(f"Images folder not found: {args.images}")
    if not args.weights.exists():
        sys.exit(f"Weights not found: {args.weights}")

    image_paths = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        sys.exit(f"No images found in {args.images}")

    from ultralytics import YOLO
    print(f"Loading model: {args.weights}")
    model = YOLO(str(args.weights))

    images: list[dict] = []
    annotations: list[dict] = []
    img_id = 1
    ann_id = 1
    total = 0

    for idx, img_path in enumerate(image_paths, 1):
        polys, boxes, (W, H) = predict_tiled(
            model, img_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
        )
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H,
            "license": 0,
        })
        for poly, box in zip(polys, boxes):
            poly_flat = poly.flatten().tolist()
            x1, y1, x2, y2 = box.tolist()
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": args.class_id,
                "segmentation": [poly_flat],
                "area": bw * bh,
                "bbox": [x1, y1, bw, bh],
                "iscrowd": 0,
            })
            ann_id += 1
        total += len(polys)
        print(f"[{idx:>3}/{len(image_paths)}] {img_path.name} ({W}x{H}) -> {len(polys)} dets")
        img_id += 1

    coco = {
        "info": {
            "description": "Pre-labels from YOLOv8-seg, tiled inference",
            "weights": str(args.weights),
            "conf": args.conf,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
        },
        "licenses": [{"id": 0, "name": "Unknown", "url": ""}],
        "categories": [{
            "id": args.class_id,
            "name": args.class_name,
            "supercategory": "",
        }],
        "images": images,
        "annotations": annotations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(image_paths)} images, {total} pre-label polygons")
    print(f"Output: {args.output.resolve()}")
    print(f"\nIn CVAT: create task with these images, then")
    print(f"  Actions -> Upload annotations -> COCO 1.0 -> {args.output.name}")


if __name__ == "__main__":
    main()
