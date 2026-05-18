"""Visual side-by-side compare: any image → predictions from top-N YOLO models.

Usage:
  python ml/v5_visual_compare.py <image_path> [--conf 0.25] [--out compare.png]

Output: PNG grid (2 rows × 4 cols by default) where each cell shows the same
image with predictions from one model overlaid. Polygons are drawn as outlines
ONLY (no "tree 0.87" text labels — cleaner for visual comparison).

If image is large, predictions are run on the full resolution using the
backend's tiled inference path (no downscale = small crowns preserved).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from v5_top_models import TOP_MODELS, filter_existing  # noqa: E402


# Distinct color per model (BGR for OpenCV)
COLORS = [
    (0, 220, 80),    # green
    (255, 140, 0),   # blue-orange
    (0, 165, 255),   # orange
    (200, 80, 255),  # purple
    (80, 220, 255),  # yellow
    (255, 80, 200),  # pink
    (180, 180, 50),  # teal
    (60, 60, 220),   # red
]


def _predict_polygons(weights_path: Path, image_path: Path, conf: float) -> list:
    """Run YOLO inference with tiled sliding-window for large images.

    Returns list of polygons (each = list of (x,y) ints in image coords).
    Reuses the backend YOLOAdapter which already has tiled inference + global NMS.
    """
    # Use backend adapter for tiled inference (matches what UI shows)
    from backend.models.yolo_adapter import YOLOAdapter
    adapter = YOLOAdapter(weights_path=str(weights_path))
    dets = adapter.predict(str(image_path), confidence=conf)
    polys = []
    for d in dets:
        # Detection.mask_polygon = [[x,y], [x,y], ...] in pixel coords
        poly = getattr(d, "mask_polygon", None)
        if poly and len(poly) >= 3:
            polys.append([(int(p[0]), int(p[1])) for p in poly])
        elif d.bbox:  # fallback for detection-only models (no mask)
            x1, y1, x2, y2 = d.bbox
            polys.append([(int(x1), int(y1)), (int(x2), int(y1)),
                          (int(x2), int(y2)), (int(x1), int(y2))])
    return polys


def _annotate(img_bgr, polygons, color, thickness=2):
    """Draw polygon outlines on image — no text labels."""
    import cv2
    import numpy as np

    out = img_bgr.copy()
    # Semi-transparent fill for visibility
    overlay = out.copy()
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    # Crisp outline on top
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
    return out


def _label_strip(img_bgr, text):
    """Add a colored strip with model name at the top of the cell."""
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    strip_h = 36
    out = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    out[strip_h:, :] = img_bgr
    # Dark strip
    cv2.rectangle(out, (0, 0), (w, strip_h), (20, 20, 20), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _grid(cells, cols):
    """Combine list of equal-sized images into a grid (rows × cols)."""
    import numpy as np

    rows = (len(cells) + cols - 1) // cols
    h, w = cells[0].shape[:2]
    grid_img = np.full((h * rows, w * cols, 3), 255, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = i // cols, i % cols
        grid_img[r * h:(r + 1) * h, c * w:(c + 1) * w] = cell
    return grid_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="path to image file")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--out", type=str, default=None, help="output PNG (default: <image>_compare.png)")
    ap.add_argument("--cols", type=int, default=4, help="columns in grid")
    ap.add_argument("--max-width", type=int, default=900, help="max width per cell")
    ap.add_argument("--models", default="all",
                    help="comma-separated model names or 'all' or 'top4' for the user-selected v4_x,exp1_m,v4_s,v2-finetune")
    ap.add_argument("--ensemble", action="store_true",
                    help="add an ensemble cell (IoU-merged result of selected models)")
    ap.add_argument("--ensemble-strategy", default="vote_2", choices=["nms", "vote_2", "vote_3", "vote_all"])
    ap.add_argument("--ensemble-iou", type=float, default=0.5)
    args = ap.parse_args()

    import cv2
    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"image not found: {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        sys.exit(f"failed to read image: {img_path}")
    print(f"Loaded {img_path.name} — {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    models = filter_existing(TOP_MODELS)

    # Model filtering
    if args.models == "top4":
        wanted = {"v4_x_clean (champ)", "exp1_m (tuned)", "v4_s_clean", "v2-finetune (legacy)"}
        models = [m for m in models if m[0] in wanted]
    elif args.models != "all":
        wanted = set(s.strip() for s in args.models.split(","))
        models = [m for m in models if any(w in m[0] for w in wanted)]
    print(f"Running {len(models)} models at conf={args.conf}: {[m[0] for m in models]}")
    print()

    cells = []
    for i, (name, weights, note) in enumerate(models):
        color = COLORS[i % len(COLORS)]
        print(f"[{i+1}/{len(models)}] {name} ({weights.name})", flush=True)
        try:
            polys = _predict_polygons(weights, img_path, args.conf)
        except Exception as e:
            print(f"  FAILED: {e}")
            polys = []
        annotated = _annotate(img_bgr, polys, color)
        # Downscale to max-width if needed
        h, w = annotated.shape[:2]
        if w > args.max_width:
            scale = args.max_width / w
            annotated = cv2.resize(annotated, (args.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        label = f"{name}  ·  {len(polys)} trees"
        cells.append(_label_strip(annotated, label))
        print(f"  detected {len(polys)} polygons")

    # Optional ensemble cell
    if args.ensemble and len(models) >= 2:
        from v5_ensemble import ensemble_predict
        ens_models = [(n, p) for n, p, _ in models]
        print()
        print(f"[ensemble · {args.ensemble_strategy}] merging across {len(models)} models...")
        try:
            ens_dets = ensemble_predict(
                str(img_path), ens_models,
                strategy=args.ensemble_strategy,
                iou_threshold=args.ensemble_iou,
                conf=args.conf,
            )
        except Exception as e:
            print(f"  ensemble FAILED: {e}")
            ens_dets = []
        ens_polys = [d.polygon for d in ens_dets if len(d.polygon) >= 3]
        # Distinct color for ensemble — bright cyan
        ens_color = (255, 255, 100)
        annotated = _annotate(img_bgr, ens_polys, ens_color, thickness=3)
        h, w = annotated.shape[:2]
        if w > args.max_width:
            scale = args.max_width / w
            annotated = cv2.resize(annotated, (args.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        label = f"ENSEMBLE {args.ensemble_strategy} (IoU>={args.ensemble_iou})  ·  {len(ens_polys)} trees"
        cells.append(_label_strip(annotated, label))
        print(f"  ensemble: {len(ens_polys)} unified detections")

    grid_img = _grid(cells, cols=args.cols)

    out_path = Path(args.out) if args.out else img_path.with_name(f"{img_path.stem}_compare.png")
    cv2.imwrite(str(out_path), grid_img)
    print()
    print(f"Saved comparison: {out_path}  ({grid_img.shape[1]}×{grid_img.shape[0]})")
    print()
    print("Tip: open in image viewer at 100% zoom — each cell shows the same source image with")
    print("one model's predictions. Easy to spot which models over/under-detect on specific scenes.")


if __name__ == "__main__":
    main()
