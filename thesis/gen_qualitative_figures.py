"""Regenerate Figure 3.3 (v1 vs v2-finetune comparison) and Figure 3.5 (val batch).

The previous Figure 3.3 (`yolo_v1_vs_v2_side_by_side.png`) actually showed
v1 vs v2-fromscratch, not v1 vs v2-finetune — mismatch with caption.
The previous Figure 3.5 was two separate vertical batch images stacked
in the PDF, overflowing page bounds and overlapping the page number.

This script:
  1. Picks val tile img_val_007 (representative dense scene) and runs
     v1 best.pt + v2-finetune best.pt on it. Produces a 3-panel
     side-by-side: GT | v1 | v2-finetune.
  2. Picks 4 val tiles and produces a single horizontal 4-panel strip
     for Fig 3.5, fitting page width without overflow.

Outputs:
  thesis/figures/yolo_v1_vs_v2_finetune_comparison.png
  thesis/figures/yolo_v2_finetune_val_4tile_strip.png
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parent
FIGURES = HERE / "figures"

V1_WEIGHTS = ROOT / "runs" / "segment" / "astana_tiled_x_max" / "weights" / "best.pt"
V2FT_WEIGHTS = ROOT / "weights" / "archive" / "yolo" / "yolo_satellite_v2_finetune.pt"

VAL_IMG_DIR = ROOT / "yolov train dataset" / "v3_yolo_v2val_tiled" / "images" / "val"
VAL_LABEL_DIR = ROOT / "yolov train dataset" / "v3_yolo_v2val_tiled" / "labels" / "val"


def _gt_polygons_for(tile_stem: str, h: int, w: int) -> list[np.ndarray]:
    """Read YOLO-format polygon labels for a tile and rescale to pixel coords."""
    label_path = VAL_LABEL_DIR / f"{tile_stem}.txt"
    if not label_path.exists():
        return []
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        toks = line.strip().split()
        if len(toks) < 7:  # class + at least 3 pairs
            continue
        coords = [float(x) for x in toks[1:]]
        # toks[1:] are normalised (x, y) pairs
        pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
        polygons.append(np.array(pts, dtype=np.int32))
    return polygons


def _draw_gt(img_bgr: np.ndarray, tile_stem: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    overlay = out.copy()
    polys = _gt_polygons_for(tile_stem, h, w)
    for poly in polys:
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], (0, 200, 0))
    out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)
    for poly in polys:
        if len(poly) >= 3:
            cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, (0, 200, 0), 2, cv2.LINE_AA)
    return out


def _predict_and_render(model, img_bgr: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    res = model(img_bgr, conf=0.25, verbose=False)[0]
    out = img_bgr.copy()
    overlay = out.copy()
    if res.masks is not None:
        for poly in res.masks.xy:
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], colour)
    out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)
    if res.masks is not None:
        for poly in res.masks.xy:
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.polylines(out, [pts], True, colour, 2, cv2.LINE_AA)
    return out


def _label_strip(img: np.ndarray, text: str, strip_h: int = 40) -> np.ndarray:
    """Add a coloured strip with title at the top of the cell."""
    h, w = img.shape[:2]
    out = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    out[:strip_h, :] = (245, 245, 245)
    out[strip_h:, :] = img
    cv2.putText(out, text, (12, strip_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return out


def figure_v1_vs_v2_finetune():
    """Figure 3.3 — GT | v1 | v2-finetune on a single representative tile."""
    from ultralytics import YOLO

    print("Loading v1 and v2-finetune weights...", flush=True)
    v1_model = YOLO(str(V1_WEIGHTS))
    v2ft_model = YOLO(str(V2FT_WEIGHTS))

    # Pick a representative tile — img_val_007 is dense residential
    candidate_stems = ["img_val_007__y0000_x0000", "img_val_003__y0000_x0000",
                       "img_val_001__y0000_x0000"]
    tile_path = None
    tile_stem = None
    for stem in candidate_stems:
        p = VAL_IMG_DIR / f"{stem}.png"
        if p.exists():
            tile_path = p
            tile_stem = stem
            break
    if tile_path is None:
        raise FileNotFoundError(f"No val tile found in {VAL_IMG_DIR}")
    print(f"Using val tile: {tile_path.name}", flush=True)

    img = cv2.imread(str(tile_path))
    if img is None:
        raise RuntimeError(f"Could not read {tile_path}")
    h, w = img.shape[:2]

    print("Rendering GT polygons...", flush=True)
    gt_panel = _draw_gt(img, tile_stem)
    print("Running v1 prediction...", flush=True)
    v1_panel = _predict_and_render(v1_model, img, (80, 80, 255))
    print("Running v2-finetune prediction...", flush=True)
    v2ft_panel = _predict_and_render(v2ft_model, img, (255, 140, 60))

    gap = 16
    titles = ["Ground truth", "YOLOv8x-seg v1", "YOLOv8x-seg v2-finetune"]
    cells = [_label_strip(p, t) for p, t in zip([gt_panel, v1_panel, v2ft_panel], titles)]
    th, tw = cells[0].shape[:2]
    out = np.full((th, tw * 3 + gap * 2, 3), 255, dtype=np.uint8)
    for i, cell in enumerate(cells):
        x = (tw + gap) * i
        out[:, x:x + tw] = cell

    out_path = FIGURES / "yolo_v1_vs_v2_finetune_comparison.png"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}  size={out.shape[1]}x{out.shape[0]}", flush=True)
    return out_path


def figure_v2_finetune_4tile_strip():
    """Figure 3.5 — single horizontal strip of 4 val tiles with v2-finetune predictions.

    Replaces the previous two stacked vertical batch images that overflowed page bounds.
    """
    from ultralytics import YOLO

    print("Loading v2-finetune for Fig 3.5...", flush=True)
    v2ft_model = YOLO(str(V2FT_WEIGHTS))

    # Pick 4 representative tiles
    stems = ["img_val_001__y0000_x0000", "img_val_003__y0000_x0000",
             "img_val_007__y0000_x0000", "img_val_009__y0000_x0000"]
    panels = []
    for stem in stems:
        p = VAL_IMG_DIR / f"{stem}.png"
        if not p.exists():
            print(f"  missing tile {stem}, skipping", flush=True)
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        # Resize all tiles to common height (smaller for compact figure)
        target_h = 480
        scale = target_h / img.shape[0]
        img_s = cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)
        # Predict on the original resolution
        rendered = _predict_and_render(v2ft_model, img, (255, 140, 60))
        rendered_s = cv2.resize(rendered, (int(rendered.shape[1] * scale), target_h),
                                interpolation=cv2.INTER_AREA)
        panels.append((stem, rendered_s))

    if not panels:
        raise RuntimeError("No val tiles found for Fig 3.5")

    # Side-by-side strip with thin separators
    gap = 12
    h = panels[0][1].shape[0]
    total_w = sum(p[1].shape[1] for p in panels) + gap * (len(panels) - 1)
    out = np.full((h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for stem, p in panels:
        out[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap

    out_path = FIGURES / "yolo_v2_finetune_val_4tile_strip.png"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}  size={out.shape[1]}x{out.shape[0]}", flush=True)
    return out_path


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not V1_WEIGHTS.exists():
        sys.exit(f"V1 weights missing: {V1_WEIGHTS}")
    if not V2FT_WEIGHTS.exists():
        sys.exit(f"V2-finetune weights missing: {V2FT_WEIGHTS}")
    figure_v1_vs_v2_finetune()
    figure_v2_finetune_4tile_strip()
    print("Done.")


if __name__ == "__main__":
    main()
