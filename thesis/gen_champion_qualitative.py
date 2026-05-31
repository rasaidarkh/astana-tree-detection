# -*- coding: utf-8 -*-
"""YOLOv8-seg (v4 champion) qualitative results on held-out merged-val (M14) tiles.

Produces two slide-ready figures:
  (1) yolo_v4_champion_qualitative.png  — per scene: Ground truth (green) over
      Champion prediction (orange), 4 representative tiles (dense -> medium).
  (2) yolo_v4_champion_results_strip.png — champion predictions only (lime), 4 tiles.

Inference: champion = weights/yolo_satellite.pt (= v4_x_clean), run directly on
each 640-px val tile at the deployed default conf=0.25 (no extra tiling needed —
these images ARE the post-tiling val tiles).
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parent
FIG = HERE / "figures"
WEIGHTS = ROOT / "weights" / "yolo_satellite.pt"        # = v4_x_clean champion
VALIMG = ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "images" / "val"
VALLBL = ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "labels" / "val"
CONF = 0.25

# representative tiles (dense -> medium), chosen from GT crown counts
TILES = [
    "img_val_013__y0000_x0000",       # 103 GT — dense residential
    "img_val_010__y0000_x0467",       # 88  GT — dense block
    "img_val_004__y0000_x0000",       # 66  GT — medium
    "img_val_007__y0000_x0000",       # 36  GT — street / mixed
]
GT_COLOR   = (0, 200, 0)      # green  (BGR)
PRED_COLOR = (0, 150, 255)    # orange (BGR) for the GT-vs-pred figure
LIME       = (60, 230, 90)    # lime   (BGR) for the champion-only strip
TARGET_H   = 320


def gt_polys(stem, h, w):
    p = VALLBL / f"{stem}.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) < 7:
            continue
        c = [float(x) for x in t[1:]]
        pts = [(c[i] * w, c[i + 1] * h) for i in range(0, len(c) - 1, 2)]
        if len(pts) >= 3:
            out.append(np.array(pts, dtype=np.int32))
    return out


def render(img, polys, color):
    out = img.copy()
    ov = out.copy()
    for poly in polys:
        cv2.fillPoly(ov, [poly.reshape(-1, 1, 2)], color)
    out = cv2.addWeighted(ov, 0.28, out, 0.72, 0)
    for poly in polys:
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    return out


def caption(img, text, fg=(35, 35, 35)):
    h, w = img.shape[:2]
    strip = 34
    out = np.full((h + strip, w, 3), 245, dtype=np.uint8)
    out[strip:, :] = img
    cv2.putText(out, text, (10, strip - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, fg, 2, cv2.LINE_AA)
    return out


def fit_h(img, H=TARGET_H):
    s = H / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), H), interpolation=cv2.INTER_AREA)


def main():
    from ultralytics import YOLO
    assert WEIGHTS.exists(), WEIGHTS
    print(f"Loading champion {WEIGHTS.name} ...", flush=True)
    model = YOLO(str(WEIGHTS))

    pair_cols, strip_cells = [], []
    for stem in TILES:
        ip = VALIMG / f"{stem}.png"
        if not ip.exists():
            print(f"  missing {stem}, skipping"); continue
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        res = model(img, conf=CONF, verbose=False)[0]
        pred = [p.astype(np.int32) for p in (res.masks.xy if res.masks is not None else []) if len(p) >= 3]
        gt = gt_polys(stem, h, w)
        print(f"  {stem}: GT={len(gt)}  champion={len(pred)}", flush=True)

        gt_img   = caption(fit_h(render(img, gt, GT_COLOR)),     f"Ground truth: {len(gt)}")
        pred_img = caption(fit_h(render(img, pred, PRED_COLOR)), f"v4 champion: {len(pred)}")
        # stack GT over prediction for this scene (align widths)
        wmax = max(gt_img.shape[1], pred_img.shape[1])
        col = np.full((gt_img.shape[0] + pred_img.shape[0] + 6, wmax, 3), 255, dtype=np.uint8)
        col[:gt_img.shape[0], :gt_img.shape[1]] = gt_img
        col[gt_img.shape[0] + 6:, :pred_img.shape[1]] = pred_img
        pair_cols.append(col)

        strip_cells.append(caption(fit_h(render(img, pred, LIME)), f"{len(pred)} trees"))

    def hcat(cells, gap=12):
        H = max(c.shape[0] for c in cells)
        cells = [np.vstack([c, np.full((H - c.shape[0], c.shape[1], 3), 255, np.uint8)]) if c.shape[0] < H else c for c in cells]
        W = sum(c.shape[1] for c in cells) + gap * (len(cells) - 1)
        out = np.full((H, W, 3), 255, dtype=np.uint8)
        x = 0
        for c in cells:
            out[:, x:x + c.shape[1]] = c; x += c.shape[1] + gap
        return out

    out1 = hcat(pair_cols)
    p1 = FIG / "yolo_v4_champion_qualitative.png"
    cv2.imwrite(str(p1), out1); print(f"Saved {p1}  {out1.shape[1]}x{out1.shape[0]}")

    out2 = hcat(strip_cells)
    p2 = FIG / "yolo_v4_champion_results_strip.png"
    cv2.imwrite(str(p2), out2); print(f"Saved {p2}  {out2.shape[1]}x{out2.shape[0]}")
    print("Done.")


if __name__ == "__main__":
    main()
