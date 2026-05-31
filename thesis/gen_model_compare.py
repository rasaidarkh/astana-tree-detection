# -*- coding: utf-8 -*-
"""Compare champion vs v2-finetune vs exp1_m on the SAME merged-val tiles,
to show the 'aggregate mAP != per-detection coverage' point (S3.5/3.7).
Prints a count table and renders a grid (rows=models, cols=tiles) @ conf 0.25.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent; ROOT = HERE.parent; FIG = HERE / "figures"
VALIMG = ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "images" / "val"
VALLBL = ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "labels" / "val"
CONF = 0.25
TILES = ["img_val_013__y0000_x0000", "img_val_010__y0000_x0467",
         "img_val_004__y0000_x0000", "img_val_007__y0000_x0000"]
MODELS = [
    ("Ground truth",   None,                                                        (0, 200, 0)),
    ("v4 champion",    ROOT / "weights" / "yolo_satellite.pt",                       (60, 230, 90)),
    ("v2-finetune",    ROOT / "weights" / "archive" / "yolo" / "yolo_satellite_v2_finetune.pt", (0, 150, 255)),
    ("exp1_m",         ROOT / "weights" / "v3_runs" / "exp1_m_cocostart_v3val0.287_mergedval0.308.pt", (220, 90, 220)),
]
H = 230


def gt_polys(stem, h, w):
    p = VALLBL / f"{stem}.txt"
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) < 7: continue
            c = [float(x) for x in t[1:]]
            pts = [(c[i] * w, c[i + 1] * h) for i in range(0, len(c) - 1, 2)]
            if len(pts) >= 3: out.append(np.array(pts, np.int32))
    return out


def draw(img, polys, color):
    out = img.copy(); ov = out.copy()
    for p in polys: cv2.fillPoly(ov, [p.reshape(-1, 1, 2)], color)
    out = cv2.addWeighted(ov, 0.28, out, 0.72, 0)
    for p in polys: cv2.polylines(out, [p.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    return out


def cap(img, text):
    s = 30; h, w = img.shape[:2]
    o = np.full((h + s, w, 3), 245, np.uint8); o[s:] = img
    cv2.putText(o, text, (8, s - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 2, cv2.LINE_AA)
    return o


def fit(img):
    return cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H), interpolation=cv2.INTER_AREA)


def main():
    from ultralytics import YOLO
    loaded = {}
    for name, wp, _ in MODELS:
        if wp is not None:
            if not wp.exists(): print(f"[!] missing {name}: {wp}"); continue
            loaded[name] = YOLO(str(wp))
    counts = {name: [] for name, _, _ in MODELS}
    rows = []
    for name, wp, color in MODELS:
        cells = []
        for stem in TILES:
            img = cv2.imread(str(VALIMG / f"{stem}.png")); h, w = img.shape[:2]
            if name == "Ground truth":
                polys = gt_polys(stem, h, w)
            else:
                if name not in loaded: polys = []
                else:
                    res = loaded[name](img, conf=CONF, verbose=False)[0]
                    polys = [p.astype(np.int32) for p in (res.masks.xy if res.masks is not None else []) if len(p) >= 3]
            counts[name].append(len(polys))
            cells.append(cap(fit(draw(img, polys, color)), f"{name}: {len(polys)}"))
        # hconcat row
        gap = 10; Wt = sum(c.shape[1] for c in cells) + gap * (len(cells) - 1); Ht = max(c.shape[0] for c in cells)
        row = np.full((Ht, Wt, 3), 255, np.uint8); x = 0
        for c in cells: row[:c.shape[0], x:x + c.shape[1]] = c; x += c.shape[1] + gap
        rows.append(row)
    Wm = max(r.shape[1] for r in rows)
    grid = np.full((sum(r.shape[0] for r in rows) + 8 * (len(rows) - 1), Wm, 3), 255, np.uint8); y = 0
    for r in rows: grid[y:y + r.shape[0], :r.shape[1]] = r; y += r.shape[0] + 8
    out = FIG / "yolo_model_compare_tiles.png"; cv2.imwrite(str(out), grid)

    print(f"\nDetections @ conf={CONF}  (tiles: 013 / 010 / 004 / 007)")
    print(f"{'model':16s} {'013':>5} {'010':>5} {'004':>5} {'007':>5} {'TOTAL':>7}")
    for name, _, _ in MODELS:
        c = counts[name]; print(f"{name:16s} " + " ".join(f"{x:5d}" for x in c) + f" {sum(c):7d}")
    print(f"\nSaved {out}  {grid.shape[1]}x{grid.shape[0]}")


if __name__ == "__main__":
    main()
