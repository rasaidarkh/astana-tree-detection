# -*- coding: utf-8 -*-
"""YOLOv8x v4 champion — qualitative results in MULTIPLE STYLES to choose from.

Runs the champion (weights/yolo_satellite.pt = v4_x_clean) on diverse M14 val
tiles (dense -> sparse) at deployed conf=0.25 and renders six visual styles:

  styleA_masks_lime    — filled lime instance masks (clean "product" look)
  styleB_boxes         — detection bounding boxes + confidence labels
  styleC_gt_vs_pred    — side-by-side Ground truth (green) | Prediction (orange)
  styleD_outline       — mask outlines only, no fill (shows crown shape, busy scenes)
  styleE_heat          — translucent canopy heat (filled masks, no outline)
  styleF_confidence    — masks colour-graded by confidence (green>0.7 / teal / amber)

Each style is written as:
  - one PER-TILE image  figures/qual/<style>__<tile>.png      (pick singles)
  - one 4-TILE STRIP     figures/qual/<style>_strip.png         (slide-ready row)

Pick whichever you like; nothing here is wired into the deck yet.
Run:  venv/Scripts/python.exe thesis/gen_yolo_qualitative_styles.py
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
OUT = HERE / "figures" / "qual"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = ROOT / "weights" / "yolo_satellite.pt"            # = v4_x_clean champion
VALIMG = ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "images" / "val"
VALLBL = ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "labels" / "val"
CONF = 0.25
TARGET_H = 360

# diverse scenes: dense / dense / medium / sparse-street
TILES = [
    ("img_val_013__y0000_x0000", "dense residential"),
    ("img_val_010__y0000_x0467", "dense block"),
    ("img_val_004__y0000_x0000", "medium mixed"),
    ("img_val_001__y0000_x0000", "sparse / street"),
]

# BGR colours
LIME   = (60, 230, 90)
GREEN  = (0, 200, 0)
ORANGE = (0, 150, 255)
TEAL   = (170, 200, 70)
AMBER  = (40, 170, 240)
WHITE  = (255, 255, 255)


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


def fill(img, polys, color, alpha=0.30, outline=2):
    out = img.copy(); ov = out.copy()
    for poly in polys:
        cv2.fillPoly(ov, [poly.reshape(-1, 1, 2)], color)
    out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)
    if outline:
        for poly in polys:
            cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, color, outline, cv2.LINE_AA)
    return out


def outline_only(img, polys, color, w=2):
    out = img.copy()
    for poly in polys:
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, color, w, cv2.LINE_AA)
    return out


def boxes(img, dets, color=LIME):
    out = img.copy()
    for (x1, y1, x2, y2, cf) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return out


def conf_color(cf):
    if cf > 0.7:  return (80, 200, 80)     # green
    if cf > 0.5:  return TEAL              # teal
    return AMBER                           # amber


def fill_byconf(img, polys, confs, alpha=0.32):
    out = img.copy(); ov = out.copy()
    for poly, cf in zip(polys, confs):
        cv2.fillPoly(ov, [poly.reshape(-1, 1, 2)], conf_color(cf))
    out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)
    for poly, cf in zip(polys, confs):
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, conf_color(cf), 2, cv2.LINE_AA)
    return out


def caption(img, text, fg=(40, 40, 40), bg=245):
    h, w = img.shape[:2]; strip = 36
    out = np.full((h + strip, w, 3), bg, dtype=np.uint8)
    out[strip:, :] = img
    cv2.putText(out, text, (10, strip - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)
    return out


def fit_h(img, H=TARGET_H):
    s = H / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), H), interpolation=cv2.INTER_AREA)


def hcat(cells, gap=12, bg=255):
    H = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.full((H - c.shape[0], c.shape[1], 3), bg, np.uint8)]) if c.shape[0] < H else c for c in cells]
    W = sum(c.shape[1] for c in cells) + gap * (len(cells) - 1)
    out = np.full((H, W, 3), bg, dtype=np.uint8)
    x = 0
    for c in cells:
        out[:, x:x + c.shape[1]] = c; x += c.shape[1] + gap
    return out


def main():
    from ultralytics import YOLO
    assert WEIGHTS.exists(), WEIGHTS
    print(f"Loading champion {WEIGHTS.name} (= v4_x_clean) ...", flush=True)
    model = YOLO(str(WEIGHTS))

    strips = {k: [] for k in ["A_masks_lime", "B_boxes", "C_gt_vs_pred", "D_outline", "E_heat", "F_confidence"]}

    for stem, desc in TILES:
        ip = VALIMG / f"{stem}.png"
        if not ip.exists():
            print(f"  missing {stem}; skip"); continue
        img = cv2.imread(str(ip)); h, w = img.shape[:2]
        res = model(img, conf=CONF, verbose=False)[0]
        polys = [p.astype(np.int32) for p in (res.masks.xy if res.masks is not None else []) if len(p) >= 3]
        confs = [float(c) for c in (res.boxes.conf.tolist() if res.boxes is not None else [])]
        dets = []
        if res.boxes is not None:
            for b, cf in zip(res.boxes.xyxy.tolist(), confs):
                dets.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), cf))
        gt = gt_polys(stem, h, w)
        n = len(polys)
        print(f"  {stem} ({desc}): GT={len(gt)} champion={n}", flush=True)

        # ---- style A: filled lime masks ----
        a = caption(fit_h(fill(img, polys, LIME, 0.30, 2)), f"{desc} - {n} trees")
        # ---- style B: boxes + conf ----
        b = caption(fit_h(boxes(img, dets, LIME)), f"{desc} - {n} detections")
        # ---- style C: GT | pred stacked ----
        cg = caption(fit_h(fill(img, gt, GREEN, 0.28, 2)),   f"Ground truth: {len(gt)}")
        cp = caption(fit_h(fill(img, polys, ORANGE, 0.28, 2)), f"v4 prediction: {n}")
        wmax = max(cg.shape[1], cp.shape[1])
        cstack = np.full((cg.shape[0] + cp.shape[0] + 6, wmax, 3), 255, np.uint8)
        cstack[:cg.shape[0], :cg.shape[1]] = cg
        cstack[cg.shape[0] + 6:, :cp.shape[1]] = cp
        # ---- style D: outline only ----
        d = caption(fit_h(outline_only(img, polys, LIME, 2)), f"{desc} - {n} crowns")
        # ---- style E: heat (filled, no outline, stronger alpha) ----
        e = caption(fit_h(fill(img, polys, LIME, 0.45, 0)), f"{desc} - canopy")
        # ---- style F: confidence-graded ----
        f = caption(fit_h(fill_byconf(img, polys, confs)), f"{desc} - by confidence")

        strips["A_masks_lime"].append(a)
        strips["B_boxes"].append(b)
        strips["C_gt_vs_pred"].append(cstack)
        strips["D_outline"].append(d)
        strips["E_heat"].append(e)
        strips["F_confidence"].append(f)

        # per-tile singles (full res, no downscale) for the best scenes
        cv2.imwrite(str(OUT / f"A_masks_lime__{stem}.png"), fill(img, polys, LIME, 0.30, 2))
        cv2.imwrite(str(OUT / f"B_boxes__{stem}.png"),      boxes(img, dets, LIME))
        cv2.imwrite(str(OUT / f"D_outline__{stem}.png"),    outline_only(img, polys, LIME, 2))
        cv2.imwrite(str(OUT / f"E_heat__{stem}.png"),       fill(img, polys, LIME, 0.45, 0))
        cv2.imwrite(str(OUT / f"F_confidence__{stem}.png"), fill_byconf(img, polys, confs))

    for key, cells in strips.items():
        if not cells:
            continue
        out = hcat(cells)
        p = OUT / f"style{key}_strip.png"
        cv2.imwrite(str(p), out)
        print(f"Saved {p}  {out.shape[1]}x{out.shape[0]}")

    # legend note for confidence style
    print("\nStyles: A=filled lime masks · B=boxes+conf · C=GT vs pred · D=outline · E=heat · F=by-confidence")
    print("Per-tile singles + 4-tile strips written to", OUT)
    print("Done.")


if __name__ == "__main__":
    main()
