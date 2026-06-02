# -*- coding: utf-8 -*-
"""YOLOv8x v4 champion — qualitative results on FULL held-out scenes, predictions
only (no ground truth), in several styles to choose from.

Uses the REAL app pipeline (backend YOLOAdapter: 640+128 sliding-window tiling +
global NMS), the same code path the deployed Canopy app runs — so these are
honest, deployment-equivalent results, not single-tile crops.

Scenes = the larger held-out merged-val (M14) source images (the model did NOT
train on these). Predictions are rendered in 5 styles:

  A_masks   — filled lime instance masks + outline   (clean product look)
  B_boxes   — bounding boxes only
  C_outline — mask outlines only (no fill)
  D_heat    — strong fill, no outline (canopy look)
  E_conf    — masks colour-graded by confidence (green>0.7 / teal>0.5 / amber)

Output: thesis/figures/qual_scenes/<style>__<scene>.png  (full-res singles)
        thesis/figures/qual_scenes/<style>_grid.png       (2x2 slide grid)

Run:  venv/Scripts/python.exe thesis/gen_yolo_qual_scenes.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                       # import backend.*
OUT = HERE / "figures" / "qual_scenes"
OUT.mkdir(parents=True, exist_ok=True)

VAL = ROOT / "yolov train dataset" / "v3_yolo_mergedval" / "images" / "val"
CONF = 0.25

# larger held-out val scenes (untiled source images, model did not train on these)
SCENES = [
    ("img_val_010.png", "dense residential block"),
    ("img_val_011.png", "mixed yards"),
    ("img_val_004.png", "buildings + trees"),
    ("img_val_013.png", "street + courtyard"),
]

LIME  = (60, 230, 90)
TEAL  = (170, 200, 70)
AMBER = (40, 170, 240)


def conf_color(cf):
    if cf > 0.7:  return (80, 200, 80)
    if cf > 0.5:  return TEAL
    return AMBER


def to_int_poly(poly):
    return np.array(poly, dtype=np.int32).reshape(-1, 2)


def render(img, dets, style):
    out = img.copy()
    ov = img.copy()
    polys = [to_int_poly(d.mask_polygon) for d in dets if d.mask_polygon and len(d.mask_polygon) >= 3]
    confs = [d.confidence for d in dets if d.mask_polygon and len(d.mask_polygon) >= 3]

    if style == "B_boxes":
        for d in dets:
            b = d.box
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), LIME, 2, cv2.LINE_AA)
        return out

    if style == "C_outline":
        for p in polys:
            cv2.polylines(out, [p.reshape(-1, 1, 2)], True, LIME, 2, cv2.LINE_AA)
        return out

    if style == "E_conf":
        for p, cf in zip(polys, confs):
            cv2.fillPoly(ov, [p.reshape(-1, 1, 2)], conf_color(cf))
        out = cv2.addWeighted(ov, 0.34, out, 0.66, 0)
        for p, cf in zip(polys, confs):
            cv2.polylines(out, [p.reshape(-1, 1, 2)], True, conf_color(cf), 2, cv2.LINE_AA)
        return out

    # A_masks (fill+outline) and D_heat (fill only, stronger)
    alpha = 0.46 if style == "D_heat" else 0.30
    for p in polys:
        cv2.fillPoly(ov, [p.reshape(-1, 1, 2)], LIME)
    out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)
    if style == "A_masks":
        for p in polys:
            cv2.polylines(out, [p.reshape(-1, 1, 2)], True, LIME, 2, cv2.LINE_AA)
    return out


def caption(img, text):
    h, w = img.shape[:2]; strip = 40
    out = np.full((h + strip, w, 3), 245, np.uint8)
    out[strip:, :] = img
    cv2.putText(out, text, (12, strip - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (40, 40, 40), 2, cv2.LINE_AA)
    return out


def fit(img, H=420):
    s = H / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), H), interpolation=cv2.INTER_AREA)


def grid2x2(cells, gap=14, bg=255):
    cells = [fit(c) for c in cells]
    while len(cells) < 4:
        cells.append(np.full_like(cells[0], bg))
    rowH = max(c.shape[0] for c in cells)
    def row(a, b):
        W = a.shape[1] + b.shape[1] + gap
        r = np.full((rowH, W, 3), bg, np.uint8)
        r[:a.shape[0], :a.shape[1]] = a
        r[:b.shape[0], a.shape[1] + gap:a.shape[1] + gap + b.shape[1]] = b
        return r
    r1 = row(cells[0], cells[1]); r2 = row(cells[2], cells[3])
    W = max(r1.shape[1], r2.shape[1])
    out = np.full((r1.shape[0] + r2.shape[0] + gap, W, 3), bg, np.uint8)
    out[:r1.shape[0], :r1.shape[1]] = r1
    out[r1.shape[0] + gap:, :r2.shape[1]] = r2
    return out


def main():
    from backend.models.yolo_adapter import YOLOAdapter
    wp = ROOT / "weights" / "yolo_satellite.pt"
    assert wp.exists(), wp
    print(f"Loading champion {wp.name} via app YOLOAdapter (640+128 tiling + global NMS) ...", flush=True)
    adapter = YOLOAdapter(weights_path=str(wp))

    styles = ["A_masks", "B_boxes", "C_outline", "D_heat", "E_conf"]
    grids = {s: [] for s in styles}

    for fname, desc in SCENES:
        ip = VAL / fname
        if not ip.exists():
            print(f"  missing {fname}; skip"); continue
        img = cv2.imread(str(ip)); h, w = img.shape[:2]
        dets = adapter.predict(str(ip), confidence=CONF)
        n = len([d for d in dets if d.mask_polygon])
        print(f"  {fname} ({desc}) {w}x{h}: {n} crowns", flush=True)
        for s in styles:
            r = render(img, dets, s)
            cv2.imwrite(str(OUT / f"{s}__{ip.stem}.png"), r)
            grids[s].append(caption(r, f"{desc} - {n} trees"))

    for s, cells in grids.items():
        if not cells:
            continue
        g = grid2x2(cells)
        p = OUT / f"{s}_grid.png"
        cv2.imwrite(str(p), g)
        print(f"Saved {p}  {g.shape[1]}x{g.shape[0]}")
    print("\nStyles: A=masks · B=boxes · C=outline · D=heat · E=by-confidence  (predictions only, no GT)")
    print("Done.")


if __name__ == "__main__":
    main()
