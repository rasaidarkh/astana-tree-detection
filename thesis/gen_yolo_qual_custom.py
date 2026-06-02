# -*- coding: utf-8 -*-
"""Champion YOLOv8x v4 on two custom park scenes (predictions only, no GT),
5 styles. Uses the real app YOLOAdapter (640+128 tiling + global NMS)."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = HERE / "figures" / "qual_custom"
OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.25

SCENES = [
    (Path.home() / "Pictures" / "Screenshots" / "asss.png", "park grove"),
    (Path.home() / "Pictures" / "Screenshots" / "asdf.png", "botanical garden"),
]

LIME = (60, 230, 90); TEAL = (170, 200, 70); AMBER = (40, 170, 240)
def conf_color(cf): return (80,200,80) if cf>0.7 else (TEAL if cf>0.5 else AMBER)
def ipoly(p): return np.array(p, np.int32).reshape(-1,2)

def render(img, dets, style):
    out = img.copy(); ov = img.copy()
    polys = [ipoly(d.mask_polygon) for d in dets if d.mask_polygon and len(d.mask_polygon)>=3]
    confs = [d.confidence for d in dets if d.mask_polygon and len(d.mask_polygon)>=3]
    if style=="B_boxes":
        for d in dets:
            b=d.box; cv2.rectangle(out,(int(b.x1),int(b.y1)),(int(b.x2),int(b.y2)),LIME,2,cv2.LINE_AA)
        return out
    if style=="C_outline":
        for p in polys: cv2.polylines(out,[p.reshape(-1,1,2)],True,LIME,2,cv2.LINE_AA)
        return out
    if style=="E_conf":
        for p,cf in zip(polys,confs): cv2.fillPoly(ov,[p.reshape(-1,1,2)],conf_color(cf))
        out=cv2.addWeighted(ov,0.34,out,0.66,0)
        for p,cf in zip(polys,confs): cv2.polylines(out,[p.reshape(-1,1,2)],True,conf_color(cf),2,cv2.LINE_AA)
        return out
    alpha = 0.46 if style=="D_heat" else 0.30
    for p in polys: cv2.fillPoly(ov,[p.reshape(-1,1,2)],LIME)
    out=cv2.addWeighted(ov,alpha,out,1-alpha,0)
    if style=="A_masks":
        for p in polys: cv2.polylines(out,[p.reshape(-1,1,2)],True,LIME,2,cv2.LINE_AA)
    return out

def main():
    from backend.models.yolo_adapter import YOLOAdapter
    wp = ROOT/"weights"/"yolo_satellite.pt"; assert wp.exists(), wp
    print(f"Loading champion {wp.name} (640+128 tiling + global NMS) ...", flush=True)
    adapter = YOLOAdapter(weights_path=str(wp))
    styles = ["A_masks","B_boxes","C_outline","D_heat","E_conf"]
    for ip, desc in SCENES:
        if not ip.exists(): print("missing", ip); continue
        img = cv2.imread(str(ip)); h,w = img.shape[:2]
        dets = adapter.predict(str(ip), confidence=CONF)
        n = len([d for d in dets if d.mask_polygon])
        print(f"  {ip.name} ({desc}) {w}x{h}: {n} crowns", flush=True)
        for s in styles:
            cv2.imwrite(str(OUT/f"{s}__{ip.stem}.png"), render(img, dets, s))
    print("Saved per-scene singles to", OUT); print("Done.")

if __name__ == "__main__":
    main()
