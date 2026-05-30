"""Honest recall measurement for the production YOLO model on the M14 tiled val set.

Answers the plain question: "what fraction of the labelled trees does the model
actually find?" — overall and per-tile, at a couple of confidence thresholds.

GT = YOLO-seg polygon labels (converted to boxes). A GT tree counts as FOUND if
some predicted box overlaps it with IoU >= 0.5 (greedy 1-to-1 matching).

Usage:  python ml/eval_yolo_recall.py [path/to/model.pt]
Default model = weights/v4_clean/v4_x_clean (the 0.315 champion).
"""
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL_IMG = ROOT / "yolov train dataset/v3_yolo_mergedval_tiled/images/val"
VAL_LBL = ROOT / "yolov train dataset/v3_yolo_mergedval_tiled/labels/val"
DEFAULT = ROOT / "weights/v4_clean/v4_x_clean_v3val0.313_mergedval0.315.pt"
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
THRESHS = [0.10, 0.25, 0.50]
IOU_HIT = 0.5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_gt_boxes(txt: Path, W: int, H: int):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text().strip().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        nums = list(map(float, v[1:]))
        if len(nums) == 4:  # cx cy w h
            cx, cy, w, h = nums
            boxes.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
        else:               # polygon x y x y ...
            xs, ys = nums[0::2], nums[1::2]
            boxes.append([min(xs)*W, min(ys)*H, max(xs)*W, max(ys)*H])
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    from ultralytics import YOLO
    from PIL import Image

    assert MODEL.exists(), MODEL
    print(f"Model : {MODEL.relative_to(ROOT)}")
    print(f"Val   : {VAL_IMG.relative_to(ROOT)}  ({len(list(VAL_IMG.glob('*.png')))} tiles)\n")
    model = YOLO(str(MODEL))

    out = {"model": str(MODEL.relative_to(ROOT)), "iou_hit": IOU_HIT, "by_conf": {}}
    for TH in THRESHS:
        TP = FP = nGT = 0
        per = []
        for img in sorted(VAL_IMG.glob("*.png")):
            W, H = Image.open(img).size
            gt = load_gt_boxes(VAL_LBL / (img.stem + ".txt"), W, H)
            nGT += len(gt)
            res = model.predict(str(img), conf=TH, imgsz=640, verbose=False)[0]
            pb = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
            matched = [False]*len(gt)
            tp = 0
            for p in pb:
                best, bj = IOU_HIT, -1
                for j, g in enumerate(gt):
                    if matched[j]:
                        continue
                    i = iou(p, g)
                    if i >= best:
                        best, bj = i, j
                if bj >= 0:
                    matched[bj] = True
                    tp += 1
            TP += tp
            FP += (len(pb) - tp)
            per.append({"tile": img.stem, "gt": len(gt), "found": tp,
                        "recall": (tp/len(gt) if gt else None)})
        overall = TP/nGT if nGT else 0.0
        precision = TP/(TP+FP) if (TP+FP) else 0.0
        recs = [p["recall"] for p in per if p["gt"] >= 5]
        out["by_conf"][f"{TH}"] = {
            "overall_recall": round(overall, 4), "precision": round(precision, 4),
            "found": TP, "false_pos": FP, "total_gt": nGT,
            "per_tile_min": round(min(recs), 3) if recs else None,
            "per_tile_max": round(max(recs), 3) if recs else None,
            "per_tile": per,
        }
        rng = f"{min(recs):.0%}–{max(recs):.0%}" if recs else "n/a"
        print(f"conf={TH:<4}  recall = {overall:6.1%}  precision = {precision:6.1%}   "
              f"({TP} found / {FP} false / {nGT} real)   per-tile {rng}")

    OUT = ROOT / "results/yolo_v4_recall.json"
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
