# -*- coding: utf-8 -*-
"""Wrapper: evaluate DeepForest+SAM2 on OUR 14-image val (4 v1 + 5 v2 + 5 v3,
excluding 194422.png which is in YOLO train corpus).

Same pipeline as ml/eval_deepforest_sam2.py but with:
  - VAL_JSON pointing to annotations_merged_14img_val.json (built by
    ml/build_14img_val_coco.py)
  - IMG_DIRS includes the v3 photos folder
"""
from __future__ import annotations
import io, json, sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.stdout.reconfigure(encoding="utf-8")

ROOT     = Path(__file__).parent.parent
VAL_JSON = ROOT / "yolov train dataset" / "annotations_merged_14img_val.json"
IMG_DIRS = [
    ROOT / "yolov train dataset" / "фотографии",
    ROOT / "yolov train dataset" / "новые фотографии",
    ROOT / "yolov train dataset" / "v3 фотографии для finetune",
]
DF_CKPT   = ROOT / "weights" / "deepforest_astana.pl"
SAM2_ID   = "facebook/sam2.1-hiera-base-plus"
CONF_THR  = 0.30
PATCH     = 400
OVERLAP   = 0.05


def find_image(fname: str) -> Path | None:
    for d in IMG_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def encode_rle(mask: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ── Load models ─────────────────────────────────────────────────────────────
print("Loading DeepForest fine-tuned...")
from deepforest import main as df_main
df_model = df_main.deepforest()
df_model.load_model(model_name="weecology/deepforest-tree", revision="main")
if DF_CKPT.exists():
    ckpt = torch.load(str(DF_CKPT), map_location="cpu", weights_only=False)
    df_model.model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    print(f"  Fine-tuned weights: {DF_CKPT.name}")
df_model.config["score_thresh"] = 0.05

print("Loading SAM2...")
from sam2.sam2_image_predictor import SAM2ImagePredictor
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = SAM2ImagePredictor.from_pretrained(SAM2_ID, device=device)
print(f"  device={device}")

# ── Load COCO GT ─────────────────────────────────────────────────────────────
import builtins
_orig_open = builtins.open
def _utf8_open(file, mode="r", **kwargs):
    if "b" not in mode:
        kwargs.setdefault("encoding", "utf-8")
    return _orig_open(file, mode, **kwargs)
builtins.open = _utf8_open

coco_gt = COCO(str(VAL_JSON))
builtins.open = _orig_open
img_ids  = sorted(coco_gt.getImgIds())

# ── Run inference ─────────────────────────────────────────────────────────────
coco_preds_box  = []
coco_preds_segm = []
det_id = 1

for img_id in img_ids:
    info   = coco_gt.loadImgs(img_id)[0]
    fname  = info["file_name"]
    img_path = find_image(fname)
    if img_path is None:
        print(f"  SKIP (missing): {fname}")
        continue

    pil_img = Image.open(str(img_path)).convert("RGB")
    img_rgb = np.array(pil_img)
    H, W    = img_rgb.shape[:2]

    boxes_df = df_model.predict_tile(path=str(img_path),
                                     patch_size=PATCH, patch_overlap=OVERLAP)
    if boxes_df is None or len(boxes_df) == 0:
        print(f"  {fname[:45]}  dets=0")
        continue

    boxes_df = boxes_df[boxes_df["score"] >= CONF_THR].reset_index(drop=True)
    if len(boxes_df) == 0:
        print(f"  {fname[:45]}  dets=0 (after threshold)")
        continue

    det_boxes  = boxes_df[["xmin","ymin","xmax","ymax"]].values.astype(np.float32)
    det_scores = boxes_df["score"].values.astype(np.float32)

    try:
        with torch.inference_mode():
            sam.set_image(img_rgb)
            sam_masks, _, _ = sam.predict(
                box=det_boxes,
                multimask_output=False,
            )
        if sam_masks.ndim == 4:
            sam_masks = sam_masks[:, 0]
    except Exception as e:
        print(f"  SAM2 error on {fname}: {e}")
        sam_masks = np.zeros((len(det_boxes), H, W), dtype=bool)

    print(f"  {fname[:45]}  dets={len(det_boxes)}")

    for i in range(len(det_boxes)):
        x1, y1, x2, y2 = det_boxes[i]
        score = float(det_scores[i])
        bbox_coco = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
        rle = encode_rle(sam_masks[i].astype(np.uint8))

        coco_preds_box.append({
            "id": det_id, "image_id": img_id, "category_id": 1,
            "bbox": bbox_coco, "score": score,
            "segmentation": rle, "area": float((x2-x1)*(y2-y1)),
        })
        coco_preds_segm.append({
            "id": det_id, "image_id": img_id, "category_id": 1,
            "bbox": bbox_coco, "score": score,
            "segmentation": rle,
            "area": float(np.sum(sam_masks[i])),
        })
        det_id += 1


def run_eval(coco_gt, preds, iou_type):
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate(); ev.accumulate()
    buf = io.StringIO()
    with redirect_stdout(buf):
        ev.summarize()
    return ev.stats, buf.getvalue()


print("\n" + "="*60)
print("DeepForest + SAM2 on 14-image val — Results")
print("="*60)

stats_box,  summary_box  = run_eval(coco_gt, coco_preds_box,  "bbox")
stats_segm, summary_segm = run_eval(coco_gt, coco_preds_segm, "segm")

print("\n[Box]")
print(summary_box)
print("[Segmentation/Mask]")
print(summary_segm)

print("="*60)
print(f"Box  mAP@50   : {stats_box[1]:.4f}")
print(f"Box  mAP@50:95: {stats_box[0]:.4f}")
print(f"Mask mAP@50   : {stats_segm[1]:.4f}")
print(f"Mask mAP@50:95: {stats_segm[0]:.4f}")
print("="*60)

# Save
out_dir = ROOT / "results" / "df_sam2_14img_eval"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps({
    "box": {
        "mAP_50_95": float(stats_box[0]),
        "mAP_50": float(stats_box[1]),
        "mAP_75": float(stats_box[2]),
    },
    "segm": {
        "mAP_50_95": float(stats_segm[0]),
        "mAP_50": float(stats_segm[1]),
        "mAP_75": float(stats_segm[2]),
    },
    "n_val_images": len(img_ids),
    "conf_threshold": CONF_THR,
    "patch_size": PATCH,
    "overlap": OVERLAP,
}, indent=2), encoding="utf-8")
print(f"Saved -> {out_dir.relative_to(ROOT)}/metrics.json")
