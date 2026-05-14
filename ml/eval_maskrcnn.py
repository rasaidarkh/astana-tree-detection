"""COCO mAP evaluation + visualisations for a Mask R-CNN checkpoint.

Computes Box + Mask mAP via pycocotools on the merged val set, writes:
  - results/maskrcnn_eval/metrics.json (full COCOeval stats + P/R at conf)
  - results/maskrcnn_eval/predictions.json (COCO-format predictions)
  - results/maskrcnn_eval/predictions/*.png (5 example overlays)
  - results/maskrcnn_eval/comparison_table.md (vs YOLO baseline)

Example:
    python -m ml.eval_maskrcnn --checkpoint weights/maskrcnn_astana.pt
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.maskrcnn_adapter import MaskRCNNAdapter  # noqa: E402
from ml.maskrcnn_dataset import CocoMaskRCNNDataset  # noqa: E402

log = logging.getLogger("astana-tree")

# Team baseline (production YOLOv8-seg v2-finetune on merged val) — to beat
YOLO_BASELINE = {
    "box_map_50": 0.372,
    "mask_map_50": 0.331,
    "box_precision": 0.425,
    "box_recall": 0.391,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--val-json",
        default="yolov train dataset/annotations_merged/instances_Validation.json",
    )
    p.add_argument(
        "--images-roots",
        nargs="+",
        default=[
            "yolov train dataset/фотографии",
            "yolov train dataset/новые фотографии",
        ],
    )
    p.add_argument("--output-dir", default="results/maskrcnn_eval")
    p.add_argument("--device", default="auto")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    return p.parse_args()


def _polygon_to_mask(polygon: list[list[float]], h: int, w: int) -> np.ndarray:
    """Rasterize a simplified polygon back to a binary (H, W) uint8 mask."""
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def _encode_rle(mask: np.ndarray) -> dict:
    """Binary mask -> pycocotools RLE dict (counts as str so it serialises to JSON)."""
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _coco_summarize(coco_eval: COCOeval) -> dict:
    """Capture COCOeval.summarize() printed output and return its stats as a dict."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()
    stats = coco_eval.stats.tolist()
    return {
        "mAP_50_95": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "mAP_small": stats[3],
        "mAP_medium": stats[4],
        "mAP_large": stats[5],
        "AR_1": stats[6],
        "AR_10": stats[7],
        "AR_100": stats[8],
        "summary": buf.getvalue(),
    }


def _iou_match(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    iou_thresh: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy IoU matching (preds sorted by score desc). Returns (tp, fp, fn)."""
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    pb = np.asarray(pred_boxes, dtype=np.float32)
    gb = np.asarray(gt_boxes, dtype=np.float32)
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32))
    pb = pb[order]

    x1 = np.maximum(pb[:, None, 0], gb[None, :, 0])
    y1 = np.maximum(pb[:, None, 1], gb[None, :, 1])
    x2 = np.minimum(pb[:, None, 2], gb[None, :, 2])
    y2 = np.minimum(pb[:, None, 3], gb[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a_p = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    a_g = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    union = a_p[:, None] + a_g[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)

    used_gt: set[int] = set()
    tp = fp = 0
    for pi in range(len(pb)):
        best_gi = -1
        best_iou = iou_thresh
        for gi in range(len(gb)):
            if gi in used_gt:
                continue
            if iou[pi, gi] >= best_iou:
                best_iou = iou[pi, gi]
                best_gi = gi
        if best_gi >= 0:
            tp += 1
            used_gt.add(best_gi)
        else:
            fp += 1
    fn = len(gb) - len(used_gt)
    return tp, fp, fn


def _draw_examples(
    images_paths: list[Path],
    coco_gt: COCO,
    image_ids: list[int],
    predictions_by_image: dict[int, list[dict]],
    output_dir: Path,
    max_examples: int = 5,
) -> None:
    """Save up to N visualisation PNGs: original + coloured masks + bbox + score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    for path, image_id in zip(images_paths[:max_examples], image_ids[:max_examples]):
        info = coco_gt.loadImgs(image_id)[0]
        h, w = int(info["height"]), int(info["width"])
        img = np.array(Image.open(path).convert("RGB"))

        fig, ax = plt.subplots(figsize=(min(w / 100, 16), min(h / 100, 12)))
        ax.imshow(img)

        preds = predictions_by_image.get(image_id, [])
        cmap = plt.get_cmap("tab20")
        for i, pred in enumerate(preds):
            color = cmap(i % 20)
            rle = pred["segmentation"]
            mask = coco_mask.decode(rle)
            overlay = np.zeros((h, w, 4), dtype=float)
            overlay[mask > 0] = (*color[:3], 0.45)
            ax.imshow(overlay)
            x, y, bw, bh = pred["bbox"]
            ax.add_patch(
                patches.Rectangle(
                    (x, y), bw, bh,
                    linewidth=1.2, edgecolor=color, facecolor="none",
                )
            )
            ax.text(
                x, max(0, y - 3), f"{pred['score']:.2f}",
                color="white", fontsize=7,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
            )

        ax.set_axis_off()
        out_png = output_dir / f"{Path(info['file_name']).stem}.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=120)
        plt.close(fig)
        log.info("Saved example: %s", out_png)


def _write_comparison(metrics: dict, output_path: Path) -> None:
    """Write markdown comparison table vs team YOLOv8-seg baseline."""
    box_50 = metrics["box"].get("mAP_50") if "box" in metrics else None
    mask_50 = metrics["segm"].get("mAP_50") if "segm" in metrics else None
    box_p = metrics.get("box_precision_at_conf")
    box_r = metrics.get("box_recall_at_conf")

    def cell(v) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    body = "\n".join([
        "| Метрика | Mask R-CNN (моя) | YOLOv8-seg (команда) |",
        "|---|---|---|",
        f"| Box mAP@50 | {cell(box_50)} | {YOLO_BASELINE['box_map_50']:.3f} |",
        f"| Mask mAP@50 | {cell(mask_50)} | {YOLO_BASELINE['mask_map_50']:.3f} |",
        f"| Box Precision | {cell(box_p)} | {YOLO_BASELINE['box_precision']:.3f} |",
        f"| Box Recall | {cell(box_r)} | {YOLO_BASELINE['box_recall']:.3f} |",
    ])
    output_path.write_text(body + "\n", encoding="utf-8")
    log.info("Comparison table -> %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"

    adapter = MaskRCNNAdapter(
        checkpoint_path=args.checkpoint,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    val_ds = CocoMaskRCNNDataset(args.val_json, args.images_roots)
    coco_gt = val_ds.coco

    predictions: list[dict] = []
    predictions_by_image: dict[int, list[dict]] = {}
    image_paths: list[Path] = []
    image_ids: list[int] = []

    tp = fp = fn = 0
    for image_id in val_ds.image_ids:
        info = coco_gt.loadImgs(image_id)[0]
        path = val_ds._resolve_image_path(info["file_name"])
        image_paths.append(path)
        image_ids.append(image_id)
        h, w = int(info["height"]), int(info["width"])

        detections = adapter.predict(str(path), confidence=args.confidence_threshold)

        gt_ann_ids = coco_gt.getAnnIds(imgIds=image_id, iscrowd=None)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        gt_boxes = [
            [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            for a in gt_anns
        ]
        pred_boxes = [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections]
        pred_scores = [d.confidence for d in detections]
        i_tp, i_fp, i_fn = _iou_match(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5)
        tp += i_tp
        fp += i_fp
        fn += i_fn

        per_image: list[dict] = []
        for det in detections:
            if det.mask_polygon is None or len(det.mask_polygon) < 3:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                mask = _polygon_to_mask(det.mask_polygon, h, w)
            rle = _encode_rle(mask)
            x1, y1, x2, y2 = det.box.x1, det.box.y1, det.box.x2, det.box.y2
            entry = {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": det.confidence,
                "segmentation": rle,
            }
            predictions.append(entry)
            per_image.append(entry)

        predictions_by_image[image_id] = per_image
        log.info(
            "%s | pred=%d gt=%d | tp=%d fp=%d fn=%d",
            info["file_name"], len(detections), len(gt_anns), i_tp, i_fp, i_fn,
        )

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    log.info("Predictions JSON -> %s (%d entries)", predictions_path, len(predictions))

    metrics: dict = {}
    if predictions:
        coco_dt = coco_gt.loadRes(str(predictions_path))
        for iou_type, key in (("bbox", "box"), ("segm", "segm")):
            ce = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            ce.evaluate()
            ce.accumulate()
            metrics[key] = _coco_summarize(ce)
    else:
        metrics["box"] = {"mAP_50": 0.0}
        metrics["segm"] = {"mAP_50": 0.0}
        log.warning("No predictions above confidence threshold — metrics are zero")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics["box_precision_at_conf"] = precision
    metrics["box_recall_at_conf"] = recall
    metrics["confidence_threshold"] = args.confidence_threshold
    metrics["n_val_images"] = len(val_ds)
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Metrics JSON -> %s", metrics_path)

    _draw_examples(image_paths, coco_gt, image_ids, predictions_by_image, predictions_dir)
    _write_comparison(metrics, output_dir / "comparison_table.md")

    log.info("=" * 60)
    log.info(
        "Box mAP@50: %.4f (baseline %.3f)",
        metrics["box"].get("mAP_50", 0.0), YOLO_BASELINE["box_map_50"],
    )
    log.info(
        "Mask mAP@50: %.4f (baseline %.3f)",
        metrics["segm"].get("mAP_50", 0.0), YOLO_BASELINE["mask_map_50"],
    )
    log.info(
        "Box P/R @ conf=%.2f: %.3f / %.3f (baseline %.3f / %.3f)",
        args.confidence_threshold, precision, recall,
        YOLO_BASELINE["box_precision"], YOLO_BASELINE["box_recall"],
    )


if __name__ == "__main__":
    main()
