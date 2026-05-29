"""Replicate exp1 exactly — sanity check whether 0.308 was reproducible.

Copies exp1's config verbatim from v3_experiment_runner.py:
  - yolov8m-seg from COCO
  - merged v1+v2+v3 train + merged val (v3_yolo_mergedval_tiled)
  - AdamW auto (Ultralytics picks lr=0.002)
  - batch=4, imgsz=640, single_cls=True, cos_lr=True
  - v2-proven augmentation
  - epochs=150, patience=30, time=1.5h  ← key difference from exp21 which had 0.75h
  - cache=disk, workers=2

Goal: get a second data point for variance estimation of exp1's 0.308.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "yolov train dataset"
RESULTS_FILE = PROJECT_ROOT / "results" / "v3_experiments.json"
ARCHIVE_DIR = PROJECT_ROOT / "weights" / "v3_runs"

VAL_DATASETS = {
    "v2-val": str(DATASET_ROOT / "v3_yolo_v2val_tiled" / "dataset.yaml"),
    "v3-val": str(DATASET_ROOT / "v3_yolo_v3val_tiled" / "dataset.yaml"),
    "merged": str(DATASET_ROOT / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
}

EXP_ID = "exp22_exp1_replicate_for_variance"
EXP_DESC = "Exact replicate of exp1 — same config, same data, same time budget (1.5h). Tests reproducibility of 0.308 result."


def main():
    from ultralytics import YOLO
    import torch

    results = []
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    if any(r.get("id") == EXP_ID and r.get("status") == "completed" for r in results):
        print(f"SKIP {EXP_ID}: already completed")
        return

    print("=" * 80)
    print(f"[{EXP_ID}] START — replicating exp1 exactly")
    print(f"  Original exp1: Box mAP50 = 0.308 on merged val, 31 min wall")
    print(f"  exp21 (same config but time=0.75h): 0.268 on merged val, 13 min wall")
    print(f"  This run uses time=1.5h identical to exp1")
    print("=" * 80, flush=True)

    t0 = time.time()

    # === EXACT exp1 config from v3_experiment_runner.py COMMON_HP + exp1 overrides ===
    args = dict(
        data=str(DATASET_ROOT / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
        epochs=150,
        patience=30,
        time=1.5,                # ← key: this is what exp1 had
        imgsz=640,
        batch=4,                 # exp1 override (from m-seg's lighter VRAM)
        device=0,
        name="v3_exp22_exp1_replicate",
        exist_ok=True,
        single_cls=True,
        optimizer="auto",        # auto picks AdamW lr=0.002
        lrf=0.01,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        nbs=64,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=20, translate=0.1, scale=0.4, shear=2.0,
        perspective=0.0,
        flipud=0.5, fliplr=0.5,
        bgr=0.0,
        mosaic=1.0, close_mosaic=10,
        mixup=0.1, copy_paste=0.1, erasing=0.2,
        amp=True,
        cache="disk",
        multi_scale=False,
        workers=2,
        save=True,
        plots=True,
        verbose=False,
    )

    model = YOLO("yolov8m-seg.pt")
    model.train(**args)
    save_dir = Path(getattr(model.trainer, "save_dir",
                            PROJECT_ROOT / "runs" / "segment" / args["name"]))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not at {best}")

    train_min = (time.time() - t0) / 60
    print(f"[{EXP_ID}] training done in {train_min:.1f} min", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Eval on 3 vals
    metrics = {}
    for vname, vyaml in VAL_DATASETS.items():
        print(f"[{EXP_ID}] eval on {vname}", flush=True)
        m = YOLO(str(best))
        r = m.val(data=vyaml, imgsz=640, batch=2, device=0,
                  plots=False, save=False, verbose=False)
        metrics[vname] = {
            "box_map50": float(r.box.map50),
            "box_map": float(r.box.map),
            "box_p": float(r.box.mp),
            "box_r": float(r.box.mr),
            "mask_map50": float(r.seg.map50),
            "mask_map": float(r.seg.map),
            "mask_p": float(r.seg.mp),
            "mask_r": float(r.seg.mr),
        }
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    merged_box = metrics["merged"]["box_map50"]
    v3_box = metrics["v3-val"]["box_map50"]
    archive_name = f"{EXP_ID}_v3val{v3_box:.3f}_mergedval{merged_box:.3f}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy(best, archive_path)

    wall_min = (time.time() - t0) / 60
    result = {
        "id": EXP_ID,
        "description": EXP_DESC,
        "status": "completed",
        "wall_time_min": round(wall_min, 1),
        "best_pt_archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }
    results.append(result)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"[{EXP_ID}] COMPLETED in {wall_min:.1f} min")
    print(f"  merged Box mAP50 = {merged_box:.4f}  (exp1 baseline: 0.308)")
    print(f"  v3-val Box mAP50 = {v3_box:.4f}")
    print(f"  v2-val Box mAP50 = {metrics['v2-val']['box_map50']:.4f}")
    if abs(merged_box - 0.308) < 0.015:
        print(f"  ✓ MATCH — within ±0.015 of exp1's 0.308")
    elif merged_box > 0.308:
        print(f"  🎯 EXCEEDED exp1 — gain of +{merged_box - 0.308:.4f}")
    else:
        print(f"  ⚠ DEVIATION — {0.308 - merged_box:.4f} below exp1, variance estimate")
    print("=" * 70)


if __name__ == "__main__":
    main()
