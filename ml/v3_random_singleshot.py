"""Single-shot baselines on random phase subsets — control for exp17 chain.

For each random phase dataset built by v3_random_chain.py, train yolov8m-seg
from COCO single-shot (= exp1 config) and compare to the chain stage output
at the same data size:

  exp19 ← single-shot on random_phase1 (21 imgs / ~50 tiles)
          Compare to exp17 stage1 (= chain start, same data).
  exp20 ← single-shot on random_phase2 (42 imgs / ~100 tiles)
          Compare to exp17 stage2 output (chain after 2 stages on same data).
  exp21 ← single-shot on random_phase3 (63 imgs / ~152 tiles) = full merged.
          MUST give ≈ 0.308 (= exp1 result) — sanity check that our random
          shuffle didn't break the dataset. If significantly different,
          we have a bug.

Conclusion patterns:
  - exp19 ≥ exp17 stage1 → chain stage 1 added no value.
  - exp20 ≥ exp17 final → chain didn't beat partial-data single-shot.
  - exp21 ≈ exp1 (0.308) → dataset rebuild is sound, results comparable.

All use exp1 config — yolov8m-seg from COCO, AdamW auto, batch=4, imgsz=640,
v2-proven aug, patience=30, time=0.75h, epochs=150.
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

# Exp1 winner config exactly
COMMON_HP = dict(
    weights_start="yolov8m-seg.pt",
    epochs=150,
    patience=30,
    imgsz=640,
    batch=4,
    device=0,
    exist_ok=True,
    single_cls=True,
    optimizer="auto",
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
    time=0.75,
)

EXPERIMENTS = [
    {
        "id": "exp19_singleshot_random_phase1",
        "description": "Single-shot exp1 config on random_phase1 (21 imgs / 50 tiles). Compares to exp17 stage 1.",
        "data": DATASET_ROOT / "random3_phase1_yolo_mergedval_tiled" / "dataset.yaml",
        "name": "v3_exp19_singleshot_p1",
    },
    {
        "id": "exp20_singleshot_random_phase2",
        "description": "Single-shot exp1 config on random_phase2 (42 imgs / 100 tiles). Compares to exp17 stages 1+2.",
        "data": DATASET_ROOT / "random3_phase2_yolo_mergedval_tiled" / "dataset.yaml",
        "name": "v3_exp20_singleshot_p2",
    },
    {
        "id": "exp21_singleshot_random_phase3",
        "description": "Single-shot exp1 config on random_phase3 (63 imgs / 152 tiles) = full merged. Sanity check: must give ~0.308.",
        "data": DATASET_ROOT / "random3_phase3_yolo_mergedval_tiled" / "dataset.yaml",
        "name": "v3_exp21_singleshot_full",
    },
]


def run(exp: dict) -> None:
    results = []
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    if any(r.get("id") == exp["id"] and r.get("status") == "completed" for r in results):
        print(f"SKIP {exp['id']}: already completed", flush=True)
        return

    from ultralytics import YOLO
    import torch

    print("=" * 80)
    print(f"[{exp['id']}] START")
    print(f"  {exp['description']}")
    print("=" * 80, flush=True)

    t0 = time.time()

    args = dict(COMMON_HP)
    weights_in = args.pop("weights_start")
    args["data"] = str(exp["data"])
    args["name"] = exp["name"]

    model = YOLO(weights_in)
    model.train(**args)
    save_dir = Path(getattr(model.trainer, "save_dir",
                            PROJECT_ROOT / "runs" / "segment" / exp["name"]))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not produced at {best}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Eval on 3 vals
    metrics = {}
    for vname, vyaml in VAL_DATASETS.items():
        print(f"[{exp['id']}] eval on {vname}", flush=True)
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
    archive_name = f"{exp['id']}_v3val{v3_box:.3f}_mergedval{merged_box:.3f}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy(best, archive_path)

    wall_min = (time.time() - t0) / 60
    result = {
        "id": exp["id"],
        "description": exp["description"],
        "status": "completed",
        "wall_time_min": round(wall_min, 1),
        "best_pt_archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }
    results.append(result)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"[{exp['id']}] DONE in {wall_min:.1f} min")
    print(f"  merged Box mAP50 = {merged_box:.4f}")
    print(f"  v3-val Box mAP50 = {v3_box:.4f}")
    print(f"  v2-val Box mAP50 = {metrics['v2-val']['box_map50']:.4f}")
    print(flush=True)


def main():
    print(f"Running {len(EXPERIMENTS)} single-shot baselines on random phase subsets...")
    for exp in EXPERIMENTS:
        try:
            run(exp)
        except Exception as e:
            import traceback
            print(f"[{exp['id']}] FAILED: {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            results = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else []
            results.append({
                "id": exp["id"],
                "description": exp["description"],
                "status": "failed",
                "error": str(e),
            })
            RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
