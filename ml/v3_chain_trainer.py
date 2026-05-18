"""3-stage continual-learning chain training.

Реализация рекомендации paper #13 (remotesensing-14-01317-v2) которая
показала +16 Box mAP от pre-train-then-finetune на больших vs малых
датасетах.

Stages:
  Stage 1: yolov8m-seg ← COCO, train on **v1-only** (58 tiles).
           Цель — научить базовое представление trees на нашем satellite domain.
           Patience высокая (20), без агрессивного fine-tune posture.

  Stage 2: best.pt из stage 1 → train on **v1+v2** (111 tiles).
           Continual learning: модель уже знает v1, теперь добавляем v2.
           lr=0.001 (10× ниже COCO start), patience=15.

  Stage 3: best.pt из stage 2 → train on **v1+v2+v3** (152 tiles).
           Финальный fine-tune. Самый низкий lr=0.0001 (paper #7 рекомендация),
           patience=12 (paper #14: "overfit at >1 epoch" — будь готов рано стопиться).

Output: одна best.pt из stage 3 = financial chain output.
Auto-eval после на 3 vals + record в results/v3_experiments.json под id
'exp11_chain_3stage'.
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
RESULTS_FILE = PROJECT_ROOT / "results" / "v3_experiments.json"
ARCHIVE_DIR = PROJECT_ROOT / "weights" / "v3_runs"

STAGES = [
    {
        "name": "stage1_v1only",
        "data": str(PROJECT_ROOT / "yolov train dataset" / "v1only_yolo_mergedval_tiled" / "dataset.yaml"),
        "weights_in": "yolov8m-seg.pt",   # COCO
        "epochs": 100,
        "patience": 20,
        # use optimizer=auto — Ultralytics picks AdamW lr=0.002 like exp1
        "extra": {},
    },
    {
        "name": "stage2_v1v2",
        "data": str(PROJECT_ROOT / "yolov train dataset" / "v1v2_yolo_mergedval_tiled" / "dataset.yaml"),
        "weights_in": None,  # set to stage1 best.pt at runtime
        "epochs": 80,
        "patience": 15,
        "extra": dict(optimizer="AdamW", lr0=0.001, lrf=0.01),
    },
    {
        "name": "stage3_v1v2v3",
        "data": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
        "weights_in": None,  # set to stage2 best.pt
        "epochs": 60,
        "patience": 12,
        "extra": dict(optimizer="AdamW", lr0=0.0001, lrf=0.01),
    },
]

# Common hyperparams — match exp1 winner's setup (v2-proven aug + size m).
COMMON = dict(
    imgsz=640,
    batch=4,
    device=0,
    exist_ok=True,
    single_cls=True,
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
    time=0.75,  # max 45 min per stage (3 stages = ~2.25h max wall)
)

VAL_DATASETS = {
    "v2-val": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_v2val_tiled" / "dataset.yaml"),
    "v3-val": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_v3val_tiled" / "dataset.yaml"),
    "merged": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
}


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))


def save_results(results: list[dict]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def run_stage(stage: dict, prev_best: Path | None) -> Path:
    from ultralytics import YOLO
    import torch

    weights = prev_best if prev_best is not None else stage["weights_in"]
    args = dict(COMMON)
    args.update({
        "data": stage["data"],
        "epochs": stage["epochs"],
        "patience": stage["patience"],
        "name": f"v3_exp11_chain_{stage['name']}",
    })
    args.update(stage["extra"])

    print("=" * 70)
    print(f"[chain · {stage['name']}]")
    print(f"  weights_in: {weights}")
    print(f"  data: {stage['data']}")
    print(f"  epochs={args['epochs']} patience={args['patience']} lr0={args.get('lr0','auto')}")
    print("=" * 70, flush=True)

    model = YOLO(str(weights))
    model.train(**args)
    save_dir = Path(getattr(model.trainer, "save_dir",
                            PROJECT_ROOT / "runs" / "segment" / args["name"]))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not produced at {best}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[chain · {stage['name']}] done. best.pt -> {best}", flush=True)
    return best


def eval_on_all_vals(best_pt: Path) -> dict:
    from ultralytics import YOLO
    import torch
    metrics = {}
    for vname, vyaml in VAL_DATASETS.items():
        print(f"[chain · eval on {vname}]", flush=True)
        m = YOLO(str(best_pt))
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
    return metrics


def main():
    exp_id = "exp11_chain_3stage_v1_v2_v3"
    results = load_results()
    if any(r.get("id") == exp_id and r.get("status") == "completed" for r in results):
        print(f"SKIP {exp_id}: already completed")
        return

    print("=" * 80)
    print("v3 EXP11 — 3-stage continual learning chain")
    print("  Per paper #13 recipe (pre-train then fine-tune on cleaner set)")
    print("=" * 80, flush=True)

    t0 = time.time()
    prev_best: Path | None = None
    stage_metrics = {}

    for stage in STAGES:
        prev_best = run_stage(stage, prev_best)
        stage_metrics[stage["name"]] = str(prev_best.relative_to(PROJECT_ROOT))

    print()
    print("[chain] all 3 stages done, running final eval...", flush=True)
    final_metrics = eval_on_all_vals(prev_best)

    # Archive final best.pt с descriptive именем
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    merged_box = final_metrics["merged"]["box_map50"]
    v3_box = final_metrics["v3-val"]["box_map50"]
    archive_name = f"exp11_chain_3stage_v3val{v3_box:.3f}_mergedval{merged_box:.3f}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy(prev_best, archive_path)

    wall_min = (time.time() - t0) / 60
    result = {
        "id": exp_id,
        "description": "3-stage continual learning: yolov8m-seg COCO→v1-only→v1+v2→v1+v2+v3 (paper #13 recipe)",
        "status": "completed",
        "wall_time_min": round(wall_min, 1),
        "stage_checkpoints": stage_metrics,
        "best_pt_archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "metrics": final_metrics,
    }
    results.append(result)
    save_results(results)

    print()
    print("=" * 70)
    print(f"[chain] COMPLETED in {wall_min:.1f} min")
    print(f"  Final merged Box mAP50 = {merged_box:.4f}")
    print(f"  Final v3-val Box mAP50 = {v3_box:.4f}")
    print(f"  Final v2-val Box mAP50 = {final_metrics['v2-val']['box_map50']:.4f}")
    print(f"  Archived: {archive_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
