"""Clean YOLO model sweep — no tuning, Ultralytics defaults, merged dataset.

Forgets everything from exp1-23 tuning history. Runs each YOLOv8 segmentation
variant (n/s/m/l/x) with:
  - Ultralytics default optimizer/lr/augmentation/loss weights
  - Same merged v1+v2+v3 dataset (152 train tiles, 17 val tiles randomly shuffled)
  - imgsz=640 (matches tile size)
  - batch=-1 (Ultralytics AutoBatch — finds 60% VRAM utilization)
  - epochs=150, patience=50, time=1.5h per model
  - single_cls=True (we have one class: tree)

Results saved to results/v4_clean_modelsweep.json (separate from v3 experiments
to keep clean comparison).

Expected wall time: ~2.5-3 hours total (n fastest, x slowest).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "yolov train dataset"
RESULTS_FILE = PROJECT_ROOT / "results" / "v4_clean_modelsweep.json"
ARCHIVE_DIR = PROJECT_ROOT / "weights" / "v4_clean"

# Same merged dataset everyone tests against. Ultralytics shuffles training
# samples within epochs by default — that's our randomization.
TRAIN_DATA = str(DATASET_ROOT / "v3_yolo_mergedval_tiled" / "dataset.yaml")

VAL_DATASETS = {
    "v2-val": str(DATASET_ROOT / "v3_yolo_v2val_tiled" / "dataset.yaml"),
    "v3-val": str(DATASET_ROOT / "v3_yolo_v3val_tiled" / "dataset.yaml"),
    "merged": str(DATASET_ROOT / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
}

EXPERIMENTS = [
    {"id": "v4_n_clean", "model": "yolov8n-seg.pt", "params_M": 3.4},
    {"id": "v4_s_clean", "model": "yolov8s-seg.pt", "params_M": 11.8},
    {"id": "v4_m_clean", "model": "yolov8m-seg.pt", "params_M": 27.2},
    {"id": "v4_l_clean", "model": "yolov8l-seg.pt", "params_M": 45.9},
    {"id": "v4_x_clean", "model": "yolov8x-seg.pt", "params_M": 71.7},
]


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))


def save_results(results: list[dict]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(exp: dict) -> dict:
    from ultralytics import YOLO
    import torch

    t0 = time.time()
    print("=" * 80)
    print(f"[{exp['id']}] START — {exp['model']} ({exp['params_M']}M params)")
    print(f"  CLEAN run: Ultralytics defaults only, no manual tuning")
    print("=" * 80, flush=True)

    model = YOLO(exp["model"])
    model.train(
        data=TRAIN_DATA,
        imgsz=640,
        batch=-1,             # AutoBatch — Ultralytics finds 60% VRAM util
        epochs=150,
        patience=50,
        time=1.5,             # safety cap
        device=0,
        single_cls=True,      # we have 1 class
        # Everything else = Ultralytics defaults
        # optimizer=auto, lr0=auto, augmentation=default, loss weights=default
        name=exp["id"],
        exist_ok=True,
        cache="disk",
        workers=2,
        amp=True,
        save=True,
        plots=True,
        verbose=False,
    )

    save_dir = Path(getattr(model.trainer, "save_dir",
                            PROJECT_ROOT / "runs" / "segment" / exp["id"]))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not at {best}")

    train_min = (time.time() - t0) / 60
    print(f"[{exp['id']}] training done in {train_min:.1f} min", flush=True)

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

    wall = (time.time() - t0) / 60
    return {
        "id": exp["id"],
        "model": exp["model"],
        "params_M": exp["params_M"],
        "status": "completed",
        "wall_time_min": round(wall, 1),
        "best_pt_archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 110)
    print(f"{'model':<20} {'params':>7} {'time':>5} {'v2-Box':>7} {'v3-Box':>7} {'mrg-Box':>8} {'mrg-Mask':>9}")
    print("=" * 110)
    sort = sorted(results, key=lambda r: r.get("metrics", {}).get("merged", {}).get("box_map50", -1), reverse=True)
    for r in sort:
        if r.get("status") != "completed":
            print(f"{r['id']:<20} FAIL: {r.get('error','?')[:50]}")
            continue
        m = r["metrics"]
        print(f"{r['id']:<20} {r['params_M']:>5.1f}M {r['wall_time_min']:>5.0f}m "
              f"{m['v2-val']['box_map50']:>7.3f} {m['v3-val']['box_map50']:>7.3f} "
              f"{m['merged']['box_map50']:>8.3f} {m['merged']['mask_map50']:>9.3f}")
    print("=" * 110)


def main():
    print()
    print("=" * 80)
    print("V4 CLEAN MODEL SWEEP — Ultralytics defaults, merged dataset")
    print(f"  Models: {len(EXPERIMENTS)} (n/s/m/l/x)")
    print(f"  Train data: {TRAIN_DATA}")
    print(f"  Results: {RESULTS_FILE}")
    print("=" * 80)

    results = load_results()
    completed_ids = {r["id"] for r in results if r.get("status") == "completed"}

    for exp in EXPERIMENTS:
        if exp["id"] in completed_ids:
            print(f"SKIP {exp['id']}: already in results JSON", flush=True)
            continue
        try:
            result = run_one(exp)
            results.append(result)
            save_results(results)
            m = result["metrics"]
            print()
            print(f"[{exp['id']}] DONE in {result['wall_time_min']:.0f} min")
            print(f"  merged Box mAP50 = {m['merged']['box_map50']:.4f}")
            print(f"  v3-val Box mAP50 = {m['v3-val']['box_map50']:.4f}")
            print(f"  v2-val Box mAP50 = {m['v2-val']['box_map50']:.4f}")
            print(flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            print(f"[{exp['id']}] FAILED: {err}", flush=True)
            print(tb, flush=True)
            results.append({
                "id": exp["id"],
                "status": "failed",
                "error": err,
                "traceback": tb,
            })
            save_results(results)

    print_summary(results)


if __name__ == "__main__":
    main()
