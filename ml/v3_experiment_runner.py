"""Autonomous v3 experiment runner — sequential training + auto-eval pipeline.

Запускает несколько training runs подряд, каждый run заканчивает evaluation-ом
на 3 separate val'ах (v2-only, v3-only, merged) и сохранением best.pt с
descriptive именем. Результаты пишутся incrementally в JSON чтобы при крэше
не терять прогресс.

Гипотезы которые проверяем:
  exp1 — yolov8m-seg from COCO: smaller model на noisy data, обычно
         generalize лучше чем yolov8x на small datasets (per Google/общая
         literature consensus).
  exp2 — yolov8l-seg from COCO: medium-large, баланс между m и x.
  exp3 — yolov8x-seg from v2-finetune: continuation на merged data,
         то что v2-finetune был с v2 add v3. Аналог "v2-finetune redone
         with v3 data added".

Все используют v2-proven augmentation (которая дала 0.372 на v2-val),
`optimizer=auto` (matches v2-finetune methodology), val=merged для
early-stopping (lower variance чем v3-only).

После всех 3 — winner определяется по Box mAP50 на merged val.
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

# Все training runs валятся на train+val merged dataset.
TRAIN_DATA = str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "dataset.yaml")

# Для post-training evaluation — 3 разных val.
VAL_DATASETS = {
    "v2-val": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_v2val_tiled" / "dataset.yaml"),
    "v3-val": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_v3val_tiled" / "dataset.yaml"),
    "merged": str(PROJECT_ROOT / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
}

RESULTS_FILE = PROJECT_ROOT / "results" / "v3_experiments.json"
ARCHIVE_DIR = PROJECT_ROOT / "weights" / "v3_runs"

# v2-finetune backup (старая production до моего run1).
V2_FINETUNE_WEIGHTS = str(PROJECT_ROOT / "weights" / "archive" / "yolo" / "yolo_satellite_v2_finetune.pt")

# Общие hyperparams — v2-proven values + minimal tweaks для reproducibility.
COMMON_HP = dict(
    data=TRAIN_DATA,
    epochs=150,                  # max — обычно early-stop срабатывает раньше
    patience=30,                 # 30 эпох без нового best → stop (vs 50 ранее)
    time=1.5,                    # safety: max 1.5 часа per experiment даже если patience не сработал
    imgsz=640,
    device=0,
    exist_ok=True,
    single_cls=True,
    # Optimizer = auto (matches v2-finetune methodology)
    optimizer="auto",
    lrf=0.01,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    nbs=64,
    # Loss weights — defaults
    box=7.5, cls=0.5, dfl=1.5,
    # Augmentation — v2-proven values (доказали 0.372 на v2-val)
    hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
    degrees=20, translate=0.1, scale=0.4, shear=2.0,
    perspective=0.0,
    flipud=0.5, fliplr=0.5,
    bgr=0.0,
    mosaic=1.0, close_mosaic=10,
    mixup=0.1, copy_paste=0.1, erasing=0.2,
    # Training meta
    amp=True,
    cache="disk",
    multi_scale=False,
    workers=2,
    save=True,
    plots=True,
    verbose=False,
)

EXPERIMENTS = [
    {
        "id": "exp1_m_cocostart",
        "description": "yolov8m-seg (27M params) from COCO on merged v1+v2+v3 — hypothesis: smaller model generalize lучше on noisy small dataset",
        "weights_start": "yolov8m-seg.pt",
        "train_overrides": dict(name="v3_exp1_m_cocostart", batch=4),
    },
    {
        "id": "exp2_l_cocostart",
        "description": "yolov8l-seg (46M params) from COCO on merged v1+v2+v3 — hypothesis: medium-large model is the sweet spot",
        "weights_start": "yolov8l-seg.pt",
        "train_overrides": dict(name="v3_exp2_l_cocostart", batch=2),
    },
    {
        "id": "exp3_x_from_v2ft",
        "description": "yolov8x-seg (71M params) from v2-finetune weights on merged v1+v2+v3 — continuation of v2-finetune with v3 data",
        "weights_start": V2_FINETUNE_WEIGHTS,
        "train_overrides": dict(name="v3_exp3_x_from_v2ft", batch=2),
    },
    {
        "id": "exp4_x_from_v2ft_sgd_lr01",
        "description": "yolov8x-seg from v2-finetune, SGD lr=0.01 explicit (matches original v2-finetune recipe) — does SGD beat AdamW-auto on this dataset",
        "weights_start": V2_FINETUNE_WEIGHTS,
        "train_overrides": dict(
            name="v3_exp4_x_from_v2ft_sgd",
            batch=2,
            optimizer="SGD",
            lr0=0.01,
        ),
    },
    {
        "id": "exp5_l_from_v2ft_attempt",
        "description": "yolov8l-seg from COCO with HIGHER lr — explore if smaller model + faster LR converges to better minimum",
        "weights_start": "yolov8l-seg.pt",
        "train_overrides": dict(
            name="v3_exp5_l_higher_lr",
            batch=2,
            optimizer="SGD",
            lr0=0.01,
        ),
    },
    # ===== Round 2 (exp6-10) — adaptive, based on Round 1 results =====
    # Winner of Round 1: exp1_m_cocostart (yolov8m-seg from COCO, Box mAP50 0.308 on merged val).
    # Confirms Google's hypothesis: smaller model better on noisy small dataset. Now we
    # probe along 5 orthogonal axes от exp1's setup.
    {
        "id": "exp6_m_imgsz896",
        "description": "yolov8m-seg from COCO + imgsz=896 — higher resolution for small crowns (20-40px range moves to 28-56px)",
        "weights_start": "yolov8m-seg.pt",
        "train_overrides": dict(
            name="v3_exp6_m_imgsz896",
            batch=2,           # batch=4 at 896 will OOM on 8GB
            imgsz=896,
        ),
    },
    {
        "id": "exp7_s_cocostart",
        "description": "yolov8s-seg (12M params) from COCO — continue size-down sweep, test if even smaller generalizes better",
        "weights_start": "yolov8s-seg.pt",
        "train_overrides": dict(
            name="v3_exp7_s_cocostart",
            batch=8,           # s-seg is light, can fit more
        ),
    },
    {
        "id": "exp8_m_dropout015",
        "description": "yolov8m-seg from COCO + dropout=0.15 — extra regularization для noisy polygon labels",
        "weights_start": "yolov8m-seg.pt",
        "train_overrides": dict(
            name="v3_exp8_m_dropout015",
            batch=4,
            dropout=0.15,
        ),
    },
    {
        "id": "exp9_m_heavy_aug",
        "description": "yolov8m-seg from COCO + heavier aug (mixup=0.3, copy_paste=0.3) — opposite of run2's 'less aug', test if more diversity helps small dataset",
        "weights_start": "yolov8m-seg.pt",
        "train_overrides": dict(
            name="v3_exp9_m_heavy_aug",
            batch=4,
            mixup=0.3,
            copy_paste=0.3,
            mosaic=1.0,
            erasing=0.4,
        ),
    },
    {
        "id": "exp10_m_chain_from_exp1",
        "description": "yolov8m-seg chained — start from exp1's best.pt, continue with lr=0.0005 для gentle polish (continuation finetune)",
        # `exp1_m_cocostart_v3val0.287_mergedval0.308.pt` — exp1 best.pt
        "weights_start": str(PROJECT_ROOT / "weights" / "v3_runs" / "exp1_m_cocostart_v3val0.287_mergedval0.308.pt"),
        "train_overrides": dict(
            name="v3_exp10_m_chain",
            batch=4,
            optimizer="AdamW",
            lr0=0.0005,         # very gentle for already-trained model
        ),
    },
]


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_results(results: list[dict]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def already_completed(exp_id: str, results: list[dict]) -> bool:
    return any(r.get("id") == exp_id and r.get("status") == "completed" for r in results)


def run_one_experiment(exp: dict) -> dict:
    """Train + evaluate + archive. Returns result dict."""
    from ultralytics import YOLO
    import torch

    t0 = time.time()
    print("=" * 80)
    print(f"[{exp['id']}] START")
    print(f"  {exp['description']}")
    print(f"  weights: {exp['weights_start']}")
    print("=" * 80, flush=True)

    train_args = dict(COMMON_HP)
    train_args.update(exp["train_overrides"])

    # Training
    model = YOLO(exp["weights_start"])
    model.train(**train_args)

    # Locate best.pt
    save_dir = Path(getattr(model.trainer, "save_dir",
                            PROJECT_ROOT / "runs" / "segment" / train_args["name"]))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not produced at {best_pt}")

    train_wall_min = (time.time() - t0) / 60
    print(f"[{exp['id']}] training done in {train_wall_min:.1f} min, best.pt at {best_pt}", flush=True)

    # Free VRAM before eval
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Eval on all 3 vals
    metrics = {}
    for vname, vyaml in VAL_DATASETS.items():
        print(f"[{exp['id']}] eval on {vname}...", flush=True)
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

    # Archive best.pt с descriptive именем
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    merged_box = metrics["merged"]["box_map50"]
    v3_box = metrics["v3-val"]["box_map50"]
    archive_name = f"{exp['id']}_v3val{v3_box:.3f}_mergedval{merged_box:.3f}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy(best_pt, archive_path)
    print(f"[{exp['id']}] archived to {archive_path}", flush=True)

    wall_min = (time.time() - t0) / 60
    return {
        "id": exp["id"],
        "description": exp["description"],
        "weights_start": exp["weights_start"],
        "train_args": {k: v for k, v in train_args.items() if k not in {"verbose"}},
        "status": "completed",
        "wall_time_min": round(wall_min, 1),
        "train_time_min": round(train_wall_min, 1),
        "best_pt_archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }


def print_summary_table(results: list[dict]) -> None:
    print()
    print("=" * 110)
    print("SUMMARY TABLE (sorted by merged val Box mAP50)")
    print("=" * 110)
    print(f"{'experiment':<28} {'status':<10} {'time(min)':>9} {'v2-Box':>7} {'v3-Box':>7} {'mrg-Box':>8} {'mrg-Mask':>9}")
    print("-" * 110)
    completed = [r for r in results if r.get("status") == "completed"]
    completed.sort(key=lambda r: r["metrics"]["merged"]["box_map50"], reverse=True)
    for r in completed:
        m = r["metrics"]
        print(f"{r['id']:<28} {'OK':<10} {r['wall_time_min']:>9.1f} "
              f"{m['v2-val']['box_map50']:>7.3f} "
              f"{m['v3-val']['box_map50']:>7.3f} "
              f"{m['merged']['box_map50']:>8.3f} "
              f"{m['merged']['mask_map50']:>9.3f}")
    failed = [r for r in results if r.get("status") != "completed"]
    for r in failed:
        print(f"{r['id']:<28} FAIL: {r.get('error', 'unknown')}")
    print("=" * 110)
    print()
    # Baseline rows для context
    print("Reference (from earlier eval):")
    print(f"  v2-finetune (production до этого) on merged val Box mAP50 = 0.1667")
    print(f"  v3-finetune run1 on merged val Box mAP50 = 0.2681 (current production)")
    print()


def main():
    print()
    print("=" * 80)
    print("v3 Autonomous Experiment Runner")
    print("=" * 80)
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Train data: {TRAIN_DATA}")
    print(f"  Results JSON: {RESULTS_FILE}")
    print(f"  Best.pt archive dir: {ARCHIVE_DIR}")
    print("=" * 80)

    results = load_results()
    if results:
        completed_ids = {r["id"] for r in results if r.get("status") == "completed"}
        print(f"Found existing results for: {sorted(completed_ids)}")

    for exp in EXPERIMENTS:
        if already_completed(exp["id"], results):
            print(f"SKIP {exp['id']}: already completed in prior run", flush=True)
            continue

        try:
            result = run_one_experiment(exp)
            results.append(result)
            save_results(results)
            m = result["metrics"]
            print(f"[{exp['id']}] COMPLETED in {result['wall_time_min']:.1f} min")
            print(f"  v2-val:  Box mAP50 = {m['v2-val']['box_map50']:.4f}")
            print(f"  v3-val:  Box mAP50 = {m['v3-val']['box_map50']:.4f}")
            print(f"  merged:  Box mAP50 = {m['merged']['box_map50']:.4f}")
            print(flush=True)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            print(f"[{exp['id']}] FAILED: {error_msg}", flush=True)
            print(tb, flush=True)
            results.append({
                "id": exp["id"],
                "description": exp["description"],
                "status": "failed",
                "error": error_msg,
                "traceback": tb,
            })
            save_results(results)
            # Continue to next experiment

    print_summary_table(results)
    print("ALL EXPERIMENTS DONE.")


if __name__ == "__main__":
    main()
