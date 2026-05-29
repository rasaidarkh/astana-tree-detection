"""Random-shuffle chain learning — control experiment for exp11's failure.

Hypothesis being tested:
  exp11 (3-stage chain on v1→v1+v2→v1+v2+v3) failed at 0.210 (worst of 16).
  Was it because:
    (A) Sequential stage training itself hurts, OR
    (B) v1/v2/v3 batches are distributionally different (each covers different
        Astana districts at different annotation quality), so stages bias the
        model toward early stages and Stage 3 can't recover?

This script tests (B) directly by replacing version-based splits with
**random splits of the same merged train pool**. If random chain ≈ exp1 single-shot,
hypothesis (B) is supported. If random chain << exp1, hypothesis (A) wins.

## Experiments

exp17_random_chain_3stage_cumulative:
  Phase 1: random 21 imgs (33% of 63), train from COCO, lr=auto, patience=20
  Phase 2: random 42 imgs (66% — includes phase 1), train from p1.pt, lr=0.001, patience=15
  Phase 3: all 63 imgs, train from p2.pt, lr=0.0001, patience=12
  → same LR schedule as exp11, just random splits instead of version splits.

exp18_random_chain_2stage_hot:
  Phase 1: random 31 imgs (50%), train from COCO, lr=auto, patience=30
  Phase 2: all 63 imgs, train from p1.pt, lr=0.001 (NOT 0.0001), patience=20
  → mimics v2-finetune's 2-stage chain pattern but with random splits + hot LR.
  This tests if 2-stage chain with proper "hot" LR beats single-shot.

Both use yolov8m-seg + v2-proven aug (= exp1 winner's config).
Cumulative splits (phase 2 ⊇ phase 1) to match real-world data-growth pattern.
Seed=42 for reproducibility.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import random
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

# Hyperparameters — exp1 winner config (v2-proven aug + yolov8m + AdamW auto)
COMMON_HP = dict(
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
    time=0.75,  # max 45 min per stage
)


def _run(cmd: list[str]) -> None:
    """Run an Ultralytics-related subprocess command and bubble up failures."""
    import subprocess
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {cmd}")


def build_random_phase_datasets(phase_sizes: list[int], suffix: str, seed: int = 42) -> list[Path]:
    """Take v3_merged/instances_Train.json, shuffle images with seed, write
    cumulative-subset COCO JSONs (each phase ⊇ previous), then run coco_to_yolo_seg
    + tile_dataset for each. Returns list of tiled dataset.yaml paths."""
    src = DATASET_ROOT / "v3_merged" / "instances_Train.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    images = data["images"]
    all_ids = [im["id"] for im in images]
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    img_by_id = {im["id"]: im for im in images}
    anns = data["annotations"]
    anns_by_img: dict[int, list] = {}
    for a in anns:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    yaml_paths = []
    cumulative_ids: list[int] = []
    for phase_idx, size in enumerate(phase_sizes, 1):
        # Take first `size` items from shuffled list (cumulative: phase 2 includes phase 1)
        phase_ids = all_ids[:size]
        phase_imgs = [img_by_id[i] for i in phase_ids]
        phase_anns = [a for img_id in phase_ids for a in anns_by_img.get(img_id, [])]

        out_coco = DATASET_ROOT / "v3_merged" / f"instances_Train_{suffix}_phase{phase_idx}.json"
        out_data = {
            "images": phase_imgs,
            "annotations": phase_anns,
            "categories": data["categories"],
            "info": data.get("info", {}),
        }
        out_coco.write_text(json.dumps(out_data, ensure_ascii=False), encoding="utf-8")
        print(f"phase {phase_idx}: {len(phase_imgs)} imgs, {len(phase_anns)} anns -> {out_coco.name}", flush=True)

        # Convert to YOLO
        yolo_out = DATASET_ROOT / f"{suffix}_phase{phase_idx}_yolo_mergedval"
        _run([
            str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe"),
            str(PROJECT_ROOT / "ml" / "coco_to_yolo_seg.py"),
            "--train-coco", str(out_coco),
            "--val-coco", str(DATASET_ROOT / "v3_merged" / "instances_Validation.json"),
            "--images-dir", str(DATASET_ROOT / "v3_merged" / "images"),
            "--output", str(yolo_out),
        ])

        # Tile
        tiled_out = DATASET_ROOT / f"{suffix}_phase{phase_idx}_yolo_mergedval_tiled"
        _run([
            str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe"),
            str(PROJECT_ROOT / "ml" / "tile_dataset.py"),
            "--input", str(yolo_out),
            "--output", str(tiled_out),
            "--tile-size", "640", "--overlap", "128", "--min-area", "25",
        ])
        yaml_paths.append(tiled_out / "dataset.yaml")

    return yaml_paths


def run_chain(exp_id: str, description: str, stage_configs: list[dict]) -> None:
    """Run a multi-stage chain training. stage_configs is a list of dicts with:
      'data': dataset.yaml path
      'weights_in': initial weights (or None to use prev stage's best.pt)
      'name': run name (for runs/segment/<name>/)
      'epochs', 'patience': training budget
      'lr0': learning rate (optional, AdamW auto-picks if omitted)
    """
    results = []
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    if any(r.get("id") == exp_id and r.get("status") == "completed" for r in results):
        print(f"SKIP {exp_id}: already completed", flush=True)
        return

    from ultralytics import YOLO
    import torch

    print("=" * 80)
    print(f"EXPERIMENT: {exp_id}")
    print(f"  {description}")
    print("=" * 80, flush=True)

    t0 = time.time()
    prev_best: Path | None = None
    stage_artifacts = {}

    for cfg in stage_configs:
        weights = prev_best if prev_best is not None else cfg["weights_in"]
        args = dict(COMMON_HP)
        args.update({
            "data": str(cfg["data"]),
            "epochs": cfg["epochs"],
            "patience": cfg["patience"],
            "name": cfg["name"],
        })
        if "lr0" in cfg:
            args["optimizer"] = "AdamW"
            args["lr0"] = cfg["lr0"]
            args["lrf"] = 0.01
        else:
            args["optimizer"] = "auto"

        print("=" * 70)
        print(f"[{exp_id}] stage: {cfg['name']}")
        print(f"  weights_in: {weights}")
        print(f"  data: {cfg['data']}")
        print(f"  epochs={args['epochs']} patience={args['patience']} optimizer={args['optimizer']} lr0={args.get('lr0','auto')}")
        print("=" * 70, flush=True)

        model = YOLO(str(weights))
        model.train(**args)
        save_dir = Path(getattr(model.trainer, "save_dir",
                                PROJECT_ROOT / "runs" / "segment" / cfg["name"]))
        best = save_dir / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"best.pt not at {best}")
        stage_artifacts[cfg["name"]] = str(best.relative_to(PROJECT_ROOT))
        prev_best = best

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Eval on 3 vals
    print(f"[{exp_id}] all stages done, evaluating on 3 vals...", flush=True)
    metrics = {}
    for vname, vyaml in VAL_DATASETS.items():
        print(f"[{exp_id}] eval on {vname}", flush=True)
        m = YOLO(str(prev_best))
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

    # Archive
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    merged_box = metrics["merged"]["box_map50"]
    v3_box = metrics["v3-val"]["box_map50"]
    archive_name = f"{exp_id}_v3val{v3_box:.3f}_mergedval{merged_box:.3f}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy(prev_best, archive_path)

    wall_min = (time.time() - t0) / 60
    result = {
        "id": exp_id,
        "description": description,
        "status": "completed",
        "wall_time_min": round(wall_min, 1),
        "stage_checkpoints": stage_artifacts,
        "best_pt_archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }
    results.append(result)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"[{exp_id}] COMPLETED in {wall_min:.1f} min")
    print(f"  merged Box mAP50 = {merged_box:.4f}  (target to beat: exp1 = 0.308)")
    print(f"  v3-val Box mAP50 = {v3_box:.4f}")
    print(f"  v2-val Box mAP50 = {metrics['v2-val']['box_map50']:.4f}")
    print(f"  Archived: {archive_path}")
    print("=" * 70, flush=True)


def main():
    # ===== Build random phase datasets (only once) =====
    print("Building random 3-phase datasets (33%/66%/100%)...", flush=True)
    yaml_3stage = build_random_phase_datasets([21, 42, 63], suffix="random3", seed=42)

    print("Building random 2-phase datasets (50%/100%)...", flush=True)
    yaml_2stage = build_random_phase_datasets([31, 63], suffix="random2", seed=42)

    # ===== exp17: 3-stage cumulative random chain =====
    run_chain(
        exp_id="exp17_random_chain_3stage_cumulative",
        description="3-stage cumulative random chain (21→42→63 imgs, seed=42). Same LR schedule as exp11 but random splits — isolates chain mechanism from version drift.",
        stage_configs=[
            {"data": yaml_3stage[0], "weights_in": "yolov8m-seg.pt", "name": "v3_exp17_random_p1", "epochs": 100, "patience": 20},
            {"data": yaml_3stage[1], "weights_in": None, "name": "v3_exp17_random_p2", "epochs": 80, "patience": 15, "lr0": 0.001},
            {"data": yaml_3stage[2], "weights_in": None, "name": "v3_exp17_random_p3", "epochs": 60, "patience": 12, "lr0": 0.0001},
        ],
    )

    # ===== exp18: 2-stage random chain with HOT LR =====
    run_chain(
        exp_id="exp18_random_chain_2stage_hot",
        description="2-stage random chain (31→63 imgs, seed=42). Stage 2 lr=0.001 (hot enough to actually learn). Mimics v2-finetune chain pattern with random splits — tests whether 2-stage + hot LR beats single-shot.",
        stage_configs=[
            {"data": yaml_2stage[0], "weights_in": "yolov8m-seg.pt", "name": "v3_exp18_random2_p1", "epochs": 100, "patience": 30},
            {"data": yaml_2stage[1], "weights_in": None, "name": "v3_exp18_random2_p2", "epochs": 80, "patience": 20, "lr0": 0.001},
        ],
    )


if __name__ == "__main__":
    main()
