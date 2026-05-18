"""Unified evaluation: run top-N models on all 3 vals, produce one comparison table.

Used as a single source of truth for thesis ablation table. Each model is loaded
once, evaluated on v2-val / v3-val / merged val, results aggregated.

Output: console table + saved JSON to results/v5_unified_eval.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "yolov train dataset"
sys.path.insert(0, str(PROJECT_ROOT / "ml"))
from v5_top_models import TOP_MODELS, filter_existing  # noqa: E402

OUT_FILE = PROJECT_ROOT / "results" / "v5_unified_eval.json"

VAL_DATASETS = {
    "v2-val (10 tiles, 258 polys)": str(DATASET_ROOT / "v3_yolo_v2val_tiled" / "dataset.yaml"),
    "v3-val (7 tiles, 497 polys)": str(DATASET_ROOT / "v3_yolo_v3val_tiled" / "dataset.yaml"),
    "merged (17 tiles, 755 polys)": str(DATASET_ROOT / "v3_yolo_mergedval_tiled" / "dataset.yaml"),
}


def main():
    from ultralytics import YOLO
    import torch

    models = filter_existing(TOP_MODELS)
    print(f"Evaluating {len(models)} models × {len(VAL_DATASETS)} val sets")
    print()

    all_rows = []
    t_total = time.time()

    for name, path, note in models:
        print(f"[{name}] loading {path.name}...", flush=True)
        m = YOLO(str(path))
        row = {"model": name, "weights": str(path.relative_to(PROJECT_ROOT)), "note": note, "metrics": {}}
        for vlabel, vyaml in VAL_DATASETS.items():
            t0 = time.time()
            r = m.val(data=vyaml, imgsz=640, batch=2, device=0,
                      plots=False, save=False, verbose=False)
            row["metrics"][vlabel] = {
                "box_map50": float(r.box.map50),
                "box_map": float(r.box.map),
                "box_p": float(r.box.mp),
                "box_r": float(r.box.mr),
                "mask_map50": float(r.seg.map50),
                "mask_map": float(r.seg.map),
            }
            print(f"  {vlabel:<32}  Box={r.box.map50:.4f}  Mask={r.seg.map50:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        all_rows.append(row)
        print()
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save JSON
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print formatted table
    print()
    print("=" * 130)
    print(f'{"model":<22} | {"v2-val Box":>10} | {"v3-val Box":>10} | {"merged Box":>10} | {"merged Mask":>11} | note')
    print("=" * 130)
    sort = sorted(all_rows, key=lambda r: r["metrics"]["merged (17 tiles, 755 polys)"]["box_map50"], reverse=True)
    for r in sort:
        m = r["metrics"]
        v2b = m["v2-val (10 tiles, 258 polys)"]["box_map50"]
        v3b = m["v3-val (7 tiles, 497 polys)"]["box_map50"]
        mrgb = m["merged (17 tiles, 755 polys)"]["box_map50"]
        mrgmask = m["merged (17 tiles, 755 polys)"]["mask_map50"]
        print(f'{r["model"]:<22} | {v2b:>10.4f} | {v3b:>10.4f} | {mrgb:>10.4f} | {mrgmask:>11.4f} | {r["note"]}')
    print("=" * 130)
    print(f"\nTotal eval time: {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
