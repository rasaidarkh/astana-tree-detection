"""Прогон 4 YOLO чекпоинтов на 14-image merged val (17 tiles).

Этот val собран в `v3_yolo_mergedval_tiled/` — 4 v1 + 5 v2 + 5 v3 source images,
17 tiles после tiling 640+128. ОТЛИЧАЕТСЯ от 15-image val Ануара/Берика на
одну v1 image (`Снимок экрана 2026-04-01 194422.png`), которая в YOLO train
corpus из-за `--dup-policy keep-train` для pre-split duplicate.

Выход — JSON в stdout + сохранение в `results/yolo_mergedval_eval.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CKPTS = {
    "v1": PROJECT_ROOT / "runs/segment/astana_tiled_x_max/weights/best.pt",
    "v2-fromscratch": PROJECT_ROOT / "runs/segment/astana_tiled_x_v2_fromscratch/weights/best.pt",
    "v2-finetune": PROJECT_ROOT / "weights/archive/yolo/yolo_satellite_v2_finetune.pt",
    # The GENUINE run1 checkpoint. NOTE: weights/yolo_satellite.pt is byte-identical
    # to the v4_x_clean champion (MD5 58fb1c00...), so pointing this key there
    # previously re-scored the champion and mislabelled it as "run1" (0.287) in
    # Table 3.2 and Figs 3.1/3.6. The real run1 (MD5 bd8d923b...) scores 0.268.
    "v3-finetune-run1": PROJECT_ROOT / "weights/v3_runs/v3_finetune_run1_ep58_v3val0220_mergedval0268.pt",
}
DATA_YAML = PROJECT_ROOT / "yolov train dataset/v3_yolo_mergedval_tiled/dataset.yaml"
OUT_JSON = PROJECT_ROOT / "results/yolo_mergedval_eval.json"


def main() -> None:
    from ultralytics import YOLO
    import torch

    assert DATA_YAML.exists(), DATA_YAML
    for ckpt in CKPTS.values():
        assert ckpt.exists(), ckpt

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, ckpt in CKPTS.items():
        print(f"\n{'='*70}\n{name}  ({ckpt.relative_to(PROJECT_ROOT)})\n{'='*70}")
        model = YOLO(str(ckpt))
        r = model.val(
            data=str(DATA_YAML),
            imgsz=640,
            device=0,
            plots=False,
            save_json=False,
            verbose=False,
            split="val",
        )

        def _f(x):
            try:
                return float(x)
            except Exception:
                try:
                    return float(x.mean())
                except Exception:
                    return None

        rec = {
            "box": {
                "map50": _f(r.box.map50),
                "map50_95": _f(r.box.map),
                "p": _f(r.box.mp),
                "r": _f(r.box.mr),
            },
            "mask": {
                "map50": _f(r.seg.map50) if hasattr(r, "seg") else None,
                "map50_95": _f(r.seg.map) if hasattr(r, "seg") else None,
                "p": _f(r.seg.mp) if hasattr(r, "seg") else None,
                "r": _f(r.seg.mr) if hasattr(r, "seg") else None,
            },
        }
        results[name] = rec
        print(f"Box  mAP@50={rec['box']['map50']:.4f}  mAP@50:95={rec['box']['map50_95']:.4f}  P={rec['box']['p']:.4f}  R={rec['box']['r']:.4f}")
        if rec["mask"]["map50"] is not None:
            print(f"Mask mAP@50={rec['mask']['map50']:.4f}  mAP@50:95={rec['mask']['map50_95']:.4f}  P={rec['mask']['p']:.4f}  R={rec['mask']['r']:.4f}")

        del model
        torch.cuda.empty_cache()

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Summary table
    print("\n" + "="*100)
    print(f"{'Model':<22}{'Box mAP@50':>12}{'Box mAP@50:95':>15}{'Box P':>10}{'Box R':>10}{'Mask mAP@50':>13}{'Mask mAP@50:95':>16}")
    print("="*100)
    for name, rec in results.items():
        b = rec["box"]; m = rec["mask"]
        mm50 = f"{m['map50']:.4f}" if m["map50"] is not None else "-"
        mm5095 = f"{m['map50_95']:.4f}" if m["map50_95"] is not None else "-"
        print(f"{name:<22}{b['map50']:>12.4f}{b['map50_95']:>15.4f}{b['p']:>10.4f}{b['r']:>10.4f}{mm50:>13}{mm5095:>16}")
    print("="*100)


if __name__ == "__main__":
    main()
