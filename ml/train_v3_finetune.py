"""v3 YOLOv8-seg fine-tune RUN 2 — adjusted hyperparams для шумных satellite labels.

Run1 итог на trained-on-v3-val:
  v2-finetune    on merged val = Box mAP50 0.167
  v3-finetune r1 on merged val = Box mAP50 0.268 (+61%)
  v3-finetune r1 on v3-val     = 0.238 (vs v2's 0.081, +193%)
  v3-finetune r1 on v2-val     = 0.334 (vs v2's 0.363, -8% catastrophic forgetting hint)

Run 2 idea — снизить overfit к noisy labels + использовать merged signal:
  * train=merged v1+v2+v3, **val=merged** (17 tiles, lower variance чем
    v3-only's 7 tiles — даст более стабильный best-epoch signal +
    balanced между v2 и v3 distributions)
  * `optimizer="AdamW"` + `lr0=0.0015` (explicit, чуть ниже run1's auto-0.002)
  * **Drop mixup=0 и copy_paste=0** — для satellite-tree segmentation
    с шумными polygon-edges они размывают shape и ломают context. Run1
    использовал v2's mixup=0.1, copy_paste=0.1 — пробуем без них.
  * Mild geometric aug: `degrees=10`, `translate=0.05`, `scale=0.2`
    (run1 имел v2's 20/0.1/0.4) — деревья rotation-invariant, но
    aggressive translate+scale на small dataset вредит
  * `mosaic=0.5` (run1 имел v2's 1.0) — половина batches без mosaic для
    cleaner gradient
  * `mask_ratio=2` (default 4) — higher-res mask head, лучше детали
  * `patience=50` (run1 75) — tighter, plateau определяется быстрее
  * `cos_lr=True`, `single_cls=True`, `cache=disk`, `workers=2`,
    `amp=True` — как run1

Стартовые веса: `weights/yolo_satellite.pt` (v2-finetune, yolov8x-seg) —
**сохраняем** v1+v2 prior. НЕ начинаем от COCO yolov8l-seg.pt чтобы не
потерять Astana обучение.

Hardware: RTX 4060 Laptop 8 GB. imgsz=640 batch=2 — same proven envelope.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    from ultralytics import YOLO

    project_root = Path(__file__).parent.parent
    weights_in = project_root / "weights" / "yolo_satellite.pt"
    data_yaml = project_root / "yolov train dataset" / "v3_yolo_mergedval_tiled" / "dataset.yaml"

    if not weights_in.exists():
        sys.exit(f"v2-finetune weights not found at {weights_in}")
    if not data_yaml.exists():
        sys.exit(f"Tiled dataset YAML not found at {data_yaml}")

    print("=" * 70)
    print("v3 YOLOv8-seg fine-tune RUN 2")
    print("=" * 70)
    print(f"  weights:  {weights_in}")
    print(f"  data:     {data_yaml}  (train+val both merged v1+v2+v3)")
    print(f"  device:   GPU (RTX 4060 Laptop)")
    print(f"  vs run1:  val=merged, AdamW explicit, drop mixup/copy_paste,")
    print(f"            milder geo-aug, mosaic=0.5, mask_ratio=2")
    print("=" * 70)

    model = YOLO(str(weights_in))

    model.train(
        data=str(data_yaml),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=2,
        device=0,
        name="astana_tiled_x_v3_finetune_r2",
        exist_ok=True,
        single_cls=True,

        # --- Optimizer + LR (explicit AdamW, reproducible) ---
        optimizer="AdamW",
        lr0=0.0015,               # gentle fine-tune-on-fine-tune
        lrf=0.01,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        nbs=64,

        # --- Loss weights (defaults) ---
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # --- Higher-res mask head (default mask_ratio=4 = 1/4 of input) ---
        mask_ratio=2,             # mask at half input resolution = more edge detail

        # --- Augmentation (milder, drop noisy aug for shape-sensitive task) ---
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.2,
        degrees=10, translate=0.05, scale=0.2, shear=0.0,
        perspective=0.0,
        flipud=0.5, fliplr=0.5,
        bgr=0.0,
        mosaic=0.5,
        close_mosaic=15,
        mixup=0.0,                # off — размывает crown shapes
        copy_paste=0.0,           # off — ломает shadows/context на satellite
        erasing=0.1,

        # --- Training meta ---
        amp=True,
        cache="disk",
        multi_scale=False,
        workers=2,
        save=True,
        plots=True,
    )

    save_dir = Path(getattr(model.trainer, "save_dir",
                            project_root / "runs" / "segment" / "astana_tiled_x_v3_finetune_r2"))
    best = save_dir / "weights" / "best.pt"
    print()
    print("=" * 70)
    print("Training done.")
    print(f"  Best:    {best}")
    print(f"  Last:    {save_dir / 'weights' / 'last.pt'}")
    print(f"  Curves:  {save_dir}")
    print()
    print("Compare with v2-finetune + run1:")
    print(f"  v2-finetune    Box mAP50 on merged val = 0.1667 (baseline)")
    print(f"  v3-finetune r1 Box mAP50 on merged val = 0.2681")
    print(f"  v3-finetune r2 Box mAP50 on merged val = (см. best epoch)")
    print()
    print("Чтобы backend подхватил новые веса:")
    print(f"  cp '{best}' weights/yolo_satellite.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
