"""v3 YOLOv8-seg fine-tune от v2 production-весов на merged v1+v2+v3.

Noise-robust hyperparameters: dataset собирался разными людьми с
неравномерной precision разметки (некоторые polygon-ы edge-нечёткие, какие-то
кроны пропущены, какие-то — false-positive с кустами). Поэтому:

  * `label_smoothing=0.1` — модель не уверена в 100% правильности
    каждого label, более стабильна к noise
  * `mixup=0.2` и `copy_paste=0.3` — сильное "augmentation regularization",
    forces модель учить shape-invariant features а не запомнить
    конкретные мисс-аннотации
  * `mosaic=1.0` (4 кадра в один) + `close_mosaic=20` last эпох —
    стандартный YOLOv8 anti-overfit prior
  * `degrees=30` + полные flip-ы — trees rotation/mirror invariant,
    safe для augmentation
  * `lr0=0.001` — fine-tune LR (10× ниже from-scratch default 0.01)
    + cosine decay через `lrf=0.01`
  * `optimizer=AdamW` — лучше чем SGD на fine-tune small dataset
  * `patience=60` epoch — long, чтобы early-stop случился на реальном
    плато а не на шумной фазе
  * `multi_scale=True` — мульти-разрешение тренировки, регуляризатор
    для small dataset

Hardware: RTX 4060 Laptop 8 GB. yolov8x-seg + batch 4 + imgsz 640 + AMP
влезает с запасом (~6 GB VRAM). batch 8 рискует OOM на augmentation peak'ах.

Дольше тренировки за счёт agressive augmentation, но **финальная mAP должна
вырасти заметно** на нашем merged val.
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
    data_yaml = project_root / "yolov train dataset" / "v3_yolo_tiled" / "dataset.yaml"

    if not weights_in.exists():
        sys.exit(f"v2 production weights not found at {weights_in}")
    if not data_yaml.exists():
        sys.exit(f"Tiled dataset YAML not found at {data_yaml}")

    print("=" * 70)
    print("v3 YOLOv8-seg fine-tune от v2-finetune")
    print("=" * 70)
    print(f"  weights:  {weights_in}")
    print(f"  data:     {data_yaml}")
    print(f"  device:   GPU (RTX 4060 Laptop)")
    print(f"  strategy: noise-robust fine-tune (см. docstring)")
    print("=" * 70)

    model = YOLO(str(weights_in))

    model.train(
        data=str(data_yaml),
        epochs=300,
        patience=60,
        imgsz=640,
        batch=4,
        device=0,
        name="astana_tiled_x_v3_finetune",
        exist_ok=True,
        # --- Optimizer + LR (fine-tune posture) ---
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.001,
        # --- Anti-noise regularization ---
        label_smoothing=0.1,
        # --- Augmentation (aggressive for small noisy dataset) ---
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=30, translate=0.15, scale=0.5, shear=3.0,
        perspective=0.0,  # satellite — без perspective
        flipud=0.5, fliplr=0.5,
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.2,
        copy_paste=0.3,
        erasing=0.3,
        # --- Training meta ---
        amp=True,
        cache="ram",
        multi_scale=True,
        save=True,
        plots=True,
        nbs=64,
    )

    save_dir = Path(getattr(model.trainer, "save_dir",
                            project_root / "runs" / "segment" / "astana_tiled_x_v3_finetune"))
    best = save_dir / "weights" / "best.pt"
    print()
    print("=" * 70)
    print("Training done.")
    print(f"  Best:    {best}")
    print(f"  Last:    {save_dir / 'weights' / 'last.pt'}")
    print(f"  Curves:  {save_dir}")
    print()
    print("Чтобы backend подхватил новые веса:")
    print(f"  cp '{best}' weights/yolo_satellite.pt")
    print()
    print("Eval (если нужен явный val-прогон отдельно):")
    print(f"  yolo segment val model='{best}' data='{data_yaml}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
