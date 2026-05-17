"""v3 YOLOv8-seg fine-tune от v2-finetune весов, train на merged v1+v2+v3.

Стратегия — "ideal" hyperparams: стартуем от v2-finetune proven settings
(они уже доказали Box mAP50=0.372 на v2 val) и вносим минимальные,
evidence-based adjustments:

  * `lr0=0.005` (вместо v2's 0.01) — fine-tune-on-fine-tune posture,
    модель уже well-trained, не нужно её "сильно толкать"
  * `cos_lr=True` (вместо v2's linear) — smoother decay, обычно даёт
    +1-2% mAP на small datasets
  * `single_cls=True` — у нас 1 класс (tree), efficiency win
  * `patience=75` (вместо v2's 50) — больше времени найти лучший epoch
  * `cache="disk"` — faster iteration, deterministic vs "ram"
  * `workers=2` — Windows DLL race на default 8

Всё остальное (augmentation, loss weights, batch, imgsz, optimizer=auto)
— как в v2-finetune. Это даёт **честный A/B**: одинаковые conditions,
только train data добавлено v3.

Val set: ТОЛЬКО v3 (`v3 annotations/.../instances_Validation.json`,
7 tiles / 497 polygons) — тестирует ровно ту distribution что мы добавили.
v2-finetune на этом val = Box mAP50 **0.0811** (измерено, gap огромный
из-за of distribution shift). Цель v3 fine-tune — закрыть этот gap.

Hardware: RTX 4060 Laptop 8 GB. yolov8x-seg + batch=2 + imgsz=640 + AMP
устойчиво помещается, peak VRAM ≈ 6.4 GB (как у v2).
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
    data_yaml = project_root / "yolov train dataset" / "v3_yolo_v3val_tiled" / "dataset.yaml"

    if not weights_in.exists():
        sys.exit(f"v2-finetune weights not found at {weights_in}")
    if not data_yaml.exists():
        sys.exit(f"Tiled dataset YAML not found at {data_yaml}")

    print("=" * 70)
    print("v3 YOLOv8-seg fine-tune от v2-finetune")
    print("=" * 70)
    print(f"  weights:  {weights_in}")
    print(f"  data:     {data_yaml}")
    print(f"  val:      v3-only (v2 baseline on this val = Box mAP50 0.0811)")
    print(f"  device:   GPU (RTX 4060 Laptop)")
    print(f"  strategy: v2 hyperparams + minimal adjustments")
    print("=" * 70)

    model = YOLO(str(weights_in))

    model.train(
        data=str(data_yaml),
        epochs=300,
        patience=75,
        imgsz=640,
        batch=2,
        device=0,
        name="astana_tiled_x_v3_finetune",
        exist_ok=True,
        single_cls=True,

        # --- Optimizer + LR ---
        # `optimizer=auto` = same as v2-finetune (Ultralytics picks SGD/AdamW
        # by dataset size; with ~8800 polygons typically picks SGD).
        optimizer="auto",
        lr0=0.005,                # half of v2's 0.01 — fine-tune-on-fine-tune
        lrf=0.01,                 # final LR = lr0 * lrf = 0.00005
        cos_lr=True,              # cosine schedule, smoother than linear
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        nbs=64,

        # --- Loss weights (v2 defaults) ---
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # --- Augmentation (v2-finetune proven values) ---
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=20, translate=0.1, scale=0.4, shear=2.0,
        perspective=0.0,
        flipud=0.5, fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.2,

        # --- Training meta ---
        amp=True,
        cache="disk",
        multi_scale=False,
        workers=2,
        save=True,
        plots=True,
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
    print("Compare directly with v2-finetune:")
    print(f"  v2-finetune Box mAP50 on v3-val = 0.0811 (baseline)")
    print(f"  v3-finetune Box mAP50 on v3-val = (см. best epoch)")
    print()
    print("Чтобы backend подхватил новые веса:")
    print(f"  cp '{best}' weights/yolo_satellite.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
