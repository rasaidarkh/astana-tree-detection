"""Тренировка YOLOv8-seg на расширенном датасете Астаны.

Стратегия для 100% защиты:
  1. Старт с весов из 70%-защиты (best.pt, 97 эпох на 134 LabelMe-кадрах)
  2. Дообучение на объединённом датасете: старые 134 + новые 15 из CVAT
  3. Augmentation подобрана под satellite imagery (поворот, flip, цвет)

Пример:
    python ml/train_yolo.py \
        --data data/processed/combined/dataset.yaml \
        --weights weights/yolo_satellite.pt \
        --epochs 50 \
        --imgsz 1024 \
        --batch 4 \
        --name astana_v2

Резюме обучения после прерывания:
    python ml/train_yolo.py --resume runs/segment/astana_v2/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for Astana trees")
    parser.add_argument("--data", required=True, help="Путь к dataset.yaml")
    parser.add_argument("--weights", default="yolov8m-seg.pt",
                        help="Стартовые веса. Можно: yolov8n-seg.pt | yolov8s-seg.pt | yolov8m-seg.pt | путь к best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Большой imgsz нужен для satellite — деревья мелкие")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU id, 'cpu', либо '0,1' для multi-GPU")
    parser.add_argument("--name", default="astana_yolo", help="Имя ран папки")
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--resume", default=None, help="Путь к last.pt для resume")
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        print(f"Resuming from {args.resume}")
    else:
        model = YOLO(args.weights)
        print(f"Starting from {args.weights}")

    print("=" * 60)
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Imgsz:   {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {args.device}")
    print("=" * 60)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        save=True,
        plots=True,
        # Augmentation tuned for aerial/satellite imagery:
        # сильнее flip, умеренный rotate, цветовые искажения скромные (satellite колориметрия стабильная)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=20,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.2,
    )

    weights_dir = Path(args.project) / args.name / "weights"
    print("\nTraining complete.")
    print(f"  Best:  {weights_dir / 'best.pt'}")
    print(f"  Last:  {weights_dir / 'last.pt'}")
    print(f"\nКопируй best.pt → ../weights/yolo_satellite.pt чтобы backend подхватил.")


if __name__ == "__main__":
    main()
