"""Ablation study: YOLO vs DeepForest vs Ensemble.

Помимо стандартных mAP-метрик (которые недооценивают модель из-за шума разметки),
считает практические метрики для тезиса:

  - Count accuracy:   |predicted_count - true_count| / true_count, по каждому снимку
  - F1 @ radius:      детекция верна, если центр в N пикселях от истинного
  - Mean centroid error: средняя дистанция между сматченными центрами

Аргумент в Discussion: "при low-res satellite разметка имеет inherent ambiguity —
для практической задачи (инвентаризации) count- и radius-метрики более релевантны
чем mAP@IoU=0.5".

Пример:
    python ml/evaluate.py \
        --data data/processed/combined/dataset.yaml \
        --models yolo deepforest ensemble \
        --conf 0.25 \
        --radius-px 30 \
        --output docs/ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


# --- Добавить корень проекта в PYTHONPATH чтобы импортить backend.* ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.base import ModelAdapter  # noqa: E402
from backend.models.yolo_adapter import YOLOAdapter  # noqa: E402
from backend.models.deepforest_adapter import DeepForestAdapter  # noqa: E402
from backend.models.ensemble_adapter import EnsembleAdapter  # noqa: E402


def load_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Возвращает список центров деревьев из YOLO-разметки."""
    if not label_path.exists():
        return []
    centers = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class + минимум 3 точки (6 координат)
            continue
        coords = list(map(float, parts[1:]))
        xs = coords[::2]
        ys = coords[1::2]
        cx = sum(xs) / len(xs) * img_w
        cy = sum(ys) / len(ys) * img_h
        centers.append((cx, cy))
    return centers


def match_centers(
    pred_centers: list[tuple[float, float]],
    true_centers: list[tuple[float, float]],
    radius_px: float,
) -> tuple[int, int, int, list[float]]:
    """Greedy матчинг: для каждого pred находим ближайший unmatched true в радиусе.
    Возвращает (TP, FP, FN, list_of_distances)."""
    if not pred_centers and not true_centers:
        return 0, 0, 0, []
    used_true = set()
    tp, fp = 0, 0
    distances = []

    # Сортируем pred по убыванию confidence — здесь confidence нет, идём как есть
    for px, py in pred_centers:
        best_idx = -1
        best_dist = radius_px
        for i, (tx, ty) in enumerate(true_centers):
            if i in used_true:
                continue
            d = float(np.hypot(px - tx, py - ty))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            tp += 1
            used_true.add(best_idx)
            distances.append(best_dist)
        else:
            fp += 1

    fn = len(true_centers) - len(used_true)
    return tp, fp, fn, distances


def evaluate_adapter(
    name: str,
    adapter: ModelAdapter,
    images: list[Path],
    labels: list[Path],
    confidence: float,
    radius_px: float,
) -> dict:
    """Прогоняет адаптер по всем изображениям, возвращает агрегированные метрики."""
    import cv2

    all_tp, all_fp, all_fn = 0, 0, 0
    all_dist: list[float] = []
    per_image_count_err = []
    per_image_records = []

    for img_path, lbl_path in zip(images, labels):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] {img_path.name}")
            continue
        h, w = img.shape[:2]

        true_centers = load_yolo_label(lbl_path, w, h)
        detections = adapter.predict(str(img_path), confidence=confidence)
        pred_centers = [(d.box.cx, d.box.cy) for d in detections]

        tp, fp, fn, dists = match_centers(pred_centers, true_centers, radius_px)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_dist.extend(dists)

        true_count = len(true_centers)
        pred_count = len(pred_centers)
        count_err = abs(pred_count - true_count) / max(true_count, 1)
        per_image_count_err.append(count_err)
        per_image_records.append(
            {
                "image": img_path.name,
                "true": true_count,
                "pred": pred_count,
                "tp": tp, "fp": fp, "fn": fn,
                "count_err": round(count_err, 3),
            }
        )
        print(f"  {img_path.name:30} true={true_count:3}  pred={pred_count:3}  TP={tp:3} FP={fp:3} FN={fn:3}")

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mean_count_err = float(np.mean(per_image_count_err)) if per_image_count_err else 0
    mean_dist = float(np.mean(all_dist)) if all_dist else 0

    return {
        "model": name,
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "precision_at_radius": round(precision, 3),
        "recall_at_radius": round(recall, 3),
        "f1_at_radius": round(f1, 3),
        "mean_count_err": round(mean_count_err, 3),
        "mean_centroid_dist_px": round(mean_dist, 1),
        "n_images": len(per_image_records),
        "per_image": per_image_records,
    }


def build_adapter(name: str, weights_root: Path) -> ModelAdapter:
    if name == "yolo":
        return YOLOAdapter(weights_path=str(weights_root / "yolo_satellite.pt"))
    if name == "deepforest":
        ckpt = weights_root / "deepforest_astana.pl"
        return DeepForestAdapter(checkpoint_path=str(ckpt) if ckpt.exists() else None)
    if name == "ensemble":
        yolo = build_adapter("yolo", weights_root)
        df = build_adapter("deepforest", weights_root)
        return EnsembleAdapter(yolo_adapter=yolo, deepforest_adapter=df)
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Compare detection models on Astana validation set")
    parser.add_argument("--data", required=True, help="dataset.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--models", nargs="+", default=["yolo", "deepforest", "ensemble"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--radius-px", type=float, default=30,
                        help="Радиус для F1 @ radius (зависит от resolution и среднего диаметра кроны)")
    parser.add_argument("--weights-root", default="weights")
    parser.add_argument("--output", default="docs/ablation.csv")
    args = parser.parse_args()

    import yaml as _yaml

    with open(args.data) as f:
        cfg = _yaml.safe_load(f)
    base = Path(cfg.get("path", Path(args.data).parent))
    img_dir = base / cfg[args.split]
    lbl_dir = base / cfg[args.split].replace("images", "labels")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = [lbl_dir / (p.stem + ".txt") for p in images]
    print(f"Found {len(images)} images in split '{args.split}'")

    weights_root = Path(args.weights_root)

    rows = []
    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")
        adapter = build_adapter(model_name, weights_root)
        result = evaluate_adapter(model_name, adapter, images, labels, args.conf, args.radius_px)
        rows.append(result)

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_images", "TP", "FP", "FN", "P@r", "R@r", "F1@r", "count_err", "centroid_dist_px"])
        for r in rows:
            w.writerow([
                r["model"], r["n_images"], r["tp"], r["fp"], r["fn"],
                r["precision_at_radius"], r["recall_at_radius"], r["f1_at_radius"],
                r["mean_count_err"], r["mean_centroid_dist_px"],
            ])

    print("\n" + "=" * 70)
    print(f"{'Model':<14}{'N':>4}{'TP':>5}{'FP':>5}{'FN':>5}{'P@r':>7}{'R@r':>7}{'F1@r':>7}{'CErr':>7}")
    print("=" * 70)
    for r in rows:
        print(f"{r['model']:<14}{r['n_images']:>4}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
              f"{r['precision_at_radius']:>7.3f}{r['recall_at_radius']:>7.3f}"
              f"{r['f1_at_radius']:>7.3f}{r['mean_count_err']:>7.3f}")
    print("=" * 70)
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
