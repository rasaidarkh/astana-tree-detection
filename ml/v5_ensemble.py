"""Multi-model YOLO ensemble — merges overlapping detections across models.

Solves the "many polygons on the same tree" problem from cross-model visual
comparison: pools predictions from N models, then deduplicates via IoU-based
merging or voting.

## Strategies

- **nms**: greedy IoU merge — for any two detections with IoU >= threshold,
  keep the one with higher confidence. Each tree gets one polygon, the
  best-confidence model wins.
- **vote_K**: only keep detection clusters where at least K different models
  agreed. Reduces false positives (stadium-roof FPs won't survive — single
  model anomalies discarded). Within surviving cluster, highest-confidence
  detection wins.

## Usage as library

  from ml.v5_ensemble import ensemble_predict
  dets = ensemble_predict(
      image_path, models=[(name, weights_path), ...],
      strategy="vote_2", iou_threshold=0.5, conf=0.25,
  )

## Usage as CLI

  python ml/v5_ensemble.py <image_path> [--strategy vote_2] [--conf 0.25]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

# Default ensemble set — user-selected 4 models that visually look complementary
V3_ARCH = PROJECT_ROOT / "weights" / "v3_runs"
V4_ARCH = PROJECT_ROOT / "weights" / "v4_clean"

DEFAULT_ENSEMBLE = [
    ("v4_x_clean",  V4_ARCH / "v4_x_clean_v3val0.313_mergedval0.315.pt"),
    ("exp1_m",      V3_ARCH / "exp1_m_cocostart_v3val0.287_mergedval0.308.pt"),
    ("v4_s_clean",  V4_ARCH / "v4_s_clean_v3val0.254_mergedval0.281.pt"),
    ("v2-finetune", PROJECT_ROOT / "weights" / "archive" / "yolo" / "yolo_satellite_v2_finetune.pt"),
]


@dataclass
class EnsembleDet:
    bbox: tuple              # (x1, y1, x2, y2) pixel coords
    polygon: list            # [(x, y), ...] pixel coords
    confidence: float
    model: str               # name of producing model
    cluster_size: int = 1    # how many models agreed (only meaningful post-merge)
    cluster_models: tuple = ()  # which models were in the cluster


def _box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


def _collect_all(image_path: str, models, conf: float) -> list[EnsembleDet]:
    """Run every model, collect into single list."""
    from backend.models.yolo_adapter import YOLOAdapter

    all_dets: list[EnsembleDet] = []
    for name, weights in models:
        if not Path(weights).exists():
            print(f"[!] missing weights for {name}: {weights} — skipping")
            continue
        print(f"  predicting with {name} ...", flush=True)
        adapter = YOLOAdapter(weights_path=str(weights))
        for d in adapter.predict(image_path, confidence=conf):
            # Detection.box is BBox(x1,y1,x2,y2) — extract to plain tuple
            bx = (d.box.x1, d.box.y1, d.box.x2, d.box.y2)
            poly = getattr(d, "mask_polygon", None) or []
            if not poly:
                # No segmentation mask — fall back to box rectangle
                poly = [(bx[0], bx[1]), (bx[2], bx[1]), (bx[2], bx[3]), (bx[0], bx[3])]
            all_dets.append(EnsembleDet(
                bbox=bx,
                polygon=[(p[0], p[1]) for p in poly],
                confidence=float(d.confidence),
                model=name,
            ))
    return all_dets


def _cluster_by_iou(dets: list[EnsembleDet], iou_thresh: float) -> list[list[int]]:
    """Union-find clustering by box IoU."""
    n = len(dets)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _box_iou(dets[i].bbox, dets[j].bbox) >= iou_thresh:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def ensemble_predict(
    image_path: str,
    models=DEFAULT_ENSEMBLE,
    strategy: str = "vote_2",
    iou_threshold: float = 0.5,
    conf: float = 0.25,
) -> list[EnsembleDet]:
    """Run ensemble inference with IoU-based dedup.

    strategy:
      'nms'      — keep highest-confidence detection per IoU cluster
      'vote_2'   — require >=2 distinct models per cluster
      'vote_3'   — require >=3
      'vote_all' — require all participating models
    """
    all_dets = _collect_all(image_path, models, conf)
    print(f"  total raw detections across models: {len(all_dets)}")

    clusters = _cluster_by_iou(all_dets, iou_threshold)

    n_models = len(models)
    if strategy == "vote_all":
        min_models = n_models
    elif strategy.startswith("vote_"):
        try:
            min_models = int(strategy.split("_", 1)[1])
        except ValueError:
            min_models = 2
    else:  # nms
        min_models = 1

    result: list[EnsembleDet] = []
    for cluster_indices in clusters:
        models_in_cluster = set(all_dets[i].model for i in cluster_indices)
        if len(models_in_cluster) < min_models:
            continue
        # Within cluster, keep the highest-confidence detection
        best_idx = max(cluster_indices, key=lambda i: all_dets[i].confidence)
        d = all_dets[best_idx]
        result.append(EnsembleDet(
            bbox=d.bbox,
            polygon=d.polygon,
            confidence=d.confidence,
            model=d.model,
            cluster_size=len(cluster_indices),
            cluster_models=tuple(sorted(models_in_cluster)),
        ))

    print(f"  after {strategy} (IoU>={iou_threshold}, min_models={min_models}): {len(result)} detections")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str)
    ap.add_argument("--strategy", default="vote_2", choices=["nms", "vote_2", "vote_3", "vote_all"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=str, default=None, help="optional output PNG with ensemble overlay")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"image not found: {img_path}")

    print(f"Image: {img_path}")
    print(f"Strategy: {args.strategy}, IoU>={args.iou}, conf>={args.conf}")
    print(f"Ensemble models: {[m[0] for m in DEFAULT_ENSEMBLE]}")
    print()

    dets = ensemble_predict(str(img_path), DEFAULT_ENSEMBLE,
                            strategy=args.strategy, iou_threshold=args.iou, conf=args.conf)

    print()
    print(f"=== Ensemble result: {len(dets)} trees ===")

    # Show per-model winner-cluster contribution
    from collections import Counter
    winner_models = Counter(d.model for d in dets)
    print("Winner model per cluster (which model's detection got picked):")
    for m, c in winner_models.most_common():
        print(f"  {m}: {c}")

    avg_cluster = sum(d.cluster_size for d in dets) / max(1, len(dets))
    print(f"Avg cluster size (how many models agreed per tree): {avg_cluster:.2f}")

    if args.out:
        import cv2, numpy as np
        img = cv2.imread(str(img_path))
        overlay = img.copy()
        for d in dets:
            poly = np.array(d.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [poly], (80, 255, 80))
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        for d in dets:
            poly = np.array(d.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.imwrite(args.out, img)
        print(f"Saved overlay: {args.out}")


if __name__ == "__main__":
    main()
