"""Cross-YOLO ensemble — IoU-merged predictions from multiple YOLO checkpoints.

Pools predictions from N YOLO variants (e.g. v4_x + exp1_m + v4_s + v2-finetune),
clusters them by box-IoU, keeps clusters where ≥K distinct models agreed.
Strong false-positive killer (single-model hallucinations like stadium-roof
FPs don't survive voting).

Backend version of `ml/v5_ensemble.py` — same algorithm, reuses existing
registered YOLOAdapter instances.
"""

from __future__ import annotations

import logging

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter

log = logging.getLogger("astana-tree")


def _box_iou(b1: BBox, b2: BBox) -> float:
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0.0, (b1.x2 - b1.x1)) * max(0.0, (b1.y2 - b1.y1))
    a2 = max(0.0, (b2.x2 - b2.x1)) * max(0.0, (b2.y2 - b2.y1))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class MultiYOLOEnsembleAdapter(ModelAdapter):
    """Cross-checkpoint YOLO ensemble with IoU-cluster voting.

    `members` — list of (label, YOLOAdapter) tuples to combine.
    `min_models` — how many distinct checkpoints must agree on a detection
      (1 = NMS-style, 2 = vote_2, etc).
    `iou_threshold` — clustering threshold for "same tree".
    """

    kind = ModelKind.YOLO_ENSEMBLE
    name = "YOLO ensemble (4× vote, IoU-merge)"

    def __init__(
        self,
        members: list[tuple[str, ModelAdapter]],
        min_models: int = 2,
        iou_threshold: float = 0.5,
    ):
        super().__init__()
        self._members = members
        self._min_models = min_models
        self._iou = iou_threshold

    def _load(self) -> None:
        # Members lazy-load on their own predict() calls.
        pass

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        # 1. Collect all detections from all member models, tagged by member name.
        pool: list[tuple[Detection, str]] = []
        for name, adapter in self._members:
            try:
                dets = adapter.predict(image_path, confidence=confidence)
            except Exception as e:
                log.warning("YOLO ensemble member '%s' failed: %s — skipping", name, e)
                continue
            pool.extend((d, name) for d in dets)

        if not pool:
            return []

        # 2. Union-find clustering by box-IoU.
        n = len(pool)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if _box_iou(pool[i][0].box, pool[j][0].box) >= self._iou:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        # 3. Filter clusters with >= min_models distinct member votes.
        out: list[Detection] = []
        for indices in clusters.values():
            members_in_cluster = set(pool[i][1] for i in indices)
            if len(members_in_cluster) < self._min_models:
                continue
            # Keep highest-confidence detection from the cluster.
            best = max(indices, key=lambda i: pool[i][0].confidence)
            out.append(pool[best][0])

        log.info(
            "YOLO ensemble: %d members, %d raw dets → %d unified (vote_%d, IoU≥%.2f)",
            len(self._members), n, len(out), self._min_models, self._iou,
        )
        return out
