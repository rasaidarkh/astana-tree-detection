"""Mask R-CNN adapter using torchvision maskrcnn_resnet50_fpn_v2.

Pipeline:
  1. Load torchvision Mask R-CNN with COCO V1 backbone
  2. Replace box + mask heads under num_classes (default 2: background + tree)
  3. Optionally load fine-tuned state_dict from checkpoint_path
  4. Inference: single-shot for small images, sliding-window tiled inference
     for large captures (matches the YOLO branch — same `tile_size=640` and
     `overlap=128` as the training pipeline), with global NMS to remove
     duplicate detections from tile-overlap regions.

Without tiled inference the network resizes large city-block screenshots to
its training scale, which crushes 20–40 px crowns down to ~7 px and triggers
massive false-positives on bare grass / canopy clusters. See Ch.2.3 of the
diploma for the original derivation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter

log = logging.getLogger("astana-tree")


# Geometry — keep aligned with YOLO branch + training pipeline.
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 128
SINGLE_SHOT_LIMIT = 768
GLOBAL_NMS_IOU = 0.5


class MaskRCNNAdapter(ModelAdapter):
    """Mask R-CNN ResNet50-FPN v2 adapter for instance segmentation of tree crowns."""

    kind = ModelKind.MASKRCNN
    name = "Mask R-CNN (R50-FPN v2)"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.05,
        mask_threshold: float = 0.5,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        single_shot_limit: int = SINGLE_SHOT_LIMIT,
        **kwargs,
    ):
        # Lowered floor from 0.5 → 0.05 — the adapter used to override any
        # user-supplied confidence below 0.5 via `max(confidence, 0.5)`, which
        # produced 0-detection scenes on tiles where MRCNN scores everything
        # below 0.5. Now the floor is just a safety net against degenerate 0.0.
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            confidence_threshold=confidence_threshold,
            mask_threshold=mask_threshold,
            **kwargs,
        )
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._mask_threshold = mask_threshold
        self._tile_size = tile_size
        self._overlap = overlap
        self._single_shot_limit = single_shot_limit
        self._model = None

    @staticmethod
    def build_model(num_classes: int = 2):
        """Build maskrcnn_resnet50_fpn_v2 with COCO V1 backbone, heads replaced for num_classes.

        Shared by adapter (_load) and training script — single source of truth for the
        architecture so train-time and inference-time graphs match exactly.
        """
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        )
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, dim_reduced=256, num_classes=num_classes
        )
        return model

    def _load(self) -> None:
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        model = self.build_model(num_classes=2)

        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            try:
                state_dict = torch.load(
                    self._checkpoint_path,
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(state_dict)
                log.info("Mask R-CNN fine-tuned weights loaded: %s", self._checkpoint_path)
            except Exception as e:
                log.error(
                    "Failed to load Mask R-CNN checkpoint %s: %s",
                    self._checkpoint_path, e,
                )
                raise
        else:
            log.info(
                "Mask R-CNN using torchvision COCO V1 pretrained (no fine-tune checkpoint)"
            )

        model.eval()
        model.to(device)
        self._model = model

    # ------------------------------------------------------------------
    # Inference entry point — chooses between single-shot and tiled.
    # ------------------------------------------------------------------
    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        from PIL import Image

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        with Image.open(image_path) as pil:
            rgb = pil.convert("RGB")
            w, h = rgb.size

        if w <= self._single_shot_limit and h <= self._single_shot_limit:
            return self._predict_on_pil(rgb, confidence, offset=(0, 0))

        return self._predict_tiled(rgb, w, h, confidence)

    def _predict_tiled(self, img_rgb, width: int, height: int, confidence: float) -> list[Detection]:
        stride = self._tile_size - self._overlap

        def _tile_origins(extent: int) -> list[int]:
            if extent <= self._tile_size:
                return [0]
            starts = list(range(0, extent - self._tile_size + 1, stride))
            last_aligned = extent - self._tile_size
            if not starts or starts[-1] != last_aligned:
                starts.append(last_aligned)
            return starts

        xs = _tile_origins(width)
        ys = _tile_origins(height)

        all_dets: list[Detection] = []
        for y in ys:
            for x in xs:
                tile = img_rgb.crop((x, y, x + self._tile_size, y + self._tile_size))
                all_dets.extend(self._predict_on_pil(tile, confidence, offset=(x, y)))

        return _global_nms(all_dets, iou_threshold=GLOBAL_NMS_IOU)

    def _predict_on_pil(self, pil_img, confidence: float, offset: tuple[int, int]) -> list[Detection]:
        import torch
        from torchvision.transforms import functional as F

        ox, oy = offset
        tensor = F.pil_to_tensor(pil_img).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output = self._model(tensor)[0]

        scores = output["scores"].cpu().numpy()
        boxes = output["boxes"].cpu().numpy()
        masks = output["masks"].cpu().numpy()  # (N, 1, H, W) float in [0, 1]

        threshold = max(confidence, self._confidence_threshold)
        keep = scores >= threshold

        detections: list[Detection] = []
        for box, score, mask in zip(boxes[keep], scores[keep], masks[keep]):
            x1, y1, x2, y2 = box
            binary_mask = (mask[0] > self._mask_threshold).astype(np.uint8)
            poly = _mask_to_polygon(binary_mask)
            mask_polygon = (
                [[px + ox, py + oy] for px, py in poly] if poly is not None else None
            )
            detections.append(
                Detection(
                    id=0,
                    box=BBox(
                        x1=float(x1) + ox, y1=float(y1) + oy,
                        x2=float(x2) + ox, y2=float(y2) + oy,
                    ),
                    confidence=float(score),
                    label="tree",
                    mask_polygon=mask_polygon,
                    crown_area_px=float(binary_mask.sum()),
                )
            )

        return detections


def _mask_to_polygon(
    mask: np.ndarray, simplify_eps: float = 1.5
) -> list[list[float]] | None:
    """Binary mask -> simplified polygon [[x, y], ...] via Suzuki-Abe contour + approxPolyDP.

    Returns None if no contour with area >= 1 px is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None
    approx = cv2.approxPolyDP(largest, simplify_eps, closed=True)
    if len(approx) < 3:
        # Polygon with < 3 vertices is degenerate — drop it instead of letting
        # downstream consumers (geo conversion, Leaflet rendering) blow up on
        # a 1- or 2-point "polygon".
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Plain axis-aligned IoU on pixel-space BBox."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, (a.x2 - a.x1)) * max(0.0, (a.y2 - a.y1))
    ub = max(0.0, (b.x2 - b.x1)) * max(0.0, (b.y2 - b.y1))
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _global_nms(dets: list[Detection], iou_threshold: float = GLOBAL_NMS_IOU) -> list[Detection]:
    """Greedy NMS поверх детекций из всех тайлов — нужен потому что одно и то
    же дерево может появиться в двух соседних тайлах в overlap-зоне."""
    if not dets:
        return dets
    sorted_dets = sorted(dets, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for d in sorted_dets:
        if any(_bbox_iou(d.box, k.box) > iou_threshold for k in kept):
            continue
        kept.append(d)
    return kept
