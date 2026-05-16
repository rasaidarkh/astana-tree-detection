"""Mask R-CNN adapter using torchvision maskrcnn_resnet50_fpn_v2.

Pipeline:
  1. Load torchvision Mask R-CNN with COCO V1 backbone
  2. Replace box + mask heads under num_classes (default 2: background + tree)
  3. Optionally load fine-tuned state_dict from checkpoint_path
  4. Inference returns Detection with bbox + polygon mask + confidence
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


class MaskRCNNAdapter(ModelAdapter):
    """Mask R-CNN ResNet50-FPN v2 adapter for instance segmentation of tree crowns."""

    kind = ModelKind.MASKRCNN
    name = "Mask R-CNN (R50-FPN v2)"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        **kwargs,
    ):
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

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Use a context manager so the source file handle is released even if
        # .convert() or pil_to_tensor() raises later in the call chain.
        with Image.open(image_path) as pil:
            rgb = pil.convert("RGB")
        tensor = F.pil_to_tensor(rgb).float() / 255.0
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
            polygon = _mask_to_polygon(binary_mask)
            detections.append(
                Detection(
                    id=0,
                    box=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(score),
                    label="tree",
                    mask_polygon=polygon,
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
