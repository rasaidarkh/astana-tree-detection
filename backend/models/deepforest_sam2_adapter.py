"""DeepForest + SAM2 post-processor.

Pipeline:
  1. DeepForest (fine-tuned) → bounding boxes
  2. SAM2 (facebook/sam2.1-hiera-base-plus) → precise crown masks per bbox
  3. Returns Detection with mask_polygon filled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter
from .deepforest_adapter import DeepForestAdapter

log = logging.getLogger("astana-tree")

SAM2_MODEL_ID = "facebook/sam2.1-hiera-base-plus"


class DeepForestSAM2Adapter(ModelAdapter):
    kind = ModelKind.DEEPFOREST_SAM2
    name = "DeepForest + SAM2 (crown masks)"

    def __init__(
        self,
        df_checkpoint_path: Optional[str] = None,
        sam2_checkpoint_path: Optional[str] = None,
        patch_size: int = 400,
        patch_overlap: float = 0.05,
    ):
        super().__init__()
        self._df = DeepForestAdapter(
            checkpoint_path=df_checkpoint_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
        self._sam2_checkpoint_path = sam2_checkpoint_path
        self._predictor = None

    def _load(self) -> None:
        # Load DeepForest
        self._df._load()
        self._df._loaded = True

        # Load SAM2
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if self._sam2_checkpoint_path and Path(self._sam2_checkpoint_path).exists():
                from sam2.build_sam import build_sam2
                model = build_sam2("sam2.1_hiera_b+.yaml", self._sam2_checkpoint_path, device=device)
                self._predictor = SAM2ImagePredictor(model)
                log.info("SAM2 loaded from local checkpoint: %s (device=%s)", self._sam2_checkpoint_path, device)
            else:
                self._predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, device=device)
                log.info("SAM2 loaded from HuggingFace: %s (device=%s)", SAM2_MODEL_ID, device)
        except Exception as e:
            log.error("SAM2 failed to load: %s", e)
            raise

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        # Step 1: DeepForest bboxes
        detections = self._df._predict_raw(image_path, confidence)
        if not detections:
            return []

        # Step 2: Load image for SAM2
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            log.warning("SAM2: could not read image %s, returning DF detections without masks", image_path)
            return detections
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Step 3: SAM2 mask prediction (batch all boxes in one pass)
        import torch
        boxes_np = np.array([[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections])

        try:
            with torch.inference_mode():
                self._predictor.set_image(image_rgb)
                masks, _, _ = self._predictor.predict(
                    box=boxes_np,
                    multimask_output=False,
                )
            # masks shape: (N, 1, H, W) or (N, H, W)
            if masks.ndim == 4:
                masks = masks[:, 0]  # (N, H, W)
        except Exception as e:
            log.warning("SAM2 prediction failed: %s — returning DF detections without masks", e)
            return detections

        # Step 4: Convert masks to polygons
        for det, mask in zip(detections, masks):
            polygon = _mask_to_polygon(mask.astype(np.uint8))
            if polygon is not None:
                det.mask_polygon = polygon
                det.crown_area_px = float(np.sum(mask))

        return detections


def _mask_to_polygon(mask: np.ndarray) -> Optional[list[list[float]]]:
    """Convert binary mask to largest-contour polygon [[x, y], ...]."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return None
    return largest.reshape(-1, 2).tolist()
