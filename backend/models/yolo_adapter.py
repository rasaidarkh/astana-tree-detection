"""YOLOv8-seg adapter. Использует обученные веса из weights/yolo_satellite.pt."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter


class YOLOAdapter(ModelAdapter):
    kind = ModelKind.YOLO
    name = "YOLOv8-seg (Astana fine-tuned)"

    def __init__(self, weights_path: str = "weights/yolo_satellite.pt", imgsz: int = 1024):
        super().__init__(weights_path=weights_path, imgsz=imgsz)
        self._weights_path = weights_path
        self._imgsz = imgsz
        self._model = None

    def _load(self) -> None:
        from ultralytics import YOLO

        weights = Path(self._weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights.resolve()}. "
                f"Скопируй pipeline/yolov8seg/runs/segment/train/weights/best.pt → {self._weights_path}"
            )
        self._model = YOLO(str(weights))

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        results = self._model.predict(
            source=image_path,
            conf=confidence,
            imgsz=self._imgsz,
            retina_masks=True,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return []

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        masks_data = None
        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()

        detections: list[Detection] = []
        for i, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
            x1, y1, x2, y2 = box
            mask_polygon = None
            crown_area_px = None

            if masks_data is not None and i < len(masks_data):
                mask = masks_data[i]
                # Resize маски к размеру оригинала, если нужно
                if mask.shape != (results.orig_shape[0], results.orig_shape[1]):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (results.orig_shape[1], results.orig_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                crown_area_px = float(np.sum(mask > 0.5))
                mask_polygon = _mask_to_polygon(mask)

            detections.append(
                Detection(
                    id=0,  # будет проставлен в base.predict()
                    box=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(conf),
                    label="tree",
                    mask_polygon=mask_polygon,
                    crown_area_px=crown_area_px,
                )
            )

        return detections


def _mask_to_polygon(mask: np.ndarray, simplify_eps: float = 1.5) -> list[list[float]] | None:
    """Бинарная маска → упрощённый полигон. Возвращает список [[x,y], ...]"""
    binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None
    approx = cv2.approxPolyDP(largest, simplify_eps, closed=True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]
