"""YOLOv8-seg adapter. Использует обученные веса из weights/yolo_satellite.pt.

Inference modes:
  * Single-shot — для маленьких изображений (≤ tile_size + overlap по обоим
    осям). Один вызов `model.predict()`, никакого тайлинга.
  * Sliding-window tiled — для больших изображений (большие capture-from-map
    или загруженный city block screenshot). Окно `tile_size × tile_size`
    с шагом `tile_size - overlap` (по умолчанию 640/128 — совпадает с
    тренировочным распределением). Каждая детекция переводится в глобальные
    координаты + global NMS убирает дубли в зонах перекрытия.

Без тайлинга Ultralytics ресайзит большое изображение к `imgsz` по длинной
стороне, и крона ~30 px на zoom 18 проседает до ~7 px, что ниже порога
надёжной детекции — модель начинает «находить» траву и тени вместо деревьев.
Этот эффект описан в Ch.2.3 диплома; здесь — соответствующая inference-ветка.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter


# Точно совпадает с тренировочной геометрией: ml/tile_dataset.py использует
# 640-pixel tiles + 128-pixel overlap.
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 128
# Картинки до этого размера обрабатываются single-shot — мелкие screenshots
# с zoom 19 без проблем влезают.
SINGLE_SHOT_LIMIT = 768
# IoU-порог для global NMS поверх результата тайлов. Чуть выше стандартного
# 0.45 потому что предсказания из соседних тайлов в overlap-зоне рисуют
# почти идентичный bbox на одно и то же дерево.
GLOBAL_NMS_IOU = 0.5


class YOLOAdapter(ModelAdapter):
    kind = ModelKind.YOLO
    name = "YOLOv8-seg (Astana fine-tuned)"

    def __init__(
        self,
        weights_path: str = "weights/yolo_satellite.pt",
        imgsz: int = DEFAULT_TILE_SIZE,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        single_shot_limit: int = SINGLE_SHOT_LIMIT,
    ):
        super().__init__(weights_path=weights_path, imgsz=imgsz)
        self._weights_path = weights_path
        self._imgsz = imgsz
        self._tile_size = tile_size
        self._overlap = overlap
        self._single_shot_limit = single_shot_limit
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

    # ------------------------------------------------------------------
    # Inference entry point — chooses between single-shot and tiled.
    # ------------------------------------------------------------------
    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        from PIL import Image as PILImage

        with PILImage.open(image_path) as img:
            w, h = img.size
            # Hold a copy so the underlying file handle can close immediately
            # — we may need to crop many times below.
            img_rgb = img.convert("RGB") if img.mode != "RGB" else img.copy()

        if w <= self._single_shot_limit and h <= self._single_shot_limit:
            return self._predict_on_pil(img_rgb, confidence, offset=(0, 0))

        return self._predict_tiled(img_rgb, w, h, confidence)

    # ------------------------------------------------------------------
    # Tiled inference: slide a tile_size window with overlap, run the
    # model on each crop, translate to global coords, global-NMS dedupe.
    # ------------------------------------------------------------------
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

        # Global NMS — same tree may be detected in two overlapping tiles.
        return _global_nms(all_dets, iou_threshold=GLOBAL_NMS_IOU)

    # ------------------------------------------------------------------
    # Single PIL crop → list[Detection] in global coords (offset added).
    # ------------------------------------------------------------------
    def _predict_on_pil(self, pil_img, confidence: float, offset: tuple[int, int]) -> list[Detection]:
        ox, oy = offset
        results = self._model.predict(
            source=pil_img,
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
                poly = _mask_to_polygon(mask)
                if poly is not None:
                    # Translate polygon to global frame
                    mask_polygon = [[px + ox, py + oy] for px, py in poly]

            detections.append(
                Detection(
                    id=0,  # будет проставлен в base.predict()
                    box=BBox(
                        x1=float(x1) + ox, y1=float(y1) + oy,
                        x2=float(x2) + ox, y2=float(y2) + oy,
                    ),
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
    if len(approx) < 3:
        # Polygon with < 3 vertices is degenerate; downstream consumers
        # (geo conversion, area heuristics, Leaflet rendering) assume a real
        # closed polygon, so drop it.
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
    """Greedy NMS поверх детекций из всех тайлов. Размерности обычно
    в пределах нескольких тысяч, O(N²) укладывается в <100 мс."""
    if not dets:
        return dets
    sorted_dets = sorted(dets, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for d in sorted_dets:
        if any(_bbox_iou(d.box, k.box) > iou_threshold for k in kept):
            continue
        kept.append(d)
    return kept
