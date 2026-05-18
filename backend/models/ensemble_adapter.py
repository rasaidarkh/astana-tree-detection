"""Ensemble adapter — Weighted Box Fusion поверх YOLO + DeepForest.

WBF (https://arxiv.org/abs/1910.13302) объединяет предсказания нескольких моделей
лучше, чем стандартный NMS, потому что использует confidence как вес.

Сильный аргумент для тезиса: "ни одна модель в одиночку не побеждает,
а ensemble стабильно даёт +X% по count accuracy".
"""

from __future__ import annotations

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter


class EnsembleAdapter(ModelAdapter):
    kind = ModelKind.ENSEMBLE
    name = "Ensemble (YOLO + DeepForest, WBF)"

    def __init__(
        self,
        yolo_adapter: ModelAdapter,
        deepforest_adapter: ModelAdapter,
        yolo_weight: float = 1.0,
        df_weight: float = 1.0,
        iou_threshold: float = 0.5,
        skip_box_threshold: float = 0.0,
    ):
        super().__init__()
        self._yolo = yolo_adapter
        self._df = deepforest_adapter
        self._weights = [yolo_weight, df_weight]
        self._iou = iou_threshold
        self._skip = skip_box_threshold

    def _load(self) -> None:
        # Подзависимости загрузятся лениво при их собственных predict()
        pass

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        from ensemble_boxes import weighted_boxes_fusion

        # Получаем размер изображения для нормализации боксов.
        # PIL (вместо cv2.imread) надёжнее на Windows-путях с кириллицей.
        from PIL import Image
        with Image.open(image_path) as pil:
            w, h = pil.size

        yolo_dets = self._yolo.predict(image_path, confidence=confidence)
        df_dets = self._df.predict(image_path, confidence=confidence)

        # WBF принимает нормализованные [0,1] боксы
        boxes_yolo = [[d.box.x1 / w, d.box.y1 / h, d.box.x2 / w, d.box.y2 / h] for d in yolo_dets]
        boxes_df = [[d.box.x1 / w, d.box.y1 / h, d.box.x2 / w, d.box.y2 / h] for d in df_dets]

        scores_yolo = [d.confidence for d in yolo_dets]
        scores_df = [d.confidence for d in df_dets]

        labels_yolo = [0] * len(yolo_dets)
        labels_df = [0] * len(df_dets)

        if not yolo_dets and not df_dets:
            return []

        boxes, scores, labels = weighted_boxes_fusion(
            [boxes_yolo, boxes_df],
            [scores_yolo, scores_df],
            [labels_yolo, labels_df],
            weights=self._weights,
            iou_thr=self._iou,
            skip_box_thr=self._skip,
        )

        detections: list[Detection] = []
        for box, score, _label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    id=0,
                    box=BBox(
                        x1=float(x1) * w,
                        y1=float(y1) * h,
                        x2=float(x2) * w,
                        y2=float(y2) * h,
                    ),
                    confidence=float(score),
                    label="tree",
                )
            )
        return detections
