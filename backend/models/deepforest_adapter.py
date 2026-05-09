"""DeepForest adapter. Использует pre-trained веса weecology/deepforest-tree
или дообученный чекпоинт из deepforest/models/."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter


class DeepForestAdapter(ModelAdapter):
    kind = ModelKind.DEEPFOREST
    name = "DeepForest (Astana fine-tuned)"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,  # weights/deepforest_astana.pl или None
        patch_size: int = 800,
        patch_overlap: float = 0.2,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
        self._checkpoint_path = checkpoint_path
        self._patch_size = patch_size
        self._patch_overlap = patch_overlap
        self._model = None

    def _load(self) -> None:
        from deepforest import main as df_main

        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            self._model = df_main.deepforest.load_from_checkpoint(self._checkpoint_path)
        else:
            self._model = df_main.deepforest()
            self._model.load_model(model_name="weecology/deepforest-tree", revision="main")

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        # DeepForest порог считывает из конфига, выставим перед запуском
        if hasattr(self._model, "config"):
            self._model.config["score_thresh"] = confidence

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # predict_tile делает sliding window — обязательно для крупных снимков
        boxes_df = self._model.predict_tile(
            image=img_rgb,
            patch_size=self._patch_size,
            patch_overlap=self._patch_overlap,
        )

        if boxes_df is None or len(boxes_df) == 0:
            return []

        # Filter by confidence
        boxes_df = boxes_df[boxes_df["score"] >= confidence]

        detections: list[Detection] = []
        for _, row in boxes_df.iterrows():
            detections.append(
                Detection(
                    id=0,
                    box=BBox(
                        x1=float(row["xmin"]),
                        y1=float(row["ymin"]),
                        x2=float(row["xmax"]),
                        y2=float(row["ymax"]),
                    ),
                    confidence=float(row["score"]),
                    label="tree",
                )
            )
        return detections
