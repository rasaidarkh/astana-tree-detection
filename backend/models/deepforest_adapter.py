"""DeepForest adapter. Использует pre-trained веса weecology/deepforest-tree
или дообученный чекпоинт из weights/deepforest_astana.pl."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..schemas import BBox, Detection, ModelKind
from .base import ModelAdapter

log = logging.getLogger("astana-tree")


class DeepForestAdapter(ModelAdapter):
    kind = ModelKind.DEEPFOREST
    name = "DeepForest (Astana fine-tuned)"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        patch_size: int = 400,
        patch_overlap: float = 0.05,
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
        self._is_finetuned = False

    def _load(self) -> None:
        from deepforest import main as df_main

        # Сначала грузим pretrained архитектуру
        self._model = df_main.deepforest()
        self._model.load_model(model_name="weecology/deepforest-tree", revision="main")

        # Если есть чекпоинт — накладываем fine-tuned веса через torch
        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            try:
                import torch
                ckpt = torch.load(self._checkpoint_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                self._model.model.load_state_dict(state_dict, strict=False)
                self._is_finetuned = True
                log.info("Fine-tuned weights loaded from %s", self._checkpoint_path)
            except Exception as e:
                log.warning("Fine-tuned weights failed to load (%s), using pretrained", e)

    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        if hasattr(self._model, "config"):
            self._model.config["score_thresh"] = confidence

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        boxes_df = self._model.predict_tile(
            path=image_path,
            patch_size=self._patch_size,
            patch_overlap=self._patch_overlap,
        )

        if boxes_df is None or len(boxes_df) == 0:
            return []

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
