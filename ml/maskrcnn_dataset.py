"""COCO dataset loader for Mask R-CNN training.

Reads annotations_merged COCO 1.0 JSON, resolves image paths across multiple
photo roots (the team's v1 / v2 folders), and returns (image_tensor, target_dict)
tuples in the torchvision detection format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

log = logging.getLogger("astana-tree")


class CocoMaskRCNNDataset(Dataset):
    """COCO instance-segmentation dataset for torchvision Mask R-CNN.

    Resolves each image's `file_name` against a list of candidate roots
    (first-existing wins). Always uses H/W from the COCO json, not from disk —
    защита от руками обрезанных файлов (например, `Снимок экрана 2026-05-10
    102326.png` 1613x862 в COCO vs 1613x1138 в исходном CVAT-экспорте).
    """

    def __init__(
        self,
        annotations_json: str,
        images_roots: list[str],
        transforms: Optional[Callable] = None,
    ):
        # pycocotools.COCO(path) opens with locale encoding (cp1251 on RU Windows),
        # which mojibakes Cyrillic file_name fields. Load JSON ourselves as UTF-8
        # and populate the COCO object manually — works on every OS.
        with open(annotations_json, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        self.coco = COCO()
        self.coco.dataset = dataset
        self.coco.createIndex()

        self.image_ids: list[int] = sorted(self.coco.getImgIds())
        self.images_roots: list[Path] = [Path(r) for r in images_roots]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, file_name: str) -> Path:
        for root in self.images_roots:
            candidate = root / file_name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Image {file_name!r} not found in any of: "
            f"{[str(r) for r in self.images_roots]}"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        image_id = int(self.image_ids[idx])
        img_info = self.coco.loadImgs(image_id)[0]
        path = self._resolve_image_path(img_info["file_name"])

        pil = Image.open(path).convert("RGB")
        image = F.pil_to_tensor(pil).float() / 255.0

        # H/W from COCO json — single source of truth (see class docstring)
        h_coco = int(img_info["height"])
        w_coco = int(img_info["width"])

        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes_list: list[list[float]] = []
        masks_list: list[np.ndarray] = []
        areas_list: list[float] = []
        iscrowd_list: list[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes_list.append([x, y, x + w, y + h])

            rle = coco_mask.frPyObjects(ann["segmentation"], h_coco, w_coco)
            m = coco_mask.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2).astype(np.uint8)
            else:
                m = m.astype(np.uint8)
            masks_list.append(m)

            areas_list.append(float(ann.get("area", 0.0)))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

        n = len(anns)
        if n == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, h_coco, w_coco), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks_list, axis=0), dtype=torch.uint8)
            labels = torch.ones((n,), dtype=torch.int64)
            areas = torch.as_tensor(areas_list, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd_list, dtype=torch.int64)

        target: dict = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """DataLoader collate that keeps lists-of-tensors (torchvision detection convention)."""
    return tuple(zip(*batch))
