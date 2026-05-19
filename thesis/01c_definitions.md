# Definitions

The following terms are used in this work:

**DeepForest** — a Python library for tree-crown detection built on a RetinaNet single-stage detector, pre-trained on National Ecological Observatory Network (NEON) lidar data over forested sites in the United States.

**Ground Sampling Distance (GSD)** — the distance on the ground corresponding to one pixel in a satellite or aerial image, expressed in metres per pixel.

**Instance segmentation** — a computer-vision task that simultaneously detects individual object instances and predicts a pixel-level binary mask for each instance, as opposed to semantic segmentation (per-pixel class labels without instance identity) or bounding-box detection (boxes only).

**M14 (14-image merged validation set)** — the cross-model validation set constructed in this work for like-for-like comparison between all model branches. It contains 14 source images (4 v1 + 5 v2 + 5 v3) and 702 polygon annotations, stored as `annotations_merged_14img_val.json`.

**mAP (mean Average Precision)** — the primary metric for object detection. mAP@50 uses an IoU threshold of 0.5 to determine whether a predicted box matches a ground-truth box; mAP@50:95 averages the AP at ten IoU thresholds from 0.50 to 0.95 in increments of 0.05.

**Mask R-CNN** — a two-stage instance-segmentation network that extends the Faster R-CNN detector with a fully-convolutional mask head running in parallel with the bounding-box regression head. The `maskrcnn_resnet50_fpn_v2` variant is used.

**SAM 2 (Segment Anything Model 2)** — a foundation model for class-agnostic image segmentation released by Meta AI in 2024, capable of producing precise object masks from box or point prompts without task-specific fine-tuning.

**Tiled inference** — the practice of partitioning a large input image into overlapping fixed-size tiles, running the model independently on each tile, and merging the per-tile predictions into a single global result via Non-Maximum Suppression.

**WBF (Weighted Box Fusion)** — an ensemble strategy that replaces Non-Maximum Suppression by averaging the coordinates of overlapping bounding boxes from multiple detectors, weighted by their confidence scores.

**YOLOv8-seg** — the instance-segmentation variant of the YOLOv8 single-stage object detector by Ultralytics, which adds a prototype mask head to the standard detection architecture to produce per-instance polygon masks alongside bounding boxes.

\newpage
