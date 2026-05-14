# Definitions

The following terms are used in this work:

**Bounding box** — a rectangular region in pixel coordinates defined by its top-left corner $(x_1, y_1)$ and bottom-right corner $(x_2, y_2)$, used to localise a detected object within an image.

**Crown** — the visible upper part of a tree as observed from above in satellite or aerial imagery, approximated by a polygon mask in this work.

**CVAT (Computer Vision Annotation Tool)** — an open-source web-based tool for manual annotation of images and video, used in this project to label tree crowns as polygon masks and bounding boxes.

**DeepForest** — a Python library for tree-crown detection built on a RetinaNet single-stage detector, pre-trained on National Ecological Observatory Network (NEON) lidar data over forested sites in the United States.

**GeoJSON** — an open standard format for encoding geographic data structures using JSON, used in this project to export tree inventories with per-crown polygon geometries in WGS-84.

**GeoTIFF** — a TIFF image file format that embeds geographic metadata (coordinate reference system, affine transform) directly in the file, enabling automatic pixel-to-geographic-coordinate conversion.

**Ground Sampling Distance (GSD)** — the distance on the ground corresponding to one pixel in a satellite or aerial image, expressed in metres per pixel.

**Instance segmentation** — a computer-vision task that simultaneously detects individual object instances and predicts a pixel-level binary mask for each instance, as opposed to semantic segmentation (per-pixel class labels without instance identity) or bounding-box detection (boxes only).

**mAP (mean Average Precision)** — the primary metric for object detection, defined as the mean of per-class Average Precision values; mAP@50 uses an IoU threshold of 0.5 to determine whether a predicted box matches a ground-truth box.

**SAM 2 (Segment Anything Model 2)** — a second-generation foundation model for class-agnostic image segmentation released by Meta AI in 2024, capable of producing precise object masks from box or point prompts without task-specific fine-tuning.

**Tiled inference** — the practice of partitioning a large input image into overlapping fixed-size tiles, running the model independently on each tile, and merging the per-tile predictions into a single global result via Non-Maximum Suppression.

**WBF (Weighted Box Fusion)** — an ensemble strategy that replaces Non-Maximum Suppression by averaging the coordinates of overlapping bounding boxes from multiple detectors, weighted by their confidence scores.

**WGS-84** — the World Geodetic System 1984, the standard coordinate reference system used by GPS and most web mapping services, representing positions as latitude and longitude in decimal degrees.

**YOLOv8-seg** — the instance-segmentation variant of the YOLOv8 single-stage object detector by Ultralytics, which adds a prototype mask head to the standard detection architecture to produce per-instance polygon masks alongside bounding boxes.

\newpage
