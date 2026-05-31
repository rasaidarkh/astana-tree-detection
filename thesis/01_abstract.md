# Abstract

**Topic:** Development of a Deep Learning Model for Automated Tree Recognition and Green Space Mapping in Urban Environments. **Authors:** Anuar Totin, Rasul Aidarkhanov, Berik Sharipov. Educational program: 6B06101 — Computer Science. Scientific supervisor: Syndar Satbayev.

**Relevance.** Urban tree inventories are a basic input to municipal planning, environmental monitoring and climate adaptation, but the manual field surveys that traditionally supply them are slow, expensive and quickly become obsolete for a city of Astana's size. Deep-learning detectors now work on freely-available satellite imagery at the per-tree level — yet the published literature does not contain a single evaluation of these models on Central-Asian imagery.

**Aim.** To design, implement and evaluate an end-to-end system that, given a satellite image of an area of Astana, automatically produces an inventory of its trees with a per-tree polygon mask, confidence score and geographic coordinates.

**Scientific novelty.** This is the first published evaluation of state-of-the-art deep-learning tree-detection models on Astana satellite imagery, and on any major Central-Asian capital. The work delivers (i) the first measured magnitude of the geographic-generalisation gap for the region — Box mAP@50 = **0.012** for the off-the-shelf NEON-pretrained DeepForest checkpoint on Astana; (ii) a four-model architecture combining YOLOv8-seg, Mask R-CNN, fine-tuned DeepForest and zero-shot SAM 2 mask refinement, together with two ensemble strategies — a Weighted-Box-Fusion ensemble and a cross-YOLO voting ensemble (both implemented in the system; their quantitative M14 evaluation is left to future work); and (iii) a custom Astana dataset of ≈ 100 source images with ≈ 5 500 hand-labelled tree-crown polygons (≈ 8 700 polygon instances after sliding-window tiling).

**Methods.** Analysis of peer-reviewed publications of 2019–2025; deep-learning training and fine-tuning in PyTorch, Ultralytics, torchvision, DeepForest and SAM 2; dataset engineering in CVAT with model-in-the-loop pre-labelling; and quantitative evaluation with standard COCO metrics on a single 14-image cross-model validation set (M14, 702 polygons) shared as common ground truth by all branches.

**Results.** A 23-experiment hyperparameter ablation selected the production configuration: a YOLOv8x-seg checkpoint trained from public COCO weights with the Ultralytics default augmentation pipeline, reaching Box mAP@50 = **0.315** and Mask mAP@50 = **0.289** on M14 — a +140 % relative improvement over the YOLO v1 baseline. Mask R-CNN (0.166 / 0.158) and DeepForest + SAM 2 (0.146 / 0.134) follow. The deployed prototype — *Canopy*, a FastAPI + React + SQLite web application — processes a 1 km × 1 km capture at zoom 19 in about 18 seconds on a single laptop GPU and exports the inventory as GeoJSON, CSV and standalone HTML.

**Keywords:** Astana, tree detection, YOLO, Mask R-CNN, DeepForest, SAM 2, deep learning, remote sensing, urban forestry, instance segmentation, geographic-generalisation gap, ensemble.

\newpage
