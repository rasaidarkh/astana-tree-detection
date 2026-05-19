# Chapter 2. Methodology

This chapter describes the technical design of the system. Section 2.1 gives a top-level view of the architecture and the data flow. Section 2.2 describes the image input pipeline and tiled inference. Sections 2.3 – 2.6 detail the four detection branches — YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest+SAM 2 — and Section 2.7 describes the two ensemble strategies that combine them. Sections 2.8 – 2.9 cover the geographic-conversion stage and the SQLite persistence layer. Section 2.10 describes the application architecture and frontend workflows in detail, including the database schema, the REST API, the Auto-Zoom Region Scan and the four detection-display modes.

## 2.1 System architecture overview

The system follows a classical three-tier separation of concerns: a thin **presentation layer** (a single-page React 18 application served by FastAPI, with React, Babel-standalone and Leaflet loaded directly from a CDN — no Node.js build step), an **application layer** (a FastAPI REST backend in Python 3.11) and a **model layer** (four pluggable deep-learning adapters that wrap the underlying frameworks). The system architecture is summarised in Figure 2.1.

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  PRESENTATION LAYER  (React 18 UMD + Leaflet, no build step)        │
   │  ┌────────────────────────────────────────────────────────────────┐ │
   │  │  Single-image view  │  City-map view  │  Scan area / Polygon   │ │
   │  └────────────────────────────────────────────────────────────────┘ │
   └──────────────────────────────────┬──────────────────────────────────┘
                                      │ HTTP / NDJSON streaming
   ┌──────────────────────────────────▼──────────────────────────────────┐
   │  APPLICATION LAYER  (FastAPI + Pydantic, Python 3.11)               │
   │                                                                     │
   │   /api/upload       /api/capture_from_map      /api/scan_region    │
   │   /api/predict      /api/scan_region/stream    /api/export/...     │
   │                                                                     │
   │   ┌────────────────┐   ┌───────────────────┐   ┌────────────────┐  │
   │   │  Map capture   │   │  Region scanner   │   │   Geo module   │  │
   │   │ (ESRI/Google)  │   │ (3-level tiling)  │   │ (4 geo modes)  │  │
   │   └────────────────┘   └───────────────────┘   └────────────────┘  │
   └────────┬───────────────┬───────────────┬──────────────┬─────────────┘
            │               │               │              │
   ┌────────▼─────┐ ┌───────▼────────┐ ┌────▼─────────┐ ┌──▼─────────────┐
   │   YOLO       │ │  Mask R-CNN    │ │  DeepForest  │ │  DeepForest +  │
   │   adapter    │ │  adapter       │ │  adapter     │ │  SAM 2 adapter │
   │ (Ultralytics)│ │ (torchvision)  │ │ (deepforest) │ │ (sam2 1.1)     │
   └────────┬─────┘ └───────┬────────┘ └──────┬───────┘ └────────┬───────┘
            └───────────────┴────────────┬────┴──────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  WBF + cross-YOLO   │
                              │  ensembles          │
                              └──────────┬──────────┘
                                         │ List<Detection>
   ┌─────────────────────────────────────▼─────────────────────────────────┐
   │  PERSISTENCE LAYER  (SQLite, storage/app.db, ON DELETE CASCADE)       │
   │   snapshots ◄── runs ◄── detections     scan_sessions                 │
   └───────────────────────────────────────┬───────────────────────────────┘
                                           │
   ┌───────────────────────────────────────▼───────────────────────────────┐
   │  EXPORT LAYER     GeoJSON      CSV      Standalone HTML (Leaflet)     │
   └───────────────────────────────────────────────────────────────────────┘
```

**Figure 2.1 — Layered architecture of the proposed system.** Solid arrows denote a synchronous request / response; the ON-DELETE-CASCADE relations between SQLite tables ensure that deleting a snapshot or a scan-session also removes all dependent runs, detections and PNG files in a single operation.

A key architectural decision is the use of the **adapter pattern** for the model layer: every model adheres to a single abstract base class with a `predict(image_path, confidence) -> List[Detection]` method, so new models can be added without changes to the rest of the system. The base class provides lazy initialisation, automatic weight loading from a `weights/` directory and exception conversion to FastAPI HTTP responses. The complete source tree is organised in three top-level directories — `backend/`, `frontend/` and `ml/` — at approximately 6 000 lines of code.

## 2.2 Image input and tiled inference

The system accepts three categories of input: regular images (PNG, JPG, WebP) decoded with Pillow on first use; GeoTIFF images carrying an affine transform and CRS in their metadata, parsed with `rasterio.open()` and re-projected to WGS-84 (EPSG:4326) when needed; and **in-browser map capture**, where the user draws a rectangle on a Leaflet map and the backend downloads the corresponding tiles from a configurable provider.

Two tile providers are registered (Table 2.1): the default **ESRI World Imagery** endpoint (max zoom 19, no API key, stable) and an unofficial **Google Satellite** endpoint (max zoom 20, the same image base as Google Earth Pro — directly relevant because the YOLO and Mask R-CNN training data was collected from Google Earth Pro screenshots). The map-capture flow (i) converts the two corner coordinates from longitude / latitude into floating-point Web-Mercator tile coordinates; (ii) computes the integer tile range and, for large bounding boxes, splits the request into up to nine sub-regions each capped at approximately 100 tiles, so per-tile downloads remain below a configurable `MAX_TILES` limit (default 400, ≈ 5 120 × 5 120 px per sub-region) that protects the server from accidental bulk downloads; (iii) downloads the tiles in parallel through a thread pool of eight workers, replacing failed tiles by a small grey placeholder; and (iv) stitches the tiles into a single canvas and crops to the requested bounding box, returning a PNG with the geographic bounds embedded.

**Table 2.1 — Tile providers registered in `backend/map_capture.py::TILE_PROVIDERS`.**

| Provider key | Endpoint | Max zoom | Notes |
|---|---|---|---|
| `esri` (default) | `server.arcgisonline.com/.../World_Imagery/...` | 19 | Free, no API key, stable |
| `google` | `mt{0–3}.google.com/vt/lyrs=s` | 20 | Same image base as Google Earth Pro (training data source); unofficial endpoint, academic prototyping only |

**Tiled inference.** A YOLOv8-seg or DeepForest model operates on inputs of approximately 640 × 640 px, while a single Astana screenshot is typically 1 600 × 1 100 px or larger. Resizing to fit the network input shrinks each tree crown — typically 20 – 40 px in diameter at zoom 18 — below the resolution any modern detector can reliably handle. Early experiments confirmed this: a YOLOv8-seg trained at `imgsz=640` directly on full-resolution inputs detected **zero trees** on the validation set. The system therefore performs **sliding-window tiled inference** at the same resolution as training: the original image is partitioned into overlapping 640-px tiles with a 128-px overlap so that any tree crown is fully contained in at least one tile; the model is applied independently to each tile; and the per-tile detections are translated back into global image coordinates by adding the tile offset to every polygon vertex. Duplicate detections of the same tree that appears in multiple overlapping tiles are removed by a **global Non-Maximum Suppression** pass with an IoU threshold of 0.5, vectorised in NumPy. The same tiling-and-NMS pattern is reused in `ml/tile_dataset.py` for training-data preparation and in `ml/prelabel_coco.py` for model-in-the-loop pre-labelling.

## 2.3 YOLOv8-seg branch

YOLOv8 [@UltralyticsYOLO2023] is a single-stage anchor-free object detector. The segmentation variant `YOLOv8x-seg` extends the architecture with an additional **prototype mask head** that produces 32 prototype masks at one quarter the input resolution; each detected instance is then represented by a vector of 32 coefficients that, linearly combined with the prototype masks and thresholded, produces a binary segmentation mask aligned with the detected bounding box. This design — borrowed from the YOLACT family of real-time instance segmenters — separates the per-pixel and per-instance computations, so the cost of generating masks scales with the number of detections rather than with the number of pixels.

**Why instance segmentation rather than detection.** Bounding-box detection is sufficient for counting trees, but the municipal user also wants the **crown size** (a proxy for species and age) and the **green-coverage percentage** of a given area (a standard urban-planning indicator). Both require pixel-level masks rather than rectangular boxes — a poplar and a birch of the same canopy area can have radically different crown silhouettes, and the bounding box of two adjacent overlapping crowns can cover a substantial fraction of bare ground. The output of the adapter therefore contains: a bounding box, a polygon mask (the YOLO mask compressed into a closed sequence of $(x, y)$ vertices via Suzuki–Abe contour extraction), a confidence score, and — derived from the mask — the crown area in pixels and (when pixel size in metres is known) in square metres.

**Training data and pre-processing.** The training data is a collection of Astana satellite screenshots taken from Google Earth and ESRI World Imagery at zoom levels 17 – 19. The annotation effort proceeded in three iterations (v1 / v2 / v3) totalling approximately 100 source images and ≈ 8 700 polygon-level annotations after tiling. The class taxonomy is deliberately minimal — a single class "Tree" (in the source: "Дерево") — because the satellite resolution available does not allow reliable species discrimination. The dataset is split at the source-image level (no tile from a single source leaks between splits). Two custom Python tools support the workflow: `ml/coco_to_yolo_seg.py` converts CVAT-exported COCO 1.0 annotation files into the polygon-line format expected by Ultralytics YOLO, with sanitisation of Cyrillic filenames into ASCII to avoid Windows-path issues; and `ml/tile_dataset.py` performs the sliding-window tiling itself using Shapely for polygon clipping, dropping any clipped fragment whose area falls below 25 square pixels.

**Training procedure.** Three YOLOv8-seg variants are used in this work, corresponding to three phases of the project's hyperparameter ablation reported in Chapter 3. The pre-ablation generations used `yolov8x-seg` (≈ 71 M parameters) on the initial hypothesis that the extra capacity would improve performance. Rounds 1 – 3 of the systematic ablation reversed that hypothesis with the medium-size `yolov8m-seg` (≈ 27 M parameters) outperforming both larger variants by 15 – 25 % relative when trained from COCO weights with manually-tuned augmentation. Round 4 then reversed the reversal: with **Ultralytics' default augmentation pipeline** instead of the manually-tuned one, the largest `yolov8x-seg` becomes the strongest single configuration, reaching Box mAP@50 = 0.315 on M14 — the headline empirical result of the diploma. The **final production checkpoint** is therefore `yolov8x-seg` trained from fresh COCO weights with `single_cls=True` and Ultralytics' default augmentation, on the single laptop GPU (NVIDIA GeForce RTX 4060, 8 GiB VRAM); mixed-precision training (AMP) was essential to fit batch size 2 in memory.

**Loss function.** The training objective of YOLOv8-seg is the weighted sum of four components — a bounding-box regression loss, a per-class classification loss, a Distribution Focal Loss for the discrete-bin regression head, and a per-pixel mask loss:

$$
\mathcal{L} \;=\; \lambda_{\text{box}}\,\mathcal{L}_{\text{CIoU}} \;+\; \lambda_{\text{cls}}\,\mathcal{L}_{\text{BCE-cls}} \;+\; \lambda_{\text{dfl}}\,\mathcal{L}_{\text{DFL}} \;+\; \lambda_{\text{seg}}\,\mathcal{L}_{\text{BCE-mask}}
$$

The bounding-box term is the **Complete IoU loss**, which extends the plain IoU loss with a centre-point-distance term and an aspect-ratio-inconsistency term, $\mathcal{L}_{\text{CIoU}} = 1 - \mathrm{IoU} + \rho^{2}(b, b^{gt})/c^{2} + \alpha v$, where $\rho$ is the Euclidean distance between predicted and ground-truth box centres, $c$ is the diagonal of the smallest box that encloses both, $v$ is a measure of aspect-ratio inconsistency, and $\alpha$ a trade-off coefficient. The loss-weight defaults are the Ultralytics-standard $\lambda_{\text{box}} = 7.5$, $\lambda_{\text{cls}} = 0.5$, $\lambda_{\text{dfl}} = 1.5$, retained as-is for comparability with the COCO-pretrained checkpoint.

## 2.4 Mask R-CNN branch

Mask R-CNN [@MaskRCNN2017] is a two-stage instance-segmentation network that extends the Faster R-CNN [@FasterRCNN2015] detector with a fully-convolutional mask head running in parallel with the bounding-box-regression head. The two-stage design relies on a Region Proposal Network that first generates a small set of class-agnostic region proposals, which are then classified, regressed and segmented independently. The variant adopted is the standard Mask R-CNN with a ResNet-50 backbone and an FPN neck, initialised from publicly-available COCO-pretrained `maskrcnn_resnet50_fpn_v2` torchvision weights and fine-tuned on the same Astana polygon dataset used for the YOLO branch. The motivation for including this branch is methodological: it provides a like-for-like architectural comparison between a one-stage (YOLO) and a two-stage (Mask R-CNN) instance segmenter under identical training data and validation conditions, in line with the comparative analysis surveyed in Section 1.3.

Two Mask R-CNN checkpoints exist in the project. The **v1 + v2 base** model was trained from the torchvision COCO V1 weights with SGD (momentum 0.9, weight decay $5 \times 10^{-4}$), an initial learning rate of $5 \times 10^{-3}$, a `StepLR` scheduler halving the learning rate every 10 epochs, batch size 2 and AMP. The **v2 + v3 fine-tune** — which is the production checkpoint released under tag `maskrcnn-v2v3` — warm-starts from the v1 + v2 base via `--resume-from` and lowers the initial learning rate automatically to $1 \times 10^{-3}$. The principal training-side improvement is a richer Albumentations pipeline: horizontal flip ($p$ = 0.5), vertical flip ($p$ = 0.3), random 90-degree rotation ($p$ = 0.5), random brightness/contrast adjustment ($p$ = 0.3) and HSV jitter ($p$ = 0.2). Early-stopping is implemented on the validation `mask_map_50` metric with a patience of 5 epochs; the v2 + v3 fine-tune was early-stopped at epoch 16 (best at epoch 11). The adapter exposes the standard `predict(image_path, confidence) -> List[Detection]` method and reuses the same tiled-inference + global-NMS code path as the YOLO branch (Section 2.2).

## 2.5 DeepForest branch

DeepForest [@DeepForest2019] is a tree-detection library built on top of a RetinaNet single-stage detector with a ResNet-50 backbone. RetinaNet was selected by the DeepForest authors over Faster R-CNN because of its better speed–accuracy trade-off and over earlier YOLO versions because of its dedicated **focal-loss** formulation [@RetinaNet2017], $\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^{\gamma} \log(p_t)$ (with $\gamma = 2$ in the DeepForest default), which down-weights well-classified background examples and is particularly well-suited to the highly imbalanced foreground-to-background ratio of dense forest scenes.

The model is shipped with two pre-trained weight sets: the "tree" model (`weecology/deepforest-tree`), trained on hundreds of thousands of semi-supervised annotations derived from NEON aerial-lidar data over forested sites in the United States, and a "bird" model irrelevant to the present work. DeepForest's recommended inference mode is the `predict_tile()` method, which performs sliding-window patch-based inference on 400 × 400-px patches with a 5 % overlap and merges per-patch detections via internal NMS. The adapter wraps this method and translates the resulting pandas DataFrame into the same `Detection` dataclass that the YOLO adapter produces.

**Fine-tuning on Astana data.** A first DeepForest fine-tune was performed by team member Anuar Totin in early May 2026 on a separate bounding-box annotation set maintained on Roboflow. The production DeepForest checkpoint used by the deployed backend is the **v3 fine-tune**, which warm-starts from that v4 checkpoint and continues training on the same merged Astana CVAT polygon dataset used by the YOLO and Mask R-CNN branches. For the v3 fine-tune the CVAT polygon annotations are converted to DeepForest's bounding-box CSV format via `ml/coco_to_deepforest_csv.py` — every polygon is replaced by its axis-aligned bounding box. The fine-tune trains for 30 epochs with SGD + momentum at learning rate $1 \times 10^{-4}$, batch size 4 and horizontal-flip augmentation, on the RTX 4050 Laptop GPU. The fine-tuned weights are stored on disk as a Lightning checkpoint (`weights/deepforest_astana_v3.pl`, published as GitHub release `v2.0`). The effective training trajectory of the production weights is therefore **NEON pretrained → v4 (Roboflow) → v3 (CVAT)**.

## 2.6 DeepForest + SAM 2 mask-refinement branch

The fourth branch exploits the **zero-shot generalisation** property of SAM 2 [@SAM2_2024] to upgrade DeepForest's bounding-box detections into precise crown polygons without any additional annotation or training. SAM 2 is the second-generation version of Meta AI's foundation segmentation model, succeeding the original SAM [@SAM2023]. Where SAM was trained on the SA-1B dataset of more than one billion masks across eleven million images, SAM 2 extends this with the SA-V dataset of more than thirty-five million masks across ≈ 250 000 videos. For the static-image, single-frame tree-detection task addressed here only the image-level segmentation capability of SAM 2 is used; its temporal-propagation mode is reserved for future-work multi-temporal canopy monitoring.

The pipeline of this branch is:

1. The DeepForest detector is run on the input image and produces a list of bounding boxes with associated confidence scores, exactly as in Section 2.5.
2. All bounding boxes above the confidence threshold are passed as a batch of SAM 2 box prompts in the image coordinate frame.
3. The `SAM2ImagePredictor.predict()` call returns one binary mask per box (with `multimask_output=False`).
4. The mask is converted into a polygon contour through OpenCV contour extraction, and the resulting `Detection` is emitted as if it had been produced by an end-to-end instance segmenter.

The branch is implemented as a separate adapter (`DeepForestSAM2Adapter`) that takes the DeepForest adapter as a constructor argument. The SAM 2 model used is `sam2.1-hiera-base-plus`, loaded automatically from HuggingFace (`facebook/sam2.1-hiera-base-plus`) on first inference, with the device (CUDA or CPU) detected automatically. The hiera-base-plus variant was chosen as a compromise between mask quality and inference speed; the larger hiera-large variant is also supported but is too slow for the interactive use case on a laptop GPU. Conceptually, this design treats SAM 2 as a **post-processing step** that decorates an otherwise pure bounding-box detector with high-quality polygon masks; the cost is a roughly two-fold increase in inference time per image.

## 2.7 Ensemble strategies

The system implements two complementary ensemble strategies. The Weighted-Box-Fusion ensemble of Section 2.7.1 combines a YOLO checkpoint with a DeepForest checkpoint (cross-architecture, addressing complementary failure modes between an instance-segmenter and a bounding-box-only detector); the cross-YOLO voting ensemble of Section 2.7.2 combines four YOLO checkpoints with each other (within-architecture, addressing per-checkpoint training-time variance and per-checkpoint failure modes).

### 2.7.1 Weighted Box Fusion (YOLO + DeepForest)

The YOLO and DeepForest branches are trained on the same data but with different network architectures, patch sizes and loss formulations. Their errors are therefore partly de-correlated: YOLO tends to over-segment large dense canopies into several smaller crowns, while DeepForest tends to merge adjacent crowns into a single bounding box. The chosen ensemble strategy is **Weighted Box Fusion** [@WBF2021]. Where Non-Maximum Suppression keeps the single highest-confidence box and discards every overlapping box, WBF instead **averages** the coordinates of the overlapping boxes weighted by their confidence scores. For a cluster of $n$ overlapping predictions $\{(\mathbf{b}_{i}, c_{i})\}_{i=1}^{n}$, the WBF-fused box and fused confidence are

$$
\mathbf{b}_{\text{fused}} \;=\; \frac{\sum_{i=1}^{n} c_{i}\, \mathbf{b}_{i}}{\sum_{i=1}^{n} c_{i}}, \qquad
c_{\text{fused}} \;=\; \frac{\sum_{i=1}^{n} c_{i}}{n} \cdot \frac{\min(n, M)}{M}
$$

where $M$ is the number of models being ensembled (here $M = 2$). The right-hand factor $\min(n, M)/M$ down-weights clusters that contain detections from only a subset of the available models — a single-model cluster receives half of its raw average confidence, while a two-model cluster receives the full average. This factor is what makes WBF a true ensemble (it rewards agreement between models) rather than a smoothing operation. The implementation uses the open-source `ensemble-boxes` package. The IoU threshold is set to 0.55, slightly higher than the typical 0.5 used for plain NMS, to compensate for the systematic location offset between YOLO and DeepForest boxes.

### 2.7.2 Cross-YOLO voting ensemble (IoU-clustered, K-of-N majority vote)

A complementary ensemble that operates **within** the YOLO family rather than across architectures was added in the late stage of the project, motivated by two observations from Chapter 3: per-checkpoint training-time variance of sample standard deviation ≈ 0.028 Box mAP@50 across four replicates of the same configuration, and qualitative cross-checkpoint complementarity in which different YOLO checkpoints with similar aggregate mAP detect substantially different per-detection subsets. The cross-YOLO ensemble averages out the per-checkpoint variance and discards single-checkpoint hallucinations through a majority-voting rule.

**Algorithm.** Given $N$ member YOLO checkpoints, the ensemble (i) runs every member on the input image independently, (ii) pools all $M$ detections from the $N$ members into a single flat list tagged by member identity, (iii) clusters the pooled detections by box IoU using a union-find data structure (any two detections with $\text{IoU} \geq 0.5$ are merged into the same cluster), (iv) **discards any cluster whose detections come from fewer than $K$ distinct member models** (default $K = 2$), and (v) emits the highest-confidence detection from each surviving cluster as its representative. The complexity is $O(NM^2)$, dominated by the pairwise IoU computation; on a typical Astana tile with $M \approx 750$ per member the four-member ensemble runs end-to-end in approximately 4–5 seconds on the laptop GPU.

**Default member set.** Four YOLO checkpoints from the project archive are configured as the default ensemble members, chosen for the visual complementarity of their failure modes on Astana scenes: **v4_x_clean** (the final production checkpoint, strongest aggregate mAP, conservative on built-environment surfaces); **exp1_m_cocostart** (yolov8m-seg with v2-proven augmentation, recovers more partially-occluded crowns than v4_x_clean); **v4_s_clean** (yolov8s-seg with defaults, most permissive, highest raw detection count, useful for recall-priority scenes); and **v2-finetune** (the previous-generation production checkpoint, most conservative on novel surface types, no stadium-roof false-positive regression of the Round 4 / exp1 generation). The cross-YOLO ensemble is exposed in the frontend's hierarchical model picker under the **Ensemble → 4× YOLO vote** option.

## 2.8 Geographic conversion

A pixel coordinate $(x, y)$ inside the detection mask has no immediate meaning to a municipal user; the system must convert it into a $(\text{longitude}, \text{latitude})$ pair in WGS-84. The conversion is implemented in `backend/geo.py` and supports four operating modes, selected automatically depending on the metadata available.

**Mode 1 — GeoTIFF affine.** If the input is a GeoTIFF, the affine transform written in the metadata maps any pixel coordinate to a coordinate in the file's projection. The system reads the transform with `rasterio.transform.AffineTransformer.xy(row, col)` and re-projects the result to EPSG:4326 with `pyproj.Transformer` when needed. This mode is the most accurate and is the recommended workflow for production use.

**Mode 2 — Four-corner bilinear.** If the user supplies the geographic coordinates of all four corners of the image (by dragging four markers on the Leaflet map until the screenshot is aligned with the underlying satellite basemap), the conversion uses bilinear interpolation: for a pixel at relative coordinates $(u, v) \in [0,1]^{2}$,
$$\mathbf{g}(u,v) = (1-u)(1-v)\mathbf{g}_{\text{nw}} + u(1-v)\mathbf{g}_{\text{ne}} + (1-u)v\mathbf{g}_{\text{sw}} + uv\mathbf{g}_{\text{se}}.$$

**Mode 3 — Two-corner axis-aligned.** When only the NW and SE corners are known (the common case after the in-browser map-capture workflow), the conversion is a simple linear interpolation along each axis, $\lambda(x) = \lambda_{\text{nw}} + (x/W)(\lambda_{\text{se}} - \lambda_{\text{nw}})$ and $\phi(y) = \phi_{\text{nw}} + (y/H)(\phi_{\text{se}} - \phi_{\text{nw}})$. This is exact under the assumption of an axis-aligned equirectangular projection at the city scale (the meridian curvature is negligible across a few hundred metres of Astana).

**Mode 4 — None.** If the user does not supply any geographic information and the input is not a GeoTIFF, the system returns the detections in pixel coordinates only; the corresponding JSON fields are left as `null` and the Leaflet map is disabled in the frontend.

In addition to the coordinate conversion, the geographic module estimates the **pixel size in metres** at the image centre using the Haversine formula on the diagonal; this estimate is propagated through to all downstream statistics — average crown area in square metres, total green coverage in hectares, total tree count per hectare — and is reported alongside the inventory.

## 2.9 Persistent storage layer

All results are written to a local SQLite database at `storage/app.db` rather than kept in a Python process dictionary. The schema consists of four tables linked by foreign keys with `ON DELETE CASCADE`:

```
   ┌────────────────┐    ┌──────────────────┐    ┌───────────────────────┐
   │   snapshots    │◄───┤      runs        │◄───┤      detections       │
   │────────────────│    │──────────────────│    │───────────────────────│
   │ id (PK)        │    │ job_id (PK)      │    │ id (PK)               │
   │ file_path      │    │ snapshot_id (FK) │    │ run_id (FK)           │
   │ bounds_nw,se   │    │ model            │    │ bbox_px, bbox_geo     │
   │ geo_mode       │    │ confidence_thr   │    │ polygon_mask          │
   │ scan_id (FK?)  │    │ duration_ms      │    │ confidence            │
   │ created_at     │    │ created_at       │    │ area_px, area_m2      │
   └────────────────┘    └──────────────────┘    │ lon, lat (centroid)   │
           ▲                                     └───────────────────────┘
           │
   ┌───────┴────────┐
   │ scan_sessions  │
   │────────────────│
   │ id (PK)        │
   │ bbox_nw, se    │
   │ zoom, provider │
   │ model          │
   │ sub_region_cnt │
   │ status         │
   └────────────────┘
```

**Figure 2.2 — SQLite schema (`storage/app.db`).** Foreign keys cascade on delete: removing a snapshot deletes its runs, its detections and its PNG file on disk; removing a scan-session removes all its child snapshots in a single statement.

The persistence layer is implemented in `backend/db.py` and is used by every read and write path. The schema has three practical consequences. First, restarting the FastAPI process loses no detections — a critical property for a tool operated by a non-developer end user. Second, the city-map view (Section 2.10) can query the database for every detection ever produced with a single SQL statement and visualise them all on a single Leaflet layer; this is the principal aggregate-inspection workflow of the application. Third, snapshot deletion is implemented via a single `DELETE FROM snapshots WHERE id = ?` statement; the cascading foreign keys then remove all dependent runs, detections and the source image file from disk.

**Export.** Three exporters are reachable via `POST /api/export/{job_id}/{format}`. **GeoJSON** writes a FeatureCollection in which each detection is a Feature whose geometry is a Polygon (the crown mask in WGS-84) and whose properties contain confidence, crown area and bounding box — loadable directly into QGIS, ArcGIS or any compatible GIS tool. **CSV** writes a flat table with one row per detection (index, centroid coordinates, bounding box, confidence, area) for spreadsheet-based inspection and direct ingestion by *Zelenstroy*'s existing reporting workflow. **Standalone HTML** writes a single self-contained file with a Leaflet map embedded inline, OpenStreetMap and ESRI basemaps loaded from CDN, and the detections rendered as a vector layer with on-hover popups; the file is intended for sharing the inventory with a non-technical audience that does not have access to a GIS tool.

## 2.10 Application architecture and frontend workflows

The frontend is a single-page React 18 application served by FastAPI at the root URL, implemented in three files (`frontend/index.html`, `frontend/app.jsx`, `frontend/styles.css`) and a small API client (`frontend/api.js`). React, Babel-standalone and Leaflet are loaded directly from a CDN as UMD bundles — there is deliberately no build step. The motivation is operational simplicity: a municipal employee can run the system without Node.js, npm or any other JavaScript toolchain installed on the host.

### 2.10.1 Two view modes

The application exposes two main views, switchable in the sidebar. **Single-image view** is the workflow for a single satellite image: the user uploads a PNG, JPG or GeoTIFF (or captures one interactively from the map), selects a detection model and a confidence threshold, clicks *Run detection* and watches a progress indicator while the backend performs inference. The result is then visualised in three coordinated panels: a Leaflet map with the image overlaid as a semi-transparent layer and the detections rendered on top; a statistics panel showing the tree count, the green-coverage percentage, the mean confidence and the analysed area in hectares; and a confidence-filter slider that interactively hides or shows low-confidence detections without re-running the model.

![*Single-image view of the web application. The left panel shows the image upload zone, model selector (YOLO / Mask R-CNN / DeepForest / DeepForest+SAM 2 / Ensemble), confidence threshold, four-mode geographic referencing controls and the three export buttons (GeoJSON, CSV, HTML). The main panel displays the Leaflet satellite basemap with the uploaded image overlay and detected tree crowns rendered as polygon masks.*](figures/ui_single_image_view.png)

**City-map view** is the aggregate-inspection mode. It queries the persistent database for the full collection of all snapshots ever processed by the system and renders every detected tree on a single Leaflet layer (with a safety cap of 50 000 detections to protect the browser). A side panel lists each snapshot with a per-snapshot summary (number of runs, total trees, last-used model, geographic centre) and a deletion action that cascades through the database and the disk. This view is the principal demonstration deliverable of the project: a single map of Astana that grows tree-by-tree as the user processes new districts, building up an organic city-wide inventory.

![*City-map view showing 1 031 detected trees across three processed Astana snapshots. Crown polygons are colour-coded by confidence (green: high ≥ 70 %, yellow: medium 50–70 %, red: low < 50 %). The left panel shows aggregate statistics and a per-snapshot list. This view is the principal operational deliverable of the system, enabling city-wide tree inventory accumulation over time.*](figures/ui_city_map_view.png)

### 2.10.2 Geographic configuration and Auto-Zoom Region Scan

In both views the user controls the geographic mode of the active snapshot through a dedicated panel. The four modes of Section 2.8 are exposed as a segmented switch, and the user can enter corner coordinates either by typing them into form fields or by dragging NW/SE markers directly on the map until the image overlay aligns visually with the basemap; when the user moves a marker, the image overlay is re-bound to the new bounds in real time. The coordinates are written back to the database on the next inference run and persist across page reloads.

The principal map-capture workflow of the application is the **Auto-Zoom Region Scan**: the user draws a rectangle (or a freely-shaped polygon) on the basemap, and the backend automatically subdivides the request into a grid of sub-bounding-boxes at the fixed zoom level of 19 — the highest available resolution for which the YOLO and DeepForest models were trained — and processes each sub-region in turn. Three protections combine to keep the operation tractable: a hard cap of nine sub-regions per request (corresponding to approximately a 1.5 × 1.5 km area at zoom 19), the per-sub-region `MAX_TILES` cap of Section 2.2, and the use of a streaming NDJSON response on the `/api/scan_region/stream` endpoint that lets the frontend show the user incremental progress events (`plan`, `capturing`, `predicting`, `sub_complete`, `done`) as the scan proceeds, rather than blocking on the full operation. Each successful sub-region is persisted as an independent snapshot in the database and tagged with the parent scan-session identifier, so a single `DELETE /api/scans/{id}` request cascades through all its sub-region snapshots, runs, detections and PNG files in one operation. A **polygon-shaped scan** additionally applies a `shapely.Polygon.contains` filter on every detection centroid, retaining only those that fall inside the user-drawn polygon — useful for inventories of irregular districts, parks or river-front green corridors.

### 2.10.3 Detection display modes and hierarchical model picker

Every detection produced by the backend carries three independent geometric representations: a centre point (latitude / longitude of the bounding-box centroid), an axis-aligned bounding box (four corners in pixel space, lifted into geographic space through the active geo-conversion mode), and a polygon mask (for YOLO, Mask R-CNN and SAM 2-refined branches, a closed sequence of vertices following the projected crown outline). The frontend exposes these as four mutually-exclusive rendering modes through a segmented control: **Point** (circle at centroid; useful for inspecting density on coarse zoom levels), **Box** (geographic quadrilateral, important in four-corner geo-mode where the image is rotated relative to north), **Polygon** (default, projected crown mask) and **Heat-map** (kernel-density estimate weighted by per-detection confidence via the `leaflet.heat` plugin, particularly informative on the city-map view at 1 000+ detections where "hot" and "cold" districts emerge visually). Switching modes is instantaneous and does not require a backend round-trip; when a particular detection lacks data for the currently-selected mode (e.g. a DeepForest detection without a polygon mask), the frontend falls back automatically to point rendering so the detection never silently disappears.

The model selector is implemented as a **hierarchical picker** organised into three families — Single, Ensemble, Variant — that opens as a centred per-action modal when the user clicks the *Run detection* button. The Single family contains the four model branches; the Ensemble family contains the WBF ensemble and the four-member cross-YOLO vote; the Variant family contains eight YOLO checkpoints registered through the `ModelKind` enum (`v1`, `v2`, `v3_run1`, `v3_run2`, `v3_exp1`, `v4_x`, `v4_m`, `v4_s`) so the user can compare any specific generation against the production one on demand. The frontend defaults to dark mode.

### 2.10.4 REST endpoints

The backend exposes a complete REST API documented automatically by FastAPI's OpenAPI integration at `/docs`. Table 2.2 summarises the endpoints used by the frontend and by the export workflows.

**Table 2.2 — REST endpoints exposed by the backend.**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Health check and aggregate counts (snapshots, runs, total trees) |
| `GET` | `/api/providers` | Tile providers with URL templates and max-zoom |
| `POST` | `/api/upload` | Upload a satellite image; returns an `ImageMeta` with assigned id |
| `POST` | `/api/capture_from_map` | Stitch tiles for `(nw, se, zoom, provider)` and return an `ImageMeta` |
| `POST` | `/api/scan_region/stream` | Auto-Zoom Region Scan with NDJSON streaming progress; accepts optional `polygon` for point-in-polygon filtering |
| `GET` / `DELETE` | `/api/scans` , `/api/scans/{id}` | List or cascade-delete a scan-session and all its sub-region snapshots / runs / detections / PNG files |
| `POST` | `/api/predict` | Run inference: `{image_id, model, confidence, geo}` |
| `GET` | `/api/snapshots`, `/api/detections`, `/api/aggregate/stats` | Per-snapshot and database-wide aggregates for the city-map view |
| `DELETE` | `/api/snapshots/{id}`, `/api/runs/{job_id}` | Cascade-delete a snapshot or a single run |
| `POST` | `/api/export/{job_id}/{format}` | Export as GeoJSON / CSV / standalone HTML |

The endpoints are intentionally fine-grained: the frontend composes complex views from several small JSON responses rather than from a single monolithic dump, which makes the city-map view efficient even with tens of thousands of detections in the database.

\newpage
