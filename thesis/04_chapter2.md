# Chapter 2. Methods

This chapter describes the technical design of the system. Section 2.1 gives a top-level view of the architecture and the data flow. Section 2.2 describes the image input pipeline and tiled inference. Sections 2.3 – 2.6 detail the four detection branches — YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest+SAM 2 — and Section 2.7 describes the two ensemble strategies that combine them. Sections 2.8 – 2.9 cover the geographic-conversion stage and the SQLite persistence layer. Section 2.10 describes the application architecture and frontend workflows in detail, including the database schema, the REST API, the Auto-Zoom Region Scan and the four detection-display modes.

## 2.1 System architecture overview

The system follows a classical three-tier separation of concerns: a thin presentation layer (a single-page React 18 application served by FastAPI, with React, Babel-standalone and Leaflet loaded directly from a CDN — no Node.js build step), an application layer (a FastAPI REST backend in Python 3.11) and a model layer (four pluggable deep-learning adapters that wrap the underlying frameworks). The five layers and their respective responsibilities are summarised in Table 2.1.

**Table 2.1 — Layered architecture of the system and the responsibilities of each layer.**

| Layer | Technology | Responsibility | Key components |
|---|---|---|---|
| Presentation | React 18 (UMD) + Leaflet + Babel-standalone, no build step | Interactive map, image upload, detection visualisation, model selector, export buttons | Single-image view, City-map view, Scan-area / Polygon-scan workflows |
| Application | FastAPI + Pydantic + Uvicorn, Python 3.11 | REST API, request validation, lazy model loading, NDJSON streaming progress, geographic conversion | `/api/upload`, `/api/capture_from_map`, `/api/predict`, `/api/scan_region/stream`, `/api/export/{job}/{format}` |
| Model | PyTorch 2.5 + CUDA 12.1 + Ultralytics 8.4 + DeepForest 1.5 + sam2 1.1 + torchvision 0.20 | Inference on satellite tiles via the adapter pattern; tiled inference + global NMS | YOLO, Mask R-CNN, DeepForest, DeepForest+SAM 2, WBF ensemble, cross-YOLO 4-vote ensemble |
| Persistence | SQLite (`storage/app.db`) with `ON DELETE CASCADE` foreign keys | Persist every snapshot, run and detection ever produced by the system | `snapshots`, `runs`, `detections`, `scan_sessions` tables |
| Export | Python serialisers + Leaflet HTML template | Deliver the inventory in GIS-compatible formats | GeoJSON (QGIS / ArcGIS), CSV (Excel), standalone HTML (browser-only) |

The data flow through the system can be summarised as follows. The user opens the web interface and either uploads a satellite image as a file (PNG, JPG, TIFF or GeoTIFF) or selects an area directly on a Leaflet map; in the latter case the backend downloads the appropriate ESRI World Imagery or Google Satellite tiles, stitches and crops them, and returns the resulting image with embedded geographic bounds. The image is stored on the server and assigned an opaque identifier; image dimensions and, if available, GeoTIFF projection metadata are extracted via the `rasterio` library. The user selects one of four detection backends (YOLO, Mask R-CNN, DeepForest, DeepForest+SAM 2) or one of two ensembles (WBF, cross-YOLO vote) and a confidence threshold, and triggers inference; the corresponding adapter is loaded lazily on first request and reused for subsequent calls. For images larger than the network's native input resolution, the adapter performs sliding-window tiled inference, runs the model on each tile and aggregates the per-tile predictions through Non-Maximum Suppression to produce a single global set of detections. Each detection is annotated with geographic coordinates via the geographic-conversion module, which supports four operating modes depending on the metadata available. The result is stored as an immutable job record, returned to the frontend for interactive visualisation on a Leaflet map, and made available for export in three formats.

A key architectural decision is the use of the adapter pattern for the model layer: every model adheres to a single abstract base class with a `predict(image_path, confidence) -> List[Detection]` method, so new models can be added without changes to the rest of the system. The base class provides lazy initialisation, automatic weight loading from a `weights/` directory and exception conversion to FastAPI HTTP responses. The complete source tree is organised in three top-level directories — `backend/`, `frontend/` and `ml/` — at approximately 6 000 lines of code.

## 2.2 Image input and tiled inference

The system accepts three categories of input: regular images (PNG, JPG, WebP) decoded with Pillow on first use; GeoTIFF images carrying an affine transform and CRS in their metadata, parsed with `rasterio.open()` and re-projected to WGS-84 (EPSG:4326) when needed; and in-browser map capture, where the user draws a rectangle on a Leaflet map and the backend downloads the corresponding tiles from a configurable provider.

Two tile providers are registered (Table 2.2): the default ESRI World Imagery endpoint (max zoom 19, no API key, stable) and an unofficial Google Satellite endpoint (max zoom 20, the same image base as Google Earth Pro — directly relevant because the YOLO and Mask R-CNN training data was collected from Google Earth Pro screenshots). The map-capture flow (i) converts the two corner coordinates from longitude / latitude into floating-point Web-Mercator tile coordinates; (ii) computes the integer tile range and, for large bounding boxes, splits the request into up to nine sub-regions each capped at approximately 100 tiles, so per-tile downloads remain below a configurable `MAX_TILES` limit (default 400, ≈ 5 120 × 5 120 px per sub-region) that protects the server from accidental bulk downloads; (iii) downloads the tiles in parallel through a thread pool of eight workers, replacing failed tiles by a small grey placeholder; and (iv) stitches the tiles into a single canvas and crops to the requested bounding box, returning a PNG with the geographic bounds embedded.

**Table 2.2 — Tile providers registered in the map-capture module.**

| Provider key | Endpoint | Max zoom | Notes |
|---|---|---|---|
| `esri` (default) | `server.arcgisonline.com/.../World_Imagery/...` | 19 | Free, no API key, stable |
| `google` | `mt{0–3}.google.com/vt/lyrs=s` | 20 | Same image base as Google Earth Pro (training data source); unofficial endpoint, academic prototyping only |

**Tiled inference.** A YOLOv8-seg or DeepForest model operates on inputs of approximately 640 × 640 px, while a single Astana screenshot is typically 1 600 × 1 100 px or larger. Resizing to fit the network input shrinks each tree crown — typically 20 – 40 px in diameter at zoom 18 — below the resolution any modern detector can reliably handle. Early experiments confirmed this: a YOLOv8-seg trained at `imgsz=640` directly on full-resolution inputs detected zero trees on the validation set. The system therefore performs sliding-window tiled inference at the same resolution as training: the original image is partitioned into overlapping 640-px tiles with a 128-px overlap so that any tree crown is fully contained in at least one tile; the model is applied independently to each tile; and the per-tile detections are translated back into global image coordinates by adding the tile offset to every polygon vertex. Duplicate detections of the same tree that appears in multiple overlapping tiles are removed by a global Non-Maximum Suppression pass with an IoU threshold of 0.5, vectorised in NumPy. The same tiling-and-NMS pattern is reused in `ml/tile_dataset.py` for training-data preparation and in `ml/prelabel_coco.py` for model-in-the-loop pre-labelling.

## 2.3 YOLOv8-seg branch

YOLOv8 [@UltralyticsYOLO2023] is a single-stage anchor-free object detector. The segmentation variant `YOLOv8x-seg` extends the architecture with an additional prototype mask head that produces 32 prototype masks at one quarter the input resolution; each detected instance is then represented by a vector of 32 coefficients that, linearly combined with the prototype masks and thresholded, produces a binary segmentation mask aligned with the detected bounding box. This design — borrowed from the YOLACT family of real-time instance segmenters — separates the per-pixel and per-instance computations, so the cost of generating masks scales with the number of detections rather than with the number of pixels.

**Why instance segmentation rather than detection.** Bounding-box detection is sufficient for counting trees, but the municipal user also wants the crown size (a proxy for species and age) and the green-coverage percentage of a given area (a standard urban-planning indicator). Both require pixel-level masks rather than rectangular boxes — a poplar and a birch of the same canopy area can have radically different crown silhouettes, and the bounding box of two adjacent overlapping crowns can cover a substantial fraction of bare ground. The output of the adapter therefore contains: a bounding box, a polygon mask (the YOLO mask compressed into a closed sequence of $(x, y)$ vertices via Suzuki–Abe contour extraction), a confidence score, and — derived from the mask — the crown area in pixels and (when pixel size in metres is known) in square metres.

**Training data and pre-processing.** The training data is a collection of Astana satellite screenshots taken from Google Earth and ESRI World Imagery at zoom levels 17 – 19. The annotation effort proceeded in three iterations (v1 / v2 / v3) totalling approximately 100 source images and ≈ 8 700 polygon-level annotations after tiling. The class taxonomy is deliberately minimal — a single class "Tree" (in the source: "Дерево") — because the satellite resolution available does not allow reliable species discrimination. The dataset is split at the source-image level (no tile from a single source leaks between splits). Two custom Python tools support the workflow: `ml/coco_to_yolo_seg.py` converts CVAT-exported COCO 1.0 annotation files into the polygon-line format expected by Ultralytics YOLO, with sanitisation of Cyrillic filenames into ASCII to avoid Windows-path issues; and `ml/tile_dataset.py` performs the sliding-window tiling itself using Shapely for polygon clipping, dropping any clipped fragment whose area falls below 25 square pixels.

**Training procedure.** Three YOLOv8-seg variants are used in this work, corresponding to three phases of the project's hyperparameter ablation reported in Chapter 3. The pre-ablation generations used `yolov8x-seg` (≈ 71 M parameters) on the initial hypothesis that the extra capacity would improve performance. Rounds 1 – 3 of the systematic ablation softened that hypothesis: when trained from COCO weights with manually-tuned augmentation the medium-size `yolov8m-seg` (≈ 27 M parameters) edged out both larger variants, though by a margin inside the ± 0.03 single-run variance band measured in Section 3.2. Round 4 then reversed the reversal: with Ultralytics' default augmentation pipeline instead of the manually-tuned one, the largest `yolov8x-seg` gives the top aggregate Box score, reaching Box mAP@50 = 0.315 on M14 — the headline empirical result of the diploma. The final production checkpoint is therefore `yolov8x-seg` trained from fresh COCO weights with `single_cls=True` and Ultralytics' default augmentation, on the single laptop GPU (NVIDIA GeForce RTX 4060, 8 GiB VRAM); mixed-precision training (AMP) was essential to fit batch size 2 in memory. The principal hyper-parameters of the production training run are summarised in Table 2.3.

**Table 2.3 — Hyper-parameters of the YOLOv8x-seg production training run.**

| Parameter | Value |
|---|---|
| Base model | `yolov8x-seg.pt` (COCO pre-trained, 71.4 M params) |
| Input resolution | 640 × 640 |
| Batch size | AutoBatch (Ultralytics 60 % VRAM heuristic, ≈ 2 on RTX 4060 8 GiB) |
| Epochs (max) | 150 |
| Early-stopping patience | 50 on validation mAP@50 |
| Time cap | 1.5 hours |
| Optimiser | AdamW, lr = 0.001 (auto-selected by Ultralytics for small datasets) |
| Augmentation | Ultralytics defaults: HSV-S 0.7, HSV-V 0.4, random erasing 0.4, no geometric (degrees=0, mixup=0, copy-paste=0, flipud=0) |
| Single-class mode | `single_cls=True` |
| Mixed-precision (AMP) | enabled |
| Wall-clock time | ≈ 91 minutes |

**Loss function.** The training objective of YOLOv8-seg is the weighted sum of four components — a bounding-box regression loss, a per-class classification loss, a Distribution Focal Loss for the discrete-bin regression head, and a per-pixel mask loss:

$$
\mathcal{L} \;=\; \lambda_{\text{box}}\,\mathcal{L}_{\text{CIoU}} \;+\; \lambda_{\text{cls}}\,\mathcal{L}_{\text{BCE-cls}} \;+\; \lambda_{\text{dfl}}\,\mathcal{L}_{\text{DFL}} \;+\; \lambda_{\text{seg}}\,\mathcal{L}_{\text{BCE-mask}}
$$

The bounding-box term is the Complete IoU loss, which extends the plain IoU loss with a centre-point-distance term and an aspect-ratio-inconsistency term, $\mathcal{L}_{\text{CIoU}} = 1 - \mathrm{IoU} + \rho^{2}(b, b^{gt})/c^{2} + \alpha v$, where $\rho$ is the Euclidean distance between predicted and ground-truth box centres, $c$ is the diagonal of the smallest box that encloses both, $v$ is a measure of aspect-ratio inconsistency, and $\alpha$ a trade-off coefficient. The loss-weight defaults are the Ultralytics-standard $\lambda_{\text{box}} = 7.5$, $\lambda_{\text{cls}} = 0.5$, $\lambda_{\text{dfl}} = 1.5$, retained as-is for comparability with the COCO-pretrained checkpoint.

## 2.4 Mask R-CNN branch

Mask R-CNN [@MaskRCNN2017] is a two-stage instance-segmentation network that extends the Faster R-CNN [@FasterRCNN2015] detector with a fully-convolutional mask head running in parallel with the bounding-box-regression head. The two-stage design relies on a Region Proposal Network that first generates a small set of class-agnostic region proposals, which are then classified, regressed and segmented independently. The variant adopted is the standard Mask R-CNN with a ResNet-50 backbone and an FPN neck, initialised from publicly-available COCO-pretrained `maskrcnn_resnet50_fpn_v2` torchvision weights and fine-tuned on the same Astana polygon dataset used for the YOLO branch. The motivation for including this branch is methodological: it provides a like-for-like architectural comparison between a one-stage (YOLO) and a two-stage (Mask R-CNN) instance segmenter under identical training data and validation conditions, in line with the comparative analysis surveyed in Section 1.3.

Two Mask R-CNN checkpoints exist in the project. The v1 + v2 base model was trained from the torchvision COCO V1 weights with SGD (momentum 0.9, weight decay $5 \times 10^{-4}$), an initial learning rate of $5 \times 10^{-3}$, a `StepLR` scheduler halving the learning rate every 10 epochs, batch size 2 and AMP. The v2 + v3 fine-tune — which is the production checkpoint released under tag `maskrcnn-v2v3` — warm-starts from the v1 + v2 base via `--resume-from` and lowers the initial learning rate automatically to $1 \times 10^{-3}$. The principal training-side improvement is a richer Albumentations pipeline: horizontal flip ($p$ = 0.5), vertical flip ($p$ = 0.3), random 90-degree rotation ($p$ = 0.5), random brightness/contrast adjustment ($p$ = 0.3) and HSV jitter ($p$ = 0.2). Early-stopping is implemented on the validation `mask_map_50` metric with a patience of 5 epochs; the v2 + v3 fine-tune was early-stopped at epoch 16 (best at epoch 11). The adapter exposes the standard `predict(image_path, confidence) -> List[Detection]` method and reuses the same tiled-inference + global-NMS code path as the YOLO branch (Section 2.2).

## 2.5 DeepForest branch

DeepForest [@DeepForest2019] is a tree-detection library built on top of a RetinaNet single-stage detector with a ResNet-50 backbone. RetinaNet was selected by the DeepForest authors over Faster R-CNN because of its better speed–accuracy trade-off and over earlier YOLO versions because of its dedicated focal-loss formulation [@RetinaNet2017], $\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^{\gamma} \log(p_t)$ (with $\gamma = 2$ in the DeepForest default), which down-weights well-classified background examples and is particularly well-suited to the highly imbalanced foreground-to-background ratio of dense forest scenes.

The model is shipped with two pre-trained weight sets: the "tree" model (`weecology/deepforest-tree`), trained on hundreds of thousands of semi-supervised annotations derived from NEON aerial-lidar data over forested sites in the United States, and a "bird" model irrelevant to the present work. DeepForest's recommended inference mode is the `predict_tile()` method, which performs sliding-window patch-based inference on 400 × 400-px patches with a 5 % overlap and merges per-patch detections via internal NMS. The adapter wraps this method and translates the resulting pandas DataFrame into the same `Detection` dataclass that the YOLO adapter produces.

**Fine-tuning on Astana data.** A first DeepForest fine-tune was performed by team member Anuar Totin in early May 2026 on a separate bounding-box annotation set maintained on Roboflow. The production DeepForest checkpoint used by the deployed backend is the v3 fine-tune, which warm-starts from that v4 checkpoint and continues training on the same merged Astana CVAT polygon dataset used by the YOLO and Mask R-CNN branches. For the v3 fine-tune the CVAT polygon annotations are converted to DeepForest's bounding-box CSV format via `ml/coco_to_deepforest_csv.py` — every polygon is replaced by its axis-aligned bounding box. The fine-tune trains for 30 epochs with SGD + momentum at learning rate $1 \times 10^{-4}$, batch size 4 and horizontal-flip augmentation, on the RTX 4050 Laptop GPU. The fine-tuned weights are stored on disk as a Lightning checkpoint (`weights/deepforest_astana_v3.pl`, published as GitHub release `v2.0`). The effective training trajectory of the production weights is therefore NEON pretrained → v4 (Roboflow) → v3 (CVAT).

## 2.6 DeepForest + SAM 2 mask-refinement branch

The fourth branch exploits the zero-shot generalisation property of SAM 2 [@SAM2_2024] to upgrade DeepForest's bounding-box detections into precise crown polygons without any additional annotation or training. SAM 2 is the second-generation version of Meta AI's foundation segmentation model, succeeding the original SAM [@SAM2023]. Where SAM was trained on the SA-1B dataset of more than one billion masks across eleven million images, SAM 2 extends this with the SA-V dataset of more than thirty-five million masks across ≈ 51 000 videos. For the static-image, single-frame tree-detection task addressed here only the image-level segmentation capability of SAM 2 is used; its temporal-propagation mode is reserved for future-work multi-temporal canopy monitoring.

The pipeline of this branch is:

1. The DeepForest detector is run on the input image and produces a list of bounding boxes with associated confidence scores, exactly as in Section 2.5.
2. All bounding boxes above the confidence threshold are passed as a batch of SAM 2 box prompts in the image coordinate frame.
3. The `SAM2ImagePredictor.predict()` call returns one binary mask per box (with `multimask_output=False`).
4. The mask is converted into a polygon contour through OpenCV contour extraction, and the resulting `Detection` is emitted as if it had been produced by an end-to-end instance segmenter.

The branch is implemented as a separate adapter (`DeepForestSAM2Adapter`) that takes the DeepForest adapter as a constructor argument. The SAM 2 model used is `sam2.1-hiera-base-plus`, loaded automatically from HuggingFace (`facebook/sam2.1-hiera-base-plus`) on first inference, with the device (CUDA or CPU) detected automatically. The hiera-base-plus variant was chosen as a compromise between mask quality and inference speed; the larger hiera-large variant is also supported but is too slow for the interactive use case on a laptop GPU. Conceptually, this design treats SAM 2 as a post-processing step that decorates an otherwise pure bounding-box detector with high-quality polygon masks; the cost is a roughly two-fold increase in inference time per image.

## 2.7 Ensemble strategies

The system implements two complementary ensemble strategies. The Weighted-Box-Fusion ensemble of Section 2.7.1 combines a YOLO checkpoint with a DeepForest checkpoint (cross-architecture, addressing complementary failure modes between an instance-segmenter and a bounding-box-only detector); the cross-YOLO voting ensemble of Section 2.7.2 combines four YOLO checkpoints with each other (within-architecture, addressing per-checkpoint training-time variance and per-checkpoint failure modes).

### 2.7.1 Weighted Box Fusion (YOLO + DeepForest)

The YOLO and DeepForest branches are trained on the same data but with different network architectures, patch sizes and loss formulations. Their errors are therefore partly de-correlated: YOLO tends to over-segment large dense canopies into several smaller crowns, while DeepForest tends to merge adjacent crowns into a single bounding box. The chosen ensemble strategy is Weighted Box Fusion [@WBF2021]. Where Non-Maximum Suppression keeps the single highest-confidence box and discards every overlapping box, WBF instead averages the coordinates of the overlapping boxes weighted by their confidence scores. For a cluster of $n$ overlapping predictions $\{(\mathbf{b}_{i}, c_{i})\}_{i=1}^{n}$, the WBF-fused box and fused confidence are

$$
\mathbf{b}_{\text{fused}} \;=\; \frac{\sum_{i=1}^{n} c_{i}\, \mathbf{b}_{i}}{\sum_{i=1}^{n} c_{i}}, \qquad
c_{\text{fused}} \;=\; \frac{\sum_{i=1}^{n} c_{i}}{n} \cdot \frac{\min(n, M)}{M}
$$

where $M$ is the number of models being ensembled (here $M = 2$). The right-hand factor $\min(n, M)/M$ down-weights clusters that contain detections from only a subset of the available models — a single-model cluster receives half of its raw average confidence, while a two-model cluster receives the full average. This factor is what makes WBF a true ensemble (it rewards agreement between models) rather than a smoothing operation. The implementation uses the open-source `ensemble-boxes` package. The IoU threshold for fusing overlapping YOLO and DeepForest boxes is set to 0.5, the standard value for box clustering.

### 2.7.2 Cross-YOLO voting ensemble (IoU-clustered, K-of-N majority vote)

A complementary ensemble that operates within the YOLO family rather than across architectures was added in the late stage of the project, motivated by two observations from Chapter 3: per-checkpoint training-time variance of sample standard deviation ≈ 0.03 Box mAP@50 across four replicates of the same configuration, and qualitative cross-checkpoint complementarity in which different YOLO checkpoints with similar aggregate mAP detect substantially different per-detection subsets. The cross-YOLO ensemble averages out the per-checkpoint variance and discards single-checkpoint hallucinations through a majority-voting rule.

**Algorithm.** Given $N$ member YOLO checkpoints, the ensemble (i) runs every member on the input image independently, (ii) pools all $M$ detections from the $N$ members into a single flat list tagged by member identity, (iii) clusters the pooled detections by box IoU using a union-find data structure (any two detections with $\text{IoU} \geq 0.5$ are merged into the same cluster), (iv) discards any cluster whose detections come from fewer than $K$ distinct member models (default $K = 2$), and (v) emits the highest-confidence detection from each surviving cluster as its representative. The complexity is $O(NM^2)$, dominated by the pairwise IoU computation; on a typical Astana tile with $M \approx 750$ per member the four-member ensemble runs end-to-end in approximately 4–5 seconds on the laptop GPU.

**Default member set.** Four YOLO checkpoints spanning three model generations (v2, v3 and v4) are configured as the default ensemble members — generation diversity is the design goal, because a vote made up only of same-generation models would share their failure modes: v4_x_clean (the final v4 production checkpoint, strongest aggregate Box mAP, but the generation that introduced the stadium-roof false positive); exp1_m_cocostart (a v3-era yolov8m-seg with v2-proven augmentation, recovers more partially-occluded crowns); v3-finetune-run1 (a v3-era yolov8x checkpoint, a second independent v3 vote); and v2-finetune (the previous-generation production checkpoint, which predates and does not exhibit the stadium-roof regression). Because the roof artifact is specific to the v4 generation, the three earlier-generation members can outvote it under the $K = 2$ rule, whereas a v4-only ensemble could not. The cross-YOLO ensemble is exposed in the frontend's hierarchical model picker under the Ensemble → 4× YOLO vote option.

## 2.8 Geographic conversion

A pixel coordinate $(x, y)$ inside the detection mask has no immediate meaning to a municipal user; the system must convert it into a $(\text{longitude}, \text{latitude})$ pair in WGS-84. The conversion is implemented in `backend/geo.py` and supports four operating modes, selected automatically depending on the metadata available.

**Mode 1 — GeoTIFF affine.** If the input is a GeoTIFF, the affine transform written in the metadata maps any pixel coordinate to a coordinate in the file's projection. The system reads the transform with `rasterio.transform.AffineTransformer.xy(row, col)` and re-projects the result to EPSG:4326 with `pyproj.Transformer` when needed. This mode is the most accurate and is the recommended workflow for production use.

**Mode 2 — Four-corner bilinear.** If the user supplies the geographic coordinates of all four corners of the image (by dragging four markers on the Leaflet map until the screenshot is aligned with the underlying satellite basemap), the conversion uses bilinear interpolation: for a pixel at relative coordinates $(u, v) \in [0,1]^{2}$,
$$\mathbf{g}(u,v) = (1-u)(1-v)\mathbf{g}_{\text{nw}} + u(1-v)\mathbf{g}_{\text{ne}} + (1-u)v\mathbf{g}_{\text{sw}} + uv\mathbf{g}_{\text{se}}.$$

**Mode 3 — Two-corner axis-aligned.** When only the NW and SE corners are known (the common case after the in-browser map-capture workflow), the conversion is a simple linear interpolation along each axis, $\lambda(x) = \lambda_{\text{nw}} + (x/W)(\lambda_{\text{se}} - \lambda_{\text{nw}})$ and $\phi(y) = \phi_{\text{nw}} + (y/H)(\phi_{\text{se}} - \phi_{\text{nw}})$. This is exact under the assumption of an axis-aligned equirectangular projection at the city scale (the meridian curvature is negligible across a few hundred metres of Astana).

**Mode 4 — None.** If the user does not supply any geographic information and the input is not a GeoTIFF, the system returns the detections in pixel coordinates only; the corresponding JSON fields are left as `null` and the Leaflet map is disabled in the frontend.

In addition to the coordinate conversion, the geographic module estimates the pixel size in metres at the image centre using the Haversine formula on the diagonal; this estimate is propagated through to all downstream statistics — average crown area in square metres, total green coverage in hectares, total tree count per hectare — and is reported alongside the inventory.

## 2.9 Persistent storage layer

All results are written to a local SQLite database rather than kept in a Python process dictionary. The schema consists of four tables linked by foreign keys with `ON DELETE CASCADE`. The tables and their principal columns are described in Table 2.4.

**Table 2.4 — SQLite schema of the persistence layer.**

| Table | What it stores | Foreign key | Typical cardinality |
|---|---|---|---|
| `snapshots` | One row per uploaded or captured image — file path, geographic bounds, geo-conversion mode | `scan_id` → `scan_sessions(id)`, ON DELETE SET NULL | 1 per image |
| `runs` | One row per model invocation on a snapshot — model name, confidence threshold, duration | `snapshot_id` → `snapshots(id)`, ON DELETE CASCADE | 1 per inference |
| `detections` | One row per detected tree — pixel + geographic bbox, polygon mask, confidence, crown area | `run_id` → `runs(job_id)`, ON DELETE CASCADE | 300 – 800 per typical 1 km² capture |
| `scan_sessions` | One row per Auto-Zoom Region Scan — bbox, zoom, provider, model, sub-region count, status | — | 1 per scan |

The relationships form a chain: `scan_sessions` ← `snapshots` ← `runs` ← `detections`, with cascade on the inner two foreign keys. Deleting a scan-session via `DELETE /api/scans/{id}` therefore propagates through all its sub-region snapshots, runs, detections and the PNG files on disk in a single SQL statement (with a complementary `os.unlink` on the file paths returned by the cascade).

The persistence layer is implemented in `backend/db.py` and is used by every read and write path. The schema has three practical consequences. First, restarting the FastAPI process loses no detections — a critical property for a tool operated by a non-developer end user. Second, the city-map view (Section 2.10) can query the database for every detection ever produced with a single SQL statement and visualise them all on a single Leaflet layer; this is the principal aggregate-inspection workflow of the application. Third, snapshot deletion is implemented via a single `DELETE FROM snapshots WHERE id = ?` statement; the cascading foreign keys then remove all dependent runs, detections and the source image file from disk.

**Export.** Three exporters are reachable via `POST /api/export/{job_id}/{format}`. GeoJSON writes a FeatureCollection in which each detection is a Feature whose geometry is a Polygon (the crown mask in WGS-84) and whose properties contain confidence, crown area and bounding box — loadable directly into QGIS, ArcGIS or any compatible GIS tool. CSV writes a flat table with one row per detection (index, centroid coordinates, bounding box, confidence, area) for spreadsheet-based inspection and direct ingestion into a municipal reporting workflow. Standalone HTML writes a single self-contained file with a Leaflet map embedded inline, OpenStreetMap and ESRI basemaps loaded from CDN, and the detections rendered as a vector layer with on-hover popups; the file is intended for sharing the inventory with a non-technical audience that does not have access to a GIS tool.

## 2.10 Application: Canopy

The frontend is a single-page React 18 application served by FastAPI at the root URL, branded *Canopy* and tagged with the leaf logo used throughout the figures in this section. The implementation lives in four files — `frontend/index.html`, `frontend/app.jsx`, `frontend/api.js` and `frontend/styles.css` — plus an optional dev tweaks panel (`frontend/tweaks-panel.jsx`). React, Babel-standalone and Leaflet are loaded directly from a CDN as UMD bundles — there is deliberately no build step. The motivation is operational simplicity: a municipal employee can run the system without Node.js, npm or any other JavaScript toolchain installed on the host. The visual design follows three principles inherited from modern map-first GIS tools: map-first (the basemap occupies the largest possible share of the viewport), progressive disclosure (only the controls relevant to the current state are shown), and dark-mode-by-default with a one-click light-mode toggle (semi-transparent detection polygons are easier to read against a dark background, but a *Zelenstroy* operator working in a brightly-lit office can switch to light mode at any time).

### 2.10.1 Top bar, view switcher and global controls

The application has a thin top bar (Figure 2.1) that is identical on both views. On the left, the *Canopy* brand mark with the leaf logo and the strap-line "Astana urban tree inventory · 2026" doubles as a return-to-default link. In the centre, a segmented switch toggles between the two views — Map (city-wide aggregate inspection, the default landing view) and Image (single-image upload-and-detect workflow). On the right, two icon buttons: a sun / moon toggle for dark / light mode, and a gear that opens the Settings popover. The gear shows a small status dot (green / amber / red) in the corner that reflects backend availability — green when all model adapters are reachable, amber when at least one adapter is unreachable, red when the backend itself is down.

![*Map view of Canopy in its default dark-mode state. The thin top bar carries the Canopy brand on the left, the Map / Image segmented switch in the centre, and dark-mode + settings icon buttons on the right. The left panel hosts all interactive controls in a single scrollable column: primary actions (Scan area, Polygon), the city-wide aggregate hero stat (51 348 trees across 14 scans), secondary metrics, the confidence filter and the recent-scans list. The main canvas is the Leaflet map. The Display strip in the bottom-right corner switches the rendering mode (Point / BBox / Polygon / Heat) for the aggregate layer.*](figures/ui_canopy_map_view.png){width=98%}

The same view in light mode is shown in Figure 2.2 — the toggle re-themes the entire interface via a CSS custom-property cascade rooted in `[data-theme="dark"|"light"]` so theme changes are instantaneous without a re-render of any Leaflet layer.

![*Light-mode variant of the Map view. The colour palette inverts (light surfaces with dark text, accent colours unchanged) so a user working under bright office lighting can read the panel without strain. All other behaviour is identical to the dark-mode variant of Figure 2.1.*](figures/ui_canopy_map_view_light.png){width=98%}

### 2.10.2 Map view: the city-wide aggregate

The Map view is the principal demonstration deliverable of the project. It queries the persistent database for every detection ever produced by the system and renders the aggregate on a single Leaflet layer; with a safety cap of 50 000 detections per request to protect the browser. The left panel (Figure 2.1) is organised top-to-bottom as a vertical reading order:

* **Primary actions** — two large buttons that initiate a new capture: Scan area (rectangular Auto-Zoom Region Scan) and Polygon (free-shaped scan). Clicking either opens the centred *ScanModelModal* described in § 2.10.3 below.
* **Hero stat** — the city-wide tree count rendered at display size with the eyebrow "Astana · Canopy aggregate". A sub-line summarises the corpus (number of scans, number of snapshots) and a metric row reports total runs, average trees per scan, and the count currently visible on the map under the active filters.
* **Secondary metrics** — two compact tiles for average confidence and average crown diameter (computed from the polygon area when geographic conversion is available).
* **Filters** — a min-confidence slider with a live percentage readout, and three toggle chips for the confidence tiers (High ≥ 70 %, Med 50 – 70 %, Low < 50 %). Toggling a tier hides the corresponding detections on the map without a backend round-trip.
* **Recent scans** — a chronological list of the six most-recent scan sessions, each with a status dot (green = completed, amber = running), the display name (auto-generated as `Scan <short-id>` or user-renamed), the per-scan tree count, time-ago timestamp, and the model used. Each row carries an eye icon that toggles the scan's visibility on the aggregate layer — useful for isolating a single district visually. The *Manage* link opens the Library modal described in § 2.10.6 below.

The same screen exposes a Display strip anchored to the bottom-right corner of the map (visible in Figure 2.1) with two control groups: a four-button segmented control to switch the rendering mode of the aggregate layer (Point / BBox / Polygon / Heat) and a *Filters* chip that opens an in-context popover with the same min-confidence slider as the left panel. Switching modes is instantaneous and does not require a re-fetch: every detection in the backend response carries three geometric representations — a centre point, an axis-aligned bounding box, and a polygon mask — and the frontend simply re-renders. The Heat mode uses the `leaflet.heat` plugin to produce a kernel-density visualisation weighted by per-detection confidence; the Polygon mode is the default whenever every detection in the active selection carries a mask, and the frontend falls back automatically to Point for any detection without geometry (most often a DeepForest detection rendered before SAM 2 mask refinement).

A real-world example of the aggregate layer at scale is shown in Figure 2.3 — a single Auto-Zoom Region Scan over the Astana Botanical Park (left-bank district, approximately 1 × 1 km), captured at zoom 19 with the v4_x_clean production checkpoint. The scan produces several thousand individual tree-crown detections rendered as yellow / green bounding boxes on the basemap; the colour encodes confidence tier (yellow / orange = low–medium, green = high). The aggregate counter in the left panel sums every detection across all stored demo scans — it illustrates the running-total UI, not a validated tree census of Astana (the total changes with each re-scan and includes overlapping demo runs over the same areas).

![*Real-world scan of the Astana Botanical Park (zoom 19, v4_x_clean production model, BBox display mode). Each rectangle is a single detected tree crown; colour encodes the confidence tier (yellow / orange for low-to-medium, green for high). The dense detection cluster outlines the regular star-shaped path layout of the park's central plantings and the more sparse perimeter row-trees along the surrounding avenues — a strong qualitative validation that the production pipeline produces results consistent with what a human operator would expect at this scale. The aggregate counts in the side panel come from repeated demo scans and are not a city-wide tree census.*](figures/ui_canopy_botanical_park.jpg)

### 2.10.3 Action-aware model picker (centred modal)

A common usability problem with multi-model systems is that the model selector lives in a sidebar far from the primary action, so the user clicks "Run" without realising which model is active. Canopy avoids this by forcing a model choice at the moment of action: clicking either *Scan area* or *Polygon* opens a centred modal (Figure 2.4) that exposes the full hierarchical model picker and an explicit *Start* / *Cancel* pair. The modal closes only when the user confirms, after which the application transitions into the drawing mode (rectangle or polygon).

![*ScanModelModal — the centred per-action model picker that opens when the user clicks Scan area or Polygon. Row 1 selects the model family (YOLOv8 / DeepForest / Mask R-CNN / Ensemble); row 2 selects the variant inside the active family (here: YOLOv8 v4 x · champ — the production checkpoint, alongside six other archived YOLO variants v4 m / v4 s · fast / v3 exp1 / v3 run 1 / v3 run 2 / v2 legacy). The footer shows the active model identifier and the per-action hint "After confirmation, drag a rectangle on the map to define the scan area.".*](figures/ui_canopy_scan_model_modal.png){width=80%}

The picker is hierarchical by family — a deliberate choice to keep the row of choices short. The four families are:

* **YOLOv8** — seven variants registered through the `ModelKind` enum: `v4 x · champ` (the production champion, v4_x_clean at Box mAP@50 = 0.315), `v4 m`, `v4 s · fast`, `v3 exp1`, `v3 run 1`, `v3 run 2`, `v2 legacy` (the prior production checkpoint kept for A / B comparison).
* **DeepForest** — two variants: *with SAM 2* (default, produces polygon masks) and *boxes only* (raw RetinaNet output).
* **Mask R-CNN** — single variant (R50-FPN v2, the production checkpoint of Section 2.4).
* **Ensemble** — two variants: *4× YOLO vote* (the cross-YOLO voting ensemble of Section 2.7.2) and *WBF (YOLO + DF)* (the Weighted-Box-Fusion ensemble of Section 2.7.1).

The variant row only appears when the active family has more than one variant; for Mask R-CNN the second row is hidden, keeping the picker visually compact. A small green dot on the family button marks the *currently-active* family so the user can browse other families without losing track of what is selected. Pressing *Enter* in the modal confirms; *Escape* cancels.

### 2.10.4 Image view: progressive disclosure

The Image view (Figure 2.5) is the workflow for a single satellite image — typically a screenshot the user has already cropped to the region of interest, or a GeoTIFF received from a colleague. The sidebar adopts a numbered, progressive-disclosure layout in which only the section relevant to the current state is visible:

* **1 · Image** — drag-and-drop upload zone (PNG / JPG / TIFF / GeoTIFF / WebP up to 100 MB), or a thumbnail preview of the loaded image with its dimensions, GeoTIFF status and remove button.
* **2 · Georeferencing** — appears only after an image is loaded; lets the user pick from three modes ("None — pixel coords only", "Two corners (NW + SE)", "Four corners (handles rotation)"). GeoTIFFs skip the section entirely with a green status row reading "GeoTIFF — auto from file metadata (EPSG:4326)".
* **3 · Detection** — the hierarchical model picker (same component as the ScanModelModal, embedded inline) plus a confidence-threshold slider and a *Run detection* button.
* **4 · Results** — appears only after a successful run; shows a four-tile stat grid (tree count, average confidence, canopy coverage percentage, average crown area in m²), the same Display segmented control as the Map view, and an optional toggle to overlay the source image on the basemap with adjustable opacity.
* **5 · Export** — appears only after results exist; three buttons for GeoJSON, CSV and standalone HTML.

![*Image view in its empty state. The right sidebar shows only the first section (drop-image or click) because no image is loaded yet; sections 2 – 5 are hidden by the progressive-disclosure rule. The Leaflet basemap on the left occupies the rest of the viewport and is centred on Astana.*](figures/ui_canopy_image_view_empty.png){width=98%}

### 2.10.5 Auto-Zoom Region Scan and live progress

The principal map-capture workflow of the application is the Auto-Zoom Region Scan. Once the user confirms the model in the ScanModelModal and draws a rectangle (or a closed polygon) on the basemap, the backend automatically subdivides the request into a grid of sub-bounding-boxes at the fixed zoom level of 19 — the highest available resolution for which the YOLO and DeepForest models were trained — and processes each sub-region in turn. Three protections combine to keep the operation tractable: a hard cap of nine sub-regions per request (corresponding to approximately a 1.5 × 1.5 km area at zoom 19), the per-sub-region `MAX_TILES` cap of Section 2.2, and the streaming NDJSON response on `/api/scan_region/stream`.

While the scan runs, a ScanProgressCard floats at the top-centre of the map. Its primary widget is a square grid of cells — one per sub-region — that change colour as the backend emits NDJSON events: grey for *pending*, blue for *capturing*, orange for *predicting*, green for *done*, red for *error*. A spinning brand mark beside the title and a horizontal progress bar at the bottom give the user a peripheral-vision sense of progress without forcing them to read the grid. The total tree count accumulates as each sub-region completes. When the scan finishes the card switches to a "Scan complete" state with an optional rename input — non-blocking, so the user can ignore it and the scan keeps its auto-generated `Scan <short-id>` name. A polygon-shaped scan additionally applies a `shapely.Polygon.contains` filter on every detection centroid, retaining only those that fall inside the user-drawn polygon — useful for inventories of irregular districts, parks or river-front green corridors.

The four-mode geographic-conversion module of Section 2.8 remains in the picture for the Image view: the user can pick None, Two-corner, Four-corner or GeoTIFF mode in the "2 · Georeferencing" section of the sidebar, and the corner markers can be dragged directly on the map until the image overlay aligns visually with the basemap. The conversion mode is persisted with the snapshot so the inventory remains reproducible across sessions.

### 2.10.6 Library modal and inline management

The Library modal (Figure 2.6) is the principal management surface for the persistent inventory. It opens from the *Manage* link in the Map view's Recent-scans block and presents two tabs — Scans and Snapshots — each a searchable, sortable card grid. Cards show a per-item summary (total trees, sub-region count, tile provider, zoom, model, duration, geographic centre, creation timestamp, short ID) and expose three actions: inline rename (single-click on the title), delete with confirmation, and a visibility toggle (eye icon) that hides the scan or snapshot from the Map-view aggregate layer without deleting it. Sort options include *newest first* (default), *oldest first*, *most trees*, *fewest trees*, and *name A–Z*; the search box matches both display names and short IDs. Deleting a scan triggers the ON-DELETE-CASCADE chain of Section 2.9: all its sub-region snapshots, runs, detections and PNG files on disk are removed in a single operation.

![*Library modal — the Scans tab showing 14 scan sessions on a searchable, sortable card grid. Each card carries the inline-renameable display name, status tags (running / polygon), a stats row (trees, sub-regions), a metadata row (provider, zoom, model, duration), the bounding-box corner coordinates, and action buttons (hide-from-map eye, delete). The toolbar at the top hosts the search input and the sort dropdown.*](figures/ui_canopy_library_modal.png){width=98%}

### 2.10.7 Settings popover

The Settings popover (Figure 2.7) opens from the gear button in the top bar and is intentionally minimal: a tile-provider dropdown ("Google Satellite (same imagery base)" / "Esri World Imagery") and a model-status list. The model-status list reports the registered name, the loaded / lazy state and an availability dot for each of the thirteen registered model adapters (the twelve user-selectable picker entries of § 2.10.3 plus the hidden generic YOLO alias) — at a glance the user can see which checkpoint files are present on disk and whether they have already been warmed in memory by a previous request. The popover dismisses on click-outside or Escape.

![*Settings popover. The Satellite imagery dropdown lets the user switch the basemap (and scan-capture source) between Google Satellite — the same image base as the training data — and Esri World Imagery. Below, the Model status block enumerates the thirteen registered model adapters with per-model availability dots and "active · ready / lazy" tags so the user can see at a glance which checkpoints are loaded.*](figures/ui_canopy_settings_popover.png){width=98%}

### 2.10.8 Detection display modes

Every detection produced by the backend carries three independent geometric representations: a centre point (latitude / longitude of the bounding-box centroid), an axis-aligned bounding box (four corners in pixel space, lifted into geographic space through the active geo-conversion mode), and — for YOLO, Mask R-CNN and SAM 2-refined branches — a polygon mask (a closed sequence of vertices following the projected crown outline). The frontend exposes these as four mutually-exclusive rendering modes through the segmented control: Point (circle at centroid; useful for inspecting density at coarse zoom levels), BBox (geographic quadrilateral, important in four-corner geo-mode where the image is rotated relative to north), Polygon (default, projected crown mask) and Heat (kernel-density estimate weighted by per-detection confidence via the `leaflet.heat` plugin, particularly informative on the city-map aggregate at 1 000+ detections where "hot" and "cold" districts emerge visually). Switching modes is instantaneous; when a particular detection lacks data for the currently-selected mode (e.g. a raw DeepForest detection without a polygon mask), the frontend falls back automatically to point rendering so the detection never silently disappears from the map.

### 2.10.9 REST endpoints

The backend exposes a complete REST API documented automatically by FastAPI's OpenAPI integration at `/docs`. Table 2.5 summarises the endpoints used by the frontend and by the export workflows.

**Table 2.5 — REST endpoints exposed by the backend.**

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
