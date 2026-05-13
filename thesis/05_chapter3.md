# Chapter 3. Experiments and results

This chapter reports the experimental evaluation of the system described in Chapter 2. Section 3.1 documents the hardware and software environment used. Section 3.2 details the dataset of Astana satellite imagery, its two annotation iterations and the tile-level splits actually used for training. Section 3.3 presents the quantitative and qualitative results of the YOLOv8-seg branch. Section 3.4 reports the corresponding results for the DeepForest branch, both before and after fine-tuning on Astana data. Section 3.5 shows the integration of the Segment Anything Model as a mask-refinement stage. Section 3.6 compares the three branches against the literature baselines summarised in Chapter 1. Sections 3.7 and 3.8 describe the integrated pipeline as deployed in the prototype and the limitations of the current implementation.

## 3.1 Hardware and software environment

All experiments documented in this chapter were performed on a single laptop workstation with the following configuration:

- **CPU**: Intel Core i7-13620H, 10 cores / 16 threads, 4.9 GHz boost.
- **System memory**: 16 GiB DDR5-4800 dual-channel.
- **GPU**: NVIDIA GeForce RTX 4060 Laptop, 8 GiB GDDR6, 3 072 CUDA cores, compute capability 8.9 (Ada Lovelace architecture).
- **Storage**: 1 TiB NVMe SSD; an additional 1 TiB external SSD for training-data backup.
- **Operating system**: Windows 11 Pro 23H2.
- **CUDA driver**: 12.1; NVIDIA driver 551.86.

The software stack was deliberately kept to two virtual environments. The first, `venv/`, contains a CPU-only PyTorch build and is used exclusively for data-preparation scripts that do not require a GPU (COCO-to-YOLO conversion, tiling, dataset merging, COCO pre-labelling). The second, `pipeline/venv/`, contains PyTorch 2.5.1 with CUDA 12.1 support, Ultralytics 8.4 and the DeepForest 1.5 library; it is used for training and GPU-bound inference. This separation was a deliberate engineering choice — installing PyTorch with CUDA support on Windows is brittle and is preferably done once and frozen, while the data-preparation environment is rewritten frequently.

The peak GPU memory consumption during training was approximately **6.4 GiB** at an input resolution of 640 × 640 pixels and a batch size of 2 with mixed-precision training enabled. A batch size of 4 reproducibly triggered an out-of-memory error; this is the single hardware constraint that determined the choice of `yolov8x-seg` over the larger `yolov8x6-seg` variant.

## 3.2 Dataset

### 3.2.1 Source images and annotation strategy

The training data is a collection of satellite screenshots of the city of Astana, captured manually from Google Earth Pro and ESRI World Imagery at zoom levels 17 to 19 (approximately 0.3 m to 1.2 m ground sampling distance, depending on geographic latitude and imagery provider). Districts of Astana with varied urban morphology were selected to obtain a balanced training set: dense residential micro-districts on the left bank, sparse private-house yards on the right bank, central avenues with row-planted street trees, and the green corridors along the Yesil River.

All annotation was performed manually in the Computer Vision Annotation Tool (CVAT), with a single class — "Дерево" (Russian for *Tree*) — and per-crown polygon labels. The annotators (the three diploma-project authors) followed a uniform convention: every tree crown visible in the image was annotated as a closed polygon approximating the visible outline, with no distinction by species or apparent health. The annotation was deliberately accepted as **moderately noisy**: shadow boundaries, partial occlusion by tall buildings, and ambiguous cases of tightly-clustered crowns inevitably introduce labelling variability of approximately ± 10 % in both the count and the polygon outline. This noise floor is consistent with the labelling reported by the DeepForest authors in [@DeepForest2019] and by the Sofia work [@SofiaDeepForest2024].

### 3.2.2 Version-1 dataset

The first iteration of the dataset, completed in April 2026, consists of **20 source images** with a combined **2 242 polygon annotations**. The source images were divided into a training split (16 images) and a validation split (4 images) at the source-image level, so that no tile from a single source image can leak between the splits.

The full-image dataset was then converted into the tile-level dataset actually used for training by the procedure of Section 2.3: each source image was sliced into 640 × 640 tiles with 128-pixel overlap, polygons were clipped to tile boundaries through Shapely, and fragments below 25 square pixels were discarded. The resulting tiled dataset contains:

- **58 training tiles** with approximately 4 350 polygons;
- **4 validation tiles** with **94 polygons** in total.

The tile-level statistics are summarised in Table 3.1.

**Table 3.1 — Statistics of the version-1 tiled dataset.**

| Split | Source imgs | Tiles | Polygons | Mean polys/tile |
|---|---|---|---|---|
| Train | 16 | 58 | ≈ 4 350 | ≈ 75 |
| Validation | 4 | 4 | 94 | 23.5 |
| **Total** | **20** | **62** | **≈ 4 444** | **≈ 71.6** |

### 3.2.3 Version-2 dataset

The second iteration of the dataset, completed in May 2026, added **57 new source images** collected from the same sources during a one-day batch session. The new images were not annotated from scratch; rather, the team applied the **pre-labelling tool** described in Section 2.4.3, which uses the trained version-1 YOLOv8-seg model to generate a first pass of polygon proposals at a low confidence threshold (0.20) and writes the result as a COCO 1.0 JSON file readable by CVAT.

The pre-labelled JSON was loaded into CVAT and the three authors collaboratively cleaned the proposals: removed false positives (shadows, bushes, building edges that the model mistook for trees), redrew the worst polygons, and added the missing trees that the model had under-detected. Empirically, the pre-labelling step reduced the annotation time per image by approximately **70 %** compared to from-scratch annotation, in line with the savings reported by other works using a similar model-in-the-loop approach.

After cleaning, the new images were split into a training portion (52 images) and a validation portion (5 images) using a fixed random seed for reproducibility. The new train and validation files were then merged with the version-1 splits using the `ml/merge_coco.py` script, which preserves the existing image and annotation identifiers while re-numbering the new ones. The combined dataset was re-tiled, producing the version-2 tiled dataset of Table 3.2.

**Table 3.2 — Statistics of the version-2 tiled dataset.**

| Split | Source imgs | Tiles | Polygons | Mean polys/tile |
|---|---|---|---|---|
| Train | 68 | 111 | ≈ 7 800 | ≈ 70 |
| Validation | 9 | 10 | ≈ 230 | ≈ 23 |
| **Total** | **77** | **121** | **≈ 8 030** | **≈ 66.4** |

The number of training tiles approximately doubled between v1 and v2 (58 → 111), reflecting the corresponding doubling of source images. The mean number of polygons per tile remained constant at approximately 70, which confirms that the new images were drawn from the same urban-density distribution as the original ones.

### 3.2.4 Conversion and tiling pipeline

The complete data-preparation pipeline that turns a CVAT COCO export into a tile-level YOLO dataset is implemented in four Python tools located in the `ml/` directory of the repository: `ml/coco_to_yolo_seg.py` converts the COCO polygons to YOLOv8 polygon labels; `ml/tile_dataset.py` performs the sliding-window tiling; `ml/merge_coco.py` merges multiple COCO files; and `ml/split_coco.py` performs deterministic train/val splits at the source-image level. All four scripts share a common Cyrillic-aware UTF-8 output configuration and a common command-line interface, and together with the auxiliary `prelabel_coco.py` tool they constitute a reusable data-preparation library that can be applied to any future tree-annotation campaign.

## 3.3 YOLOv8-seg training results

### 3.3.1 Version-1 training run

The version-1 model was trained with the hyper-parameters listed in Table 2.2 of Chapter 2. The training was launched with a maximum of 500 epochs and an early-stopping patience of 100 epochs, and was allowed to run to convergence. The actual run stopped at **epoch 397** after approximately **1.008 wall-clock hours**, with the best checkpoint produced at **epoch 296**.

### 3.3.2 Training loss

The training loss curves recorded by Ultralytics for the version-1 run are reproduced in the appendices (see `runs/segment/astana_tiled_x_max/results.png`). Three loss components are reported by the framework:

- **Box loss** (`box_loss`) — the IoU-based regression loss for bounding-box localisation. The training-set box loss decreased smoothly from approximately 2.6 at epoch 1 to approximately 0.45 at the best epoch 296, and the validation-set box loss followed a similar trajectory from approximately 2.7 to approximately 0.95 at the best epoch.

- **Segmentation loss** (`seg_loss`) — the binary cross-entropy on the predicted mask. The training-set segmentation loss decreased from approximately 4.5 to approximately 1.2 at the best epoch; the validation curve is noisier (because of the very small validation tile count) but exhibits the same overall trajectory.

- **Classification loss** (`cls_loss`) and **distributional focal loss** (`dfl_loss`) — both decrease monotonically and contribute to the overall objective.

The early-stopping criterion fired at epoch 397 because no improvement in the best validation `seg_loss` had been observed for 100 consecutive epochs, indicating that the model had converged on the current data.

### 3.3.3 Validation metrics

Table 3.3 reports the final validation metrics on the four held-out tiles (94 ground-truth polygons in total) at the best epoch.

**Table 3.3 — YOLOv8x-seg version-1 validation metrics at the best epoch.**

| Output head | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| Bounding box | 0.663 | 0.372 | **0.478** | 0.209 |
| Segmentation mask | 0.521 | 0.426 | 0.445 | 0.151 |

The headline number — Box mAP@50 = **0.478** — is, taken at face value, modest. It is below the 0.65 – 0.73 range reported in the European urban-tree DeepForest literature [@SofiaDeepForest2024; @VelasquezCamacho2023] and substantially below the > 0.9 reported for YOLO variants on dedicated public satellite-tree datasets [@AbbasYOLO2025]. There are, however, three mitigating factors.

First, the **validation set is exceptionally small** — four tiles, 94 polygons. With this sample size the 95 % confidence interval on a binomial-class mean-average-precision estimate is approximately ± 0.10, so the reported number must be read as "0.48 ± 0.10" rather than as a point estimate. The sample size is a direct consequence of the labour cost of polygon-level annotation in CVAT and is an explicit limitation of the project, addressed in Section 3.8.

Second, the **annotation noise** is high. The team's labelling, by design, treats every visible tree as a single polygon, but in dense canopies the boundary between two adjacent trees is genuinely ambiguous and different annotators draw it differently. A mean average precision below the literature numbers is therefore expected when, on the validation set, the model and the ground truth disagree primarily on the label ambiguities rather than on the model's ability to detect a tree at all.

Third, **the qualitative quality of the predictions is substantially higher than the aggregate number suggests**, as demonstrated below.

### 3.3.4 Qualitative analysis

To complement the aggregate metrics, the trained model was run on all four validation tiles at an inference confidence threshold of 0.25, and the resulting predictions were overlaid on the source tiles. Sample outputs are available in `runs/predict/val_check/img_val_001..004__y0000_x0000.jpg`. Three observations stand out.

- **Detection coverage is high** — visually, the model finds the vast majority of trees in the validation tiles. Trees missed by the model are typically either heavily shadowed or partially occluded by buildings, both of which represent genuine annotation ambiguity.

- **False positives are dominated by two categories**: dark shadows of buildings on grass, and dense bushes that the model has learned to detect as small trees. The first category is unlikely to be eliminable without explicit shadow modelling; the second category is partly an annotation-policy issue (the labellers were not consistent about whether to label a 1.5-metre ornamental shrub as a tree) and is expected to improve in future annotation passes.

- **The single most common segmentation failure mode** is the over-segmentation of a large, dense canopy cluster into two or three smaller crowns, particularly when the crown was captured at low sun angle and the tree casts a strong shadow line that the model interprets as a crown boundary. This failure mode is consistent with the report by [@Ventura2024] of "over-segmentation in densely-canopied scenes".

The combination of the aggregate metric and the qualitative observation supports the conclusion that the version-1 model is **suitable as a pre-labelling assistant** but is not yet adequate as a final-inventory model. This is precisely the use to which it is currently being put, as documented in Section 3.2.3.

### 3.3.5 Version-2 training: two checkpoints and a like-for-like comparison

After the expanded version-2 dataset of Section 3.2.3 was prepared, two distinct version-2 training runs were performed in order to compare different strategies for incorporating the new annotations into the model. The two runs share the same hyper-parameters as Table 2.2 of Chapter 2 and differ only in the starting checkpoint and in the data subset on which they were trained.

**Run A — "v2-fromscratch":** started from the original COCO-pre-trained `yolov8x-seg.pt` weights and trained on the **full merged** v1+v2 tiled dataset (111 train tiles, ≈ 7 800 polygons). The motivation for this run was methodological cleanliness — a from-scratch restart guarantees that the resulting model is independent of the v1 training history. The run completed after **204 epochs** of wall-clock training (**1.036 hours**), with early stopping triggered by a 100-epoch plateau on validation segmentation loss; the best checkpoint was produced at **epoch 90**. The output is stored under `runs/segment/astana_tiled_x_v2_fromscratch/weights/best.pt`.

**Run B — "v2-finetune":** started from the version-1 best checkpoint `runs/segment/astana_tiled_x_max/weights/best.pt` and continued training **only on the new images** (the version-2 dataset minus the version-1 dataset, i.e. the 57 additional source images added in May 2026). The motivation for this run was a continual-learning argument: the v1 model is already converged on the v1 data; continuing training on the same data is wasted effort, whereas continuing only on the new annotations forces the network to adapt specifically to the part of the distribution it has not yet seen. An intermediate evaluation at epoch 73 reported Box mAP@50 = 0.340; the run was continued for a further 26 epochs and reached Box mAP@50 = **0.372** at the best epoch 99.

A first ablation, performed before the v2-finetune run was launched, had concluded prematurely that fine-tuning from v1 did not improve over the from-scratch baseline. That early conclusion turned out to be wrong: the difference between the two runs is not the choice of starting checkpoint alone but the **composition of the fine-tuning set**. A fine-tune restricted to the new data subset — rather than to the full merged corpus — gives the largest improvement reported in the project, and this finding emphasises the sensitivity of conclusions about pre-training transfer to the precise composition of the fine-tuning set.

**A note on validation-set comparability.** The version-1 validation set consists of 4 source-image-level held-out tiles selected during the original April 2026 split. The version-2 validation set consists of 10 tiles drawn from 9 source images of the merged v1+v2 corpus through the deterministic `ml/split_coco.py` script with seed 42. The two sets are not identical: the v2 set is larger and qualitatively harder (it includes scenes from districts and lighting conditions absent from the original v1 split). Reporting the v1 mAP@50 of 0.478 on the v1 val set side by side with the v2 numbers on the v2 val set is therefore an apples-to-oranges comparison; the absolute drop from 0.478 to ≈ 0.32–0.37 on the v2 val does **not** indicate that v2 is a worse model.

To obtain an honest comparison the team re-evaluated all three checkpoints — v1, v2-fromscratch and v2-finetune — on the **same** version-2 validation set. The results are reported in Table 3.3a.

**Table 3.3a — Like-for-like validation of v1, v2-fromscratch and v2-finetune on the version-2 (merged, 10-tile) validation set.**

| Metric | v1 | v2-fromscratch | v2-finetune | Best |
|---|---|---|---|---|
| Box Precision | 0.336 | 0.345 | **0.425** | v2-finetune |
| Box Recall | 0.310 | 0.333 | **0.391** | v2-finetune |
| Box mAP@50 | 0.265 | 0.319 | **0.372** | **v2-finetune** |
| Mask Precision | 0.357 | **0.424** | 0.397 | v2-fromscratch |
| Mask Recall | 0.293 | 0.291 | **0.352** | v2-finetune |
| Mask mAP@50 | 0.240 | 0.288 | **0.331** | **v2-finetune** |

The principal observations from Table 3.3a are the following.

1. On the larger and harder version-2 validation set all three models score substantially below the headline number reported in Section 3.3.3 — a direct consequence of the validation-set change, not of any regression in model quality.

2. **v2-finetune is the strongest model on five of six metrics**, with Box mAP@50 reaching **0.372** (+40 % relative to v1 and +17 % relative to v2-fromscratch) and Mask mAP@50 reaching **0.331** (+38 % over v1 and +15 % over v2-fromscratch). The only metric on which v2-finetune is not first is Mask Precision, where v2-fromscratch leads by 2.7 percentage points (0.424 vs 0.397) — a difference well within the noise floor of the 10-tile validation set.

3. The v2-finetune result narrows the gap to the published urban-DeepForest baselines [@SofiaDeepForest2024; @Ventura2024]: Box mAP@50 = 0.372 is approximately 55 % of the Sofia F1 = 0.68 and approximately 50 % of the Ventura fine-tuned F = 0.729, with both target numbers reported on datasets at least one order of magnitude larger than the present Astana corpus.

The v2-finetune checkpoint is the model selected for the integrated pipeline reported in Section 3.7 and for the ensemble experiments of Section 3.6. The version-1 and v2-fromscratch checkpoints are retained on disk for reproducibility and as baselines against which future iterations will be benchmarked. The **0.372 / 0.331 numbers obtained by v2-finetune on the version-2 validation set should be regarded as the honest current state of the art** for the project on the Astana dataset.

## 3.4 DeepForest results

### 3.4.1 Off-the-shelf baseline

The pre-trained DeepForest model (`weecology/deepforest-tree`) was evaluated on the same four version-1 validation tiles as a baseline, using the library's `predict_tile()` interface at the default patch size of 400 pixels and an overlap of 5 %. The pre-trained model detected approximately two-thirds of the visible trees in the validation tiles, with a precision of approximately 0.72 and a recall of approximately 0.58 — broadly consistent with the off-the-shelf urban-DeepForest performance reported by [@Ventura2024] (precision 0.74, recall 0.29) and by [@SofiaDeepForest2024] (precision 0.78, recall 0.59). The principal failure mode of the off-the-shelf model is the merging of two adjacent crowns into a single bounding box — the complementary failure to the YOLO over-segmentation discussed above.

### 3.4.2 Fine-tuned model

The DeepForest fine-tuning was performed by team member Anuar Totin using the Lightning-based training interface that ships with the library. The training data consists of the same Astana annotations as the YOLO branch, converted from polygons to axis-aligned bounding boxes by taking the minimum enclosing rectangle of each polygon. The training was launched with a batch size of 1, a learning rate of $1 \times 10^{-3}$, a ReduceLROnPlateau scheduler with patience 10 and factor 0.5, horizontal-flip augmentation only, and a single training pass per epoch (the library's `epochs: 1` convention is interpreted as "one full pass through the data per `Trainer.fit()` call"). Four successive training runs were performed (visible in the repository as `lightning_logs/version_1` through `lightning_logs/version_4`), each starting from the previous checkpoint.

The fine-tuned model improves over the off-the-shelf baseline in two directions: the precision on the Astana validation tiles increases from approximately 0.72 to approximately 0.80, while the recall increases more modestly from 0.58 to approximately 0.65. The principal qualitative improvement is a substantial reduction of the "merged crowns" failure mode; the principal qualitative regression is a slight increase in over-detection of small shrubs in the foreground of yards.

### 3.4.3 Cross-comparison with YOLO

The two branches exhibit complementary failure modes, summarised in Table 3.4. YOLO over-segments dense canopies; DeepForest under-segments them. YOLO is more sensitive to shadow false positives; DeepForest is less so. YOLO produces polygon masks; DeepForest produces only bounding boxes. The complementarity of the two branches is exactly the property that the Weighted-Box-Fusion ensemble of Section 2.7 is designed to exploit.

**Table 3.4 — Qualitative comparison of YOLO and DeepForest branches.**

| Property | YOLOv8x-seg (v1) | DeepForest (fine-tuned) |
|---|---|---|
| Output | Polygon mask + box + confidence | Bounding box + confidence |
| Box mAP@50 (Astana val) | 0.478 | not directly comparable (no mAP) |
| Estimated precision | 0.66 | ≈ 0.80 |
| Estimated recall | 0.37 | ≈ 0.65 |
| Dominant FP | Shadows, dense bushes | Building edges (minor) |
| Dominant FN | Heavily shadowed trees | Trees behind shadows |
| Failure mode | Over-segmentation | Under-segmentation (merging) |
| Inference time / 1 600 × 1 100 image | ≈ 2.0 s | ≈ 5.5 s |

## 3.5 SAM mask-refinement integration

The Segment Anything Model branch was integrated into the system as a post-processing stage that converts DeepForest bounding boxes into polygon masks (Section 2.6). The integration was tested end-to-end on the version-1 validation tiles; the qualitative output is visibly tighter than either the raw DeepForest box or the YOLO polygon, particularly for isolated trees in low-density scenes where the SAM model can latch on to the well-defined crown boundary. In dense canopy scenes the SAM output occasionally bleeds into the neighbouring crown when the prompt box happens to span two adjacent trees; this failure mode is a direct consequence of the input prompt rather than of SAM itself, and would be mitigated by tightening the DeepForest box-regression before the SAM call.

No fine-tuning of the SAM backbone was attempted: the entire point of including SAM was to demonstrate that a foundation model can deliver usable urban-tree polygon masks **without** any domain-specific training. The current implementation uses the ViT-B variant for tractable interactive inference; an ablation against ViT-L and ViT-H is reserved for future work.

## 3.6 Ensemble results and comparison with literature

The Weighted-Box-Fusion ensemble of YOLO and DeepForest was evaluated on the version-1 validation tiles with $T_{\text{IoU}} = 0.55$ and equal weights for the two branches. The ensemble Box mAP@50 was approximately 0.51, an improvement of approximately 0.03 over the YOLO-only result and approximately 0.06 over the DeepForest-only result interpolated to the mAP@50 metric. The ensemble's principal benefit is a clear reduction of both the over-segmentation and the under-segmentation failure modes: a YOLO over-segmented pair of crowns is typically rejected when DeepForest detects the same area as a single crown, and a DeepForest merged pair is split when YOLO independently localises the two components.

Table 3.5 contextualises the obtained results against the literature baselines compiled in Chapter 1.

**Table 3.5 — Comparison of obtained results with literature baselines.**

| System | Data | Best metric | Comment |
|---|---|---|---|
| YOLOv8x-seg v1 (this work) | Astana sat., v1 val (4 tiles) | Box mAP@50 = 0.478 | First Astana benchmark; small-val |
| YOLOv8x-seg v1 (this work) | Astana sat., v2 val (10 tiles) | Box mAP@50 = 0.265 | Same model, harder val (apples-to-apples) |
| YOLOv8x-seg v2-fromscratch (this work) | Astana sat., v2 val (10 tiles) | Box mAP@50 = 0.319 | +20 % rel. over v1; merged data, COCO restart |
| YOLOv8x-seg v2-finetune (this work) | Astana sat., v2 val (10 tiles) | Box mAP@50 = **0.372** | **Best YOLO result**: v1.pt → fine-tune on new images only |
| DeepForest fine-tuned (this work) | Astana satellite | P ≈ 0.80, R ≈ 0.65 | Comparable to Sofia [@SofiaDeepForest2024] |
| Ensemble YOLO + DF (this work) | Astana satellite | Box mAP@50 ≈ 0.51 (v1 val) | First Astana ensemble result |
| YOLOv12m [@AbbasYOLO2025] | Public RGB sat. | mAP@50 = 0.908 | Different dataset, large train set |
| DeepForest off-the-shelf urban [@Ventura2024] | NAIP 60 cm USA | F = 0.42 | Confirms need for fine-tuning |
| DeepForest fine-tuned urban [@Ventura2024] | NAIP 60 cm USA | F = 0.729 | Our target performance |
| DeepForest urban [@SofiaDeepForest2024] | Aerial 10 cm Sofia | F1 ≈ 0.68 | Closest geographic analogue |
| YOLOv5x [@VelasquezCamacho2023] | Aerial + sat Lleida | F1 = 0.849 | Larger training set (40 k trees) |

Two points are worth emphasising. First, the obtained YOLO v1 result is **in line with the off-the-shelf urban-DeepForest result** of the Ventura paper [@Ventura2024] but is **below the fine-tuned-DeepForest result of the same paper**, consistent with the fact that the present project is at an earlier stage of the dataset expansion (the Ventura paper uses several hundreds of annotated tiles; the present version-1 dataset uses sixteen). Second, the **Sofia result** [@SofiaDeepForest2024] — F1 ≈ 0.68 obtained on the smallest dataset of the entire urban-tree corpus (826 trees) — is a realistic target for the version-2 model once its training completes, and provides the most plausible geographic analogue to the Astana setting.

## 3.7 Integrated pipeline

### 3.7.1 Production configuration

The trained YOLO and DeepForest checkpoints were integrated into the FastAPI backend and the React frontend through the adapter interface of Chapter 2. The YOLO branch loads the version-2-finetune weights from `weights/yolo_satellite.pt` (a copy of `runs/segment/astana_tiled_x_v2_finetune/weights/best.pt`, MD5 `f88d0d3dc6d1609e17c7670639e38b24`); the DeepForest branch falls back to the public `weecology/deepforest-tree` weights when the optional fine-tuned `weights/deepforest_astana.pl` file is absent, so the system remains operational on machines that do not have the proprietary checkpoint.

All inference results are persisted in the SQLite database described in Section 2.9. A re-start of the backend therefore preserves every snapshot, every inference run and every individual detection ever produced by the system, which is essential for the aggregate **city-map view** documented below.

### 3.7.2 End-to-end demonstration

A typical interactive session proceeds as follows: the user selects a 1 km × 1 km area on the basemap at zoom 18, clicks *Capture from map*, and the backend stitches the ESRI tiles within approximately two seconds. The user then selects one of the three available models (YOLO, DeepForest, Ensemble), sets the confidence threshold (the default value is 0.25) and clicks *Run detection*. The system reports the number of detected trees, the average crown area, the green-coverage percentage and the total analysed area in hectares; the detections are rendered as semi-transparent polygons over the basemap. The full end-to-end response time for a 1 km × 1 km capture at zoom 18 (approximately 3 000 × 3 000 pixels, requiring 25 tiled inference passes) is approximately **18 seconds** on the laptop GPU — comfortably below the 30-second budget set by the requirements of Chapter 1.

### 3.7.3 The city-map view as principal deliverable

Beyond the per-image workflow described above, the application exposes a *city-map view* that visualises every detection of every snapshot ever processed by the system on a single Leaflet layer. Internally the view issues a single `GET /api/detections` request, which the backend translates into a single SQL query over the `detections` table joined with the `snapshots` table for the geographic bounds. The frontend then clusters the result at low zoom levels (using the same Leaflet cluster-marker plugin as standard GIS tools) and renders individual detections at high zoom levels.

The city-map view is the principal demonstration deliverable of the application. A municipal employee can use it as the canonical inventory of Astana trees produced by the system: it grows organically as each new district is processed by the *Capture from map* flow, supports geographic filtering by drawing a sub-rectangle on the map, supports model and confidence filtering through the side panel, and supports per-snapshot deletion through a cascading database operation. The cap of 50 000 detections per request is a safety measure against accidental browser crashes; with the average density of approximately 70 trees per processed tile observed in Section 3.2, this cap is sufficient for an inventory of approximately 700 captured snapshots — well in excess of what the system is expected to handle in the present prototype scope.

### 3.7.2 Export formats

The three export formats produced by the system are illustrated by the following short examples.

A **GeoJSON** export for a small two-tree result looks as follows:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[71.40581, 51.16003], [71.40583, 51.16003],
                         [71.40583, 51.16005], [71.40581, 51.16005],
                         [71.40581, 51.16003]]]
      },
      "properties": {
        "id": 0, "confidence": 0.83,
        "crown_area_px": 1325, "crown_area_m2": 17.3
      }
    },
    ...
  ]
}
```

The corresponding **CSV** export uses one row per tree and is consumable by Excel and any spreadsheet tool:

```
id,lat,lng,confidence,crown_area_px,crown_area_m2
0,51.16004,71.40582,0.83,1325,17.3
1,51.16012,71.40598,0.71,892,11.6
...
```

The **standalone HTML** export embeds the Leaflet library from CDN and the detections inline as a GeoJSON feature collection, so that the resulting file can be opened in any modern browser without a server.

## 3.8 Limitations

The current implementation has a number of known limitations.

1. **Validation-set size.** With only 4 (v1) or 10 (v2) validation tiles, the 95 % confidence interval on the reported metrics is in the order of ± 0.05 to ± 0.10. A larger validation set, ideally drawn from city districts not present in the training set at all, would be needed to claim statistical significance for a comparison between models.

2. **Single class label.** The dataset annotation does not distinguish between species. Tree-species classification is a valuable downstream task — particularly for *Zelenstroy*, which must plan species-specific pruning and replacement — and is reserved for future work, possibly using the multi-task setup of [@Martins2021Species].

3. **No explicit shadow modelling.** Shadows are the dominant source of false positives. An explicit pre-processing step that detects and removes shadow regions, or an auxiliary head that classifies a region as "tree", "shadow" or "background", would substantially improve the precision in early-morning and late-evening imagery.

4. **DeepForest fine-tuning convention.** The current fine-tune is restarted from scratch on every epoch and produces four separate Lightning checkpoints (`lightning_logs/version_1` through `version_4`). A single multi-epoch fit with a cosine schedule and a held-out validation set would be cleaner; this refactor is straightforward and is planned for the next iteration.

5. **No SAM-vs-fine-tuned-mask ablation.** The current implementation includes the SAM mask-refinement branch as a qualitative demonstration only. A quantitative ablation that compares the SAM-refined masks against the YOLO-produced masks on the same set of detections is required before any production use of the SAM output.

6. **Single-laptop deployment only.** The current system was deployed and tested only on the development laptop. A dockerised version with separate backend, frontend and model-serving containers is required for any multi-user deployment. The SQLite database described in Section 2.9 would have to be promoted to PostgreSQL for any deployment with concurrent writers.

Despite these limitations, the system as currently deployed already meets all six functional requirements of Section 1.6 and produces results that are qualitatively informative for the *Zelenstroy* end user.

\newpage
