# Chapter 3. Experiments and results

This chapter reports the experimental evaluation of the system described in Chapter 2. Section 3.1 documents the hardware and software environment used. Section 3.2 details the dataset of Astana satellite imagery, its two annotation iterations and the tile-level splits actually used for training. Section 3.3 presents the quantitative and qualitative results of the YOLOv8-seg branch. Section 3.4 reports the training and validation of the Mask R-CNN branch. Section 3.5 reports the corresponding results for the DeepForest branch, both before and after fine-tuning on Astana data. Section 3.6 shows the integration of SAM 2 as a mask-refinement stage. Section 3.7 compares the four branches against the literature baselines summarised in Chapter 1. Sections 3.8 and 3.9 describe the integrated pipeline as deployed in the prototype and the limitations of the current implementation.

## 3.1 Hardware and software environment

Because the three model branches of the project (YOLOv8x-seg, Mask R-CNN, DeepForest) are owned by three different team members, the training and evaluation experiments documented in this chapter were performed on **three separate laptop workstations** — one per branch owner. The configurations are summarised in Table 3.0.

**Table 3.0 — Workstations used for the three model branches.**

| Branch | Owner | GPU | Phys. VRAM | Peak training VRAM | Wall-clock training |
|---|---|---|---|---|---|
| YOLOv8x-seg (v1, v2-fs, v2-ft, v3-ft) | R. Aidarkhanov | NVIDIA RTX 4060 Laptop | 8 GiB GDDR6 | ≈ 6.4 GiB | 50 – 60 min per run |
| Mask R-CNN v1+v2 base / v2+v3 ft | B. Sharipov | NVIDIA RTX 4070 Laptop | 8 GiB GDDR6 | ≈ 17 GiB (via Windows shared-memory) | ≈ 1 h 50 min total |
| DeepForest v4 / v3 ft | A. Totin | NVIDIA RTX 4050 Laptop | 6 GiB GDDR6 | ≈ 5 GiB | ≈ 8 min per run |

Inference and cross-model evaluation of every checkpoint (Sections 3.3.6, 3.4, 3.5, 3.6, 3.7) was consolidated on the YOLOv8x-seg workstation — an Intel Core i7-13620H (10 cores / 16 threads, 4.9 GHz boost), 16 GiB DDR5-4800 system memory, Windows 11 Pro 23H2, CUDA 12.1, NVIDIA driver 551.86 — so that all reported M14 numbers are produced under identical inference conditions and the same NMS / confidence-threshold defaults.

The Python software stack was deliberately kept to two virtual environments. The first, `venv/`, contains a CPU-only PyTorch build and is used exclusively for data-preparation scripts that do not require a GPU (COCO-to-YOLO conversion, tiling, dataset merging, COCO pre-labelling). The second, `pipeline/venv/`, contains PyTorch 2.5.1 with CUDA 12.1 support, Ultralytics 8.4, the DeepForest 1.5 library, the SAM 2.1 inference package and pycocotools for COCO-style evaluation. This separation is a deliberate engineering choice — installing PyTorch with CUDA support on Windows is brittle and is preferably done once and frozen, while the data-preparation environment is rewritten frequently.

The principal hardware constraint of the project was the **8 GiB VRAM** ceiling of the YOLO training laptop. Peak GPU memory consumption during YOLO training reaches approximately **6.4 GiB** at an input resolution of 640 × 640 pixels and a batch size of 2 with mixed-precision training enabled; a batch size of 4 reproducibly triggered an out-of-memory error and determined the choice of `yolov8x-seg` over the larger `yolov8x6-seg` variant. The Mask R-CNN training, by contrast, peaks at approximately 17 GiB through Windows shared-memory extension on the RTX 4070 Laptop — over twice the physical VRAM of either of the other two cards — confirming that the two-stage Mask R-CNN architecture is substantially more memory-intensive than the one-stage YOLO design at comparable batch size.

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

### 3.2.4 Version-3 batch and the M14 cross-model validation set

A third batch of **24 additional satellite photographs** was added in May 2026 to cover Astana districts and time-of-day conditions that were under-represented in v1 and v2 — sub-urban yards, river-front green corridors and the densest historical-park canopy outside the central avenues. Annotation followed the same model-in-the-loop workflow as v2 (Section 3.2.3), now using the v2-finetune YOLO checkpoint as the pre-labeller; the per-image annotation budget dropped to approximately 4 minutes per image at this stage, against approximately 7 minutes for v2 and approximately 25 minutes for the from-scratch v1 work. The new images were split into 19 training and 5 validation source images, contributing roughly 1 700 polygon annotations after tiling.

The combined v1 + v2 + v3 dataset is the corpus used in the production training of every branch reported in this chapter — the YOLO v3-finetune (Section 3.3.6), the Mask R-CNN v2+v3 fine-tune (Section 3.4) and the DeepForest v3 fine-tune (Section 3.5.2). For the cross-model **evaluation** the same source-image-level held-out set is used by all three branches: the canonical configuration is the **M14 set** — 4 v1 + 5 v2 + 5 v3 source images, 17 tiles after 640 + 128 sliding-window tiling, 702 polygon annotations at the source-image level and 755 polygon instances at the tile level. The construction and rationale of this M14 set, including the explicit exclusion of `Снимок экрана 2026-04-01 194422.png` from v1 to avoid YOLO train/val leakage, are documented in detail in Section 3.7.1. The COCO JSON used by the Mask R-CNN and DeepForest evaluation scripts is reproducibly generated from the v1 + v2 + v3 source COCO files by the helper script `ml/build_14img_val_coco.py`.

### 3.2.5 Conversion and tiling pipeline

The complete data-preparation pipeline that turns a CVAT COCO export into a tile-level YOLO dataset is implemented in four Python tools located in the `ml/` directory of the repository: `ml/coco_to_yolo_seg.py` converts the COCO polygons to YOLOv8 polygon labels; `ml/tile_dataset.py` performs the sliding-window tiling; `ml/merge_coco.py` merges multiple COCO files; and `ml/split_coco.py` performs deterministic train/val splits at the source-image level. All four scripts share a common Cyrillic-aware UTF-8 output configuration and a common command-line interface, and together with the auxiliary `prelabel_coco.py` tool they constitute a reusable data-preparation library that can be applied to any future tree-annotation campaign.

## 3.3 YOLOv8-seg training results

Throughout this chapter, detection quality is reported using the standard COCO-style metrics. For a single prediction $p$ and a ground-truth annotation $g$, the **Intersection-over-Union** is defined as

$$
\mathrm{IoU}(p, g) \;=\; \frac{|p \cap g|}{|p \cup g|}.
$$

A prediction is counted as a True Positive when it has an IoU of at least $0.5$ against any unmatched ground-truth instance; otherwise it is a False Positive. Ground-truth instances with no matching prediction are False Negatives. **Precision**, **Recall** and the **F1-score** are then

$$
P \;=\; \frac{TP}{TP + FP}, \qquad R \;=\; \frac{TP}{TP + FN}, \qquad F_{1} \;=\; \frac{2 \cdot P \cdot R}{P + R}.
$$

The **Average Precision at IoU = 0.5** (AP@50) is the area under the Precision-Recall curve obtained by sweeping the detector's confidence threshold from 1 to 0. **mAP@50** is the mean of AP@50 across object classes (here only one — *Tree*). The stricter **mAP@50:95** averages the AP at ten IoU thresholds $\{0.50, 0.55, \ldots, 0.95\}$ — a metric that penalises imprecise localisation as well as missed detections. The same definitions apply to the segmentation-mask outputs, with the IoU computed over per-pixel mask sets rather than over rectangular boxes (denoted *Mask mAP@50* in the tables that follow).

### 3.3.1 Version-1 training run

The version-1 model was trained with the hyper-parameters listed in Table 2.2 of Chapter 2. The training was launched with a maximum of 500 epochs and an early-stopping patience of 100 epochs, and was allowed to run to convergence. The actual run stopped at **epoch 397** after approximately **1.008 wall-clock hours**, with the best checkpoint produced at **epoch 296**.

### 3.3.2 Training loss

The training loss curves recorded by Ultralytics for the version-1 run are reproduced in Figure 3.1 below. Three loss components are reported by the framework:

![*Ultralytics-generated training and validation curves for the YOLOv8x-seg v1 run. Top row: box, segmentation, classification and DFL losses on the training set. Bottom row: the same losses on the validation set together with Box / Mask precision, recall, mAP@50 and mAP@50-95 over the 397 trained epochs.*](figures/yolo_v1_results.png)

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

First, the **validation set is exceptionally small** — four tiles, 94 polygons. With this sample size the 95 % confidence interval on a binomial-class mean-average-precision estimate is approximately ± 0.10, so the reported number must be read as "0.48 ± 0.10" rather than as a point estimate. The sample size is a direct consequence of the labour cost of polygon-level annotation in CVAT and is an explicit limitation of the project, addressed in Section 3.9.

Second, the **annotation noise** is high. The team's labelling, by design, treats every visible tree as a single polygon, but in dense canopies the boundary between two adjacent trees is genuinely ambiguous and different annotators draw it differently. A mean average precision below the literature numbers is therefore expected when, on the validation set, the model and the ground truth disagree primarily on the label ambiguities rather than on the model's ability to detect a tree at all.

Third, **the qualitative quality of the predictions is substantially higher than the aggregate number suggests**, as demonstrated below.

### 3.3.4 Qualitative analysis

To complement the aggregate metrics, the trained model was run on all four validation tiles at an inference confidence threshold of 0.25, and the resulting predictions were overlaid on the source tiles. A representative sample is shown in Figure 3.2 below; the full set is available in `runs/predict/val_check/`. Three observations stand out.

![*Sample YOLOv8x-seg v1 prediction on a sparse-residential Astana validation tile: predicted crown polygons (blue masks) with per-detection confidence scores overlaid on the source imagery. On a tile with roughly twenty visible crowns the model produces only a partial inventory at confidence ≥ 0.25 — a recall typical of the v1 baseline and one of the principal motivations for the version-2 fine-tune reported in Section 3.3.5 below.*](figures/yolo_val_qualitative_001.jpg)

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

2. **v2-finetune is the strongest checkpoint of the v1 + v2 era**, with Box mAP@50 reaching **0.372** (+40 % relative to v1 and +17 % relative to v2-fromscratch) and Mask mAP@50 reaching **0.331** (+38 % over v1 and +15 % over v2-fromscratch). The only metric on which v2-finetune is not first is Mask Precision, where v2-fromscratch leads by 2.7 percentage points (0.424 vs 0.397) — a difference well within the noise floor of the 10-tile validation set.

3. The v2-finetune result narrows the gap to the published urban-DeepForest baselines [@SofiaDeepForest2024; @Ventura2024]: Box mAP@50 = 0.372 is approximately 55 % of the Sofia F1 = 0.68 and approximately 50 % of the Ventura fine-tuned F = 0.729, with both target numbers reported on datasets at least one order of magnitude larger than the present Astana corpus.

The v2-finetune checkpoint became the first system-production model of the project (`weights/yolo_satellite.pt` from 2026-05-13) and remained in production until the v3 dataset extension and the subsequent 16-experiment hyperparameter ablation (Sections 3.3.6 – 3.3.7) selected the smaller yolov8m-seg architecture as the final production checkpoint. The v1, v2-fromscratch and v2-finetune checkpoints are retained on disk as baselines for the cross-version ablation chain (Table 3.3f), and the v2-finetune checkpoint specifically is archived at `weights/archive/yolo/yolo_satellite_v2_finetune.pt`.

![*Like-for-like qualitative comparison on the held-out validation tile `img_val_007` (dense residential micro-district). Left: ground-truth polygon annotation in green. Centre: YOLOv8x-seg v1 prediction in blue. Right: YOLOv8x-seg v2-finetune prediction in orange. The v2-finetune model recovers a substantially larger fraction of the partially-shadowed crowns in the centre of the tile and produces tighter crown boundaries on the row-planted street trees along the lower edge. The 3-panel comparison is reproducible from the script `thesis/gen_qualitative_figures.py` using the archived v1 and v2-finetune checkpoints.*](figures/yolo_v1_vs_v2_finetune_comparison.png)

The full training and validation curves of the v2-finetune run, complementary to those of the v1 run shown above, are reproduced in the figure below.

![*Ultralytics-generated training and validation curves for the YOLOv8x-seg v2-finetune run over its 173 trained epochs (best checkpoint at epoch 99). The relative regularity of the validation loss compared to the v1 run reflects the larger version-2 validation set and the slower effective learning rate that comes from fine-tuning the already-converged v1 checkpoint on the new-image subset only.*](figures/yolo_v2_finetune_results.png)

Qualitative predictions of the v2-finetune checkpoint on two four-tile samples drawn from the version-2 validation set are shown in the figures below. Each panel is one held-out Astana tile with predicted crown polygons (blue masks) and per-detection confidence scores. Compared to the v1 sample of the previous sub-section, the v2-finetune model recovers almost every visible crown and is noticeably more confident on clearly-resolved trees in the foreground.

![*YOLOv8x-seg v2-finetune predictions on a four-tile sample from the version-2 validation set (tiles `img_val_001`, `003`, `007`, `009`), shown as a single horizontal strip to fit page width. The model recovers nearly all visible crowns on the sparse-residential and dense-canopy scenes alike, and assigns higher confidence (0.6 – 0.9) to clearly-resolved trees in the foreground. The residual over-detection failure mode discussed for the v1 model in Section 3.3.4 — small ornamental shrubs along the road shoulder tagged as trees at low confidence (≈ 0.3 – 0.4) — persists in v2-finetune at a reduced frequency. The strip is reproducible from `thesis/gen_qualitative_figures.py`.*](figures/yolo_v2_finetune_val_4tile_strip.png)

### 3.3.6 Initial version-3 fine-tune attempts

After the May 2026 batch of 24 additional satellite photographs (Section 3.2.4) was annotated and added to the training corpus, a third training round was launched with the explicit goal of closing the **out-of-distribution gap** that the v2-finetune model exhibited on the new images. A preliminary evaluation of the v2-finetune checkpoint on a v3-only validation set of 7 tiles and 497 polygons gave a Box mAP@50 of only **0.0811** — a factor of approximately 4.5 below the same checkpoint's 0.363 on the v2-only validation set. The drop is consistent with the natural domain heterogeneity of the May 2026 batch, which sampled different Astana districts and time-of-day conditions from those covered by the original v1 and v2 captures.

The first two version-3 fine-tune attempts both started from the v2-finetune `best.pt` checkpoint and continued training on the merged v1 + v2 + v3 dataset, varying only the hyper-parameter posture. Table 3.3b summarises the trade-offs.

**Table 3.3b — Initial v3 fine-tune attempts from the v2-finetune checkpoint (yolov8x-seg backbone).**

| Run | Posture | Best ep | v3-val Box mAP@50 | merged Box mAP@50 |
|---|---|---|---|---|
| First attempt (killed) | AdamW lr=0.001, aggressive aug (`mixup=0.2`, `copy_paste=0.3`, `degrees=30`), `label_smoothing=0.1` | 60 (plateau) | ≈ 0.16 | ≈ 0.25 |
| v3-finetune-**run1** | v2-proven aug, lr0=0.005 cosine, `single_cls=True`, `patience=75`, val = v3-only | 58 | **0.220** | 0.268 |
| v3-finetune-**run2** | run1 setup with `mixup=0`, `copy_paste=0`, milder geo aug, `mask_ratio=2` (after external paper advice) | 5 | 0.18 | 0.246 |

Three lessons follow from this first cycle. First, the killed attempt confirms that **aggressive augmentation combined with a low learning rate on an already-fine-tuned checkpoint locks the model into the starting basin** — at epoch 60 it had not moved meaningfully from the v2-finetune starting point. Second, run1 — which uses the same v2-proven augmentation that produced the v2-finetune itself — produced a substantial Box mAP@50 gain on the v3 distribution (0.220 vs the 0.0811 baseline, a 2.7× improvement) while keeping the v2-only performance close to its original level (0.334 vs 0.363, a small but real catastrophic-forgetting penalty). Third, run2's milder augmentation produced an immediate plateau at epoch 5 and final merged Box mAP@50 of 0.246 — *worse* than run1, consistent with the small-dataset observation that some augmentation diversity is required even when the inputs are already clean. Run1 was deployed as the initial version-3 production checkpoint.

Even the run1 result, however, left two open questions: was `yolov8x-seg` the right architecture for a dataset of this size, and was the continuation from v2-finetune the right starting point? Both questions were resolved by the systematic 16-experiment ablation reported below.

### 3.3.7 Systematic hyperparameter ablation: a 16-experiment factorial study

Having established v3-finetune-run1 as a working baseline, a structured ablation was conducted along five orthogonal axes — model size, starting weights, optimiser, augmentation level and inference resolution — across 16 experiments grouped into three rounds. The experiment runner is implemented in `ml/v3_experiment_runner.py` and writes results incrementally to `results/v3_experiments.json`. Every completed experiment is automatically evaluated on the same three validation splits used throughout the chapter (v2-only, v3-only, merged 14-image / 17-tile M14 set), so all numbers compare directly.

**Round 1 — architecture × start-weights × optimiser sweep (exp1 – exp5)**

The first round varies one of three principal axes per experiment while holding augmentation, batch size 4 and patience 30 constant. Results on the M14 merged val are reported in Table 3.3c.

**Table 3.3c — Round 1 ablation on M14, sorted by Box mAP@50.**

| Exp | Architecture | Start weights | Optimiser | merged Box mAP@50 | merged Mask mAP@50 |
|---|---|---|---|---|---|
| **exp1** | yolov8m-seg (27 M params) | COCO | auto → AdamW lr ≈ 0.002 | **0.308** | **0.305** |
| exp5 | yolov8l-seg (46 M params) | v2-finetune | SGD lr = 0.01 | 0.273 | 0.247 |
| exp3 | yolov8x-seg (71 M params) | v2-finetune | auto → AdamW | 0.254 | 0.236 |
| exp2 | yolov8l-seg (46 M params) | COCO | auto → AdamW | 0.230 | 0.234 |
| exp4 | yolov8x-seg (71 M params) | v2-finetune | SGD lr = 0.01 | 0.219 | 0.213 |

The headline finding is that **yolov8m-seg with only 27 million parameters and a fresh COCO start beats every larger or warm-started configuration**, including the architecture used for the v1, v2 and v3-finetune-run1 production weights (yolov8x-seg, 71 M parameters). The absolute gap is +15 % relative over the best Large variant (exp5) and +21 % relative over the best XLarge variant (exp3). This result is consistent with the satellite-tree literature finding that smaller YOLO backbones generalise better on datasets below approximately 200 source images with moderately noisy polygon annotations [@AbbasYOLO2025; @Sun2025], and is to the best of the authors' knowledge the first explicit ablation of this effect on Central-Asian satellite imagery. The single worst configuration in Round 1 was exp4 (XLarge backbone, SGD with the original YOLO paper learning rate 0.01) — a hot LR applied to an already-fine-tuned checkpoint destabilised convergence and confirmed the choice of `optimizer="auto"` (Ultralytics' small-dataset heuristic, which picks AdamW with lr ≈ 0.002) as the safe default.

**Round 2 — orthogonal probes from the exp1 winner (exp6 – exp10)**

The second round varies one factor at a time from the exp1 winner to test specific hypotheses raised by external sources and the satellite-tree literature. Results are reported in Table 3.3d.

**Table 3.3d — Round 2 probes from the exp1 setup; every change degraded the merged Box mAP@50.**

| Exp | Probe | Hypothesis | merged Box mAP@50 | Δ vs exp1 |
|---|---|---|---|---|
| **exp1** | baseline | — | **0.308** | — |
| exp10 | continuation from exp1, lr = 5×10⁻⁴ | gentle polish | 0.283 | −0.025 |
| exp8 | exp1 + `dropout = 0.15` | regularise noisy labels | 0.278 | −0.030 |
| exp7 | yolov8s-seg (12 M params) ← COCO | size-down sweep | 0.272 | −0.036 |
| exp9 | exp1 + heavier aug (`mixup = 0.3`, `copy_paste = 0.3`) | more diversity | 0.267 | −0.041 |
| exp6 | exp1 + `imgsz = 896` | higher resolution for small crowns | 0.261 | −0.047 |

Every Round-2 perturbation hurt the merged Box mAP@50, a strong empirical signal that the exp1 configuration sits at a local optimum on this dataset scale. The most negative result, `imgsz = 896`, was the one that external advice (search-engine summaries and a contemporaneous large-language-model consultation) had predicted to help most. The size-down sweep (exp7 with yolov8s-seg) shows that the model-size landscape is U-shaped rather than monotonic on our regime: 27 M parameters is the sweet spot, not 12 M and not 71 M.

**Round 3 — paper-informed experiments (exp11 – exp14)**

The third round implements four specific recipes mined from the project's literature corpus: a three-stage continual-learning chain (exp11), an aggressive low-LR finish from the exp1 best.pt (exp12), a freeze-then-unfreeze schedule (exp13) and the augmentation ranges reported in a recent YOLO satellite-tree benchmark [@AbbasYOLO2025] (exp14). Results are reported in Table 3.3e.

**Table 3.3e — Round 3 paper-informed experiments. None beat exp1.**

| Exp | Description | Paper source | merged Box mAP@50 |
|---|---|---|---|
| exp12 | exp1 → fine-tune v3-only, lr = 1×10⁻⁴, patience 12 | [@SofiaDeepForest2024] | 0.286 |
| exp13 | exp1 setup + `freeze=10` for 40 epochs, then unfreeze + lr = 1×10⁻⁴ | [@Chen2023] | 0.264 |
| exp14 | exp1 + augmentation ranges (`degrees=21`, `shear=15`, `hsv_v=0.44`) from satellite benchmark | [@AbbasYOLO2025] | 0.256 |
| exp11 | 3-stage continual chain: COCO → v1 only → v1 + v2 → v1 + v2 + v3 | [@Chen2023] Table 5 (similar regime) | 0.210 |

The aggressive low-LR continuation finish (exp12, 0.286) and the freeze-then-unfreeze schedule (exp13, 0.264) are the closest second-place contenders to exp1 (0.308), but none of the paper-informed recipes overtakes it. The three-stage continual chain (exp11), which trains separately on v1, then on v1 + v2, then on v1 + v2 + v3, was specifically expected to help based on the staged-pretrain-then-finetune result of [@Chen2023] — empirically it underperforms substantially, suggesting that with only 63 training source images the chain partitions the data too thinly to produce a useful intermediate prior. The root cause is investigated in Section 3.3.10 below.

### 3.3.8 Round 4 — clean-defaults model-size sweep

The 16-experiment ablation summarised in Tables 3.3c – 3.3e holds the augmentation pipeline constant at the **v2-proven** values inherited from the v1 / v2 YOLO checkpoints. Round 4 asks the orthogonal question of whether the v2-proven augmentation itself is the right choice for the satellite-tree distribution, by stripping the augmentation pipeline back to the **Ultralytics framework defaults** and running a fresh model-size sweep at five scales (yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg). The implementation lives in `ml/v4_clean_modelsweep.py` and writes results to `results/v4_clean_modelsweep.json`. All five runs share the same merged v1 + v2 + v3 dataset, the same `imgsz = 640`, `batch = -1` (Ultralytics AutoBatch — finds 60 % VRAM utilisation), `epochs = 150`, `patience = 50`, `time = 1.5 h` cap, and the framework default augmentation: aggressive HSV colour jitter (`hsv_s = 0.7`, `hsv_v = 0.4`), aggressive random erasing (`erasing = 0.4`), and **no geometric augmentation at all** (`degrees = 0`, `mixup = 0`, `copy_paste = 0`, `flipud = 0`). The results on M14 are reported in Table 3.3f.

**Table 3.3f — Round 4 clean-defaults model-size sweep on M14, sorted by Box mAP@50.**

| Backbone | Params | Wall time | v2-only Box | v3-only Box | merged Box mAP@50 | merged Mask mAP@50 |
|---|---|---|---|---|---|---|
| **v4_x_clean** | **71 M** | 91 min | **0.319** | **0.313** | **0.315** | **0.289** |
| v4_m_clean | 27 M | 18 min | 0.344 | 0.267 | 0.291 | 0.280 |
| v4_s_clean | 12 M | 11 min | 0.329 | 0.254 | 0.281 | 0.270 |
| v4_n_clean | 3 M | 9 min | 0.254 | 0.262 | 0.261 | 0.251 |
| v4_l_clean | 46 M | 52 min | 0.282 | 0.257 | 0.260 | 0.253 |

Three findings emerge from this table. First, the **v4_x_clean configuration — yolov8x-seg with Ultralytics defaults — overtakes the exp1 winner of Section 3.3.7 by +2 % relative on merged Box mAP@50** (0.315 vs 0.308) and dominates on the v3-only (out-of-distribution) subset by a larger margin (+9 %, 0.313 vs 0.287). This is the strongest single configuration measured anywhere in the project, and the M14 number 0.315 is the headline empirical result of this diploma. Second, **the model-size landscape is genuinely U-shaped** rather than monotonic: the medium-size m-seg (Round 1 winner with v2-proven aug at 0.308) and the extra-large x-seg (Round 4 winner with defaults at 0.315) both occupy local optima, while l-seg (0.260 with defaults) lies in a valley between them — an empirical pattern that the present project is, to the best of the authors' knowledge, the first to report on Central-Asian satellite-tree data. Third, and most surprising for a practitioner conditioned by the satellite-tree literature on aggressive geometric augmentation, **the Ultralytics default augmentation pipeline outperforms the manually-tuned v2-proven pipeline that produced every preceding YOLO checkpoint on this same dataset**. The plausible explanation is that satellite trees viewed from above are rotationally consistent (any rotation introduces unnatural data variance not present in the test distribution), so the geometric component of the v2-proven pipeline (degrees ± 20°, full mixup, full copy-paste) actively hurts generalisation, while the strong colour-jitter and random-erasing components of the Ultralytics defaults correctly target the natural sources of variation in satellite imagery (illumination, season, sensor settings, partial occlusion).

The v4_x_clean checkpoint is therefore promoted to **the final production checkpoint** of the project, replacing the previous exp1 deployment. The current `weights/yolo_satellite.pt` is a copy of `weights/v4_clean/v4_x_clean_v3val0.313_mergedval0.315.pt` (MD5 `58fb1c0018db3fd3dd49fae436bedcca`). All earlier checkpoints — v1, v2-fromscratch, v2-finetune, v3-finetune-run1, exp1 m-seg — are archived on disk under `weights/v3_runs/` and `weights/archive/yolo/` and remain selectable through the frontend's hierarchical model picker for A/B comparison.

### 3.3.9 Round 5 — multi-replicate variance check

The 16-experiment ablation of Section 3.3.7 and the model-size sweep of Section 3.3.8 each report a single training run per configuration. To estimate the variance of these point estimates — and in particular to test whether the exp1 winner number (0.308) was a stable median or a lucky upper tail — four independent replicate runs of the **exact exp1 configuration** were performed in Round 5. All four use yolov8m-seg from fresh COCO weights, AdamW with `optimizer=auto` (Ultralytics picks AdamW with lr ≈ 0.002), v2-proven augmentation, batch size 4, input resolution 640, single-class head, patience 30, time cap 1.5 h. Only the random seed differs across runs (default seed = 0 for all, with non-determinism arising from CUDA cudnn kernel selection and AMP mixed-precision arithmetic order). The results on the merged M14 validation set are reported in Table 3.3g.

**Table 3.3g — Round 5 multi-replicate variance of the exp1 configuration on merged M14 Box mAP@50.**

| Run | Wall time | merged Box mAP@50 |
|---|---|---|
| exp1 (original) | 31 min | 0.308 |
| exp21 (replicate 1, time = 0.75 h budget) | 13 min | 0.268 |
| exp22 (replicate 2, time = 1.5 h budget) | 19 min | 0.269 |
| exp23 (replicate 3, time = 1.5 h budget) | 10 min | 0.239 |
| **Sample mean** | — | **0.271** |
| **Sample standard deviation** | — | **0.028** |

The four-replicate mean is 0.271 with a standard deviation of 0.028, placing the **realistic single-shot variance band** of an exp1-style run at approximately ± 0.03 Box mAP@50. The original exp1 number (0.308) is therefore a +1.3 standard-deviation outlier above the mean, and the worst replicate (exp23 at 0.239) is a −1.1 standard-deviation outlier below. This is a methodologically important observation for two reasons. First, on this dataset scale **single-seed mAP comparisons between configurations that differ by less than approximately 0.03 are inside the noise floor** and cannot be trusted to discriminate one configuration from another. The Round 1 to Round 3 ablation rankings of Tables 3.3c – 3.3e remain individually informative — most differences are larger than 0.03 — but small differences in the lower half of those tables should be read with this variance band in mind. Second, the v4_x_clean number of 0.315 from Section 3.3.8 is itself a single-run measurement and inherits a similar variance band, which is one motivation for the cross-YOLO voting ensemble of Section 3.7.5 below: a vote across multiple checkpoints averages out the per-checkpoint training-time variance and produces a more stable inventory at inference time.

### 3.3.10 Round 6 — random-split chain-learning control

The chain-learning experiment exp11 of Section 3.3.7 (three-stage continual training v1 → v1 + v2 → v1 + v2 + v3) underperformed substantially — Box mAP@50 = 0.210 versus single-shot baseline 0.308. Two competing hypotheses can explain this 32 % relative regression: either (H1) the multi-stage chain mechanism itself thins the gradient signal per stage and hurts on small datasets independently of the data partitioning, or (H2) the version-batch boundaries of the original v1, v2, v3 splits are the real culprit — each batch was captured on a different date by a different annotator with different district coverage, so the chain progressively pulls the model toward each batch's specific distribution and degrades performance on the earlier batches.

Round 6 tests the two hypotheses directly with two random-split chain controls. Experiment **exp17** runs a 3-stage cumulative chain (33 % → 66 % → 100 % of the merged 63-image training set, with cumulative inclusion: phase 2 contains phase 1's images plus 33 % more, phase 3 is the full corpus) on **random splits** generated with a fixed seed; the per-phase hyper-parameters match the original exp11 schedule (`patience = 20 / 15 / 12`, `lr0 = auto / 0.001 / 0.0001`). Experiment **exp18** is a 2-stage random chain (50 % → 100 %) mimicking the structure of the historical v1 → v1 + v2 v2-finetune transition, with hot LR (`lr0 = 0.001` for stage 2) rather than the gentle continuation LR. The implementation is in `ml/v3_random_chain.py` and writes results to `results/v3_experiments.json`. The results are reported in Table 3.3h.

**Table 3.3h — Round 6 random-split chain controls on merged M14, sorted by Box mAP@50.**

| Setup | Stages | Splits | Stage-2/3 LR | Box mAP@50 | Δ vs single-shot exp1 (0.308) |
|---|---|---|---|---|---|
| Single-shot exp1 (reference) | 1 | full | — | **0.308** | — |
| exp17 random 3-stage cumulative | 3 | random 33 % / 66 % / 100 % | auto / 0.001 / 0.0001 | 0.287 | −0.021 |
| exp18 random 2-stage hot LR | 2 | random 50 % / 100 % | auto / 0.001 | 0.270 | −0.038 |
| exp11 version-split 3-stage (Section 3.3.7) | 3 | v1 / v1 + v2 / v1 + v2 + v3 | auto / 0.001 / 0.0001 | 0.210 | **−0.098** |

The empirical result decisively favours hypothesis H2. Random-split 3-stage chain (exp17) loses only 0.021 Box mAP@50 against the single-shot baseline — well inside the variance band measured in Section 3.3.9 — while the version-split 3-stage chain (exp11) loses 0.098, a 4.7× larger gap. The 2-stage hot-LR variant (exp18) sits in the middle at −0.038. Approximately **77 % of the original exp11 chain-learning penalty is therefore attributable to distribution drift between the v1, v2 and v3 annotation batches**, not to the chain mechanism per se. The remaining 23 % is a small genuine staging penalty caused by the per-phase gradient-signal thinning at the early stages, which the random control reproduces faithfully.

A useful methodological implication follows: chain learning is appropriate when there is a clear gradient of label quality along the chain (paper #13's recipe: weak large pre-training, clean small fine-tune), but for our regime of three uniformly-noisy small annotation batches the optimal strategy is **single-shot training on the union of all available data**, as already adopted by the v4_x_clean production checkpoint of Section 3.3.8.

### 3.3.11 Final production choice and complete YOLO ablation chain on M14

After the six rounds of ablation reported above (Round 1 – Round 3 in Section 3.3.7, Round 4 – Round 6 in Sections 3.3.8 – 3.3.10), the **v4_x_clean configuration** — yolov8x-seg from public COCO weights, Ultralytics default augmentation pipeline, AutoBatch, `imgsz = 640`, `patience = 50`, `single_cls = True`, no manual hyper-parameter tuning — was promoted to production. The current `weights/yolo_satellite.pt` is a copy of `weights/v4_clean/v4_x_clean_v3val0.313_mergedval0.315.pt` (MD5 `58fb1c0018db3fd3dd49fae436bedcca`). All twenty-three intermediate experiments (exp1 – exp23) are archived under `weights/v3_runs/` and `weights/v4_clean/` and remain selectable from the frontend's hierarchical model picker for runtime A/B comparison. The previous v2-finetune production checkpoint is archived at `weights/archive/yolo/yolo_satellite_v2_finetune.pt`. The complete YOLO ablation chain — from the v1 baseline through the v3 era to the v4 final production — is reported in Table 3.3i.

**Table 3.3i — Full YOLO ablation chain on M14 (14 source images / 17 tiles / 755 polygons).**

| Checkpoint | Backbone | Params | Start weights | Augmentation | Box mAP@50 | Mask mAP@50 |
|---|---|---|---|---|---|---|
| v1 (16 train images) | yolov8x-seg | 71 M | COCO | v1 hand-tuned | 0.131 | 0.134 |
| v2-fromscratch (77 train images) | yolov8x-seg | 71 M | COCO | v2-proven | 0.156 | 0.147 |
| v2-finetune (Was production) | yolov8x-seg | 71 M | v1 best.pt | v2-proven | 0.187 | 0.185 |
| v3-finetune-run1 (intermediate) | yolov8x-seg | 71 M | v2-finetune best.pt | v2-proven | 0.268 | 0.244 |
| exp1 v3 m-seg (lucky single run) | yolov8m-seg | 27 M | COCO (fresh) | v2-proven | 0.308 | 0.305 |
| exp1 v3 m-seg (4-replicate mean) | yolov8m-seg | 27 M | COCO (fresh) | v2-proven | 0.271 ± 0.028 | 0.282 ± 0.025 |
| **v4_x_clean (FINAL production)** | **yolov8x-seg** | **71 M** | **COCO (fresh)** | **Ultralytics defaults** | **0.315** | **0.289** |

The complete trajectory yields a **+140 % relative improvement** on Box mAP@50 from the v1 baseline (0.131 → 0.315) and a **+116 % relative improvement** on Mask mAP@50 (0.134 → 0.289). The final v4_x_clean production model improves over the previous v2-finetune production by **+69 % on Box** mAP@50 and **+56 % on Mask** mAP@50, with the two largest contributors being the v3 dataset expansion (24 additional source images from previously-uncovered Astana districts) and the architectural / augmentation switch to yolov8x-seg with Ultralytics defaults from Round 4.

### 3.3.12 Out-of-distribution evaluation

The 23-experiment ablation above optimises the *aggregate* M14 metric, which mixes in-distribution (v1 / v2 imagery) and out-of-distribution (v3 imagery) tiles. Because the principal motivation of the entire v3 effort was the OOD failure of v2-finetune on v3 imagery (Box mAP@50 = 0.0811 on the v3-only val), it is informative to break the headline numbers down by validation subset. Table 3.3j reports the v2-only and v3-only Box mAP@50 of every YOLO checkpoint in the project.

**Table 3.3j — Per-distribution evaluation. The v2-finetune → v4_x_clean transition closes essentially all of the OOD gap.**

| Checkpoint | v2-only Box mAP@50 | v3-only Box mAP@50 | OOD ratio (v3 / v2) |
|---|---|---|---|
| v2-finetune | 0.363 | **0.081** | 0.22 |
| v3-finetune-run1 (yolov8x v2-proven aug) | 0.334 | 0.220 | 0.66 |
| exp1 v3 m-seg (lucky single run) | 0.367 | 0.287 | 0.78 |
| **v4_x_clean (FINAL production)** | **0.319** | **0.313** | **0.98** |

Two empirical observations follow. First, **the v4_x_clean production model recovers essentially all of the v2-distribution performance** of the v2-finetune checkpoint (0.319 vs 0.363, a small but real 12 % relative loss that the exp1 m-seg variant did not exhibit — the architectural / augmentation switch trades a small fraction of in-distribution accuracy for a much larger gain on the new distribution). Second, the **OOD ratio** — the fraction of in-distribution Box mAP@50 recovered on the out-of-distribution subset — rises from 22 % at the v2-finetune baseline through 78 % at the exp1 intermediate to **98 % at the v4_x_clean production**, i.e. by Round 4 the model performs almost identically on previously-seen and previously-unseen Astana districts. The residual asymmetry of 2 % is within the noise floor measured in Section 3.3.9 and is no longer a quantitative target for further dataset expansion; instead the future-work direction shifts to broadening the test-time distribution itself (Section 3.9 — built-environment scenes, river-front imagery, multi-season acquisitions).

## 3.4 Mask R-CNN training and results

The Mask R-CNN branch was trained by team member Berik Sharipov on the same merged Astana polygon dataset as the YOLO branch. Two checkpoints exist in the project: a **v1+v2 base** model trained from the public torchvision COCO V1 weights, and a **v2+v3 fine-tune** that warm-starts from the v1+v2 base on the expanded dataset.

### 3.4.1 Experimental setup

The Mask R-CNN training and inference use the same merged Astana CVAT polygon dataset described in Section 3.2 and the same M14 validation set defined in Section 3.7 — this is what makes the cross-model comparison reported below directly comparable with the YOLO and DeepForest numbers. Hardware is a single workstation with an NVIDIA RTX 4070 Laptop GPU; peak VRAM during training reaches approximately 17 GB through Windows shared-memory extension, well in excess of the 8 GB physical limit (the YOLO branch by comparison peaks at 6.4 GB on an RTX 4060 Laptop). The framework, base model, training schedule and hyper-parameters of both Mask R-CNN runs are documented in Section 2.5 and are not repeated here.

### 3.4.2 Training results

The v2+v3 fine-tune (release tag `maskrcnn-v2v3`, file `maskrcnn_astana_v2v3.pt`, 176 MB) is the production checkpoint and is the one used for the cross-model comparison of Section 3.7. The training was launched with a maximum of 30 epochs and an early-stopping patience of 5 epochs on the validation `mask_map_50` metric. The run early-stopped at epoch 16 with the best checkpoint produced at **epoch 11**. The Albumentations augmentation pipeline — horizontal flip ($p$ = 0.5), vertical flip ($p$ = 0.3), random 90-degree rotation ($p$ = 0.5), random brightness / contrast adjustment ($p$ = 0.3) and HSV jitter ($p$ = 0.2) — was applied only to the training split. The warm-start option (`--resume-from weights/maskrcnn_astana.pt`) automatically lowered the initial learning rate from the from-scratch default of $5 \times 10^{-3}$ to $1 \times 10^{-3}$, which is appropriate for continuing training from an already-converged checkpoint. Mixed-precision via `torch.amp.autocast("cuda")` with a `GradScaler` was used throughout.

The headline validation metrics on the M14 set (14 source images, 702 polygons) are summarised in Table 3.4a. For comparability with the literature, Berik also reports the same metrics on the larger 15-image superset that includes the excluded v1 image (Section 3.7.1) and the **before-vs-after** numbers relative to the v1+v2 base.

**Table 3.4a — Mask R-CNN v2+v3 fine-tune validation metrics on the 14-image M14 set.**

| Output head | mAP@50 | mAP@50:95 | mAP@75 | Precision @ conf 0.5 | Recall @ conf 0.5 |
|---|---|---|---|---|---|
| Bounding box | **0.166** | 0.062 | 0.040 | 0.437 | 0.224 |
| Segmentation mask | **0.158** | 0.055 | 0.030 | — | — |

**Table 3.4b — Improvement over the v1+v2 base on the 15-image superset (Berik's own measurement).**

| Metric | v1+v2 base | v2+v3 fine-tune | Δ |
|---|---|---|---|
| Box mAP@50 | 0.121 | **0.187** | +54 % |
| Box mAP@50:95 | 0.047 | 0.068 | +45 % |
| Box mAP@75 | 0.033 | 0.043 | +30 % |
| Mask mAP@50 | 0.116 | **0.183** | +58 % |
| Mask mAP@50:95 | 0.040 | 0.062 | +56 % |
| Box P @ conf 0.5 | 0.292 | **0.460** | +57 % |
| Box R @ conf 0.5 | 0.198 | 0.242 | +22 % |
| False-positive count | 349 | **207** | −41 % |

The principal observations are the following. First, the fine-tune produces a **monotonic improvement on every metric** — a 54–58 % relative gain in mAP@50 / mAP@50:95 for both heads, with a particularly strong precision gain (+57 %) that drives the 41 % reduction in absolute false-positive count between the two checkpoints. Second, the early-stop at epoch 11 — well below the 30-epoch budget — confirms that with the warm-start strategy and the stronger augmentation pipeline the model converges fast, and that further training would not have yielded additional value. Third, the on-M14 numbers in Table 3.4a are systematically 11–14 % lower than the 15-image numbers in Table 3.4b because of the removed v1 duplicate image (Section 3.7.1), as expected.

### 3.4.3 Qualitative analysis

A representative sample of the Mask R-CNN v2+v3 predictions on M14 tiles is provided in `results/maskrcnn_14img_eval/predictions/`. Visually, the trained model detects most of the larger, well-isolated crowns with sharp polygon boundaries — a direct benefit of the two-stage RoI-aligned mask head — but it misses a substantial fraction of small or partially-occluded crowns and is sensitive to the same dense-scene under-detection failure mode reported in Section 3.5 for the DeepForest branch. The dominant precision-vs-recall trade-off therefore differs from the YOLOv8x-seg branch: YOLO tends to over-segment dense canopies (producing many partial-crown detections at low confidence), whereas Mask R-CNN tends to under-detect small crowns altogether at the same confidence threshold.

### 3.4.4 Comparison with YOLOv8-seg

Placing the Mask R-CNN v2+v3 numbers side-by-side with the YOLOv8x-seg v3-finetune (production) and YOLOv8x-seg v2-finetune (previous production) checkpoints on the same M14 validation set yields a clear architectural ranking on the Astana satellite dataset:

| Metric | YOLOv8x-seg v3-ft | YOLOv8x-seg v2-ft | Mask R-CNN v2+v3 |
|---|---|---|---|
| Box mAP@50 | **0.287** | 0.187 | 0.166 |
| Mask mAP@50 | **0.263** | 0.185 | 0.158 |
| Box mAP@50:95 | **0.095** | 0.067 | 0.062 |
| Mask mAP@50:95 | **0.084** | 0.062 | 0.055 |

The YOLOv8x-seg v3-finetune dominates by a comfortable margin on every metric. Even the previous-generation YOLO v2-finetune is within statistical noise of the Mask R-CNN v2+v3 on Box mAP@50 (0.187 vs 0.166) and slightly ahead on Mask mAP@50 (0.185 vs 0.158). This result is at odds with the literature precedent of Lv et al. [@Lv2023] (Det AP 92.40 %, Seg AP 97.70 % for the MCAN Mask R-CNN variant on UAV imagery), but is consistent with the very different signal regime of the present setting: low-resolution satellite imagery (0.3–1 m GSD vs 1–5 cm UAV), a small Astana dataset (≈ 100 source images and ≈ 8 700 polygons vs the multi-thousand-image UAV datasets), and the modern one-stage YOLO design that closes most of the historical gap to two-stage methods on small datasets through its CIoU + DFL loss formulation and aggressive augmentation pipeline.

\newpage

## 3.5 DeepForest results

### 3.5.1 Off-the-shelf NEON baseline on Astana

The first empirical question for the DeepForest branch is how well the public `weecology/deepforest-tree` checkpoint — pre-trained on NEON aerial lidar of forested sites in the United States — transfers to Astana satellite imagery without any fine-tuning. To the best of the authors' knowledge no published number exists for this configuration on any Central-Asian city, so the experiment was performed from scratch in this project.

The pre-trained checkpoint was evaluated on the 15-image merged validation set originally used by team member Anuar Totin (a superset of the 14-image M14 set defined in Section 3.7) using the library's `predict_tile()` interface at the default patch size of 400 pixels and an overlap of 5 %, with a confidence threshold of 0.30 — the same protocol that is applied to the fine-tuned configurations in Section 3.5.2 below. The obtained Box mAP@50 is **0.012**, with a corresponding mAP@50:95 of effectively zero. The number is one to two orders of magnitude below any of the fine-tuned configurations reported later in the chapter, and three orders of magnitude below the public Ventura et al. off-the-shelf result of F = 0.42 on NAIP USA imagery [@Ventura2024]. The qualitative inspection of the predictions confirms the aggregate metric: the NEON-pretrained model produces only a handful of detections per Astana tile, all of which are concentrated on the few canopy patches that visually resemble the dense North-American broadleaf shapes of the NEON training set.

This first-time measurement establishes the empirical magnitude of the geographic-generalisation gap discussed in Section 1.5 for the specific Astana satellite domain: the floristic composition (dominated by *Populus* with tall narrow crowns), the urban morphology (Soviet-era micro-districts with row-planted street trees) and the acquisition geometry of the satellite imagery jointly produce a gap that is much larger than the urban-domain gap previously documented on European or North-American cities. A useful baseline cannot be obtained without fine-tuning on local data.

### 3.5.2 Fine-tuned DeepForest v3

The fine-tuning of the DeepForest branch was performed by team member Anuar Totin using the PyTorch-Lightning-based training interface that ships with the DeepForest 1.5 library. The training data is the **same merged v1 + v2 + v3 Astana CVAT polygon dataset** that is used by the YOLO and Mask R-CNN branches, converted to DeepForest's bounding-box CSV format via the helper script `ml/coco_to_deepforest_csv.py`. The split is 63 training images and 15 validation images for a total of 4 733 training bounding boxes and 726 validation bounding boxes.

A previous version of the fine-tune (checkpoint `astana_trees_v4_10epochs.pl`) had been trained on a separate Astana annotation set maintained on Roboflow by Anuar; that set has since been deprecated as access to the original Roboflow workspace was lost. The v3 fine-tune used in the final system is a continuation of the v4 checkpoint on the merged CVAT dataset, so the effective training trajectory of the production weights is **NEON → v4 (Roboflow) → v3 (CVAT)**.

The training hyper-parameters of the v3 run, archived in the GitHub release `v2.0` of the project repository, are summarised in Table 3.6a.

**Table 3.6a — DeepForest v3 fine-tune hyper-parameters (final production checkpoint).**

| Parameter | Value |
|---|---|
| Architecture | RetinaNet (ResNet-50 + FPN), 32.1 M parameters |
| Starting weights | `astana_trees_v4_10epochs.pl` (NOT NEON) |
| Train images | 63 (16 v1 + 28 v2 + 19 v3) |
| Train bboxes | 4 733 |
| Val images | 15 (5 v1 + 5 v2 + 5 v3) |
| Val bboxes | 726 |
| Optimiser | SGD with momentum (DeepForest default) |
| Learning rate | $1 \times 10^{-4}$, no scheduler |
| Batch size | 4 |
| Epochs | 30 (single run) |
| Augmentations | HorizontalFlip ($p$ = 0.5) |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| Wall time | ≈ 8 minutes |

The fine-tuned model raises the Box mAP@50 from the NEON baseline of 0.012 to **0.146 on the 14-image M14 validation set** (Section 3.7) — a factor of approximately twelve. The corresponding precision and recall at the standard score threshold of 0.30 are 0.39 and 0.10 respectively, in line with the recall-limited regime that is consistently reported for DeepForest in dense-canopy urban scenes [@SofiaDeepForest2024]. The per-image breakdown of detections versus ground-truth crowns is reported in Table 3.6b.

**Table 3.6b — DeepForest v3 fine-tune per-image detection counts on the 15-image superset of M14.**

| Image (truncated stem) | GT crowns | Predicted | Mean score |
|---|---|---|---|
| 195425 (v1) | 67 | 47 | 0.377 |
| 194422 (v1, excluded from M14) | 24 | 21 | 0.501 |
| 195234 (v1) | 11 | 17 | 0.413 |
| 195221 (v1) | 1 | 31 | 0.284 |
| 195214 (v1) | 15 | 28 | 0.285 |
| 103205 (v2) | 51 | 51 | 0.225 |
| 103142 (v2) | 37 | 35 | 0.387 |
| 103019 (v2) | 37 | 28 | 0.390 |
| 103214 (v2) | 23 | 51 | 0.292 |
| 102339 (v2) | 9 | 13 | 0.255 |
| 124541 (v3) | 138 | 104 | 0.264 |
| 124628 (v3) | 88 | 47 | 0.356 |
| 124634 (v3) | 104 | 55 | 0.326 |
| 124711 (v3) | 74 | 57 | 0.294 |
| 124556 (v3) | 47 | 35 | 0.350 |

Two qualitative patterns dominate. The model **under-detects on dense scenes** — on the densest tile (138 ground-truth crowns), the model produces only 104 predictions — and **over-detects on sparse scenes** — on the sparsest tile (1 ground-truth crown), the model produces 31 candidate detections at low confidence. The first pattern is characteristic of small-dataset DeepForest fine-tunes [@SofiaDeepForest2024] and points to additional dense-scene training images as the most productive next dataset expansion direction. The second pattern is a confidence-calibration artefact and would be partially mitigated by raising the inference confidence threshold from 0.30 to 0.50 at the expense of further reducing recall.

### 3.5.3 Cross-comparison with YOLO

The two branches exhibit **complementary failure modes** that are visible both on the aggregate metric and on per-image inspection. The YOLO branch tends to **over-segment dense canopies**, splitting a single cluster of tightly-overlapping crowns into several smaller polygons — most commonly when the cluster casts a strong internal shadow line that the model interprets as a crown boundary. The DeepForest branch tends to do the opposite: it **under-segments dense canopies**, merging adjacent crowns into a single bounding box, especially in tiles like the densest v3 sample (138 ground-truth crowns, 104 predicted) reported in Table 3.6b above. YOLO is more sensitive to **shadow false positives**, predicting low-confidence detections on the dark grass patches beside buildings; DeepForest is far less sensitive to that mode. The qualitative comparison is summarised in Table 3.4.

**Table 3.4 — Qualitative comparison of YOLO and DeepForest branches on Astana.**

| Property | YOLOv8x-seg v3-finetune | DeepForest v3 fine-tuned |
|---|---|---|
| Output | Polygon mask + box + confidence | Bounding box + confidence (polygons via optional SAM 2) |
| Box mAP@50 (M14) | **0.287** | 0.146 |
| Mask mAP@50 (M14) | **0.263** | 0.134 (via SAM 2) |
| Dominant FP | Shadows, dense bushes | Sparse-scene over-detection of small shrubs |
| Dominant FN | Heavily shadowed crowns | Dense-scene merged crowns |
| Failure mode | Over-segmentation | Under-segmentation in dense scenes |
| Inference time / 1 600 × 1 100 image | ≈ 2.0 s | ≈ 5.5 s (DeepForest) + 1.5 s (SAM 2) |

These complementary patterns are exactly what the Weighted-Box-Fusion ensemble of Section 2.8 is designed to combine — see Section 3.7 for the integrated cross-model comparison and Section 3.7.3 for the WBF ensemble result.

## 3.6 SAM 2 mask-refinement integration

The SAM 2 [@SAM2_2024] branch (`DeepForestSAM2Adapter`) was integrated into the system as a post-processing stage that converts DeepForest bounding boxes into polygon masks (Section 2.7). The model used is `sam2.1-hiera-base-plus`, loaded automatically from HuggingFace on first inference. All bounding boxes from a single image are passed to SAM 2 in a single batched call, which keeps inference overhead low compared to per-box sequential calls. The integration was evaluated end-to-end on the 14-image M14 validation set (Section 3.7) using the dedicated evaluation script `ml/eval_df_sam2_14img.py`, which feeds the DeepForest v3 detections directly to SAM 2 with `multimask_output=False` and computes the mask mAP via pycocotools COCOeval.

No fine-tuning of the SAM 2 backbone was attempted. The entire point of including SAM 2 in the architecture is to demonstrate that a second-generation foundation model can deliver usable urban-tree polygon masks **without** any domain-specific training — the model's only input is the DeepForest bounding box and its only output is the corresponding refined binary mask. The current implementation uses the `hiera-base-plus` variant of SAM 2 for tractable interactive inference on a laptop GPU; an ablation against the larger `hiera-large` variant is reserved for future work.

**Quantitative result on M14.** The SAM 2 mask-refinement stage raises the Mask mAP@50 of the DeepForest branch from zero (no native mask output) to **0.134** on the 14-image M14 validation set. The corresponding Mask mAP@50:95 is **0.042**. Box-level metrics are essentially unchanged from the standalone DeepForest configuration of Section 3.5.2 (Box mAP@50 = 0.146 vs 0.146 ± noise) because SAM 2 does not modify the detection scores or bounding-box coordinates — it operates as a pure post-processor on the boxes produced by the underlying detector.

The previously reported Mask mAP@50 of 0.004 for this branch was obtained against the now-deprecated v4 DeepForest checkpoint (`astana_trees_v4_10epochs.pl`) trained on a separate Roboflow Astana annotation set whose bounding-box labelling convention did not match the polygon-derived ground truth of the CVAT-annotated validation set used by the YOLO and Mask R-CNN branches. With the v3 fine-tune of Section 3.5.2, which is now trained on the same CVAT polygon dataset, this annotation-policy mismatch is resolved and the SAM 2 mask metric becomes comparable to the YOLO mask metric.

**Qualitative observations.** The SAM 2 output is **visibly tighter than either the raw DeepForest box or the YOLO polygon** on isolated trees in low-density residential scenes where the crown boundary is clearly defined in the satellite imagery (e.g. street trees against a uniform pavement background). In dense canopy scenes the SAM 2 output occasionally bleeds into the neighbouring crown when the prompt box spans two adjacent trees; this failure mode is a direct consequence of the imprecise input prompt and is not a limitation of SAM 2 itself, and would be mitigated by tightening the DeepForest box-regression before the SAM 2 call. A comparison of SAM 2 masks (0.134) with the Mask R-CNN v2+v3 masks (0.158) and the YOLO v3-finetune masks (0.263) is reported in Table 3.5 of Section 3.7.

## 3.7 Cross-model comparison and contextualisation against the literature

This section presents the central empirical contribution of the diploma project: a head-to-head comparison of every model branch — three YOLO checkpoints, two Mask R-CNN checkpoints, fine-tuned DeepForest with and without SAM 2 mask refinement, and the off-the-shelf NEON baseline — evaluated under a single protocol on a single validation set. Six of the seven configurations are then compared, in turn, with the literature baselines collected in Chapter 1.

### 3.7.1 The 14-image merged validation set

For the cross-model ablation reported below we constructed a dedicated 14-image merged validation set, denoted **M14** in the remainder of this section. The set unifies five validation images from each of the v2 and v3 dataset batches with four held-out images from the v1 batch — the fifth v1 candidate, `Снимок экрана 2026-04-01 194422.png`, was excluded because it was duplicated between the original v1 train and val splits and the YOLO data-preparation pipeline kept it in train via the `--dup-policy keep-train` flag, so retaining it in any cross-model val would silently leak training data for every YOLO checkpoint. The resulting M14 set contains 14 source images and 702 polygon annotations.

This M14 set is identical to the 15-image set used by the team members responsible for the DeepForest and Mask R-CNN branches **except for the excluded v1 duplicate**. The DeepForest fine-tune trained by team member Anuar Totin and the Mask R-CNN v2+v3 checkpoint trained by team member Berik Sharipov were both originally evaluated on the 15-image set. In the present section both branches were **re-evaluated on M14** with the unchanged scripts `ml/eval_df_sam2_14img.py` and `ml/eval_maskrcnn.py` against the COCO file `yolov train dataset/annotations_merged_14img_val.json`, ensuring that every cell of Table 3.5 is computed on exactly the same ground-truth annotations. The drop between the 15-image and 14-image numbers, of the order of 11–14 % relative, is consistent across both branches and is explained by the removal of the single dense image (24 polygons) that previously inflated both branches' counts.

### 3.7.2 Cross-model ablation on M14

The seven configurations are summarised in Table 3.5. Numbers are computed via the pycocotools COCOeval evaluator at all IoU thresholds from 0.50 to 0.95; the headline mAP@50 column is the area-under-PR-curve at IoU 0.50.

**Table 3.5 — Cross-model ablation on the 14-image merged validation set (M14, 702 polygons).**

| Model | Box mAP@50 | Box mAP@50:95 | Mask mAP@50 | Mask mAP@50:95 |
|---|---|---|---|---|
| NEON pretrained (off-the-shelf) † | 0.012 | — | — | — |
| DeepForest v4 (old Roboflow ckpt) † | 0.004 | — | — | — |
| YOLOv8x-seg v1 | 0.131 | 0.047 | 0.134 | 0.042 |
| DeepForest v3 + SAM 2 | 0.146 | 0.046 | 0.134 | 0.042 |
| YOLOv8x-seg v2-fromscratch | 0.156 | 0.056 | 0.147 | 0.049 |
| Mask R-CNN v2+v3 (warm-start) | 0.166 | 0.062 | 0.158 | 0.055 |
| YOLOv8x-seg v2-finetune | 0.187 | 0.067 | 0.185 | 0.062 |
| YOLOv8m-seg exp1 (v2-proven aug, lucky single run) | 0.308 | 0.110 | 0.305 | 0.097 |
| YOLOv8m-seg exp1 (4-replicate mean ± std) | 0.271 ± 0.028 | — | 0.282 ± 0.025 | — |
| **YOLOv8x-seg v4_x_clean (FINAL production)** | **0.315** | **0.115** | **0.289** | **0.099** |

† NEON and DeepForest v4 were measured on the 15-image superset by team member Anuar Totin; on M14 their numbers are bounded above by these values and remain essentially zero relative to the other configurations. The full YOLO ablation chain — including all intermediate Round 1 – Round 6 experiments — is reported separately in Table 3.3i of Section 3.3.11.

Five empirical observations follow from this table.

First, the **YOLOv8x-seg v4_x_clean production checkpoint is the strongest single configuration on every metric**, reaching Box mAP@50 = 0.315 and Mask mAP@50 = 0.289 on M14. This number is **+140 %** relative to the YOLO v1 baseline (0.131) and **+69 %** relative to the previous v2-finetune production (0.187). The full justification of this architectural choice is given in the Round 4 clean-defaults sweep of Section 3.3.8: contrary to the working hypothesis through Rounds 1 – 3 of the ablation (which had favoured the smaller yolov8m-seg variant with manually-tuned augmentation), the largest YOLOv8 variant with the framework's default augmentation pipeline outperforms every manually-tuned configuration. The intermediate exp1 m-seg checkpoint at 0.308 is retained in the table for ablation context and as a member of the cross-YOLO voting ensemble described in Section 3.7.4 above; its four-replicate variance band (Section 3.3.9, mean 0.271 ± 0.028) places the single-run 0.308 result on the upper tail of its expected distribution.

Second, **YOLO v2-finetune (0.187) and Mask R-CNN v2+v3 (0.166) are within statistical noise of each other** by Box mAP@50, but YOLO is consistently ahead by Mask mAP@50 (0.185 vs 0.158). The one-stage YOLOv8 design — anchor-free detection head, CIoU box-regression loss, distribution focal loss for the discrete-bin regressor, prototype-mask head — closes the historical gap to the two-stage Mask R-CNN family for instance segmentation on a small low-resolution-satellite dataset, and the medium-size m-seg variant extends the lead further to a roughly 2× margin over Mask R-CNN.

Third, the **fine-tuned DeepForest + SAM 2 pipeline (0.146 / 0.134)** is the weakest of the three fine-tuned branches by Box mAP@50, although it improves over the NEON pretrained baseline (0.012) by a factor of approximately twelve. The principal source of the absolute gap to YOLO and Mask R-CNN is the DeepForest backbone's training distribution — NEON aerial lidar of American forested sites — which is more remote from the Astana satellite domain than the COCO pre-training distribution that initialised the YOLO and Mask R-CNN backbones.

Fourth, the **NEON pretrained baseline (0.012)** is the first published measurement of out-of-the-box DeepForest on Central-Asian satellite imagery. It confirms the geographic-generalisation hypothesis of Section 1.5 with a number that is one to two orders of magnitude below any fine-tuned configuration. The old Roboflow-trained `astana_trees_v4_10epochs.pl` checkpoint (0.004) is *even worse* than NEON on M14 — counter-intuitively, because the v4 checkpoint was trained on a separate Astana annotation set (a Roboflow workspace maintained independently by Anuar Totin) whose bounding-box labelling convention does not match the polygon-derived M14 ground truth. The result demonstrates that **annotation-policy mismatch within a city is in this case larger than the floristic gap between continents** — an empirical finding of interest beyond the immediate project.

Fifth, the **SAM 2 mask-refinement stage** brings the Mask mAP@50 of the DeepForest branch from zero (no native mask output) to 0.134 — comparable in magnitude to the Mask mAP@50 of YOLO v1 (0.134) and not far behind that of Mask R-CNN v2+v3 (0.158). The Box quality of the DeepForest predictions does not change with SAM 2 (the model is a pure post-processor of the bounding-box detector), so the Box mAP@50 of 0.146 is also the standalone DeepForest v3 number on M14.

### 3.7.3 Weighted-Box-Fusion ensemble — implementation status

The YOLO and DeepForest branches are combined through the Weighted-Box-Fusion ensemble described in Section 2.8 (`backend/models/ensemble_adapter.py`, $T_{\text{IoU}} = 0.55$, equal per-branch weights). Qualitative inspection on M14 images confirms that the ensemble suppresses the complementary failure modes of the two branches — a YOLO-over-segmented pair of crowns is typically rejected when DeepForest detects the same area as a single crown, and a DeepForest-merged pair is split when YOLO independently localises the two components.

**The Weighted-Box-Fusion ensemble has not yet been quantitatively evaluated on the M14 validation set under the same protocol as the individual branches.** A pre-v3 prototype evaluation on the v1 validation tiles gave a Box mAP@50 of approximately 0.51, but this number is not directly comparable with the M14 numbers reported in Table 3.5 (different validation set, different ground-truth corpus). A rigorous M14 ensemble evaluation, together with per-class WBF weight calibration favouring the dominant YOLO branch, is reserved for future work (Section 3.9).

### 3.7.4 Cross-YOLO voting ensemble (4× IoU-merged checkpoints)

A complementary ensemble strategy that operates within the YOLO family rather than across architectures was developed to mitigate the per-checkpoint training-time variance documented in Section 3.3.9 and to address the qualitative observation, discussed in Section 3.9 below, that different YOLO checkpoints with statistically indistinguishable aggregate mAP detect substantially different per-detection subsets on the same input scene. The ensemble is implemented in `backend/models/yolo_ensemble_adapter.py` (`MultiYOLOEnsembleAdapter`) and exposed in the frontend's hierarchical model picker under the "Ensemble → 4× YOLO vote" option. Its CLI counterpart is `ml/v5_ensemble.py`.

**Algorithm.** Predictions from N member checkpoints are pooled into a single detection list tagged by member name. Detections are clustered by box Intersection-over-Union — for any two detections with IoU ≥ 0.5 the cluster grows by union-find — and clusters that contain detections from fewer than K distinct member models are discarded. Within each surviving cluster the highest-confidence detection is kept and emitted as the cluster's representative. The implementation is order-independent in the member list and runs in $O(N M^2)$ for $N$ models and $M$ detections per image, dominated by the pairwise IoU computation rather than the union-find.

**Default member set.** Four YOLO checkpoints from the project's archive are used as the default ensemble members, chosen for their visually complementary failure modes on Astana scenes:
- **v4_x_clean** — the final production checkpoint (yolov8x-seg, Ultralytics defaults, Section 3.3.8), strongest aggregate mAP, conservative on built-environment surfaces;
- **exp1_m_cocostart** — yolov8m-seg with v2-proven augmentation, the Round 1 winner of Section 3.3.7, finds more partially-occluded crowns than v4_x_clean;
- **v4_s_clean** — yolov8s-seg with defaults, the smallest reasonable variant, most permissive (highest raw detection count, useful for recall-priority scenes);
- **v2-finetune** — yolov8x-seg with v2-proven aug, the previous production checkpoint, most conservative on novel surfaces (no stadium-roof regression).

**Default voting threshold.** K = 2 (a detection is retained when at least two of the four members agree on it), which removes single-model hallucinations without requiring three-of-four agreement that would substantially reduce recall.

**Qualitative result on a representative test scene.** Running the four-checkpoint vote-2 ensemble at `conf = 0.15` on the 1 236 × 1 159 pixel Astana tennis-court complex tile produces the per-checkpoint and unified detection counts reported in Table 3.5a.

**Table 3.5a — Per-checkpoint detection counts on a single 1 236 × 1 159 pixel Astana tile at `conf = 0.15`. The vote-2 ensemble keeps only the IoU-clusters where at least two members agreed.**

| Checkpoint | Raw detection count |
|---|---|
| v4_x_clean | 687 |
| exp1_m_cocostart | 738 |
| v4_s_clean | 819 |
| v2-finetune (legacy) | 756 |
| **Total raw pool (with cross-checkpoint duplicates)** | **3 000** |
| **4× vote-2 unified ensemble** | **790** |

The 3 000-to-790 reduction is dominated by cross-checkpoint duplicates — most trees appear in three or four of the four checkpoint outputs and survive vote-2, while single-model hallucinations (the most common being false-positive crowns on stadium roof structures specifically present in the v4_x_clean output, the typical visual regression discussed in Section 3.9) are discarded. The visual cross-checkpoint comparison and the ensemble output are reproduced in Figure 3.7a / 3.7b below.

![*Cross-checkpoint qualitative comparison on a representative 1 236 × 1 159 pixel Astana tile (tennis-court complex with surrounding row-planted street trees). Each cell shows the same input image overlaid with one checkpoint's predicted polygons (no per-detection text labels — outline + 25 % alpha fill only). The 132-tree spread between the most conservative checkpoint (v4_x_clean at 687) and the most permissive (v4_s_clean at 819) on the same input is a direct visual illustration of the aggregate-mAP limitation discussed in Section 3.9: models with statistically indistinguishable Box mAP@50 detect substantially different per-detection subsets of the same ground-truth crowns.*](figures/cross_model_8way_comparison.png)

![*Top-4 member checkpoints and the resulting cross-YOLO vote-2 ensemble on the same input tile. The ensemble cell (bottom right, cyan outline with 3-pixel thickness) shows the 790 unified detections that survive IoU ≥ 0.5 clustering with at least two member-model votes. Single-model hallucinations — most visibly the stadium-roof false positives present in the v4_x_clean output — are discarded by the voting requirement.*](figures/cross_yolo_ensemble_4way.png)

A full quantitative M14 evaluation of this ensemble — Box mAP@50 versus the individual checkpoints in Table 3.5 — is currently in progress at the time of writing and is reported in Section 3.7.5 of the next thesis revision. The implementation itself is production-ready and the option is selectable from the frontend at runtime.

### 3.7.5 Comparison with the literature

Table 3.6 contextualises the obtained numbers against the published baselines compiled in Chapter 1.

**Table 3.6 — Best obtained results compared with the literature.**

| System | Data | Best metric |
|---|---|---|
| **YOLOv8x-seg v4_x_clean (this work, FINAL production)** | Astana sat., M14 | **Box mAP@50 = 0.315, Mask mAP@50 = 0.289** |
| YOLOv8m-seg exp1 (this work, lucky run) | Astana sat., M14 | Box mAP@50 = 0.308, Mask mAP@50 = 0.305 |
| Mask R-CNN v2+v3 (this work) | Astana sat., M14 | Box mAP@50 = 0.166, Mask mAP@50 = 0.158 |
| DeepForest v3 + SAM 2 (this work) | Astana sat., M14 | Box mAP@50 = 0.146, Mask mAP@50 = 0.134 |
| NEON off-the-shelf (this work) | Astana sat., M14 | Box mAP@50 = 0.012 |
| YOLOv12m [@AbbasYOLO2025] | Public RGB satellite | mAP@50 = 0.908 |
| DeepForest off-the-shelf urban [@Ventura2024] | NAIP 60 cm USA | F = 0.42 |
| DeepForest fine-tuned urban [@Ventura2024] | NAIP 60 cm USA | F = 0.729 |
| DeepForest urban Sofia [@SofiaDeepForest2024] | Aerial 10 cm Sofia | F1 ≈ 0.68 |
| YOLOv5x Lleida [@VelasquezCamacho2023] | Aerial + sat Lleida | F1 = 0.849 |
| MCAN (Mask R-CNN variant) [@Lv2023] | UAV RGB Zhejiang | Det AP = 0.924 |

Three observations follow.

First, the absolute mAP@50 of 0.315 is **below** the best published numbers — 0.91 for YOLOv12m on a much larger public dataset, 0.85 for YOLOv5x on Lleida — but it has been obtained on a single-city dataset of approximately 100 satellite images, two orders of magnitude smaller than those benchmarks. The headline number is therefore best read as a *first-of-its-kind* result on Astana rather than as a competitive entry on a public benchmark.

Second, the comparison with the off-the-shelf-vs-fine-tuned DeepForest gap of Ventura et al. is now empirically tight: the present project measures **0.012 → 0.146** on Astana M14 (factor 12) where Ventura measures **0.42 → 0.729** on NAIP USA (factor 1.7). The much larger relative jump in Astana reflects two compounding factors — the smaller floristic mismatch between the NEON training set and the European-Mediterranean target (Ventura's NAIP) compared with the much larger mismatch with the *Populus*-dominated Astana canopy, and the smaller absolute starting point on Astana imagery.

Third, the closest geographic analogue — the Sofia DeepForest work of Dakov and Petrova-Antonova [@SofiaDeepForest2024] at F1 ≈ 0.68 on 826 annotated trees in a Soviet-era European capital — establishes a realistic medium-term target for the Astana project, attainable through the dataset-expansion strategy outlined in the future-work discussion of the conclusion.

![*Visual comparison of Box mAP@50 and Mask mAP@50 for the seven configurations on the M14 validation set. The YOLOv8m-seg v3 production model (exp1) is the strongest model on both metrics; the previous YOLOv8x-seg v3-run1 intermediate checkpoint is the second-strongest; the three smaller-gap Astana-fine-tuned branches (DeepForest+SAM 2, Mask R-CNN v2+v3, YOLO v2-finetune) cluster in the 0.13–0.19 range; the NEON pretrained and Roboflow-v4 checkpoints are near zero, confirming the necessity of Astana-domain fine-tuning.*](figures/model_comparison_barchart.png)

## 3.8 Integrated pipeline

### 3.8.1 Production configuration

The four production checkpoints — **YOLOv8m-seg v3 (exp1, `weights/yolo_satellite.pt`,** copied from `runs/segment/v3_exp1_m_cocostart/weights/best.pt`, MD5 `c6aada99dd9261e39dabeb52f5ad19ff`), Mask R-CNN v2+v3 fine-tune (`weights/maskrcnn_astana.pt`, copied from the GitHub release `maskrcnn-v2v3`), DeepForest v3 fine-tune (`weights/deepforest_astana.pl`, copied from the release `v2.0`) and the SAM 2 backbone (auto-downloaded from HuggingFace `facebook/sam2.1-hiera-base-plus` on first inference) — are loaded by the FastAPI backend through the adapter interface of Chapter 2. Each adapter performs sliding-window tiled inference at 640 + 128 overlap and applies global Non-Maximum Suppression on the merged per-tile predictions; the DeepForest adapter falls back to the public `weecology/deepforest-tree` weights when the v3 file is absent, so the system remains operational on machines without the proprietary checkpoint. For interactive A/B comparison the backend additionally registers the historical YOLO variants (v2-finetune, v3-finetune-run1, v3-finetune-run2) through the `ModelKind` enum and the `_register_yolo_variant()` helper, so the user can switch between any of the archived checkpoints from the frontend's settings popover at runtime.

All inference results are persisted in the SQLite database described in Section 2.10. A re-start of the backend therefore preserves every snapshot, every inference run and every individual detection ever produced by the system, which is essential for the aggregate **city-map view** documented below.

### 3.8.2 End-to-end demonstration

A typical interactive session proceeds as follows: the user draws a 1 km × 1 km rectangle on the basemap, clicks *Scan area*, and the backend automatically subdivides the request into a small grid of sub-bounding-boxes at the fixed zoom level of 19 (the highest available resolution for which the production checkpoints were trained), stitches the ESRI or Google Satellite tiles for each sub-region in approximately two seconds per sub-region, and runs the chosen model on each captured snapshot. The user can monitor the per-sub-region progress in real time through the streaming NDJSON endpoint `/api/scan_region/stream`. The system reports the number of detected trees, the average crown area, the green-coverage percentage and the total analysed area in hectares; the detections are rendered as semi-transparent polygons over the basemap and persisted in the database as a single scan-session. The full end-to-end response time for a 1 km × 1 km capture at zoom 19 (approximately 3 000 × 3 000 pixels, requiring roughly 25 tiled inference passes on the YOLO branch) is approximately **18 seconds** on the laptop GPU — comfortably below the 30-second budget set by the requirements of Chapter 1. With the YOLO v3-finetune checkpoint as the model the detection count for a typical residential 1 km × 1 km block in Astana is in the range of 300 – 800 trees.

### 3.8.3 The city-map view as principal deliverable

Beyond the per-image workflow described above, the application exposes a *city-map view* that visualises every detection of every snapshot ever processed by the system on a single Leaflet layer. Internally the view issues a single `GET /api/detections` request, which the backend translates into a single SQL query over the `detections` table joined with the `snapshots` table for the geographic bounds. The frontend then clusters the result at low zoom levels (using the same Leaflet cluster-marker plugin as standard GIS tools) and renders individual detections at high zoom levels.

The city-map view is the principal demonstration deliverable of the application. A municipal employee can use it as the canonical inventory of Astana trees produced by the system: it grows organically as each new district is processed by the *Capture from map* flow, supports geographic filtering by drawing a sub-rectangle on the map, supports model and confidence filtering through the side panel, and supports per-snapshot deletion through a cascading database operation. The cap of 50 000 detections per request is a safety measure against accidental browser crashes; with the average density of approximately 70 trees per processed tile observed in Section 3.2, this cap is sufficient for an inventory of approximately 700 captured snapshots — well in excess of what the system is expected to handle in the present prototype scope.

![*City-map view of the web application showing 1 031 detected trees across 3 processed snapshots of Astana, rendered as semi-transparent crown polygons on the ESRI World Imagery satellite basemap. The left panel shows aggregate statistics (trees, runs, average crown diameter, avg. confidence) and a list of processed snapshots. The detection legend distinguishes high-confidence (≥ 70 %), medium (50–70 %) and low (< 50 %) detections by colour. The map demonstrates the city-wide accumulation workflow: each new district processed by the system adds its detections to the persistent database, building an organic inventory of the urban canopy over time.*](figures/ui_city_map_view.png)

![*Single-image view of the web application. The left sidebar shows the upload zone, model selector (DeepForest Astana fine-tuned), confidence threshold, geographic referencing controls (2-corner axis-aligned mode with NW/SE coordinates) and export buttons. The main panel displays the Leaflet map with the uploaded satellite image overlaid as a semi-transparent layer and the model selector ready to run detection.*](figures/ui_single_image_view.png)

### 3.8.4 Export formats

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

## 3.9 Limitations

The current implementation has a number of known limitations.

0. **The aggregate mAP@50 metric does not adequately discriminate between checkpoints with similar headline numbers.** A central methodological observation that emerged from the cross-checkpoint qualitative comparison of Section 3.7.4 is that different YOLO checkpoints with statistically indistinguishable Box mAP@50 (e.g. v4_x_clean at 0.315, exp1 at 0.308, v4_m_clean at 0.291 — all within twice the single-run variance band of Section 3.3.9) detect substantially different per-detection subsets of the ground-truth crowns on the same input scene. On the 1 236 × 1 159 pixel Astana tile of Figure 3.7a the per-checkpoint raw detection count ranges from 687 (v4_x_clean, most conservative) to 819 (v4_s_clean, most permissive) — a 19 % spread that the aggregate mAP@50 cannot reveal because each checkpoint approximately matches a different subset of the ground-truth labels rather than always agreeing on which subset is "the right one". This is a well-known limitation of aggregate detection metrics on small heterogeneous datasets, but rarely surfaces in the published satellite-tree literature, where authors typically report only the highest single-run mAP. Two practical consequences follow for the present project: (i) the cross-YOLO voting ensemble of Section 3.7.4 is a partial mitigation, because it averages out the per-checkpoint training-time variance and per-checkpoint failure modes at inference time, but it remains an ensemble of qualitatively similar models and does not in principle add information; (ii) any production deployment of this system at *Zelenstroy* should be accompanied by per-district visual cross-checkpoint review on the actual scenes of interest, not only by a single aggregate metric.

1. **Validation-set size.** The cross-model M14 validation set comprises only 14 source images and 702 polygon annotations. With this sample size the 95 % confidence interval on the reported mAP@50 numbers is of the order of ± 0.05, and the four-replicate variance experiment of Section 3.3.9 places the single-shot training-time variance at approximately ± 0.03 mAP@50 (independently of the validation-set sample size). A larger validation set drawn from city districts not present in the training corpus at all — and a multi-seed protocol that averages over training-time variance — would be required to claim full statistical significance for the model-vs-model comparisons of Section 3.7.

2. **Single class label.** The dataset annotation does not distinguish between species. Tree-species classification is a valuable downstream task — particularly for *Zelenstroy*, which must plan species-specific pruning and replacement — and is reserved for future work, possibly using the multi-task setup of [@Martins2021Species].

3. **No explicit shadow modelling.** Shadows are the dominant source of false positives across all branches. An explicit pre-processing step that detects and removes shadow regions, or an auxiliary head that classifies a region as "tree", "shadow" or "background", would substantially improve the precision in early-morning and late-evening imagery.

4. **No per-class WBF ensemble calibration and no quantitative M14 evaluation of the ensemble.** The Weighted-Box-Fusion ensemble of Section 2.8 is implemented in `backend/models/ensemble_adapter.py` and works end-to-end at inference time, but it has not been evaluated on the M14 validation set under the same protocol as the individual branches. The pre-v3 prototype evaluation on the v1 validation tiles gave Box mAP@50 ≈ 0.51, but this number is not directly comparable with the M14 numbers reported in Table 3.5. Both the rigorous M14 ensemble evaluation and the per-class weight calibration favouring the dominant YOLO branch (which on M14 outperforms DeepForest by approximately 2× on Box mAP@50) are reserved for future work.

5. **Stadium-roof false positives — a qualitative regression of the v3 production model.** During interactive testing of the v3 production model on Astana scenes that include sports infrastructure (notably the Botanical Garden area, which is adjacent to a covered arena), the model produced false-positive detections on stadium and arena roof structures that the previous v2-finetune model had correctly ignored. The v2-finetune model is more conservative on novel non-natural surfaces; the v3 production model has learned crown-like patterns that misfire on certain roof textures. Crucially, this regression is **not captured by the M14 aggregate metric** because none of the validation tiles in v1, v2 or v3 contain stadium-style architecture. The implication is methodological: models tuned to maximise an aggregate mAP can develop scene-specific failure modes on surface types absent from the training distribution, and the held-out validation set's representativeness determines what kinds of regression the aggregate metric can detect. Two complementary mitigations are recommended for future work: (i) augmenting the training set with negative examples of built-environment structures, and (ii) inference-time post-filtering using OpenStreetMap building footprints — drop predicted boxes whose centroid falls inside an OSM building polygon, following the approach reported in [@Ventura2024]. The OSM filter is the lower-cost option and is recommended as the immediate next implementation step in the project backlog.

5. **DeepForest v4 / Roboflow access loss.** Anuar's original Astana bounding-box annotation set on Roboflow (workspace `bads-workspace`, project `astana-trees-ndi9r`, version 4) is no longer accessible. The v4 checkpoint trained on that set is retained on disk for the ablation reported in Section 3.7 but cannot be reproduced. The current production DeepForest v3 fine-tune was trained on the CVAT polygon dataset (Section 3.5.2), so the production weights themselves are fully reproducible from the present repository.

6. **Train / serve domain shift.** All three iterative dataset batches (v1, v2, v3) are screenshots captured manually from Google Earth Pro. The production runtime fetches tiles from either ESRI World Imagery or the unofficial Google Satellite endpoint (Section 2.2 and Section 2.11.2) — different rendering pipelines, possibly different capture dates and tone-mapping. The magnitude of the resulting train / serve distribution shift has not been quantified in this thesis and is a candidate for a v4 data batch captured directly from the runtime `/api/scan_region` output.

7. **Single-laptop deployment only.** The current system was deployed and tested only on the development laptops of the three team members. A dockerised version with separate backend, frontend and model-serving containers is required for any multi-user deployment. The SQLite database described in Section 2.10 would have to be promoted to PostgreSQL for any deployment with concurrent writers.

Despite these limitations, the system as currently deployed already meets all six functional requirements of Section 1.6 and produces results that are qualitatively informative for the *Zelenstroy* end user.

\newpage
