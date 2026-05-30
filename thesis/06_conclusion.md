# Conclusion

This diploma project set out to design, implement and evaluate a complete software system capable of automatically detecting individual trees in satellite imagery of the city of Astana. The work was motivated by the practical need of *Zelenstroy* and other municipal services for a fast, repeatable and inexpensive alternative to the manual field survey, and by the absence of any published deep-learning tree-detection benchmark on Central-Asian satellite imagery (Chapter 1).

## Achievement of the stated objectives

The six objectives of the introduction were addressed as follows.

**Objective 1 — state of the art.** Chapter 1 surveys the peer-reviewed literature of 2019 – 2025 organised by technical paradigm. Table 1.2 consolidates the best reported numbers and establishes that fine-tuned models routinely achieve F1-scores in the 0.65 – 0.85 range on European and American urban data, and that the gap between an off-the-shelf and a fine-tuned configuration is in the order of 0.30 F-score points.

**Objective 2 — quantify the gap on Astana imagery.** Sections 1.4 and 3.4.3 report the first measurement of the public NEON-pretrained DeepForest checkpoint on Astana satellite imagery: Box mAP@50 = **0.012** on M14, one to two orders of magnitude below any fine-tuned configuration. This is the empirical anchor of the geographic-generalisation gap on Central Asia.

**Objective 3 — build a custom annotated Astana dataset.** Section 3.1 documents three iterative dataset batches (v1 / v2 / v3) constructed in CVAT through a model-in-the-loop pre-labelling workflow that reduced the per-image annotation cost from approximately 25 minutes (v1, from scratch) to approximately 4 minutes (v3, pre-labelled by the v2-finetune YOLO model). The final corpus is approximately 100 source images with ≈ 5 500 hand-labelled tree-crown polygons (≈ 8 700 polygon instances after sliding-window tiling).

**Objective 4 — train and combine four complementary models.** A structured **23-experiment hyperparameter ablation** of the YOLO branch spanning six orthogonal axes was conducted (Section 3.2), together with two Mask R-CNN checkpoints, the DeepForest v3 fine-tune and the SAM 2 zero-shot mask-refinement stage. All branches were evaluated on the M14 cross-model validation set. The best single configuration is the **YOLOv8x-seg v4_x_clean production model** at **Box mAP@50 = 0.315** and **Mask mAP@50 = 0.289** on M14, a +140 % relative improvement over the YOLO v1 baseline. The DeepForest+SAM 2 pipeline (Box 0.146 / Mask 0.134) and the Mask R-CNN v2+v3 fine-tune (Box 0.166 / Mask 0.158) follow as the second and third best configurations.

**Objective 5 — end-to-end software pipeline.** Chapter 2 describes the deployed system: a FastAPI backend with a pluggable four-model adapter interface (eight YOLO checkpoint variants exposed through a hierarchical model picker, plus DeepForest, DeepForest+SAM 2, Mask R-CNN, the WBF ensemble and the cross-YOLO 4-way vote ensemble), a React 18 + Leaflet frontend with dark-mode-default styling and a centred per-action model picker, a four-mode geographic-conversion module, an in-browser map-capture feature with Auto-Zoom Region Scan (rectangular or polygon-shaped) and streaming NDJSON progress, SQLite persistent storage with `ON DELETE CASCADE` cleanup, and three export formats (GeoJSON, CSV, standalone HTML).

**Objective 6 — evaluation on realistic Astana scenes.** A 1 km × 1 km area captured at zoom 19 is processed in approximately 18 seconds on a single laptop GPU, comfortably below the 30-second budget set by the requirements of Section 1.5. Qualitative cross-checkpoint inspection confirms that meaningful per-detection complementarity exists between the YOLO variants, motivating both the cross-YOLO voting ensemble (Section 3.5) and the broader methodological discussion of aggregate-mAP limitations in Section 3.7.

## Scientific and engineering contributions

1. **The first measured magnitude of the geographic-generalisation gap on Central-Asian satellite imagery** — Box mAP@50 = 0.012 for off-the-shelf NEON DeepForest on Astana, three orders of magnitude below the same model's reported off-the-shelf F-score on NAIP USA imagery.

2. **A single cross-model validation set (M14) and the first apples-to-apples ablation** of four neural-network families (YOLOv8-seg, Mask R-CNN, fine-tuned DeepForest, SAM 2 zero-shot mask refinement) on the same Astana ground truth.

3. **A 23-experiment hyperparameter ablation of the YOLO branch** establishing that the Ultralytics default augmentation pipeline outperforms manually-tuned configurations on Astana satellite imagery, that model size has a **U-shaped** relationship with performance on small annotated datasets (yolov8m-seg at 27 M parameters and yolov8x-seg at 71 M with defaults both occupy local optima), and that chain learning across version-batch boundaries hurts by approximately 0.10 mAP relative to single-shot training, with the damage isolated to inter-batch distribution drift rather than the staging mechanism itself.

4. **A multi-replicate variance estimate of the best single YOLO configuration** — four independent training runs of the same exp1 configuration produced merged-val Box mAP@50 = 0.308 / 0.268 / 0.269 / 0.239, sample mean **0.271 ± 0.028**. This places the headline variance band of any single-shot YOLO experiment on this dataset at approximately ± 0.03 Box mAP@50.

5. **A novel cross-YOLO vote-based ensemble** (Section 3.5) that pools predictions from four YOLO members, clusters them by box IoU and discards single-model hallucinations through a $K = 2$ majority-vote rule. The ensemble is a qualitative false-positive killer for surface types absent from the training distribution (notably the stadium-roof regression of the v4_x_clean checkpoint).

6. **An unexpected secondary finding**: the deprecated Roboflow-trained DeepForest v4 checkpoint scores Box mAP@50 = 0.004 on M14 — *below* the NEON pretrained baseline — because of an annotation-policy mismatch between the Roboflow bounding-box convention and the CVAT polygon convention. The result demonstrates that intra-Astana annotation-policy mismatch can dominate the floristic gap between continents.

7. **A reusable Astana satellite-tree dataset** of approximately 100 source images with ≈ 5 500 hand-labelled tree-crown polygons (≈ 8 700 polygon instances after tiling) across three iterative batches.

8. **A deployable software prototype** that turns the trained models into a usable internal tool: pluggable adapter interface, four geographic-conversion modes, SQLite persistence with cascading deletes, rectangle and polygon Auto-Zoom Region Scan workflows with streaming NDJSON progress, three exporters, dark-mode-default UI with a centred per-action model picker, and a city-map aggregate view that grows organically as the user processes new districts.

## Methodological reflection

A central observation that surfaces from the empirical work is that **aggregate metrics such as mAP@50 are insufficient as a single criterion for production model selection on small datasets**. Cross-checkpoint visual comparison on the same Astana input scene shows that different models with statistically indistinguishable aggregate mAP can detect substantially different per-detection subsets — a 132-tree spread on a single 1 236 × 1 159 px tile between the most conservative checkpoint (v4_x_clean, 687 detections) and the most permissive (v4_s_clean, 819). This is a known limitation of detection-model evaluation but rarely surfaces in the published satellite-tree literature, where authors typically report only the highest single-run mAP. Two practical consequences follow: the cross-YOLO voting ensemble is a partial mitigation, and any production deployment at *Zelenstroy* should be accompanied by per-district visual cross-checkpoint review on the actual scenes of interest.

## Future work

The natural directions for future work are: (i) **dataset expansion to 200 – 500 source images** with the explicit goal of reaching the F1 = 0.7 range demonstrated on European urban data, in particular adding tiles of built-environment scenes that produce the dominant false-positive failure mode of the current production checkpoint; (ii) **quantitative M14 evaluation and per-class weight calibration** of the WBF ensemble and the cross-YOLO vote ensemble; (iii) **species-level annotation and a multi-class detection head**, since species-specific pruning and replacement is a separate budget line in the municipal urban-forestry plan; (iv) **OpenStreetMap building-footprint post-filtering** at inference time — a zero-training-cost mitigation of the stadium-roof regression; (v) **SAM 2 video propagation** for multi-temporal monitoring of canopy change across successive satellite acquisitions; (vi) **Sentinel-2 city-wide canopy-cover monitoring** as a complement to the very-high-resolution per-tree inventory; and (vii) **multi-user containerised deployment** with authentication and per-organisation isolation so the system can be adopted as part of *Zelenstroy*'s day-to-day operations.

## Closing remark

This work demonstrates that deep-learning techniques developed for tree detection in American forests, European cities and Asian metropolises can be adapted to the very different geographic, architectural and floristic context of a Central-Asian capital through a relatively modest annotation effort and an off-the-shelf software stack. Beyond the headline Box mAP@50 of 0.315, the project contributes a systematic empirical methodology — single cross-model M14 validation set, four-replicate variance estimation, six-axis hyperparameter ablation, qualitative cross-checkpoint complementarity analysis — that can be reused by any successor project on Astana or any analogous Central-Asian city. The resulting system delivers, for the first time, a quantitative baseline for automated urban-tree detection in Astana, a deployable prototype that can be operated by a non-technical user from a web browser, and a reusable template for future work in the region.

\newpage
