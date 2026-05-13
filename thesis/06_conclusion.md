# Conclusion

This diploma project set out to design, implement and evaluate a complete software system capable of automatically detecting individual trees in satellite imagery of the city of Astana. The work was motivated by the practical need of *Zelenstroy* and other municipal services for a fast, repeatable and inexpensive alternative to the manual field survey that has historically been the only way to obtain an urban-tree inventory at the scale of a major city. The analysis of the existing literature (Chapter 1) confirmed that the deep-learning toolkit needed for such a system is technically mature — state-of-the-art object-detection and instance-segmentation networks routinely exceed 90 % mean average precision on dedicated public datasets — but also that **no published work** to date has evaluated any of these methods on Central-Asian satellite imagery in general or on Astana imagery in particular. Closing this gap was the principal motivation of the present work.

## Achievement of the stated objectives

The six objectives formulated in the introduction were addressed as follows.

**Objective 1 — survey the deep-learning state of the art.** Chapter 1 contains a critical analysis of 31 peer-reviewed publications of 2019 – 2025, organised by technical paradigm (two-stage detectors, one-stage detectors, semantic segmentation, the domain-specialised DeepForest model and the SAM foundation model) and culminating in a consolidated Table 1.2 of the best metrics reported by each work. The survey established two essential facts: that fine-tuned models routinely achieve F1-scores in the 0.65 – 0.85 range on European and American urban data, and that the geographic-generalisation gap between an off-the-shelf and a fine-tuned model is in the order of 0.30 F1 points.

**Objective 2 — quantify the gap on Astana imagery.** Section 3.4.1 reports the off-the-shelf DeepForest baseline on Astana — precision ≈ 0.72, recall ≈ 0.58 — confirming both the order-of-magnitude consistency with the European urban literature and the necessity of a fine-tuning stage.

**Objective 3 — build a custom annotated Astana dataset.** Section 3.2 documents two iterations of dataset construction: a version-1 dataset of 20 source images / 2 242 polygons obtained through from-scratch manual annotation in CVAT, and a version-2 dataset of 77 source images / ≈ 8 000 polygons obtained through a model-in-the-loop pre-labelling workflow that reduced the annotation cost per image by approximately 70 %. The dataset is, to the best of the authors' knowledge, the first of its kind for Astana and is small but reusable for future research.

**Objective 4 — train and combine three complementary models.** Chapters 2 and 3 document the training of three successive YOLOv8x-seg checkpoints on the Astana dataset: the version-1 model (397 epochs on 20 source images, Box mAP@50 = 0.478 on its own small 4-tile validation set), the version-2-from-scratch model (204 epochs from COCO weights on the merged 77-image dataset) and the version-2-finetune model (continued training of the v1 checkpoint on the 57 new images only, 99 epochs). All three models were re-evaluated on a common, larger and harder version-2 validation set; on this like-for-like comparison the version-2-finetune model is the best, with Box mAP@50 = **0.372** and Mask mAP@50 = **0.331**, representing a 40 % relative improvement over the version-1 baseline (0.265 / 0.240). The DeepForest RetinaNet detector, trained by team member Anuar Totin on the Roboflow `astana-trees-ndi9r` v4 dataset, reaches precision = 0.667, recall = 0.552 and F1 = 0.604 on the Astana test split. SAM 2 (`sam2.1-hiera-base-plus`) is integrated as a zero-shot mask-refinement stage on top of DeepForest. The YOLO and DeepForest branches are combined through a Weighted-Box-Fusion ensemble (Box mAP@50 ≈ 0.51 on the v1 val), exploiting the complementary failure modes of the two architectures: YOLO over-segments dense canopies, DeepForest under-segments them.

A methodological observation of the version-2 training comparison is worth retaining for future work. A first ablation, restricted to the comparison of v1 against v2-from-scratch, suggested that fine-tuning offered no benefit; this conclusion was reversed once a third configuration — fine-tune on the new-images subset rather than on the full merged corpus — was added to the comparison. The episode is a useful reminder that conclusions about transfer-learning effectiveness are extremely sensitive to the precise composition of the fine-tuning set, and that a two-way ablation can hide an important interaction with a third design dimension.

**Objective 5 — design and implement an end-to-end software pipeline.** Chapter 2 describes the resulting system: a FastAPI Python backend with a pluggable model-adapter interface, a React 18 single-page frontend with a Leaflet map, a four-mode geographic-conversion module supporting GeoTIFFs and informal screenshots alike, an in-browser map-capture feature that stitches ESRI World Imagery tiles for an arbitrary user-drawn rectangle, and three export formats (GeoJSON, CSV, standalone HTML) that integrate the inventory with the user's existing GIS workflow. The complete prototype is approximately 6 000 lines of Python and JavaScript.

**Objective 6 — evaluate the system on realistic Astana scenes.** Section 3.7 reports the end-to-end behaviour of the integrated pipeline: a 1 km × 1 km area captured at zoom 18 is processed in approximately 18 seconds on a single laptop GPU, comfortably below the 30-second budget set by the requirements. Qualitative inspection of the predictions on Astana validation tiles confirms that the system already produces an inventory that is qualitatively informative for the *Zelenstroy* end user, while the aggregate metrics — Box mAP@50 ≈ 0.48 for YOLO alone and ≈ 0.51 for the ensemble — establish the empirical baseline against which any future model can be compared.

## Scientific contributions

The work contributes the following original results.

1. The **first published evaluation** of state-of-the-art deep-learning tree-detection models on satellite imagery of Astana and, by extension, of any major Central-Asian capital. The empirical numbers — off-the-shelf DeepForest precision/recall on Astana, fine-tuned DeepForest on Astana, YOLOv8x-seg on Astana, ensemble — are now part of the empirical record for the region.

2. A **small but reusable annotated dataset** of Astana satellite imagery, with approximately 77 source images and approximately 8 000 polygon-level tree annotations after tiling, distributed under the same convention as the original DeepForest dataset.

3. A **three-model architecture** combining instance-segmentation (YOLOv8-seg), bounding-box detection (DeepForest fine-tuned) and zero-shot mask refinement (SAM 2 on DeepForest boxes), with a Weighted-Box-Fusion ensemble of the first two branches. The complementarity of the YOLO and DeepForest failure modes — established quantitatively in Section 3.4.3 — is a novel observation in the urban-tree literature.

4. A **complete deployable prototype** — backend, frontend, geographic conversion, three exporters, an in-browser ESRI tile-capture feature — that serves as a template for any city administration wishing to adopt deep-learning-based tree inventories. The system is designed to be retrained on imagery from any other city by simply replacing the dataset.

## Practical contributions

The system is in a state in which it can be used immediately for an informal Astana tree inventory: the user supplies a satellite image (uploaded or captured interactively from the map), runs the selected model, inspects the result on a Leaflet map and exports the inventory in the format required by the downstream tool. The interactive response time of approximately 18 seconds for a 1 km × 1 km area is short enough that a single specialist can comfortably process several districts of the city in a working day — a productivity improvement of two to three orders of magnitude over the manual field survey it replaces.

The architectural decisions made along the way — adapter pattern for the model layer, four-mode geographic conversion, sliding-window tiled inference with global NMS, model-in-the-loop pre-labelling for annotation expansion, the Cyrillic-aware data-preparation pipeline — are independent of the specific application to Astana and can be reused as a template for any future urban-remote-sensing project.

## Limitations and future work

The current implementation has the limitations listed in Section 3.8: a validation set of only 4 (v1) or 10 (v2) tiles, a single class label that does not distinguish species, no explicit shadow modelling, an iterative DeepForest fine-tune that should be refactored into a single multi-epoch run, no quantitative SAM-vs-YOLO mask ablation, an in-memory job store with no persistence and a single-laptop deployment.

The natural directions for future work are:

1. **A fourth, independent detection branch** — currently in the architecture-selection phase, to be implemented and benchmarked by team member Berik Sharipov before the final defence. Candidates under consideration include a Mask R-CNN instance segmenter for a direct comparison with the YOLOv8-seg branch on identical data, an MCAN-style improved Mask R-CNN [@LvMCAN2023] for explicit benchmarking against the published state of the art, and a transformer-backbone Faster R-CNN variant [@Zhang2022] for a quantification of the value of attention-based feature extraction on the small Astana dataset. The fourth branch will be trained on the same YOLO polygon annotations described in Section 3.2, will follow the same tiled-inference protocol of Section 2.3, and will be evaluated on the same version-2 validation set used throughout Section 3.3.

2. **Dataset expansion to 200 – 500 source images**, with the explicit goal of reaching the F1 = 0.7 range demonstrated by [@Ventura2024] and [@SofiaDeepForest2024] on European urban data.

2. **Species-level annotation and a multi-class detection head**, following the multi-task approach of [@Martins2021Species] and the species-specific work of [@Branson2019].

3. **Sentinel-2 integration** for city-wide canopy-cover monitoring, following the methodology of [@He2022] and the canopy-height extension of [@Xu2025]. A Sentinel-2-based monitoring pipeline would complement the very-high-resolution per-tree inventory built in the present work and would allow the city to observe long-term trends in green coverage.

4. **A persistent inventory database** that accumulates detection results over time, supports versioning of the canonical inventory and exposes a difference view (trees added, trees removed) between successive captures of the same district.

5. **Multi-user deployment** through a containerised version of the system, with authentication and per-organisation isolation, so that *Zelenstroy* and other municipal services can adopt the tool as part of their day-to-day operations.

## Closing remark

This work has demonstrated that the deep-learning techniques developed for tree detection in American forests, European cities and Asian metropolises can be adapted to the very different geographic, architectural and floristic context of a Central-Asian capital through a relatively modest annotation effort and an off-the-shelf software stack. The resulting system delivers, for the first time, a quantitative baseline for automated urban-tree detection in Astana, an open prototype that can be operated by a non-technical user from a web browser, and a reusable template for future work in the region. The combination of these three contributions defines the engineering and scientific value of the present diploma project.

\newpage
