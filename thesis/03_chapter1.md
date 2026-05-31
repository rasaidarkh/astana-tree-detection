# Chapter 1. Analysis of the subject area and problem statement

This chapter establishes the technical and scientific context of the work. Section 1.1 describes urban-tree inventory as an applied problem. Section 1.2 surveys the remote-sensing data sources available for the task. Section 1.3 organises the relevant deep-learning literature of 2019 – 2025 into a single comparative table grouped by model family. Section 1.4 identifies the **geographic generalisation gap** as the dominant obstacle to applying off-the-shelf models in Astana. Section 1.5 formulates the precise problem statement that the rest of the work will solve.

## 1.1 Urban-tree inventory as an applied problem

Municipalities increasingly rely on quantitative inventories of urban green space for planning, environmental monitoring and climate adaptation. The basic unit of such inventories is the individual tree, characterised by a geographic position, a crown size and — when available — a species label and a health status. Aggregated indicators (number of trees per district, canopy-cover percentage, average crown area) follow directly from the per-tree records.

Historically, urban-tree inventories were obtained through manual field surveys with hand-held GPS receivers and paper forms. Reports from comparable cities — for example, the multi-year inventory of Sofia, Bulgaria [@Dakov2024] or the multi-month effort of Pasadena, California [@Branson2019] — confirm that for any large city the manual approach is slow, expensive and rapidly becomes obsolete. Three classes of automated alternative have emerged: airborne LiDAR [@Schmohl2022], which is the gold standard wherever it is available but is exceptionally expensive and unsuitable for cities — including Astana — that do not appear in national LiDAR programmes; high-resolution aerial or UAV imagery at 5 – 30 cm GSD, the dominant data source for urban-tree research [@Martins2021; @dosSantos2019; @VelasquezCamacho2023]; and **freely or inexpensively available very-high-resolution satellite imagery** from commercial vendors and aggregators such as ESRI World Imagery and Google Earth — the only practical option for Astana at the required scale [@He2022; @VelasquezCamacho2025; @AbbasYOLO2025].

The practical requirements imposed by *Zelenstroy* on any automated inventory are: (i) accept imagery from heterogeneous sources without re-training; (ii) output a per-tree polygon mask rather than a single pixel, so that crown area can be computed; (iii) convert pixel coordinates into geographic coordinates in WGS-84; (iv) export the inventory in formats consumable by QGIS, ArcGIS and Excel; (v) provide a confidence score for each detection so that human reviewers can prioritise their inspection time. The methodological choices documented in Chapter 2 are direct consequences of these five requirements.

## 1.2 Remote-sensing data sources

The deep-learning literature on tree detection is highly **data-source-dependent**: a model that excels on 5-cm UAV imagery typically fails when applied directly to 1-m satellite imagery. The body of work surveyed below is organised along this dimension in Table 1.1.

**Table 1.1 — Data sources reported in the surveyed literature.**

| Source | Typical GSD | Coverage | Cost | Representative works |
|---|---|---|---|---|
| Airborne LiDAR | 5 – 50 cm 3-D | National / regional campaigns | Very high | [@Schmohl2022; @DeepForest2019] |
| UAV / drone RGB | 1 – 5 cm | A single survey area | Medium | [@dosSantos2019; @Lv2023; @Xia2021] |
| Aerial orthophotos | 5 – 30 cm | Country-wide | Medium-high | [@Martins2021; @VelasquezCamacho2023; @Ventura2024] |
| Very-high-res satellite | 30 cm – 1 m | Global | Low–medium | [@AbbasYOLO2025; @VelasquezCamacho2025; @He2022] |
| Sentinel-2 multispectral | 10 m | Global, free | Free | [@He2022; @Xu2025; @Awad2021] |

The very-high-resolution satellite category is the most recent in the literature — most dedicated satellite papers appeared in 2024 and 2025 — and is the only data source practically available for Astana.

## 1.3 Deep-learning paradigms for tree detection

Four model families are relevant to the present work: two-stage detectors (Faster R-CNN, Mask R-CNN), one-stage detectors (YOLO, RetinaNet), the domain-specialised DeepForest detector, and the SAM 2 foundation model for mask refinement. Table 1.2 consolidates the best reported numbers from each family.

**Two-stage detectors.** Faster R-CNN [@FasterRCNN2015] and its instance-segmentation extension Mask R-CNN [@MaskRCNN2017] were the first deep architectures applied to individual tree detection. The most recent contribution in this family is the MCAN architecture of Lv et al. [@Lv2023], which adds a CSPNet backbone and CBAM attention to Mask R-CNN and reports a detection AP of 92.40 % on UAV imagery of a forested campus in Zhejiang, China. The principal disadvantage of the family is **computational cost**: for an interactive web tool, Mask-R-CNN-class instance segmenters are too slow on a laptop GPU. This consideration motivates the choice of a one-stage architecture for the YOLO branch of the present system.

**One-stage detectors: YOLO and RetinaNet.** The YOLO family [@YOLOv1; @UltralyticsYOLO2023] has been the dominant choice for urban-tree detection since 2022. Velasquez-Camacho et al. [@VelasquezCamacho2023] combined ground-level Google Street View with aerial and satellite imagery of Lleida, Spain; their YOLOv5x variant reached an F-score of 84.9 % on the ground-level images, and a markedly lower F-score on the from-above (aerial/satellite) task that is the setting directly comparable to the present work. The most recent benchmark of Abbas and Damaševičius [@AbbasYOLO2025] evaluated every modern YOLO variant — YOLOv8 through YOLOv12 — on a public RGB satellite tree dataset of 3 157 images and reported YOLOv12m as the best performer with mAP@50 = 90.8 %. Sun [@Sun2025] dedicated an entire PhD thesis to YOLOv7/v8 for tree-crown instance segmentation on Wellington, New Zealand aerial imagery, and reported that her improved YOLOv8 variants outperform Mask R-CNN, YOLOv5 and SOLOv2 on both Box AP and Mask AP with fewer parameters. The other relevant one-stage architecture is **RetinaNet** [@RetinaNet2017]: the benchmark by dos Santos et al. [@dosSantos2019] reported a RetinaNet AP of 92.64 %, decisively outperforming YOLOv3 (85.88 %) and Faster R-CNN (82.48 %) on UAV imagery of Campo Grande, Brazil. This empirical result motivated the developers of DeepForest [@DeepForest2019] to choose RetinaNet as their base architecture.

**The DeepForest model.** DeepForest [@DeepForest2019] is a tree-specialised RetinaNet detector packaged as a Python library with pre-trained weights and a stable `predict_tile()` API. The original release was trained on a large semi-supervised dataset derived from the NEON aerial-lidar campaigns over forested sites in the United States, and reported a baseline F1 ≈ 0.65 on held-out NEON scenes. Two empirical results from the literature decisively constrain its use on urban imagery. Ventura et al. [@Ventura2024] tested DeepForest off-the-shelf on 60-cm NAIP imagery of eight Californian cities and reported precision 0.735 but **recall of only 0.294**, for an F-score of 0.42; after fine-tuning on a few hundred urban tiles, the same model recovered an F-score of **0.729**. The Sofia DeepForest work of Dakov and Petrova-Antonova [@SofiaDeepForest2024] provides a complementary data point: trained on 826 manually-annotated trees in Sofia, Bulgaria, DeepForest reached an F1 of 0.674 – 0.685 — comparable to the original NEON benchmark but obtained on what is geographically and architecturally the closest analogue to Astana in the entire surveyed literature. Both works conclude that **DeepForest must be fine-tuned for urban use**.

**Foundation models: SAM and SAM 2.** The newest paradigm in the field is the foundation-model approach exemplified by the Segment Anything Model (SAM) [@SAM2023] and its 2024 successor SAM 2 [@SAM2_2024]. Both expose a prompt-based interface in which the user supplies a point, a bounding box or a coarse mask and the model returns a precise segmentation of the corresponding object. The critical property is **zero-shot generalisation**: the models produce sharp masks even on object categories that were never explicitly labelled at training time, including trees in satellite imagery. SAM 2 specifically can therefore be used as a mask-refinement stage on top of any bounding-box detector — DeepForest in our case — without requiring an additional fine-tune. This idea is the basis of the fourth model branch in Section 2.6.

**Table 1.2 — Best reported results from the surveyed literature, sorted by method family.**

| Method | Work | Year | Data | Best metric |
|---|---|---|---|---|
| Faster R-CNN | dos Santos et al. [@dosSantos2019] | 2019 | UAV RGB, Campo Grande | AP = 82.48 % |
| Mask R-CNN (MCAN) | Lv et al. [@Lv2023] | 2023 | UAV RGB, Zhejiang | Det AP = 92.40 %, Seg AP = 97.70 % |
| RetinaNet | dos Santos et al. [@dosSantos2019] | 2019 | UAV RGB, Campo Grande | AP = 92.64 % |
| YOLOv4-Lite | Zheng and Wu [@Zheng2022] | 2022 | Google Earth 0.27 m | Acc = 96.3 % (campus) |
| YOLOv5x | Velasquez-Camacho et al. [@VelasquezCamacho2023] | 2023 | Ground-level (GSV), Lleida | F1 = 84.9 % |
| YOLOv8 (Wellington) | Sun [@Sun2025] | 2025 | Aerial, Wellington | Best Box and Mask AP vs. Mask R-CNN / YOLOv5 / SOLOv2 |
| YOLOv12m | Abbas and Damaševičius [@AbbasYOLO2025] | 2025 | RGB satellite, 3 157 imgs | **mAP@50 = 90.8 %**, mAP@50:95 = 58.1 % |
| U-Net | Wang et al. [@Wang2021] | 2021 | Aerial 32 cm, Vaihingen | OA = 99.14 %, IoU = 96.38 % |
| DeepLabV3+ | Martins et al. [@Martins2021] | 2021 | Aerial 10 cm, Campo Grande | F1 = 91.4 %, IoU = 73.89 % |
| DeepForest off-the-shelf (urban) | Ventura et al. [@Ventura2024] | 2024 | NAIP 60 cm, 8 CA cities | F = **0.42** (P = 0.74, R = 0.29) |
| DeepForest fine-tuned (urban) | Ventura et al. [@Ventura2024] | 2024 | NAIP 60 cm, 8 CA cities | F = **0.729** |
| DeepForest urban (Sofia) | Dakov and Petrova-Antonova [@SofiaDeepForest2024] | 2024 | Aerial 10 cm, Sofia | F1 = 0.674 – 0.685 |
| Sub-pixel canopy (CASNet) | He et al. [@He2022] | 2022 | Sentinel-2, 34 Chinese cities | OA = 88.6 % |
| Canopy height (MUFCH) | Xu et al. [@Xu2025] | 2025 | Sentinel-2 + OSM, Beijing | MAE = 2.02 m |
| U-Net-DenseNet | He et al. [@He2020] | 2020 | VHR satellite, urban forests | object-based mapping |
| Faster R-CNN (urban RGB) | Zhang et al. [@Zhang2022] | 2022 | High-res RGB, urban | individual-tree detection |
| Double-branch segmenter | Zhang and Liu [@Zhang2024] | 2024 | High-res remote sensing | multi-scale street trees |
| Semantic segmentation | Chen et al. [@Chen2022] | 2022 | High-res, community green | green-space identification |
| Double-branch CNN (multi-temporal) | Chen et al. [@Chen2023] | 2023 | Multi-temporal satellite | canopy mapping |
| DeepLab/U-Net (metropolitan) | Huerta et al. [@Huerta2021] | 2021 | VHR satellite | green-space semantic seg. |
| OB-CNN (change) | Timilsina et al. [@Timilsina2020] | 2020 | Aerial, multi-temporal | tree-cover change mapping |
| Geospatial neural network | Chen et al. [@Chen2021] | 2021 | Satellite, urban | green-space mapping |
| Cascaded CNN | Dong et al. [@Dong2019] | 2019 | Google Earth | single-tree detection |
| **YOLOv8x-seg v4 (this work)** | this thesis | 2026 | Very-high-res sat., Astana M14 | **Box mAP@50 = 0.315, Mask mAP@50 = 0.289** |
| **Mask R-CNN v2+v3 (this work)** | this thesis | 2026 | Very-high-res sat., Astana M14 | Box mAP@50 = 0.166, Mask mAP@50 = 0.158 |
| **DeepForest v3 + SAM 2 (this work)** | this thesis | 2026 | Very-high-res sat., Astana M14 | Box mAP@50 = 0.146, Mask mAP@50 = 0.134 |
| **NEON off-the-shelf on Astana (this work)** | this thesis | 2026 | Very-high-res sat., Astana M14 | Box mAP@50 = **0.012** (first measurement on a Central-Asian city) |

Three observations follow from this table. First, the best reported numbers in the literature are very high — state-of-the-art detection models routinely exceed 90 % mAP at IoU = 0.5. Second, **all the best results are obtained on datasets from the United States, Europe, China, Brazil or New Zealand**: there is no paper in the surveyed corpus that evaluates any of these methods on Central-Asian imagery, no paper that uses imagery of Kazakhstan, and no paper that addresses the specific architectural context of a Soviet-era micro-district city. Third, the gap between off-the-shelf and fine-tuned DeepForest is enormous — 0.42 → 0.73 F1 on the same data after a few hundred annotations [@Ventura2024]. This is the single most important quantitative finding for the present work, because it directly justifies the time investment in building a custom Astana annotated dataset.

## 1.4 The geographic generalisation gap

The pattern — high in-domain performance, large performance drop on out-of-domain imagery — is sometimes called the **geographic generalisation gap**. It is not specific to tree detection: similar drops have been reported in building segmentation, land-cover classification and road extraction whenever a model trained in one country is applied without fine-tuning to imagery of another.

For Astana the magnitude of the gap was unknown at the start of the project, since no published number existed for any deep-learning tree-detection model on Central-Asian satellite imagery. As part of this work the public DeepForest checkpoint (trained on NEON aerial-lidar campaigns of forested sites in the United States) was evaluated out-of-the-box on the Astana M14 validation set described in Section 3.1. The result is a Box mAP@50 of **0.012** — one to two orders of magnitude below any of the fine-tuned configurations reported later, and more than an order of magnitude below the same model's off-the-shelf 0.42 F-score on NAIP USA imagery [@Ventura2024]. This single number is the empirical anchor of the entire diploma project: it quantifies, for the first time on a Central-Asian capital, the gap that motivates the dataset-construction and fine-tuning effort of Chapters 2 and 3.

Four factors are commonly invoked to explain the gap: **floristic differences** (Astana's urban canopy is dominated by *Populus*, elms, birches and apricots, which present visually different crowns from the broadleaf and conifer species typical of North-American NEON sites); **urban-morphology differences** (Soviet-era micro-district planning produces a regular grid of multi-storey apartment blocks separated by narrow strips of green space, qualitatively different from suburban-sprawl U.S. benchmarks and dense historic-city European benchmarks); **resolution and acquisition-geometry mismatch** (satellite look-angles can deviate by up to 30° from nadir, leading to large variations in tree-shadow direction within a single image); and **annotation-policy differences** (per-crown vs. per-cluster labelling conventions vary between datasets). The conclusion is that off-the-shelf models cannot be expected to deliver state-of-the-art numbers on Astana imagery and that some form of domain adaptation — fine-tuning, ensemble or both — is mandatory. This is the principal motivation for the methodology of Chapter 2.

## 1.5 Problem statement

Based on the analysis above, the problem solved in the present work can be stated formally as follows.

**Given:** a colour satellite image of an arbitrary area of Astana at a ground sampling distance of approximately 0.3 – 1 m per pixel, optionally accompanied by geographic metadata (a GeoTIFF affine transform, two or four corner coordinates supplied by the user, or none).

**Produce:** an inventory $I = \{(p_i, c_i, s_i, a_i, l_i, \lambda_i, \phi_i)\}_{i=1}^{N}$ of $N$ trees, where for each tree $i$, $p_i$ is a polygon mask approximating the projected crown, $c_i$ is the bounding box of the polygon in pixel coordinates, $s_i \in [0, 1]$ is a confidence score, $a_i$ is the crown area in pixels and (when geographic conversion is possible) in square metres, $l_i$ is the class label (restricted to the single class "Tree"), and $(\lambda_i, \phi_i)$ are the longitude and latitude of the crown centroid in WGS-84 (defined when geographic conversion is possible).

**Subject to:** per-image inference time of at most 30 seconds on a single laptop GPU of the GeForce RTX 4060 class with 8 GiB of VRAM; Box mAP@50 on a held-out Astana validation set strictly above the public NEON-pretrained baseline of 0.012 (Section 1.4) by at least one order of magnitude, with the explicit aim of being the strongest published number on Astana satellite imagery to date; support for sliding-window tiled inference so that arbitrarily large input images can be processed; support for at least three export formats — GeoJSON, CSV and standalone HTML — to satisfy the interoperability requirement; and compatibility with arbitrary geographic input modes so that the system can be used both with rigorous GeoTIFF deliverables and with informal screenshots supplied by a non-technical user.

Chapter 2 presents the system design that meets these requirements, and Chapter 3 reports its experimental evaluation against the literature baselines of Table 1.2.

\newpage
