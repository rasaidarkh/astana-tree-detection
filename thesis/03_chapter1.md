# Chapter 1. Analysis of the subject area and problem statement

This chapter establishes the technical and scientific context of the work. Section 1.1 describes urban-tree inventory as an applied problem and lists the practical requirements it imposes on any automated solution. Section 1.2 surveys the remote-sensing data sources available for the task. Section 1.3 organises the relevant deep-learning literature of 2019 – 2025 into five technical paradigms — two-stage detectors, one-stage detectors, semantic-segmentation networks, the domain-specialised DeepForest model and the foundation Segment-Anything model — and summarises the most relevant results. Section 1.4 consolidates the reported quantitative comparisons into a single table. Section 1.5 identifies the **geographic generalisation gap** as the dominant obstacle to applying off-the-shelf models in Astana. Section 1.6 formulates the precise problem statement that the rest of the work will solve.

## 1.1 Urban-tree inventory as an applied problem

Modern municipalities increasingly rely on quantitative inventories of urban green space as inputs to their planning, environmental-monitoring and climate-adaptation processes. The basic unit of such inventories is the individual tree, characterised by a geographic position, a crown size and — when available — a species label and a health status. Aggregations of individual trees yield the higher-order indicators that municipal agencies actually consume: number of trees per district, canopy-cover percentage, average crown area, density of green corridors and so on.

Historically, urban tree inventories were obtained through **manual field surveys** conducted by a team of foresters with hand-held GPS receivers, paper forms and a fixed budget per district. Reports from comparable cities — for example, the multi-year tree-inventory effort of the city of Sofia, Bulgaria [@Dakov2024] or the multi-month effort of Pasadena, California [@Branson2019] — show that for any large city the manual approach is slow, expensive and quickly becomes obsolete because of the rapid turnover of urban green plantings.

Three classes of alternatives have emerged in the last decade. The first relies on **airborne LiDAR**, which delivers a centimetre-precision three-dimensional reconstruction of the canopy and is the gold standard wherever it is available [@Schmohl2022]. LiDAR campaigns, however, are exceptionally expensive and are usually flown only on a five- or ten-year cadence by national mapping agencies; they are also unsuitable for cities that do not appear in the public LiDAR programmes of their respective countries — a category that includes Astana.

The second class relies on **high-resolution aerial or UAV imagery**, typically at a ground sampling distance of 5 – 30 cm per pixel. Such imagery offers an excellent balance between detail and acquisition cost and has been the dominant data source for urban-tree research, including foundational benchmarks on Campo Grande, Brazil [@Martins2021; @dosSantos2019] and the recent multi-city work on Lleida, Spain [@VelasquezCamacho2023].

The third class — the one chosen for the present project — relies on **freely or inexpensively available very-high-resolution satellite imagery**, accessed either through commercial vendors (Maxar, Airbus) or through aggregators such as ESRI World Imagery and Google Earth. The ground-sampling distance of such imagery is typically 30 cm – 1 m in cities, with the lowest pricing tier (Sentinel-2) at 10 m. This source is the only one practically available for Astana at the scale required by the city, and the body of literature on its use for tree detection — though still small compared to the UAV literature — has grown substantially since 2019 [@He2022; @VelasquezCamacho2025; @AbbasYOLO2025].

The practical requirements imposed by the *Zelenstroy* end-user on any automated inventory system can be summarised as follows: (i) accept satellite imagery from a heterogeneous set of sources without re-training; (ii) output a per-tree polygon mask rather than a single pixel, so that crown area can be computed; (iii) convert pixel coordinates into geographic coordinates in WGS-84; (iv) export the inventory in formats consumable by QGIS, ArcGIS and Excel; (v) provide a confidence score for each detection so that human reviewers can prioritise their inspection time. The methodological choices documented in Chapter 2 are direct consequences of these five requirements.

## 1.2 Remote-sensing data sources for tree detection

The deep-learning literature on tree detection is highly **data-source-dependent**: a model that excels on 5-cm UAV imagery typically fails when applied directly to 1-m satellite imagery, and vice versa. The body of work surveyed in this chapter can therefore be organised along the dimension of the data source, summarised in Table 1.1.

**Table 1.1 — Data sources reported in the surveyed literature.**

| Source | Typical GSD | Coverage | Cost | Representative works |
|---|---|---|---|---|
| Airborne LiDAR | 5 – 50 cm 3-D | National / regional campaigns | Very high | [@Schmohl2022; @DeepForest2019] |
| UAV / drone RGB | 1 – 5 cm | A single survey area | Medium | [@dosSantos2019; @Lv2023; @Xia2021] |
| Aerial orthophotos | 5 – 30 cm | Country-wide | Medium-high | [@Martins2021; @VelasquezCamacho2023; @Ventura2024] |
| Very-high-res satellite | 30 cm – 1 m | Global | Low–medium | [@AbbasYOLO2025; @VelasquezCamacho2025; @He2022] |
| Sentinel-2 multispectral | 10 m | Global, free | Free | [@He2022; @Xu2025; @Awad2021] |

Two observations from this table directly inform the present work. First, the **very-high-resolution satellite** category — the only data source practically available for Astana — is also the most recent: most of the dedicated satellite-imagery papers appeared in 2024 and 2025, suggesting that the area is now technically mature but the literature is still sparse. Second, the **Sentinel-2** category, despite its very coarse 10-m resolution, has been shown to yield usable urban-tree-cover products through deep-learning super-resolution and sub-pixel mapping techniques [@He2022; @Awad2021]; this is an important option for the future-work extension of the present project.

## 1.3 Deep-learning paradigms for tree detection

This section surveys the relevant deep-learning literature, organised by the technical paradigm of the model rather than by year. Five paradigms are identified and each is illustrated by the most-cited works in the field.

### 1.3.1 Two-stage detectors

Two-stage detectors — the family of Faster R-CNN [@FasterRCNN2015] and its extensions Mask R-CNN [@MaskRCNN2017] and Cascade R-CNN — were the first deep-learning architectures to be applied to individual tree detection at large scale. The general approach is to first generate a small set of class-agnostic region proposals through a Region Proposal Network, then to classify and regress each proposal independently.

For the urban-tree task the most-cited contribution is the early benchmark of dos Santos et al. [@dosSantos2019] on the *Dipteryx alata* species in Campo Grande, Brazil: on 392 UAV RGB images at 0.82 cm GSD, Faster R-CNN achieved an average precision of 82.48 %, outperformed by both YOLOv3 and — significantly — by RetinaNet (see Section 1.3.2). Subsequent works extended the Faster R-CNN backbone with a Swin Transformer [@Zhang2022] and reported improvements over the ResNet-50 baseline on the same Campo Grande dataset. Lv et al. [@Lv2023] proposed an extended Mask R-CNN — the MCAN architecture, with a CSPNet backbone and CBAM attention — and reported a detection average precision of 92.40 % and a segmentation average precision of 97.70 % on UAV imagery of a forested university campus in Zhejiang, China; the same paper reported only 79.87 % for YOLOv5 on the same data, leading to a recurring debate about the relative merits of one-stage and two-stage detectors that is revisited in Section 1.4.

The principal disadvantage of the two-stage family is its **computational cost**. For an interactive web tool the inference speed of a Faster-R-CNN-class model is too slow on a laptop GPU, and Mask R-CNN-class instance segmenters are slower still. This consideration motivated the choice of a one-stage architecture for the YOLO branch of the present system.

### 1.3.2 One-stage detectors: YOLO and RetinaNet

The YOLO family of object detectors [@YOLOv1; @UltralyticsYOLO2023] has been the dominant choice for urban-tree detection since 2022. Velasquez-Camacho et al. [@VelasquezCamacho2023] tested YOLOv5 in five sizes against DeepForest and Faster R-CNN on aerial and satellite imagery of Lleida, Spain, and reported an F-score of 84.9 % for the YOLOv5x variant — substantially better than Faster R-CNN's 35.6 %. The same authors extended their pipeline to multi-temporal satellite imagery of the San Francisco Bay Area in [@VelasquezCamacho2025], reporting a precision of 90 % and a recall of 81 % for street-tree detection across more than fifteen years of NAIP imagery.

Zheng and Wu [@Zheng2022] proposed YOLOv4-Lite with a MobileNetv3 backbone for tree detection in Google Earth imagery (GSD 0.27 m) and reported an accuracy of 96.3 % on a campus scene, outperforming traditional watershed-based and template-matching approaches by 26 – 46 percentage points; this work is one of the few that demonstrates the feasibility of YOLO-class detectors on the same source of imagery (Google Earth) used in the present project.

The most recent benchmark is the work of Abbas and Damaševičius [@AbbasYOLO2025] of Kaunas University of Technology, Lithuania. They evaluated every modern YOLO variant — YOLOv8 through YOLOv12 in the n, s and m sizes — on a public RGB satellite tree dataset of 3 157 images, and reported that **YOLOv12m** achieves the best performance with a mean average precision at IoU = 0.5 of **90.8 %**, a mAP@50:95 of 58.1 %, an F1-score of 84.7 % and precision and recall of approximately 85 %. The YOLOv8 variants are competitive in performance and consume less inference time, motivating the choice of YOLOv8x-seg as a reasonable engineering compromise for the present project.

Sun [@Sun2025] dedicated an entire PhD thesis to the application of YOLOv7/v8 to individual tree-crown instance segmentation on Wellington, New Zealand aerial imagery; she proposed three improved variants (YOLO-ITC, YOLOv8E and YOLOv8-FF) that outperform Mask R-CNN, YOLOv7, YOLOv5 and SOLOv2 on both Box AP and Mask AP with fewer parameters. The thesis is the strongest direct precedent for the YOLO branch of the present work.

The other relevant one-stage architecture is **RetinaNet** [@RetinaNet2017], which is best known for its **focal loss** formulation that down-weights the contribution of easy background examples and is therefore well-suited to the highly imbalanced foreground-background ratio of dense forest scenes. The earlier-mentioned benchmark by dos Santos et al. [@dosSantos2019] reported a RetinaNet average precision of **92.64 %**, decisively outperforming YOLOv3 (85.88 %) and Faster R-CNN (82.48 %). It is precisely this empirical result that motivated the developers of DeepForest [@DeepForest2019] to choose RetinaNet as their base architecture (Section 1.3.4).

### 1.3.3 Semantic-segmentation networks

A second, parallel line of work treats tree detection as a **semantic-segmentation** problem in which every pixel is classified independently as either *tree-canopy* or *background*. Individual trees are then recovered as connected components of the resulting binary mask, with optional post-processing through morphological opening or local-maximum filtering.

The dominant architecture in this family is the U-Net [@UNet2015], originally developed for biomedical segmentation but quickly adopted by the remote-sensing community. Wang et al. [@Wang2021] evaluated U-Net at four input scales (16, 32, 64 and 100 cm GSD) on the ISPRS Vaihingen aerial-orthophoto benchmark and reported a best overall accuracy of **99.14 %** and an intersection-over-union of **96.38 %** at 32 cm — among the highest segmentation accuracies reported in the urban-tree literature. Martins et al. [@Martins2021] performed a similar five-architecture comparison on Campo Grande and concluded that DeepLabV3+ marginally outperforms U-Net, FCN, SegNet and DDCN, with a final pixel accuracy of 96.18 % and an IoU of 73.89 % — substantially lower than Wang et al.'s number because the Campo Grande dataset covers a more diverse Cerrado biome.

Several extensions of U-Net for urban contexts are worth mentioning: the OUDN architecture of He et al. [@He2020] which couples U-Net with a DenseNet feature extractor and operates on WorldView-3 imagery; the HRNet-based segmenter of Chen et al. [@Chen2022] applied to four districts of Shanghai with an F1-score of 84.33 %; and the multi-temporal double-branch U-Net of Chen et al. [@Chen2023] applied to GaoFen-2 imagery over Beijing. The double-branch network of Zhang and Liu [@Zhang2024] combines RGB and NDVI features in a CNN-Transformer hybrid and reports a 1.13 % mIoU improvement over plain U-Net baselines on a custom street-tree dataset.

The semantic-segmentation approach is attractive when only canopy-cover statistics are required, but it does **not** produce per-tree instance identities — a critical requirement of the present project — and is therefore not selected as the primary modelling approach. Semantic segmentation appears in the present work only as a sanity-check baseline.

### 1.3.4 The domain-specialised DeepForest model

DeepForest [@DeepForest2019] is a tree-specialised RetinaNet detector packaged as a Python library with pre-trained weights and a stable API. The original release was trained on a large semi-supervised dataset derived from the National Ecological Observatory Network (NEON) lidar campaigns over forested sites in the United States, and reported a baseline F1-score of approximately 0.65 on held-out NEON scenes.

Because DeepForest is shipped with a permissive licence, pre-trained weights and a one-line `predict_tile()` API, it has become the de-facto standard baseline for any new tree-detection work. The relevant question for the present project is therefore not whether to use DeepForest at all but whether and how to **fine-tune** it for the Astana domain.

Two empirical results from the literature decisively answer this question. Ventura et al. [@Ventura2024] tested DeepForest off-the-shelf on 60-cm NAIP imagery of eight Californian cities and reported a precision of 0.735 but a **recall of only 0.294**, for a final F-score of 0.42 — substantially below the 0.65 reported on the original NEON benchmark and clearly insufficient for production use. After fine-tuning DeepForest on a few hundred urban tiles the same authors recovered an F-score of **0.729**, virtually identical to their bespoke HR-SFANet model. The conclusion is unambiguous: **DeepForest must be fine-tuned to be useful on urban imagery**.

The Sofia DeepForest work of Dakov and Petrova-Antonova [@SofiaDeepForest2024] provides a complementary data point. They trained DeepForest on 826 manually-annotated trees and 98 cluster polygons in the Lozenets district of Sofia, Bulgaria, and reported a single-tree F1 between 0.674 and 0.685 — comparable to the original NEON benchmark but obtained on what is, geographically and architecturally, the closest analogue to Astana found in the entire literature surveyed. Critically, the authors explicitly write that DeepForest must be "retrained for urban use because of obstructions by buildings, shade, and irregularly planted trees" — exactly the conditions present in Astana, where Soviet-era micro-district planning produces a highly heterogeneous urban texture.

These two data points — Ventura et al. and Dakov and Petrova-Antonova — establish the empirical foundation for the DeepForest branch of the present system and define the performance target: a fine-tuned DeepForest F1 in the 0.65 – 0.73 range, which would match the state of the art for European urban environments.

### 1.3.5 Foundation models: Segment Anything Model 2

The newest paradigm in the field is the **foundation-model** approach, whose most relevant representative for the present work is the Segment Anything Model 2 (SAM 2) [@SAM2024], released by Meta AI in 2024 as a successor to the original SAM [@SAM2023]. SAM 2 was trained on a large-scale dataset of over one billion masks and exposes a *prompt-based* interface in which the user supplies a point, a bounding box or a coarse mask, and the model returns a precise segmentation of the corresponding object. Compared to SAM v1, SAM 2 introduces a streaming memory module that improves mask consistency and zero-shot generalisation on out-of-domain imagery — a property that is particularly valuable for satellite scenes of Astana, which bear no resemblance to the web images and photographs that dominate the training distribution.

The critical property of SAM 2 is its **zero-shot generalisation**: because the model was trained on such a diverse dataset, it produces sharp masks even on object categories that were never explicitly labelled at training time, including trees in satellite imagery. The implication is that SAM 2 can be used as a **mask-refinement stage** on top of any bounding-box detector — DeepForest in our case — without requiring an additional fine-tune. This idea is explored in detail in Section 2.6 of the methodology chapter.

Although no SAM 2 paper specifically addressing tree detection is present in the surveyed corpus, the model's strong zero-shot performance on remote-sensing objects and its straightforward batch-inference API make its inclusion in the present project a low-risk, high-value design choice.

## 1.4 Quantitative results reported in the literature

Table 1.2 consolidates the best per-method results reported by the works surveyed above. It is the empirical basis on which the present project's architectural choices are made, and the target against which the experimental results of Chapter 3 will be compared.

**Table 1.2 — Best reported results from the surveyed literature, sorted by method family.**

| Method | Work | Year | Data | Best metric |
|---|---|---|---|---|
| Faster R-CNN | dos Santos et al. [@dosSantos2019] | 2019 | UAV RGB, Campo Grande | AP = 82.48 % |
| Mask R-CNN (MCAN) | Lv et al. [@Lv2023] | 2023 | UAV RGB, Zhejiang | Det AP = 92.40 %, Seg AP = 97.70 % |
| RetinaNet | dos Santos et al. [@dosSantos2019] | 2019 | UAV RGB, Campo Grande | AP = 92.64 % |
| YOLOv3 | dos Santos et al. [@dosSantos2019] | 2019 | UAV RGB, Campo Grande | AP = 85.88 % |
| YOLOv4-Lite | Zheng and Wu [@Zheng2022] | 2022 | Google Earth 0.27 m | Acc = 96.3 % (campus) |
| YOLOv5x | Velasquez-Camacho et al. [@VelasquezCamacho2023] | 2023 | Aerial + sat., Lleida | F1 = 84.9 % |
| YOLOv8 (Wellington) | Sun [@Sun2025] | 2025 | Aerial, Wellington | Best Box AP and Mask AP among Mask R-CNN / YOLOv5 / SOLOv2 |
| YOLOv12m | Abbas and Damaševičius [@AbbasYOLO2025] | 2025 | RGB satellite, 3 157 imgs | mAP@50 = **90.8 %**, mAP@50:95 = 58.1 % |
| U-Net | Wang et al. [@Wang2021] | 2021 | Aerial 32 cm, Vaihingen | OA = 99.14 %, IoU = 96.38 % |
| DeepLabV3+ | Martins et al. [@Martins2021] | 2021 | Aerial 10 cm, Campo Grande | F1 = 91.4 %, IoU = 73.89 % |
| DeepForest off-the-shelf (urban) | Ventura et al. [@Ventura2024] | 2024 | NAIP 60 cm, 8 CA cities | F = **0.42** (P = 0.74, R = 0.29) |
| DeepForest fine-tuned (urban) | Ventura et al. [@Ventura2024] | 2024 | NAIP 60 cm, 8 CA cities | F = **0.729** |
| DeepForest urban (Sofia) | Dakov and Petrova-Antonova [@SofiaDeepForest2024] | 2024 | Aerial 10 cm, Sofia | F1 = 0.674 – 0.685 |
| DeepForest urban (Lleida) | Velasquez-Camacho et al. [@VelasquezCamacho2023] | 2023 | Aerial + sat., Lleida | F = 78.0 % |
| CASNet sub-pixel | He et al. [@He2022] | 2022 | Sentinel-2, 34 Chinese cities | OA = 88.6 % |
| MUFCH (canopy height) | Xu et al. [@Xu2025] | 2025 | Sentinel-2 + OSM, Beijing | MAE = 2.02 m |

Three observations follow from this table.

First, **the best reported numbers are very high**. State-of-the-art detection models routinely exceed 90 % mean average precision at IoU = 0.5 and 95 % accuracy in semantic segmentation. The bar for any new work is therefore high in absolute terms.

Second, **all the best results are obtained on datasets from the United States, Europe, China, Brazil or New Zealand**. There is **no** paper in the surveyed corpus that evaluates any of these methods on Central-Asian imagery, no paper that uses imagery of Kazakhstan, and no paper that addresses the specific architectural and floristic context of a Soviet-era micro-district city.

Third, the **gap between off-the-shelf and fine-tuned DeepForest is enormous**: 0.42 to 0.73 F1 on the same data after a few hundred annotations [@Ventura2024]. This is the single most important quantitative finding of the entire literature for the present work, because it directly justifies the time investment that the team spent on building a custom Astana annotated dataset.

## 1.5 The geographic generalisation gap

The pattern observed in the literature — high in-domain performance, large performance drop on out-of-domain imagery — is sometimes called the **geographic generalisation gap**. It is not specific to tree detection: similar drops have been reported in building segmentation, land-cover classification and road extraction whenever a model trained in one country is applied without fine-tuning to imagery of another country.

Four factors are commonly invoked to explain the gap:

1. **Floristic differences.** Models trained on American oaks, maples and pines learn crown silhouettes that do not match the poplars, elms, birches and apricots that dominate Astana's urban canopy. The DeepForest authors themselves acknowledge that the model's NEON pre-training contains "mostly broadleaf and conifer species typical of North-American temperate forests" [@DeepForest2019]; Kazakh urban arboriculture is dominated by *Populus* species which present a tall, narrow, sparse crown that is visually very different from the dense round broadleaf canopies in the NEON data.

2. **Urban-morphology differences.** Soviet-era micro-district planning produces a regular grid of multi-storey apartment blocks separated by narrow strips of green space, with row-planted street trees along arterial avenues. This morphology is qualitatively different from the suburban-sprawl morphology that dominates the U.S. urban-tree benchmarks and from the dense historic-city morphology of European benchmarks. Shadow patterns, building-cast occlusion and ground-cover composition are therefore all different.

3. **Resolution and acquisition-geometry mismatch.** Cards trained on UAV imagery with a vertical optical axis cannot transfer trivially to satellite imagery whose look-angle can deviate by up to 30° from nadir. The Astana acquisitions from ESRI World Imagery and Google Earth are made with a variable look-angle and at a variable time of year, leading to large variations in tree-shadow direction and length within a single image.

4. **Annotation-policy differences.** Even when the model and the imagery are nominally compatible, the labelling convention can differ between datasets: some annotators label every visible tree crown regardless of whether the trunk is visible, others label only trees that can be uniquely traced to a trunk, others still label clusters of trees as a single polygon. The Astana annotation in the present work follows the per-crown convention used by the original DeepForest dataset.

The conclusion of this section is that an off-the-shelf model cannot be expected to deliver state-of-the-art numbers on Astana imagery and that some form of domain adaptation — fine-tuning, ensemble or both — is mandatory. This conclusion is the principal motivation for the methodology of Chapter 2.

## 1.6 Problem statement

Based on the analysis above, the problem solved in the present work can be stated formally as follows.

**Given:** a colour satellite image of an arbitrary area of Astana at a ground sampling distance of approximately 0.3 – 1 m per pixel, optionally accompanied by geographic metadata (a GeoTIFF affine transform, or two/four corner coordinates supplied by the user, or none).

**Produce:** an inventory $I = \{(p_i, c_i, s_i, a_i, l_i, \lambda_i, \phi_i)\}_{i=1}^N$ of $N$ trees, where for each tree $i$:

- $p_i$ is a polygon mask in pixel coordinates approximating the projected crown;
- $c_i$ is the bounding box of the polygon in pixel coordinates;
- $s_i \in [0, 1]$ is a confidence score;
- $a_i$ is the crown area in pixels and (when geographic conversion is possible) in square metres;
- $l_i$ is the class label, restricted in the present work to the single class "Tree";
- $(\lambda_i, \phi_i)$ are the longitude and latitude of the crown centroid, in WGS-84, defined when geographic conversion is possible.

**Subject to:**

- per-image inference time of at most 30 seconds on a single laptop GPU of the GeForce RTX 4060 class with 8 GiB of VRAM;
- a Box mAP@50 on a held-out Astana validation set of at least 0.45 — matching the YOLO v1 baseline reported in Chapter 3 — with the implicit target of approaching the 0.65 – 0.73 range demonstrated for fine-tuned DeepForest in European urban data;
- support for sliding-window tiled inference so that arbitrarily large input images can be processed;
- support for at least three export formats — GeoJSON, CSV and standalone HTML — to satisfy the interoperability requirement of Section 1.1;
- compatibility with arbitrary geographic input modes so that the system can be used both with rigorous GeoTIFF deliverables and with informal screenshots supplied by a non-technical user.

The remainder of the work is devoted to the design of a system that meets these requirements (Chapter 2) and to its experimental evaluation against the literature baselines summarised in Table 1.2 (Chapter 3).

\newpage
