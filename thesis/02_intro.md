# Introduction

## Relevance of the topic

Urban green space — and particularly the urban tree canopy — is a critical asset for any modern city. Trees moderate microclimate, sequester carbon, mitigate the urban heat island effect, improve air quality and contribute to the psychological well-being of residents. For Astana, a city located in a continental steppe climate with extreme winter and summer temperatures, the value of every tree is amplified: the municipal *Zelenstroy* service is responsible for the inventory, planting and maintenance of every street tree, park area and protective forest belt around the capital.

The fundamental input required by any urban-forestry decision — where to plant, where to remove diseased specimens, how to compute green-coverage indicators for new districts — is an accurate, up-to-date inventory of existing trees. The traditional method of obtaining this inventory is a manual field survey: a specialist physically walks each district and records every tree by hand or with a hand-held GPS device. For a city of Astana's size and pace of growth this approach is intractable. Comparable surveys for cities of similar size are reported to take from one to three years to complete and require substantial labour resources [@SofiaDeepForest2024]. By the time a manually-collected inventory is finished, large portions of it are already outdated because of new construction, urban renewal projects and the natural turnover of green plantings.

At the same time, high-resolution satellite and aerial imagery covering Astana is freely or inexpensively available from sources such as Google Earth, ESRI World Imagery, the European Sentinel-2 mission and commercial vendors. In the last six years deep-learning-based object detection and instance segmentation have reached a level of maturity that allows the automatic localisation of individual tree crowns directly from such imagery, in a fraction of the time and cost of any manual survey [@DeepForest2019; @AbbasYOLO2025; @LvMCAN2023].

The relevance of this diploma project therefore stems from two converging factors:

1. The practical need of *Zelenstroy* and other municipal services for a fast, repeatable and inexpensive way to obtain an inventory of Astana's urban trees.
2. The fact that modern deep-learning methods — when properly adapted to the specific geographic, climatic and architectural context of a Central-Asian city — are capable of producing such an inventory automatically from satellite imagery.

The combination of these factors defines the topic of this work as both timely and technologically feasible.

## Object and subject of research

The **object of the research** is the urban green space of the city of Astana — namely the population of individual trees observable from above in the visible spectrum.

The **subject of the research** is the set of deep-learning models and software components capable of detecting these trees in satellite imagery, segmenting their crowns, converting pixel coordinates into geographic coordinates and presenting the result in a form usable by municipal experts.

## Aim and objectives

The **aim** of the work is to design, implement and evaluate an end-to-end software system that, given a satellite image of an area of Astana as input, automatically produces an inventory of the trees present in that area, with per-tree geographic coordinates, crown geometry and a confidence score.

To achieve this aim the following **objectives** are set:

1. Survey the current state of deep-learning-based methods for individual tree detection and instance segmentation in remote-sensing imagery, with particular attention to applications in urban environments.
2. Identify the research gap with respect to Central-Asian cities and quantify the performance of off-the-shelf pre-trained models on Astana imagery.
3. Build a custom annotated dataset of Astana satellite imagery, suitable for training instance-segmentation models in the urban domain.
4. Train and fine-tune three complementary detection models (an instance-segmentation YOLOv8 network, a DeepForest RetinaNet detector and a SAM-based mask-refinement stage), and combine their outputs through a model-ensemble strategy.
5. Design and implement a complete software pipeline — backend, frontend, geographic conversion and export — exposing the trained models through a usable web interface and producing GeoJSON / CSV / standalone-HTML deliverables.
6. Evaluate the resulting system in terms of detection quality, computational cost and practical applicability, and compare the three models against each other and against published baselines.

## Methods of research

The work draws on a combination of theoretical and experimental methods:

- **Literature analysis** of peer-reviewed publications from journals such as *Remote Sensing*, *ISPRS Annals*, *Computers, Environment and Urban Systems* and conference proceedings of 2019–2025, in order to characterise the state of the art.
- **Deep-learning model training and fine-tuning** based on the PyTorch and Ultralytics frameworks, the DeepForest library and the SAM (Segment Anything) foundation model.
- **Dataset engineering**: collection of source satellite tiles, manual annotation in the CVAT tool, conversion between COCO and YOLO label formats, tiling of high-resolution images into overlapping patches and automatic pre-labelling with an iterative model-in-the-loop workflow.
- **Software engineering** in Python (FastAPI, Pydantic) and JavaScript (React 18, Leaflet) following the adapter pattern for model integration.
- **Quantitative evaluation** using standard object-detection metrics (Box mAP@50, Box mAP@50-95, Mask mAP@50, Precision, Recall) over an independent validation tile set.
- **Qualitative spot-checking** of model predictions over real Astana scenes — yards, streets, dense residential blocks, parks — to characterise model behaviour beyond what the aggregate metrics can capture.

## Scientific novelty

The scientific novelty of the work consists in the following:

- It is, to the best of the authors' knowledge, the first published evaluation of state-of-the-art deep-learning tree-detection models on satellite imagery of Astana and, more generally, of any major Central-Asian capital.
- The work proposes a hybrid three-model architecture (YOLOv8 instance segmentation; DeepForest fine-tuned on local data; SAM zero-shot mask refinement of DeepForest bounding boxes), combined through a Weighted-Box-Fusion ensemble, and quantitatively compares the three branches on a single dataset.
- It introduces a small but novel annotated dataset of Astana satellite imagery (≈ 77 source images, ≈ 5 000 polygon-level tree annotations after tiling) which is reusable for future research in the region.
- It contributes a complete reusable software template — a FastAPI backend with a pluggable model-adapter interface and a Leaflet-based frontend with in-browser tile capture from ESRI World Imagery — that turns research models into a deployable internal tool.

## Practical significance

The resulting system is directly applicable to the day-to-day work of Astana's *Zelenstroy* and of any city administration concerned with the management of urban green spaces. The user supplies a satellite image (or selects an area on an interactive map) and receives in seconds a list of detected trees with geographic coordinates, crown geometry, an interactive Leaflet visualisation and standard GIS-compatible exports (GeoJSON for QGIS, CSV for spreadsheet analysis, standalone HTML for sharing). The architecture is designed to be retrained on imagery from any other city by replacing the dataset, without changes to the pipeline.

## Structure of the work

The diploma project consists of an introduction, three chapters, a conclusion, a list of references and a set of appendices.

**Chapter 1** analyses the subject area of urban-tree detection from remote-sensing data, reviews the relevant literature of 2019–2025, identifies the research gap with respect to Central-Asian cities, and formulates the precise problem statement that the rest of the work attempts to solve.

**Chapter 2** describes the proposed methodology: the overall system architecture, the three deep-learning models (YOLOv8-seg, DeepForest, SAM), the data-preparation pipeline (annotation, tiling, augmentation), the ensemble strategy and the geographic-conversion and export components.

**Chapter 3** documents the experiments and reports the quantitative and qualitative results — training curves and validation metrics for each model, qualitative inspection of predictions, an ablation comparing the three models on the same data, the integrated pipeline as deployed in the prototype and the limitations of the current implementation.

The conclusion summarises the results obtained, evaluates the achievement of each of the six objectives stated above, and outlines directions for future work.

\newpage
