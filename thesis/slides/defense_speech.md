# Defense speech — Automated Tree Recognition & Green-Space Mapping (Astana)

**Language:** English · **Target:** ~10–11 min (pre-defense 8–10 min — trim slides 6, 14, 18 if needed)
**Speakers:** `[R]` Rasul (system + YOLO + web app) · `[B]` Berik (Mask R-CNN) · `[A]` Anuar (DeepForest + SAM 2)
**Note:** follows the current working 22-slide arc — provisional, easy to re-map. Competitor numbers on slides 12/13/15 are owned by Berik & Anuar — confirm their final figures before the talk.

---

### Slide 1 · Title — `[R]` ~20s
Good morning. Our project is an automated system that detects trees and maps urban green space in Astana, directly from satellite imagery. I'm Rasul Aidarkhanov; with me are Berik Sharipov and Anuar Totin; our supervisor is Syndar Satbayev. I'll cover the overall system and the YOLO branch; Berik and Anuar will present their models.

### Slide 2 · Problem & relevance — `[R]` ~30s
Astana is growing fast, and the city needs an up-to-date inventory of its trees — for green-cover policy, irrigation, and heat mitigation. Counting trees by hand from imagery doesn't scale. Strong detection models exist in the literature, but almost all are trained and tested on North-American or European data. Whether they work on a Central-Asian city was simply unknown. That gap is our motivation.

### Slide 3 · The gap & our aim — `[R]` ~30s
Here is that gap in one number. A state-of-the-art off-the-shelf tree detector — NEON DeepForest — scores just **0.012** on Astana imagery. Essentially blind. So our aim is precise: build and validate a deep-learning system that reliably detects trees on Astana satellite imagery, and deploy it as a usable tool. Our best model reaches **0.315** — a 140% jump — and the rest of the talk explains how.

### Slide 4 · Objectives & methods — `[R]` ~30s
We set five objectives: build an annotated Astana dataset; train and compare four model families — YOLOv8 segmentation, Mask R-CNN, DeepForest, and SAM 2; combine them through ensembles; evaluate everything on one common set; and deliver a working web application. The methods are supervised deep learning with transfer learning and sliding-window tiling.

### Slide 5 · Literature review — `[R]` ~30s
We reviewed 31 peer-reviewed papers from 2019 to 2025. This table summarises the strongest — Mask R-CNN, YOLO variants, DeepLab and U-Net, DeepForest, and SAM. They report high accuracy, but always on their own regions and sensors. The last row is ours: the first measurement of these models on Central-Asian satellite imagery.

### Slide 6 · Mathematical foundation — `[R]` ~30s *(cut if short on time)*
The models rest on convolutional feature extraction and two core ideas: anchor-free detection with a combined objectness, classification, and IoU-based box loss for YOLO; and region proposals with a mask head for Mask R-CNN. We score everything with mean Average Precision at IoU 0.5 — mAP@50 — for boxes and masks. One metric makes every model comparable.

### Slide 7 · System architecture — `[R]` ~30s
The system has three layers: a React and Leaflet front end where the user picks an area; a FastAPI back end that tiles the image, runs inference, and merges detections; and a SQLite store for captures and tree polygons. The same trained weights serve every request. We call the application **Canopy**.

### Slide 8 · Data & hardware — `[R]` ~25s
All training imagery is high-zoom satellite capture of Astana, around 0.3-metre resolution. Models were trained on a single RTX 4060 with 8 gigabytes — deliberately modest hardware, to show the system is reproducible without a cluster. The database schema is simple: scans, snapshots, and detections.

### Slide 9 · Dataset — `[R]` ~25s
We annotated trees across tiled Astana imagery and held out a validation split. Because trees touch and overlap, we annotate instance polygons, not just boxes. The common validation set — **14 images, 702 labelled trees** — is what every model in this project is scored on.

### Slide 10 · Four branches — `[R]` ~15s
From here the project splits into four model branches that we compare head-to-head. I'll take the YOLO branch; Berik and Anuar will present theirs.

### Slide 11 · YOLO + ablation — `[R]` ~45s  ← *core of my part*
YOLOv8x-segmentation is our strongest single model. We ran more than twenty experiments — fine-tuning from COCO, image size, augmentation, learning-rate schedules. Three lessons mattered: start from pretrained weights; tile at the resolution you deploy at; and keep augmentation moderate. The score climbed from **0.131** at version one to **0.315** at version four — Box mAP@50 — a **140% improvement**, and the strongest result in the project.
> *Handoff:* "I'll hand over to Berik for Mask R-CNN."

### Slide 12 · Mask R-CNN — `[B]` ~35s
*(Berik)* Trained on the same dataset and scored on the same 14-image set; reaches roughly **0.166** Box mAP@50 — clean masks, but below YOLO on this data.

### Slide 13 · DeepForest + SAM 2 — `[A]` ~35s
*(Anuar)* Off-the-shelf DeepForest scored 0.012; after fine-tuning and adding SAM 2 masks it reaches about **0.146**. Strong general-purpose masks, weaker on small urban trees.
> *Handoff back to Rasul.*

### Slide 14 · Ensembles — `[R]` ~30s *(cut if short on time)*
We then asked whether combining models beats the best single one. We tested two ensembles — weighted box fusion across families, and a cross-YOLO vote. They improve robustness on hard tiles, but on our validation set the single YOLO champion stays on top. An honest, useful finding.

### Slide 15 · Results — `[R]` ~35s
This is the head-to-head on the common 14-image set. YOLOv8x-seg leads at **0.315**, ahead of Mask R-CNN and the DeepForest–SAM 2 branch, ensembles close behind. Every model is scored identically, so the comparison is fair. To our knowledge, these are the first such numbers for Astana.

### Slide 16 · Qualitative — `[R]` ~25s
Beyond the numbers — here's what it looks like. The detections track real tree crowns closely, even in dense rows and mixed urban texture. The typical misses are tiny or heavily shaded trees.

### Slide 17 · Web application — `[R]` ~40s  ← *core of my part*
Finally, the deployed system. In Canopy the user selects an area of Astana; the back end tiles it, runs YOLO, and paints every detected tree on the map within seconds. It already indexes **over 51,000 trees** across the city, with confidence filtering, polygon and heat-map layers, and per-scan history. This turns a research model into a tool a city planner can actually use.

### Slide 18 · Discussion — `[R]` ~25s *(cut if short on time)*
Two honest limitations: accuracy is bounded by training-set size, and there is a domain shift between our training captures and live map tiles. Both are addressable — more annotation, and training on matched imagery.

### Slide 19 · Conclusions — `[R]` ~25s
To conclude, we met every objective: a built dataset, four compared models, ensembles, one common evaluation, and a working application. The headline: from **0.012** off-the-shelf to **0.315** — a deployable tree-detection system for Astana.

### Slide 20 · Contributions — `[R]` ~20s
Our contributions: the first multi-model benchmark on Central-Asian urban imagery; a YOLO detector improved by 140%; and the open Canopy application.

### Slide 21 · Future work — `[R]` ~15s
Next: more annotated data, training directly on live map tiles to close the domain gap, and city-wide canopy analytics.

### Slide 22 · Thank you — `[R]` ~10s
Thank you. We'd be glad to answer your questions.

---

**Timing check:** full read ≈ 10–11 min. For an 8-min pre-defense, cut slides 6, 14, and 18 (marked) → ≈ 8 min.
**Rasul owns:** slides 1–11, 14–22 (all but 12–13). Make slides **11 (YOLO)** and **17 (Canopy)** land hardest — that's where your individual contribution is scored.
