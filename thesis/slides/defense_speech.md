# Defense speech — Automated Tree Recognition & Green-Space Mapping (Astana)

**Language:** English · **Target:** ~9–10 min · **Tone:** confident + honest. We're proud of a *working automated system*; numbers are reported straight, framed as wins, not apologies.
**Deck:** `deck_final.html` / `deck_final.pdf` — **24 slides**. Slide numbers below match the deck exactly.

| Speaker | Slides | Block |
|---------|--------|-------|
| **Rasul** | 1, 2, 3, 4, 12, 18, 19, 20, 24 | intro · YOLO · the app · close |
| **Berik** | 5, 6, 7, 8, 9, 10, 11, 13 | methodology · system · maths · Mask R-CNN |
| **Anuar** | 14, 15, 16, 17, 21, 22, 23 | DeepForest · results · conclusions |

> ⚠️ **Slide 8 (the maths) is worth up to ~20 pts across criteria 1 & 2 — do NOT skip it.** If you must save time, trim slide **21** (discussion) only.

---

## RASUL — slides 1–4, 12, 18–20, 24

### Slide 1 · Title — `[R]` ~20s
Good morning. Our project turns a manual, one-by-one job — finding and mapping a city's trees — into an **automated system**: give it a satellite image, it returns a tree map. I'm Rasul Aidarkhanov, with me Berik Sharipov and Anuar Totin; supervisor Syndar Satbayev. I'll open and later cover the YOLO engine and the app; Berik takes the methodology; Anuar the results and conclusions.

### Slide 2 · Contents — `[R]` ~10s
Here's the plan: the problem and our aim, how we built and measured the system, the four detection engines, the results, the Canopy app, and conclusions.

### Slide 3 · Problem & relevance — `[R]` ~25s
A fast-growing city needs to know where its trees are — for green planning, irrigation, shade. Today it's done by hand, image by image; that doesn't scale. Models exist in the literature, but they're trained on American and European data. The real gap isn't accuracy — it's that **no automated process for this exists here at all**. So we built one.

### Slide 4 · The gap & our aim — `[R]` ~25s
Here's why the project exists, in one number. A popular ready-made tree detector, run as-is on Astana, scores **0.012** — effectively blind. There was nothing off-the-shelf we could just deploy. So our aim is concrete: collect Astana data, build an **automated pipeline**, let several models drive it, and measure each honestly.

> *Handoff:* "Berik will walk you through how we set this up."

---
*(Rasul returns at slide 12)*

### Slide 12 · YOLOv8 engine + ablation — `[R]` ~40s
YOLOv8 is the engine I built and tuned. About twenty experiments — starting weights, image sizes, augmentation. Three things actually mattered: start from pretrained weights, tile at the size you'll deploy at, keep augmentation moderate. The score climbed from **0.131** on my first attempt to **0.315** — that's **+140%** over my own baseline, and the strongest configuration we found. The gaps near the top are within run-to-run noise — which is itself useful: it tells us which differences are real.

> *Handoff:* "Berik — Mask R-CNN."

---
*(Rasul returns at slide 18)*

### Slide 18 · Web application — Canopy — `[R]` ~35s
This is the deliverable — **Canopy**. You pick an area of Astana, choose an engine, and it tiles the image, runs detection, and draws every tree it finds onto the map — confidence filter, several layers. The whole chain — capture, tile, infer, merge, map, export — runs end-to-end, about one square kilometre in roughly twenty seconds on a laptop GPU.

### Slide 19 · Choosing the engine — `[R]` ~20s
And this is the key UI idea made concrete: before a scan you pick the engine and size — YOLO, DeepForest, Mask R-CNN, or an ensemble. The counts you see come from areas we scanned ourselves for demos — not a full city census. The point is that anyone can run a model on an area and check the result.

### Slide 20 · Live demo — `[R]` ~25s
A quick look at it running: the v4 engine scanning the Botanical Garden end-to-end — tile, detect, map — every tree drawn live. Dense plantings like this are where it's strongest; the full recording is linked on the slide.

> *Handoff:* "Anuar will take the results and conclusions."

---
*(Rasul returns to close)*

### Slide 24 · Thank you — `[R]` ~10s
To close: we turned a manual task into a working, automated tree-mapping system for Astana, with an honest benchmark behind it. Thank you — we'll gladly answer your questions.

---

## BERIK — slides 5–11, 13

### Slide 5 · Objectives & methods — `[B]` ~25s
Five objectives: build an annotated Astana dataset; train and compare four engines — YOLOv8, Mask R-CNN, DeepForest, SAM 2; combine them; test all on one common set; and wrap it in an automated web app. Methods are standard and proven — supervised training, transfer learning, sliding-window tiling.

### Slide 6 · Problem statement: input → output — `[B]` ~20s
Formally: the **input** is a satellite capture of Astana at zoom 19 — about 0.3 m per pixel — cut into overlapping 640-pixel tiles. The **output**, per tree, is a bounding box, a pixel mask, and a confidence score; tiles are stitched back, duplicates merged, results placed on a map. So the task is detection plus instance segmentation.

### Slide 7 · Literature review — `[B]` ~25s
We analysed 31 peer-reviewed papers, 2019 to 2025. They report high accuracy — but each on its own region and sensor. The last row is ours: to our knowledge the **first measurement of these models on Astana imagery**. We don't claim to beat the literature — we're first to set the local baseline.

### Slide 8 · How we measure success — the maths — `[B]` ~40s  ← **KEY, do not skip**
Three simple ideas, one number. **First, IoU** — intersection over union: take the box the model drew and the real tree, divide the overlapping area by their combined area. One is a perfect match, zero a miss; we count a tree as found when that overlap is at least one-half. **Second, precision and recall**: precision is, of the trees we flagged, how many were real; recall is, of all real trees, how many we caught. **Third, mAP@50** — the score on every slide: we sweep the confidence threshold, plot precision against recall, and take the area under that curve. One number, every engine measured the same way — that's the 0.315 in the results.

*(If asked about the loss: YOLO trains a combined box + class + mask loss in one pass; Mask R-CNN adds a separate mask-head loss on top of its region proposals.)*

### Slide 9 · The automated pipeline — `[B]` ~30s
The system is three layers: a React + Leaflet front end where you pick an area; a FastAPI back end that tiles the image, runs the chosen engine, and merges results; and a SQLite store. The detection model is just **one swappable step** — that's the core design idea, and we'll show why it matters.

### Slide 10 · Data, database & hardware — `[B]` ~25s
Training images are high-zoom Astana captures, ~0.3 m per pixel. Trees touch, so we hand-label polygons, not just boxes — about **5,500 tree crowns across ~100 images** (which the sliding-window tiling expands to roughly 8,700 training instances). The database is four tables — scans, snapshots, runs, detections — with cascading deletes. Everything trains on a single laptop GPU, an RTX 4060 with 8 GB — on purpose, to prove it's reproducible without a cluster.

### Slide 11 · The detection engines (divider) — `[B]` ~15s
From here the project splits into four engines, all scored on the same 14 images. Rasul takes YOLO first.

> *Handoff:* "Back to Rasul for YOLO."

---
*(Berik returns at slide 13)*

### Slide 13 · Mask R-CNN — `[B]` ~35s
Mask R-CNN is the engine I built. It's a careful two-stage detector: a region-proposal network first suggests where trees might be, then a second stage refines each one and outputs a clean pixel mask — segmentation built in, on a ResNet-50 backbone. Same Astana tiles, same 14-image test set; it reaches **0.166**. Lower than YOLO overall — but it recovers some larger trees the others miss and gives the cleanest masks, which is exactly why it stays in the app as a selectable engine.

> *Handoff:* "Anuar — DeepForest and SAM 2."

---

## ANUAR — slides 14–17, 21–23

### Slide 14 · DeepForest + SAM 2 — `[A]` ~35s
This is my branch. DeepForest is a specialist tree-crown detector — but trained on American forests, and pointed at a dry steppe city it scored just **0.012**. That's not a failure, it's the clearest proof of the geographic gap. After fine-tuning on our Astana data and chaining **SAM 2** to turn each box into a precise mask — zero-shot, no extra labels — it climbs to about **0.146**, a twelve-fold jump. It still struggles with small street trees, so it's one option among several, not the only one.

### Slide 15 · Why several models — `[A]` ~35s
The most useful thing we learned, and the heart of the design. **No single model is best everywhere** — each catches trees the others miss; on the same garden, eight engines give eight different counts. So instead of crowning one winner we keep them all in the app and let the user choose — and we combine them: a weighted box fusion across models, and a cross-YOLO vote that removes single-model false alarms like stadium roofs. The automation doesn't depend on any one model.

### Slide 16 · Results — `[A]` ~30s
Head-to-head on the same 14 images. By the numbers YOLO leads at **0.315**, then Mask R-CNN at 0.166, then DeepForest + SAM 2 at 0.146 — and the off-the-shelf baseline at 0.012. Chart and table tell the same story. The gaps near the top are small, partly within noise — so we read this as "these engines are in the same ballpark on Astana," and what matters is that any of them drives the pipeline.

### Slide 17 · Qualitative — `[A]` ~25s
What it looks like in practice. Detections line up well with real tree crowns in dense plantings. Straight numbers: the production engine recovers about **a third** of the labelled trees at its default confidence, more if you loosen the threshold. We show the misses openly — small and shaded trees are hardest. That's the honest state of the system.

> *Handoff:* "Rasul will show the app, then I'll wrap up."
> *(Rasul presents 18–20, then hands back to Anuar)*

### Slide 21 · Discussion & limits — `[A]` ~20s  *(trim here first if short on time)*
Honestly: the training set is small, accuracy is modest, and there's a gap between our training captures and live map tiles. None of it is hidden — each part improves directly with more annotated data, and the pipeline is built to absorb that.

### Slide 22 · Conclusions — `[A]` ~25s
Against our objectives: we built the dataset, compared four engines fairly on one set, combined them, and delivered a working app. The headline isn't a score — it's that a **manual task is now an automated process**. Ready-made tools score near zero here; our pipeline does meaningfully better and actually runs end-to-end.

### Slide 23 · Contributions & future work — `[A]` ~25s
We contribute: the first honest multi-model benchmark on Central-Asian urban imagery; a YOLO engine improved 140% over our baseline; and Canopy, with a free choice of engine. Next: more annotated data, training on the same imagery we run on to close the domain gap, and better handling of small trees.

> *Handoff:* "Back to Rasul to close."

---

**Timing:** full read ≈ 9–10 min. To hit 8 min, trim slide **21** only — never the maths (slide 8).

**Confident-but-honest anchors** (say 2–3, don't over-apologise):
- We built the **first automated tree-mapping pipeline** measured on Astana.
- Off-the-shelf scores **0.012** here; our pipeline reaches **0.315** — same test set, fair comparison.
- No single engine wins everywhere → the app lets you **choose and combine** them.
- Numbers are straight: champion recovers ~⅓ of trees at default confidence; demo counts, not a census.

**Q&A quick-reference (numbers to defend):**
- *"Is 0.315 good?"* — modest in absolute terms, but it's the first local benchmark and +140% over our own baseline; off-the-shelf is 0.012.
- *"Recall?"* — v4 engine, default confidence 0.25: ≈30% of labelled trees found (≈44% at a looser 0.10), on the 14-image set. *(measured: `results/yolo_v4_recall.json`)*
- *"Why keep lower-scoring engines?"* — different engines catch different trees; we offer choice + ensembles, not one forced winner.
- *"mAP@50 vs 50:95?"* — we headline mAP@50 (IoU ≥ 0.5); stricter 50:95 = 0.110 box for v4, it's in the thesis.
- *"How much did you annotate?"* — by hand: **~5,500 tree crowns across ~100 Astana images**; sliding-window tiling expands that to ≈8,700 training polygon instances (a crown on a tile boundary appears in two tiles).
- *"Why is the test set only 14 images?"* — the **14-image / 702-polygon M14** set is the *common cross-model test set* (a held-out subset), so every engine is scored on identical ground truth — not the whole dataset. Training uses the full ~100 images.
- *"Why m-seg 0.308 vs x-seg 0.315?"* — both are local optima (U-shaped size curve); x-seg is final production, the chart marks it.
