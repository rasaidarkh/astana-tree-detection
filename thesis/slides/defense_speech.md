# Defense speech — Automated Tree Recognition & Green-Space Mapping (Astana)

**Language:** English · **Target:** ~9–10 min · **Tone:** honest student team — modest, clear, no overselling
**Speakers:** `[R]` Rasul (system + YOLO + web app) · `[B]` Berik (Mask R-CNN) · `[A]` Anuar (DeepForest + SAM 2)
**Note:** follows the current working 22-slide arc — provisional. Competitor numbers (slides 12/13/15) are Berik's & Anuar's — confirm their final figures before the talk. Keep the honest framing: a small test set, modest accuracy, models in the same ballpark.

---

### Slide 1 · Title — `[R]` ~20s
Good morning. Our diploma project is a system that detects trees in Astana from satellite images and shows them on a map. I'm Rasul Aidarkhanov, together with Berik Sharipov and Anuar Totin; our supervisor is Syndar Satbayev. I'll present the overall system and the YOLO part; Berik and Anuar will talk about their models.

### Slide 2 · Problem & relevance — `[R]` ~30s
A growing city like Astana needs to know where its trees are — for green planning and irrigation. Counting them by hand from images is slow. There are tree-detection models in the literature, but they're trained and tested mostly on American or European data. We wanted to see whether they work on our city — and if not, build something practical that does.

### Slide 3 · The gap & our aim — `[R]` ~30s
This number is why the project exists. A popular ready-made tree detector, used as-is on Astana imagery, scores only **0.012** — basically it doesn't work here. So our goal was practical: collect Astana data, train and compare several models, and build a simple tool around them. To be clear from the start — our accuracy is modest, and we'll be honest about what the system can and can't do.

### Slide 4 · Objectives & methods — `[R]` ~25s
Our objectives: build an annotated Astana dataset; train and compare four model families — YOLOv8, Mask R-CNN, DeepForest, and SAM 2; try combining them; test them all on one common set; and wrap the result in a web app. The methods are standard — supervised training, transfer learning, and cutting large images into tiles.

### Slide 5 · Literature review — `[R]` ~25s
We read 31 papers from 2019 to 2025. They report high accuracy — but each on its own region and camera. The last row is ours. As far as we know, this is the first time these models have been tested on Astana satellite imagery. We're not claiming to beat the literature — we're measuring how it does on new ground.

### Slide 6 · Math foundation — `[R]` ~25s *(cut if short)*
Briefly: the models use convolutional networks. YOLO predicts boxes and masks directly with a combined loss; Mask R-CNN proposes regions and then segments them. We score everything with mean Average Precision at IoU 0.5 — mAP@50 — so every model is measured the same way.

### Slide 7 · System architecture — `[R]` ~30s
The system has three parts: a React and Leaflet front end where you pick an area; a FastAPI back end that tiles the image and runs the model; and a small SQLite database. We call the app **Canopy**. One thing to note now — the app lets you choose which model to run, and I'll come back to why that matters.

### Slide 8 · Data & hardware — `[R]` ~25s
The training images are high-zoom satellite captures of Astana, about 0.3 m per pixel. We trained on a single laptop GPU — an RTX 4060 with 8 GB — to keep everything reproducible on normal hardware. The database simply stores scans, snapshots, and detections.

### Slide 9 · Dataset — `[R]` ~25s
We annotated trees on tiled Astana images and kept some aside for testing. Trees overlap, so we label them as polygons, not just boxes. Our shared test set is small — and we're upfront about that: **14 images, 702 labelled trees**. Every model in the project is scored on exactly this set, so the comparison is fair.

### Slide 10 · Four branches — `[R]` ~15s
From here the project splits into four model branches that we compare on that same test set. I'll take YOLO; Berik and Anuar will present theirs.

### Slide 11 · YOLO + ablation — `[R]` ~40s  ← *my part*
YOLOv8 is the part I worked on. I ran about twenty experiments — different starting weights, image sizes, and augmentation. A few practical lessons: start from pretrained weights; tile at the size you'll actually run at; and don't over-do augmentation. On our test set the score went from **0.131** in my first attempt to **0.315** — a 140% improvement over my own baseline. I want to be honest, though: 0.315 is a modest number, and the gap over the next-best settings is small.
> *Handoff:* "I'll pass to Berik for Mask R-CNN."

### Slide 12 · Mask R-CNN — `[B]` ~30s
*(Berik)* Same dataset, same test set; scores around **0.166**. Cleaner masks in some cases — lower than YOLO on our data, but it handles certain trees nicely.

### Slide 13 · DeepForest + SAM 2 — `[A]` ~30s
*(Anuar)* Ready-made DeepForest scored 0.012; after fine-tuning and adding SAM 2 masks, about **0.146**. Strong general-purpose masks, weaker on small city trees.
> *Handoff back to Rasul.*

### Slide 14 · Why several models — `[R]` ~35s  ← *key point*
Here's the most useful thing we learned. **No single model is best at everything** — each one catches some trees the others miss. So instead of forcing one "winner", we kept all of them in the app and let the user choose. We also tried combining them — a confidence-weighted box fusion and a YOLO voting scheme — which is more robust on hard tiles. On our small test set the single YOLO scores highest by the numbers, but that doesn't make it the right choice for every image.

### Slide 15 · Results — `[R]` ~30s
This is the side-by-side on the same 14 images. By the numbers YOLO is on top at **0.315**, then Mask R-CNN, then DeepForest with SAM 2. But the margins are small and partly within run-to-run noise — so we read this as "these models are in the same ballpark on Astana," not "one clearly wins."

### Slide 16 · Qualitative — `[R]` ~25s
And here's what it actually looks like. The detections line up with real tree crowns in many places. But honestly — it doesn't find every tree. On our test set it finds about **30%** of the labelled trees at the default confidence — more if we lower the threshold (around 44%), fewer if we raise it — and per image it swings from almost none on hard scenes to nearly all on clear ones. We'd rather show that openly than hide it.

### Slide 17 · Web application — `[R]` ~35s  ← *my part*
Finally, the app. You pick an area of Astana, choose a model, and Canopy tiles the image, runs detection, and draws the trees it found on the map — with a confidence filter and a few layers. The counts you see come from the areas we scanned for demos — it is **not** a full census of the city. The point isn't a big number; it's that anyone can run a model on an area and inspect the result for themselves.

### Slide 18 · Discussion & limits — `[R]` ~25s *(cut if short)*
Our main limits: a small training set, modest accuracy, and a gap between the images we trained on and live map tiles. We're not hiding any of it — it's the honest state of a student project, and each part improves with more data.

### Slide 19 · Conclusions — `[R]` ~25s
To sum up: we built a dataset, compared four models fairly on one set, tried combining them, and delivered a working app. The clearest result is the gap — ready-made tools score near zero on Astana (**0.012**), and our own models do meaningfully better, even if the absolute numbers stay modest.

### Slide 20 · Contributions — `[R]` ~20s
What we add: a first, honest comparison of these models on Astana imagery; a YOLO model improved over our own baseline; and the Canopy app, with a free choice of models.

### Slide 21 · Future work — `[R]` ~15s
Next: more annotated data, training on the same kind of images we actually run on, and better handling of small trees.

### Slide 22 · Thank you — `[R]` ~10s
Thank you for listening. We'll gladly answer your questions.

---

**Timing:** full read ≈ 9–10 min. For an 8-min pre-defense, cut slides 6 and 18.
**Rasul owns:** slides 1–11, 14–22. Make **11 (YOLO)**, **14 (why several models)** and **17 (Canopy)** land best — that's your individual contribution.
**Honesty anchors (say them out loud, examiners respect it):** 0.315 mAP is modest · finds ≈30% of trees at default confidence (varies a lot by image) · margins are within noise · demo counts, not a census · off-the-shelf fails on Astana (0.012).
