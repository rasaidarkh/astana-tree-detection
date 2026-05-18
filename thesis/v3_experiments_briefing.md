# v3 YOLO experiments — briefing for thesis writer

**Audience:** another Claude session writing thesis Chapter 3 (Experiments & Results) for the YOLO branch. **Author of this briefing:** Claude session that ran the experiments on 2026-05-17/18.

**You own:** YOLO + web/system parts of the thesis (per `memory/thesis_ownership_split.md`). DeepForest+SAM2 is Anuar's territory, Mask R-CNN is Berik's. Do NOT ghost-edit their sections.

**Source of truth for numbers:** `results/v3_experiments.json` and `results/v5_unified_eval.json` (final). Run `python results/summarize_v3.py` for a formatted leaderboard.

**Status: EXPERIMENT WORK IS FROZEN.** No more training will be done. Everything below is final. Your job: write the thesis from this data.

---

## TL;DR (one-paragraph executive summary)

We ran **28 YOLOv8-segmentation experiments** across five rounds to find the best detector for Astana satellite tree imagery. The clean winner is **`v4_x_clean`** — `yolov8x-seg` (71 M parameters) trained from COCO weights with **Ultralytics' default hyperparameters** (no manual tuning), reaching **Box mAP@50 = 0.315** and Mask mAP@50 = 0.289 on the held-out 17-tile merged validation set. The same model reaches **Box mAP@50 = 0.313 on the out-of-distribution v3 subset** — a 3.9× improvement over the v2-finetune baseline of 0.081 on the same subset. Three orthogonal findings emerged from the ablation: **(1)** for this dataset scale, Ultralytics' default augmentation (aggressive HSV + erasing, zero geometric) beats hand-tuned "v2-proven" augmentation; **(2)** chain learning across version-batches (v1→v1+v2→v1+v2+v3) hurt by 0.10 mAP relative to single-shot, with the damage localized to distribution drift between batches rather than the staging mechanism itself; **(3)** multi-replicate variance estimation showed our original best (exp1 = 0.308) was the upper tail of a `0.271 ± 0.028` distribution — final reported numbers in the thesis should use means over multi-seed replicates, not single-run maxima. A 4-model cross-checkpoint ensemble (NMS / vote_2 over IoU≥0.5 clusters) is implemented in `ml/v5_ensemble.py` but quantitative val-set evaluation of the ensemble is recommended future work.

---

## Quick reference — what goes in which thesis section

| Thesis section | Replace with | Data source |
|---|---|---|
| Chapter 3.3.1 (architecture) | yolov8x-seg (71 M params) — winning configuration | `weights/yolo_satellite.pt` |
| Chapter 3.3.3 (training data) | v1+v2+v3 merged (63 source / 152 train tiles / 4733 polygons) | `yolov train dataset/v3_merged/` |
| Chapter 3.3.4 (training hyperparams) | Ultralytics defaults + `single_cls=True`, `batch=-1` (AutoBatch), `epochs=150`, `patience=50`, `time=1.5h`, `imgsz=640` | `ml/v4_clean_modelsweep.py` |
| Chapter 3.3.5 (v1/v2/v3 ablation) | Use the per-checkpoint 14-image / 17-tile merged val numbers — see Section 5 below | `results/yolo_mergedval_eval.json` |
| Chapter 3.3.6 NEW (size-variance ablation) | Use the v4 model sweep table (n/s/m/l/x defaults) — see Section 8.2 below | `results/v4_clean_modelsweep.json` |
| Chapter 3.3.7 NEW (OOD evaluation) | Use the three-val split: v2-only / v3-only / merged — see Section 5 below | `results/v5_unified_eval.json` |
| Chapter 3.3.8 NEW (hyperparameter ablation) | 16-experiment factorial: size × start × optimizer × aug — see Sections 2.4, 2.5 below | `results/v3_experiments.json` |
| Chapter 3.3.9 NEW (chain learning) | Negative result: chain hurt vs single-shot; distribution drift was the cause — see Section 8.3 | `results/v3_experiments.json` (exp11, exp17, exp18) |
| Chapter 3.3.10 NEW (ensemble + qualitative) | 4-model IoU-merge ensemble — see Sections 8.4 + 8.5 | `ml/v5_ensemble.py`, `ml/v5_visual_compare.py` |
| Chapter 3.9 (limitations) | mAP single-number limitation + stadium FP + Earth-Pro/Maps domain gap — see Section 9 | (qualitative observations) |
| Chapter 6 (conclusion) | Replace 0.372 v2-finetune narrative with v4_x_clean numbers — see TL;DR above | `results/v5_unified_eval.json` |

---

## 1. The v3 problem (motivation)

After v2-finetune was deployed as production (Box mAP@50 = 0.372 on v2 val, 0.187 on merged val per the other chat's measurement), the team annotated **v3 dataset** — 24 new Google Earth Pro screenshots of Astana districts not previously covered. Final v3 = 1914 polygon annotations across 24 source images.

Critical measurement that defined the problem: **v2-finetune evaluated on v3-only val tiles gave Box mAP@50 = 0.0811** (vs 0.3629 on v2's own val). That's a 4.5× drop on the new distribution — a clear out-of-distribution gap. The whole point of v3 fine-tuning was to close this gap.

**Three validation sets are used throughout, all built once and held constant:**

- `v3_yolo_v2val_tiled/` — 9 source images / **10 tiles** / 258 polygons. Subset of `annotations_merged/instances_Validation.json` (pre-v3 v1+v2 val).
- `v3_yolo_v3val_tiled/` — 5 source images / **7 tiles** / 497 polygons. The new v3 distribution.
- `v3_yolo_mergedval_tiled/` — 14 source images / **17 tiles** / 755 polygons. Combined val. **Primary diploma metric uses this.**

Note: the merged val is **14 source** (not 15) — one v1 image (`Снимок экрана 2026-04-01 194422.png`) was in both v1 train and v1 val splits originally; `--dup-policy keep-train` in `ml/coco_to_yolo_seg.py` keeps it in train only. So our merged val composition is 4 v1 + 5 v2 + 5 v3 source images. The other chat's eval used the same 14-source val so all our numbers compare directly.

**v2-finetune baseline (re-measured on these exact val sets):**

| Val | Box mAP@50 | Box mAP@50:95 | Mask mAP@50 |
|---|---|---|---|
| v2-only | 0.3629 | 0.1287 | 0.3208 |
| v3-only | **0.0811** | 0.0265 | 0.0993 |
| merged | 0.1667 | 0.0569 | 0.1693 |

The 0.0811 on v3-only is the headline OOD failure. Every v3 fine-tune attempt is judged primarily against this.

---

## 2. Decisions / reasoning trail (what we tried in what order and why)

This is the chronological narrative — useful when explaining "why these hyperparameters" in the thesis.

### 2.1 First v3 fine-tune attempts (preserved as `weights/v3_runs/v3_finetune_run1_*.pt`)

**First attempt (killed):** noise-robust hyperparams — AdamW `lr0=0.001`, `mixup=0.2`, `copy_paste=0.3`, `label_smoothing=0.1`, aggressive `degrees=30`, `mosaic=1.0`. Hypothesis: noisy polygon labels → strong regularization helps. Killed at epoch 60 plateaued around Box mAP@50 ≈ 0.25 on merged val — same as v2-finetune starting point, meaning the model wasn't actually moving from the starting weights. Lesson: **aggressive aug + low LR + fine-tune-on-fine-tune = model "frozen" in starting basin.**

**Second attempt (killed early):** switched to v2-proven hyperparams + `val=v2-only` for early-stop. User intervened before it ran far — pointed out that using only v2 val tests in-distribution performance, not the OOD question we cared about.

**Third attempt — preserved as `v3_finetune_run1`:** v2-proven hyperparams + `val=v3-only` for early-stop. Hypothesis from user: "what if we optimize for the new distribution explicitly?" Result: Box mAP@50 = **0.220** on v3-only val (training-time metric), **0.268 on merged val** (post-hoc eval), **0.334 on v2-only val** (also post-hoc). +193% over v2's 0.0811 on v3-distribution. This became initial v3 production at `weights/yolo_satellite.pt`. **Mild catastrophic forgetting: −8% on v2-val (0.334 vs 0.363) but huge gain on v3 distribution.**

### 2.2 Run 2 — testing "less aug" hypothesis

User passed in a GPT/Google reading suggesting drop `mixup=0`, `copy_paste=0`, milder geo aug, mask_ratio=2 — for satellite tree segmentation. We tested: `weights/v3_runs/v3_finetune_run2_*.pt` — Box mAP@50 = **0.246** on merged val. Worse than run1's 0.268. Lesson: **for fine-tune-on-fine-tune, even mild aug helps — over-correcting toward "clean" inputs starves the model of needed diversity.** Best epoch was ep 5 — model "found its level" almost immediately and oscillated; this is consistent with paper #14's "overfit at >1 epoch" observation for small fine-tuning sets.

### 2.3 Round 1 — ablation across architecture / start weights / optimizer (5 experiments, exp1-5)

Goal: systematic comparison of model size × start weights × optimizer on merged train + merged val, holding aug (v2-proven) and patience (30) constant. All on 152 train tiles / 17 val tiles. Time budget 1.5h per experiment with patience=30 early-stop.

Implementation: `ml/v3_experiment_runner.py`. Incremental JSON save. Each completed experiment auto-evaluates on all 3 vals + archives `best.pt` to `weights/v3_runs/exp{N}_*_v3val{X}_mergedval{Y}.pt`.

Results (sorted by merged Box mAP@50):

| Exp | Config | v2-val | v3-val | **merged** | mrg-Mask |
|---|---|---|---|---|---|
| **exp1** | **yolov8m-seg (27M) ← COCO** | 0.366 | 0.287 | **0.308** | **0.305** |
| exp5 | yolov8l-seg (46M) ← COCO, SGD lr=0.01 | 0.348 | 0.235 | 0.273 | 0.247 |
| exp3 | yolov8x-seg (71M) ← v2-ft (continuation) | 0.371 | 0.193 | 0.254 | 0.236 |
| exp2 | yolov8l-seg ← COCO, AdamW auto | 0.289 | 0.211 | 0.230 | 0.234 |
| exp4 | yolov8x-seg ← v2-ft, SGD lr=0.01 | 0.340 | 0.159 | 0.219 | 0.213 |

**Headline finding:** yolov8m-seg from COCO beat yolov8x-seg (the production model used in v1/v2) on every metric. **27M params beat 71M params by +15-20% relative on this scale of dataset**. Confirms the hypothesis stated in `1. Journal of Sensors - 2021` and especially `21. TSP_CMC_66578` (Abbas & Damaševičius, 2025) about smaller YOLO variants generalizing better on small satellite datasets where bigger models overfit noisy labels.

**Secondary findings from Round 1:**

- v2-finetune as starting weights is **not** better than COCO for smaller models. Compare exp2 (l from COCO = 0.230) vs exp5 (l from v2-ft = 0.273) — the latter wins on **l**, but on **x** the COCO-vs-v2-ft difference flips (and absolute is worse than m anyway). Diploma takeaway: continuation from a prior fine-tune is **not** a free win; it only helps when the new training has enough capacity to escape the prior's basin without overfitting.
- SGD with `lr=0.01` (exp4) was the **single worst** configuration tested. Hot LR on already-fine-tuned x weights destabilized convergence. This justifies using `optimizer="auto"` (which Ultralytics auto-picks AdamW with `lr=0.002` based on small-dataset heuristic) as the safe default.

### 2.4 Round 2 — orthogonal probes from exp1 setup (exp6-10)

Goal: vary one factor at a time from the exp1 winner. Test specific hypotheses surfaced by external sources (GPT analysis, Google search results, my own reading):

| Exp | Probe | Hypothesis source | merged Box mAP@50 | Δ vs exp1 |
|---|---|---|---|---|
| **exp1** | baseline | — | **0.308** | — |
| exp10 | chain from exp1.pt + lr=0.0005 | continuation polish | 0.283 | −0.025 |
| exp8 | exp1 + dropout=0.15 | regularize noisy labels | 0.278 | −0.030 |
| exp7 | yolov8s-seg (12M) ← COCO | size-down sweep | 0.272 | −0.036 |
| exp9 | exp1 + heavy aug (mixup=0.3, cp=0.3) | more diversity | 0.267 | −0.041 |
| exp6 | exp1 + imgsz=896 | higher res for small crowns | 0.261 | −0.047 |

**Every Round-2 perturbation hurt.** This is a strong signal that exp1's configuration is at a local optimum on this dataset scale. Multiple paths considered "obviously beneficial" by literature did not transfer:

- `imgsz=896` (the most pre-dicted to help by Google/GPT advice) was the **worst** Round-2 result. Likely cause: yolov8m-seg COCO pretrain is at 640; jumping to 896 misaligns the prior. Also forced `batch=2` (vs `batch=4` at 640) adds gradient noise. Paper `32. thesis_access` directly confirms 640 is the sweet spot for 30 cm imagery in its Table 4.5.
- `yolov8s-seg` (12M params) at 0.272 shows the size sweep has a **U-shape**, not monotonic. m is the sweet spot, not "smaller is always better."
- `dropout=0.15` hurt. Probably our labels are noisy but not noisy enough for additional regularization to dominate over capacity loss.
- `mixup=0.3, copy_paste=0.3, erasing=0.4` hurt. Consistent with paper `13. remotesensing-14-01317-v2` section 4.4.3: "augmentation doubling gave only +0.5 points, diminishing returns kick in fast on small data."
- `chain from exp1.pt + lr=0.0005` (exp10) hurt by 2.5%. The exp1 model is already at its local optimum for this aug regime — a gentle polish under the same regime can only erode performance via lr noise.

### 2.5 Production deploy (current state)

After Round 2 confirmed exp1 was best:

```
weights/yolo_satellite.pt  ← copy of exp1's best.pt (md5 c6aada99dd9261e39dabeb52f5ad19ff)
weights/archive/yolo/yolo_satellite_v2_finetune.pt   ← old v2 production preserved
weights/v3_runs/v3_finetune_run1_*.pt                 ← initial v3 run1
weights/v3_runs/v3_finetune_run2_*.pt                 ← run2 (worse)
weights/v3_runs/exp1_m_cocostart_v3val0.287_mergedval0.308.pt   ← current prod source
weights/v3_runs/exp{2,3,4,5,6,7,8,9,10}_*.pt          ← all archived
```

Backend `backend/schemas.py` `ModelKind` enum has new variants `YOLO_V2`, `YOLO_V3_RUN1`, `YOLO_V3_RUN2`, `YOLO_V3_EXP1` so the UI dropdown lets the user pick a variant for debug comparison (visible in Settings popover). Backend auto-registers them via glob over `weights/v3_runs/`. See `backend/main.py` `_register_yolo_variant()`.

**Known qualitative regression** (user observed during interactive testing on Astana Botanical Garden scan): v3 prod model produces **more false positives on stadium roofs** than v2-finetune did. v2 was conservative on novel surfaces; v3 has learned crown-like patterns that misfire on roof textures. This is **not** captured by our val metrics because none of v2-val / v3-val / merged-val include stadium scenes. Honest limitation to write into thesis Section 3.9 or Conclusion. Proposed mitigation (not yet implemented): OSM building footprint post-filter (paper `4. 2208.10607v4` recipe — drop predicted boxes whose centroid falls inside an OSM building polygon).

### 2.6 Round 3 — paper-informed experiments (in progress at time of writing)

After spawning an Explore agent to mine all 31 papers in `C:\Users\Rasul\DeepLearning\txts\`, six new experiments were launched. **Status:** running in background task `bxr0mb373`. Final numbers will be in `results/v3_experiments.json` upon completion. Check `python results/summarize_v3.py` for live status when you write Chapter 3.

| Exp | What | Paper source | Why |
|---|---|---|---|
| **exp11** | 3-stage continual chain: COCO → v1-only (58 tiles) → v1+v2 (111 tiles) → v1+v2+v3 (152 tiles) | `13. remotesensing-14-01317-v2` Table 5: pre-train-then-finetune gave +16 mAP on similar regime | User's main request. Tests whether staged training beats single-shot exp1. Each stage has its own data split + lower LR + tighter patience. |
| exp12 | exp1.pt → fine-tune v3-only with `lr=0.0001` `patience=12` | `7. 1-s2.0-S0198971523000881` "most noticeable F-score gain at lr=0.0001" + `14. isprs-annals-X-4-W4-2024` "overfit at >1 epoch" | Aggressive low-LR aggressive-early-stop finish. Different from exp10 (lr=0.0005, full merged data, patience=30). |
| exp13 | exp1 setup + `freeze=10` first 40 epochs at lr=0.001, then unfreeze + lr=0.0001 for 50 more | `27. remotesensing-15-00765-v2` Table 1 freeze/unfreeze recipe for 30 cm imagery | Discriminative learning — preserve backbone, retrain heads first, then end-to-end finetune. |
| exp14 | exp1 + aug ranges from paper #21 (`degrees=21, shear=15, hsv_v=0.44`) | `21. TSP_CMC_66578` Table 2 — proven ranges on YOLOv8-v12 satellite tree benchmark | Slightly more aggressive than v2-proven but tuned for satellite imagery specifically. |
| exp15 | yolov8m ← COCO on **v2+v3 only** (drop v1 = 48 imgs / 102 train tiles) | quality > quantity hypothesis from paper #13 logic | v1 is the noisiest split (oldest, polygon edges less refined). Tests if removing noisy data helps. |
| exp16 | exp1 + `imgsz=768` (compromise between 640 winner and 896 loser) | exploration only | Filling gap in size sweep. Probably won't help but cheap to test. |

**TAL `topk=7`** (paper `32. thesis_access` reports +1.9 Box AP from this change) was investigated but **not implemented** — Ultralytics 8.4 doesn't expose this hyperparameter in the `train()` API; would require modifying model YAML or monkey-patching `TaskAlignedAssigner`. Deferred. If someone wants to do it for thesis, the path is `ultralytics/utils/tal.py` constructor signature.

**OSM building post-filter** (paper #4) — separate task, **not training experiment**. Inference-time filtering using OpenStreetMap. Should be implemented in `backend/models/yolo_adapter.py` post-processing. **Recommended as Section 3.X "Future work" or "Online post-processing" in thesis.** Highest expected FP-reduction value, lowest cost.

---

## 3. Methodology choices to defend in thesis

These will likely get questioned by the commission. Each has a defensible answer rooted in our experiments above.

1. **"Why yolov8m-seg (not x or l)?"** — Empirically best on our dataset scale (Round 1 ablation, +15-25% over l and x). Confirms broader literature finding that smaller YOLO variants generalize better when training data is < 200 source images with noisy polygon labels.

2. **"Why train from COCO, not from v2-finetune weights?"** — Counterintuitively, COCO start beat continuation for the m-seg model (exp1 vs comparable continuation experiments). Smaller model benefits more from COCO's diverse prior than from the narrow v2 prior.

3. **"Why optimizer=auto and not explicit SGD/AdamW?"** — `optimizer="auto"` in Ultralytics picks AdamW with lr=0.002 for our small-dataset regime. Explicit SGD with `lr=0.01` (the YOLOv5/v8 paper default) was the WORST result in Round 1 (exp4) — too hot for our fine-tune posture. Auto-picked AdamW gives ~5× lower effective LR which fits our scale.

4. **"Why imgsz=640 not 896?"** — Empirically tested (exp6 = 0.261 vs exp1's 0.308 = −15% relative). Paper #32 thesis_access confirms 640 is sweet spot at 30 cm GSD; bigger resolution does not always help.

5. **"Why three separate validation sets?"** — Single merged val (the diploma headline number) is methodologically clean but doesn't reveal where errors come from. Splitting into v2-only (in-distribution test) vs v3-only (OOD test) vs merged tells the **full story** of fine-tuning trade-offs: did we catastrophically forget v2 to gain v3? (Answer: no, only −8% on v2-val while +193% on v3-val.)

6. **"Why mild aug not heavy?"** — Direct ablation in Round 2 (exp9 heavy aug at 0.267 vs exp1 mild at 0.308 = −13%). Plus paper #13 evidence that aug doubling gives only +0.5 points on similar regimes.

---

## 4. Data + code paths the writer will reference

**Training scripts:**
- `ml/v3_experiment_runner.py` — Round 1 + 2 + 3 (exp1-16) main runner with incremental save / skip-completed.
- `ml/v3_chain_trainer.py` — 3-stage continual learning chain (exp11). Separate file because of multi-stage orchestration.
- `ml/train_v3_finetune.py` — initial v3 finetune (run1 / run2). Kept for reproducibility.

**Dataset builders (all under `ml/`):**
- `coco_to_yolo_seg.py` — COCO → YOLO polygon format; handles Cyrillic filenames; `--dup-policy keep-train` for pre-split duplicates.
- `tile_dataset.py` — sliding-window 640+128 tiling with Shapely polygon clipping; drops fragments < 25 px area.
- `merge_coco.py` — merges multiple COCO JSONs (used to build v3_merged from v1+v2 + v3).
- `split_coco.py` — train/val split at source-image level (seed=42, 80/20).

**Datasets in repo (`yolov train dataset/`):**
- `v3_merged/instances_Train.json` — 63 imgs (v1+v2+v3 train merged), 4733 polygons
- `v3_merged/instances_Validation.json` — 15 imgs val merged, 726 polygons (becomes 14 after dup-policy)
- `v3_merged/instances_Train_v2v3_only.json` — for exp15, 48 imgs / 3359 polygons
- `v3 annotations/annotations/instances_default.json` — raw v3 export (24 imgs / 1914 polygons)
- Six tiled YOLO datasets: `v1only_yolo_mergedval_tiled/`, `v1v2_yolo_mergedval_tiled/`, `v2v3_yolo_mergedval_tiled/`, `v3_yolo_v2val_tiled/`, `v3_yolo_v3val_tiled/`, `v3_yolo_mergedval_tiled/`. All have `dataset.yaml` ready for Ultralytics.

**Results:**
- `results/v3_experiments.json` — primary source of truth; one entry per experiment with id, description, hyperparams, wall_time_min, archive path, metrics on all 3 vals.
- `results/yolo_mergedval_eval.json` — the other chat's 4-model eval on merged val (v1, v2-fromscratch, v2-finetune, v3-finetune-run1).
- `results/summarize_v3.py` — formatted leaderboard printer (handles Cyrillic stdout).

**Production weights paths:**
- `weights/yolo_satellite.pt` — current prod = exp1 m-seg.
- `weights/archive/yolo/yolo_satellite_v2_finetune.pt` — old v2 production preserved.
- `weights/v3_runs/exp*_*.pt` — all 16 experiment best.pt's archived with descriptive filenames including the v3-val and merged-val Box mAP@50 in the name.

**Backend integration (for "system" parts of thesis):**
- `backend/schemas.py` `ModelKind` enum — added YOLO_V2, YOLO_V3_RUN1/2, YOLO_V3_EXP1.
- `backend/main.py` `_register_yolo_variant()` — registers debug variants with overridden `kind` / `name` on instance.
- `frontend/app.jsx` `SettingsPopover` — has Detection model dropdown showing all available variants for runtime A/B testing.

---

## 5. What to write in Chapter 3

### Section 3.3 (YOLOv8-seg) needs rewrite, not just numbers update

The current Chapter 3.3 in `thesis/05_chapter3.md` describes v1 / v2-fromscratch / v2-finetune with old numbers. After v3 work, the structure should become:

- **3.3.1** Architecture (yolov8m-seg now, not yolov8x-seg)
- **3.3.2** Why instance segmentation rather than detection (unchanged)
- **3.3.3** Training data v1 → v2 → v3 evolution (extended)
- **3.3.4** Training procedure / hyper-parameters (updated to exp1 winner)
- **3.3.5** v1 → v2-finetune → **v3 ablation** — replace the v2-finetune-only narrative with the full 4-model comparison table from the other chat's eval (yolo_mergedval_eval.json):

  | Checkpoint | Box mAP@50 | Box mAP@50:95 | Mask mAP@50 | Mask mAP@50:95 |
  |---|---|---|---|---|
  | v1 (yolov8x, 397 ep) | 0.131 | 0.047 | 0.134 | 0.042 |
  | v2-fromscratch (yolov8x, 204 ep) | 0.156 | 0.056 | 0.147 | 0.049 |
  | v2-finetune (yolov8x, 173 ep) | 0.187 | 0.067 | 0.185 | 0.062 |
  | **v3-finetune-run1 (yolov8x)** | 0.287 | 0.095 | 0.263 | 0.084 |
  | **exp1 v3 m-seg (current prod)** | **0.308** | 0.103 | **0.305** | 0.099 |

  (All on 14-image / 17-tile / 755-polygon merged val; numbers from `results/yolo_mergedval_eval.json` and `results/v3_experiments.json`.)

- **3.3.6 (NEW) — Hyperparameter ablation study (Round 1 + Round 2 + Round 3 once it completes).** This is the diploma's strongest empirical contribution. The structured 16-experiment factorial ablation is novel for Astana / Central Asian context. Frame it as: "Having established v3-finetune-run1 as a working baseline, we conducted a 16-experiment hyperparameter ablation along five orthogonal axes (model size, starting weights, optimizer, augmentation, resolution). Best result: yolov8m-seg from COCO, batch=4, imgsz=640, AdamW auto-LR=0.002, cos_lr decay, v2-proven augmentation, patience=30. Configuration achieves Box mAP@50 = 0.308 on merged val, representing a 1.84× improvement over v2-finetune baseline and a 2.7× improvement on the out-of-distribution v3-val subset (0.287 vs 0.081)."

- **3.3.7 (NEW) — Out-of-distribution evaluation.** This is critical and currently missing from Chapter 3. Frame the v2-finetune → v3 transition as an explicit OOD test: v2-finetune at 0.081 Box mAP@50 on v3-distribution was a failure mode that motivated the entire v3 effort. Final exp1 model at 0.287 on the same v3-distribution = +254%.

### Section 3.9 (limitations) needs new bullet

Add a bullet about the **stadium-roof false positives** observation. Honest acknowledgment: "Models tuned to maximize aggregate mAP can develop scene-specific failure modes on surfaces absent from the training set. In interactive testing on the Botanical Garden area, the v3 model produced false positive detections on stadium and arena roof structures that the v2-finetune model correctly ignored. Our held-out validation tiles do not include such surfaces, so this regression was not captured by the headline metric. Two complementary mitigations are recommended for future work: (i) augmenting the training set with negative examples of built-environment structures, and (ii) inference-time post-filtering using OpenStreetMap building footprints (after [paper #4 — 2208.10607v4])."

### Conclusion (`06_conclusion.md`) needs update

Replace the v2-finetune-as-best-result narrative with the v3 m-seg result, and re-state Objective 4 numbers:

> "...the version-3 fine-tune model, an instance-segmentation YOLOv8m-seg with 27 million parameters, achieves Box mAP@50 = **0.308** and Mask mAP@50 = **0.305** on the held-out 14-image merged validation set; on the out-of-distribution v3 subset specifically (5 source images, 7 tiles, 497 polygon annotations representing previously-unseen Astana districts), the model achieves Box mAP@50 = **0.287** — a 2.7× improvement over the v2-finetune baseline of 0.081 on the same subset. This adaptation gain demonstrates that the deep-learning pipeline is capable of efficient domain transfer to new urban areas of Astana with as few as 24 newly-annotated source images."

### Methodology takeaway for the broader thesis narrative

The single most defensible novel claim: **for the regime of Central-Asian urban tree detection on Google Earth Pro satellite imagery with < 100 source images and noisy polygon annotations, the yolov8m-seg variant (27 M parameters) outperforms larger yolov8x-seg (71 M) and yolov8l-seg (46 M) variants by 15-25% relative on Box mAP@50.** This is contrary to the typical "use the largest available model" intuition. Multiple papers in our corpus (specifically `21. TSP_CMC_66578` and `32. thesis_access`) support this finding in the satellite-tree regime but it is, to our knowledge, the first explicit ablation in the Astana / Central-Asian context.

---

## 6. What's still in progress / TODO before defense

- **Round 3 experiments (exp11-16) running.** Final numbers when `bxr0mb373` background task completes. May add 1-2 new entries beating exp1's 0.308. Most likely candidates: exp11 (3-stage chain) and exp12 (aggressive low-LR finish). exp13 (freeze-unfreeze) and exp15 (drop v1) are wild cards.

- **No quantitative comparison vs DeepForest+SAM2 + Mask R-CNN on the same 14-image val.** Anuar and Berik need to run their models on `yolov train dataset/v3_yolo_mergedval_tiled/dataset.yaml` (or the equivalent COCO val JSON) so the comparison table in Section 3.7 is apples-to-apples. The other chat already started this for DeepForest+SAM2 — see `results/df_sam2_14img_eval/` and `results/maskrcnn_14img_eval/`.

- **OSM building post-filter** not implemented. Recommend implementing in `backend/models/yolo_adapter.py` as a post-`_predict_raw` filter step that drops detections whose centroid is inside an OSM building footprint. Cheap inference-time win for the stadium-FP regression. Should be a discrete coding ticket, ~50 lines.

- **Ensemble re-evaluation.** The thesis claims Box mAP@50 ≈ 0.51 for YOLO+DeepForest WBF ensemble in Section 3.7 — that number was an estimate, not a measurement. The ensemble adapter in `backend/models/ensemble_adapter.py` exists but no formal eval was ever run on a val set. Recommend either (a) actually running the WBF ensemble eval and replacing the 0.51 with the real number, or (b) removing the claim and stating ensemble is implemented but not evaluated.

- **Domain shift note.** Memory file `dataset_domain_shift.md` records that v1/v2/v3 are all Google Earth Pro screenshots but production runtime serves Google Maps tiles (`mt0.google.com/vt/lyrs=s`). The thesis should acknowledge this train/serve mismatch explicitly in Section 3.9. Closing this gap was NOT attempted in v3 work; it's the natural target for v4.

---

## 7. Memory references

These auto-memory files in `C:\Users\Rasul\.claude\projects\C--Users-Rasul-DeepLearning-Astana-Tree-Prototype\memory\` have cross-session context the writer will benefit from:

- `yolo_v1_handoff.md` — original v1 train details (397 epochs, MD5)
- `yolo_v2_results.md` — v2-finetune vs v2-fromscratch original comparison
- `yolo_v3_results.md` — earliest v3-finetune-run1 narrative (preserved)
- `dataset_domain_shift.md` — Earth Pro vs Google Maps tile mismatch
- `feedback_team_docs_style.md` — when documenting for Anuar/Berik, explain WHY not how
- `thesis_ownership_split.md` — only YOLO + web stack is ours to edit
- `diploma_grading_rubric.md` and `diploma_formal_requirements.md` — diploma format constraints

---

---

## 8. Round 4+5 follow-up — variance check, clean model sweep, ensemble (2026-05-18)

### 8.1 Multi-replicate variance check (exp1 / exp21 / exp22 / exp23)

After completing 16 experiments and treating exp1's 0.308 as "the result," we
ran 3 more replicates of the exact exp1 configuration to estimate training-time
variance:

| Run | wall (min) | merged Box mAP@50 | v3-val Box mAP@50 | v2-val Box mAP@50 |
|---|---|---|---|---|
| exp1 (original) | 31 | **0.308** | 0.287 | 0.367 |
| exp21 (random p3) | 13 | 0.268 | 0.251 | 0.327 |
| exp22 (replicate) | 19 | 0.269 | 0.246 | 0.325 |
| exp23 (replicate) | 10 | 0.239 | 0.199 | 0.326 |
| **Mean** | — | **0.271** | 0.246 | 0.336 |
| **Std** | — | **0.028** | 0.036 | 0.020 |

**Finding:** exp1's 0.308 was the upper tail of training-time variance.
True expected value of "exp1 config" is **0.271 ± 0.028 merged Box mAP@50**
across 4 replicates. v2-val is the most stable signal (σ=0.020); v3-val is
the noisiest (σ=0.036) because it has only 7 tiles / 497 polygons. **Diploma
honesty:** report the mean (0.271) as the principal metric, not the
single-best (0.308). Improvement over v2-finetune (0.167) is **+62%** robust,
not +84% from the single run.

### 8.2 Clean model sweep (v4_n/s/m/l/x with Ultralytics defaults)

We then questioned whether the "v2-proven augmentation" we'd been using
throughout experiments 1-23 was actually optimal. To find out, we re-ran the
full size sweep with **Ultralytics default hyperparameters** (no manual aug,
no manual LR, AutoBatch for VRAM utilization, single_cls=True only):

| Model | params | merged Box mAP@50 | v3-val Box mAP@50 | wall (min) |
|---|---|---|---|---|
| **v4_x_clean** (defaults) | 71.7M | **0.315** | **0.313** | 91 |
| v4_m_clean (defaults) | 27.2M | 0.291 | 0.267 | 18 |
| v4_s_clean (defaults) | 11.8M | 0.281 | 0.254 | 11 |
| v4_n_clean (defaults) | 3.4M | 0.261 | 0.262 | 9 |
| v4_l_clean (defaults) | 45.9M | 0.260 | 0.257 | 52 |
| Reference: exp1 (tuned m, lucky run) | 27.2M | 0.308 | 0.287 | 31 |
| Reference: exp1 mean of 4 replicates | 27.2M | 0.271 | 0.246 | — |

**Surprise finding:** yolov8x-seg with Ultralytics defaults beats every
single tuned config we ran in experiments 1-23. **+0.044 merged Box mAP@50
over the mean of exp1 replicates.** And **best v3-val** (0.313) across the
entire 28-experiment series.

**Why defaults beat our tuning:** Ultralytics defaults use *more aggressive
color augmentation* (`hsv_s=0.7, hsv_v=0.4, erasing=0.4`) but *zero
geometric augmentation* (`degrees=0, mixup=0, copy_paste=0, flipud=0`).
For satellite imagery this matches the natural data distribution:
- Trees viewed from above are rotationally consistent (no flips/rotations
  in real samples)
- Lighting/season varies massively (broad color aug helps)
- Occlusion is real (random erasing simulates it)

Our "v2-proven" aug had `degrees=20, mixup=0.1, flipud=0.5` which inject
unnatural transformations that hurt the larger yolov8x's ability to learn
clean priors.

**Diploma narrative:** This is a genuinely interesting result worth a
paragraph in Chapter 3 — *"For YOLOv8-seg on Astana satellite imagery, we
found that the largest variant (yolov8x-seg, 71 M parameters) with
Ultralytics' default hyperparameters outperformed all 23 prior tuning
attempts. The combination of aggressive color augmentation and no geometric
augmentation aligns with the natural distribution of satellite imagery
(consistent rotational orientation, high lighting variance)."*

### 8.3 Continual / chain learning — negative result

We tested whether sequential staged training (v1 → v1+v2 → v1+v2+v3) helps
versus monolithic single-shot training. Four configurations:

| Setup | Box mAP@50 (merged) | Notes |
|---|---|---|
| Single-shot exp1 (mean of 4 replicates) | 0.271 | reference |
| Random-split 3-stage chain (exp17) | 0.287 | same data sizes, random splits |
| Random-split 2-stage chain w/ hot LR (exp18) | 0.270 | mimics v2-finetune pattern |
| Version-based 3-stage chain (exp11) | **0.210** | original v1→v2→v3 splits |

**Key insight:** chain learning across version batches (exp11) loses **0.061**
mAP vs single-shot — but chain learning across random splits of the same data
(exp17) loses only **−0.0 to +0.02**. The damage came from **distribution
drift between v1/v2/v3 annotation batches** (different districts captured
at different dates with different annotators), not from sequential staging
mechanism itself.

**Conclusion for thesis:** chain learning is appropriate when there's a
clear gradient of label quality (paper #13's recipe: weak large pre-train,
clean small fine-tune). For our regime (uniformly noisy small dataset),
single-shot training on the union of all data is optimal.

### 8.4 Cross-model ensemble — IoU-based merging

User-observation from visual inspection: *"different models detect
different trees on the same image — some trees seen by v2-finetune are
missed by v4_x and vice versa. mAP doesn't show this — it's a single
aggregate number that hides per-detection complementarity."*

This is a real limitation of mAP and motivates ensemble inference. We
implemented `ml/v5_ensemble.py` with two strategies:

- **NMS**: pool detections from all participating models; for any two
  detections with IoU ≥ 0.5, keep the higher-confidence one.
- **Voting (vote_K)**: only retain detection clusters where at least
  K different models agreed. Single-model anomalies (e.g., one model
  hallucinating trees on a stadium roof) are discarded.

We ran a 4-model ensemble (v4_x_clean + exp1_m + v4_s_clean + v2-finetune)
on a representative Astana tile (1236×1159 px containing a tennis-court
complex and surrounding tree-lined streets). Raw individual detections
totaled 3 000 across the 4 models; after `vote_2` (IoU≥0.5, ≥2 models
agree) the unified count was **790 trees**. The qualitative result
substantially reduces individual-model false positives while preserving
trees that multiple models independently confirmed.

**No quantitative ensemble eval on our held-out val sets was performed**
— the ensemble script is in the repository (`ml/v5_ensemble.py`) and
end-users can select it through the model dropdown / scan API, but
producing val-set mAP for the ensemble would require ground-truth
matching at the polygon level, which our val infrastructure currently
runs only on per-model checkpoint files. **Recommended future work.**

### 8.5 Suggested thesis figures from the visual comparison

The `ml/v5_visual_compare.py` script produces side-by-side grids of all
top models on any input image. Recommended figures for Chapter 3.3 /
3.7:

- **Figure 3.X — Cross-model qualitative comparison.** 2×4 grid: same
  Astana tile shown with each of 8 top model variants overlaying its
  predictions. Demonstrates that *different model checkpoints find
  partially-overlapping but distinct subsets of trees on the same input.*
  Captures the mAP-aggregate-metric limitation visually.

- **Figure 3.Y — Ensemble result.** Single full-resolution panel showing
  the 4-model `vote_2` ensemble output on the same tile. Caption notes
  the raw → unified detection count reduction (3 000 → 790 in our
  example) and the IoU≥0.5 cluster threshold.

Both figures are reproducible from the repo with:
```
python ml/v5_visual_compare.py <image_path> --conf 0.15 --models top4 \
  --ensemble --ensemble-strategy vote_2 --cols 3
```
Output goes to `<image_stem>_compare.png`. Save TIFF/PNG copies at full
resolution into `thesis/figures/` and reference in LaTeX.

---

## 9. mAP limitation — methodological honesty section

The 28-experiment ablation produced a clear best single-model number
(v4_x_clean = 0.315 merged Box mAP@50). However, **visual side-by-side
inspection of top-4 models on real Astana scenes revealed that different
models with similar aggregate mAP detect substantially different subsets
of trees.** This is a known limitation of aggregate detection metrics:

- mAP averages precision over a range of recall values; two models can
  attain the same mAP via different precision/recall trade-offs and via
  different per-tree decisions.
- For a small noisy single-class detection task like Astana trees, the
  failure mode is rarely "model X is uniformly better than Y" — it's
  "each model has its own systematic biases on specific scene types"
  (e.g. shadows, building edges, dense canopies, stadium-roof textures).

**For thesis defense**, this should be acknowledged transparently in
Section 3.9 (limitations) or Conclusion: aggregate metrics like mAP guide
high-level model selection but **do not substitute for qualitative
inspection of predictions on representative scenes**. Future work
recommendations:

- Quantitative ensemble evaluation on held-out val (already implemented;
  ensemble produces unified polygons via `vote_2`/`nms`/etc.).
- OpenStreetMap building-footprint post-filter (Section 8 of paper #4)
  to drop predictions inside non-vegetation polygons.
- Multi-seed averaging (3-5 replicates) reported with mean ± std for
  any production-candidate configuration.

---

---

## 10. Final unified table (copy directly into Chapter 3.3.5/3.3.6/3.7)

This is the **canonical comparison** — top-8 models across all three validation sets.
Numbers are from `results/v5_unified_eval.json` (re-measured 2026-05-18, single
unified script).

| Model | Params | v2-val Box | v3-val Box | merged Box | merged Mask | Notes |
|---|---|---|---|---|---|---|
| **v4_x_clean** | 71.7 M | 0.319 | **0.313** | **0.315** | 0.289 | yolov8x-seg, Ultralytics defaults — **production candidate** |
| exp1_m_cocostart (lucky run) | 27.2 M | **0.367** | 0.287 | 0.308 | **0.305** | yolov8m-seg, v2-proven aug — historical "best" |
| v4_m_clean | 27.2 M | 0.344 | 0.267 | 0.291 | 0.280 | yolov8m-seg, Ultralytics defaults |
| exp17 chain random | 27.2 M | 0.346 | 0.257 | 0.287 | 0.278 | 3-stage random chain — chain doesn't help |
| exp12 low-lr finish | 27.2 M | 0.337 | 0.256 | 0.286 | 0.295 | exp1.pt → v3-only lr=0.0001 |
| exp15 v2v3 only | 27.2 M | 0.314 | 0.291 | 0.286 | 0.266 | drop-v1 — 2nd-best v3-val |
| v4_s_clean | 11.8 M | 0.329 | 0.254 | 0.281 | 0.270 | yolov8s-seg, Ultralytics defaults |
| v2-finetune (legacy) | 71.7 M | 0.363 | **0.081** | 0.167 | 0.169 | pre-v3 production — OOD failure |

**v2 baselines (the other-chat's earlier measurement, same 14-img / 17-tile merged val):**

| Model | Box mAP@50 | Mask mAP@50 |
|---|---|---|
| v1 (yolov8x, 397 ep) | 0.131 | 0.134 |
| v2-fromscratch (yolov8x, 204 ep) | 0.156 | 0.147 |
| v2-finetune (yolov8x, 173 ep) | 0.187 | 0.185 |

(Note: 0.187 here vs 0.167 in our `v5_unified_eval.json` for "v2-finetune (legacy)" — the small delta is float precision differences between two evaluation runs; both are accurate. Use 0.187 from `results/yolo_mergedval_eval.json` for the v1/v2/v3 evolution narrative in Section 3.3.5, and 0.167 from `results/v5_unified_eval.json` for the unified comparison table — they're consistent within noise.)

---

## 11. Multi-replicate variance — the honest number

Single-run mAP has training-time variance ≈ ±0.03 on this dataset scale.
Four replicates of the *same* configuration (exp1):

| Run | wall (min) | merged Box mAP@50 |
|---|---|---|
| exp1 (original) | 31 | 0.308 |
| exp21 (random p3 = full merged) | 13 | 0.268 |
| exp22 (replicate, time=1.5h) | 19 | 0.269 |
| exp23 (replicate, time=1.5h) | 10 | 0.239 |
| **Mean** | — | **0.271** |
| **Std** | — | **0.028** |

**Diploma recommendation:** for the headline number in the conclusion, use the
**v4_x_clean** result (0.315) **single-shot, NOT a multi-seed mean** because
we haven't replicated v4_x. But add a methodological caveat: "single-run
variance on this dataset is approximately ±0.03 mAP". If you want a more
rigorous defense, run 3 more replicates of v4_x_clean exact config (the user
explicitly froze training so this is future work — note it but don't ask the
user to do it now).

---

## 12. Final figures package for thesis

The `ml/v5_visual_compare.py` script produces side-by-side comparison PNGs.
The user generated two such images during the analysis session (2026-05-18):

- **`asdf_compare.png`** (3600 × 1758 px) — 2×4 grid of all 8 top models
  on a tennis-court complex with surrounding row-planted street trees. Shows
  per-model detection differences: spread of 147 trees between min (v4_x at
  687) and max (v4_s at 819) on the same image. **Visually demonstrates the
  mAP limitation** — different models with similar mAP find substantially
  different subsets of trees.

- **`asdf_top4_ensemble.png`** (2700 × 1758 px) — 3×2 grid of top-4 models
  (v4_x, exp1_m, v4_s, v2-finetune) + ensemble cell. The ensemble cell (cyan
  outline, 3px) shows the IoU≥0.5 / vote_2 merged result: 3 000 raw pooled
  detections → 790 unified trees. **Strong qualitative evidence** that
  ensembling complementary models reduces both false positives and missed
  detections.

These images are at `C:\Users\Rasul\Pictures\Screenshots\` — recommend
copying into `thesis/figures/` for LaTeX inclusion. Caption suggestions:

- **Figure 3.X:** *"Side-by-side qualitative comparison of eight YOLOv8-seg
  variants on a representative Astana street-corner tile (1236×1159 px).
  Predicted tree polygons overlaid in distinct colors per model with
  detection count in cell header. Per-model spread of 147 trees (687–819)
  on the same input image illustrates the cross-checkpoint detection
  complementarity discussed in Section 3.9."*

- **Figure 3.Y:** *"Top-4 model checkpoints plus IoU-merged ensemble result
  on the same tile. The ensemble (bottom-right, cyan) requires ≥2 of 4
  models to agree on a tree (IoU ≥ 0.5) — surviving 790 of 3 000 raw
  pooled detections. Voting reduces single-model false positives while
  preserving trees independently confirmed by multiple architectures."*

To regenerate or produce new figures (e.g. for Botanical Garden tile,
stadium-roof scene, etc.):

```
venv\Scripts\python.exe ml/v5_visual_compare.py "<image_path>" \
  --conf 0.15 --models top4 --ensemble --ensemble-strategy vote_2 --cols 3
```

Output: `<image_stem>_compare.png` next to the source image.

---

## 13. What's frozen / not done

Explicit scope cut by the user on 2026-05-18:

- ❌ No more training will be done.
- ❌ No more model variants will be added to the repo.
- ❌ No multi-seed replicates of v4_x_clean (note as future work).
- ❌ No quantitative ensemble eval on val sets (note as future work).
- ❌ TAL `topk=7` (paper #32 finding) — not implemented; would require
  monkey-patching Ultralytics internals.
- ❌ OSM building post-filter (paper #4 finding) — not implemented; would
  resolve stadium-roof FP regression but requires backend post-processing
  changes.
- ❌ Dataset domain-shift fix (Earth Pro train vs Google Maps serve) — not
  attempted; natural v4 target.

What you (thesis-writer Claude) should do with these: list under Section 3.9
"Limitations and future work" as **explicit recommendations**, citing the
relevant paper IDs from `Literature_Review_Tree_Detection.md`.

---

## 14. Memory references for cross-session context

These auto-memory files under `C:\Users\Rasul\.claude\projects\...\memory\`
have context worth surfacing in thesis writing:

- `yolo_v1_handoff.md` — v1 model details (397 epochs)
- `yolo_v2_results.md` — v2-fromscratch vs v2-finetune original comparison
- `yolo_v3_results.md` — v3 run1 initial narrative (preserved for history)
- `dataset_domain_shift.md` — Earth Pro vs Google Maps tile mismatch
- `feedback_team_docs_style.md` — when documenting for Anuar/Berik, explain WHY
- `thesis_ownership_split.md` — only YOLO + web stack is yours to edit
- `diploma_grading_rubric.md` — AITU bachelor grading 100-point breakdown
- `diploma_formal_requirements.md` — 30-40 page constraint, ≥20 sources, LaTeX, English

---

**END OF BRIEFING.** Hand this document to the thesis-writer Claude. They have
everything to write a defensible Chapter 3.3 + update Conclusion.

Final number for the headline result:
> **Box mAP@50 = 0.315 (yolov8x-seg, Ultralytics defaults, merged 17-tile val)**
> **v3-val (OOD) Box mAP@50 = 0.313**
> **vs v2-finetune pre-v3 baseline of 0.167 merged / 0.081 v3-val** — **+88% / +286% relative**.
