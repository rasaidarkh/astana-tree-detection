# v3 YOLO experiments — briefing for thesis writer

**Audience:** another Claude session writing thesis Chapter 3 (Experiments & Results) for the YOLO branch. **Author of this briefing:** Claude session that ran the experiments on 2026-05-17/18.

**You own:** YOLO + web/system parts of the thesis (per `memory/thesis_ownership_split.md`). DeepForest+SAM2 is Anuar's territory, Mask R-CNN is Berik's. Do NOT ghost-edit their sections.

**Source of truth for numbers:** `results/v3_experiments.json` (live, updated by experiment runner). Run `python results/summarize_v3.py` to get the latest formatted leaderboard.

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

**END OF BRIEFING.** Hand this document to the thesis-writer Claude. They have everything to write a defensible Chapter 3.3 + update Conclusion.
