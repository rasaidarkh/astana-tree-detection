# Diploma rewrite plan

**Status of read-through (2026-05-18 night):**
- 00_title.md ✓
- 01_abstract.md ✓
- 01b_declaration.md (boilerplate, OK)
- 01c_definitions.md (boilerplate, OK)
- 02_intro.md ✓
- 03_chapter1.md (Literature Review) ✓
- 04_chapter2.md (Methodology) ✓
- 05_chapter3.md (Experiments) ✓
- 06_conclusion.md ✓
- 07_references.md (not deeply read, needs check)
- Diploma_70_Defense_FINAL.pptx (historical context, April 2026) ✓
- AITU 2025-2026 LaTeX template ✓
- Classmate diploma reference (ML fraud detection, 76p) ✓
- v3_experiments_briefing.md (this session's full record) ✓

---

## Critique — what's good and what's broken

### ✅ Strengths

1. **Comprehensive scaffold** — all 6 objectives addressed, problem statement, 31 papers, M14 cross-model val
2. **Strong methodology** — adapter pattern, tiled inference, four-mode geo, three exporters all real
3. **Honest empirics** — 0.012 NEON baseline, 16-experiment ablation, cross-model M14 comparison
4. **Working artifact** — full backend/frontend running, 8 YOLO variants + 3 other detectors
5. **Limitations section exists** (3.9) — covers M14 size, single class, stadium FP, ensemble gap
6. **Voice is academic & defensible** — not overhyped, acknowledges trade-offs

### ❌ Critical issues blocking defense

**1. Numbers are stale — v4 sweep is missing entirely from the thesis.**
- Section 3.3.7 says "16-experiment ablation". Actual count today: **28+ experiments** across 5 rounds
- **v4_x_clean (0.315 merged Box mAP@50)** is unbeaten champion across all 28 experiments. Thesis still has exp1 (0.308) as production hero.
- **Multi-replicate variance** (exp21/22/23 = 0.268/0.269/0.239 vs exp1's lucky 0.308) — mean 0.271±0.028 — is in briefing, missing from thesis.
- Random-chain ablation (exp17/18) is missing — strong negative result showing distribution drift was the real killer in exp11, not staging mechanism.

**2. Cross-YOLO ensemble missing from thesis.**
- `ml/v5_ensemble.py` (4-model vote_2 over IoU≥0.5 clusters) is implemented and working in production. Backend has `yolo_ensemble` adapter registered.
- Empirical demonstration: 3000 raw → 790 unified detections on test image.
- Thesis Section 3.7.3 says "WBF not evaluated on M14, future work" — but the cross-YOLO vote-based ensemble exists and the user has visual proof.

**3. Production weights ambiguity.**
- Conclusion says "exp1 (yolov8m, 27M) is production".
- v4_x_clean (yolov8x, 71M, defaults) is the measured best at 0.315.
- Current `weights/yolo_satellite.pt` = exp1 (verified MD5).
- Must decide: deploy v4_x as production (then thesis hero becomes 0.315 + new architecture story), OR explain that v4 was added during thesis writing as "additional validation that the size sweep is U-shaped".

**4. Abstract & intro overstate scope.**
- Abstract: "16-experiment hyperparameter ablation across five orthogonal axes". Actual delivered = 28 experiments × 7 vals = 196 metric measurements, not 16.
- Intro Section "Scientific novelty" mentions the 16-exp ablation. Needs update to mention v4 sweep + variance check + random-chain control.

**5. Stadium-FP discussion (3.9) is correct but underdeveloped.**
- User's actual observation was broader: **different models with similar mAP detect different trees** (mAP limitation).
- Briefing Section 9 has the full discussion. Thesis only mentions stadium-roof FPs as one instance.
- Should be promoted to a separate sub-section or expanded paragraph: "The aggregate mAP metric is insufficient for production model selection; visual cross-checkpoint inspection on Astana scenes reveals substantial per-detection complementarity..."

**6. Visual comparison figures missing.**
- User generated two excellent figures during 2026-05-18 session:
  - `asdf_compare.png` — 2×4 grid of 8 model variants on a tennis-court complex tile, 687–819 detection range = visual proof of mAP limitation
  - `asdf_top4_ensemble.png` — 4-model + ensemble cell (3×2 grid)
- Currently sit at `C:\Users\Rasul\Pictures\Screenshots\`. Should be copied to `thesis/figures/` and referenced as Figure 3.X / 3.Y.

**7. Title still says "Astana Tree Detection"; product brand is "Canopy".**
- Defensible — title is the academic topic, "Canopy" is the product. But abstract / intro should mention the product name once for grounding.

**8. PowerPoint historical context (70% defense, April 2026) shows enormous progress that the thesis should narrate.**
- April: "DeepForest ~70% on Astana, YOLOv8 failed, Wellington failed". This was the starting point.
- May: full ablation, 0.308 YOLO, working pipeline.
- The progress narrative is hidden in the chronology. Could be a paragraph in the introduction or conclusion's "what we learned" section.

**9. References file (07_references.md) needs verification.**
- 31 papers cited per memory. Need to confirm:
  - All `[@Citekey]` references in body resolve to references.md entries
  - Bibtex equivalent (`thesisbiblio.bib`) is in sync for LaTeX build
- ALSO: the new evidence base from this session (Ultralytics docs, SAM2 paper, paper #21 satellite-tree benchmark) — need to check if all already in the 31.

**10. LaTeX skeleton at `thesis/latex/` is incomplete.**
- `chapter01/introduction.tex` — Tengxiang Li template placeholder, not our content
- `chapter02/main.tex`, `chapter03/conclusion.tex` — exist but small
- The markdown is the source of truth; need to render to LaTeX for actual defense submission per AITU formal requirements.
- AITU template extracted to `C:\Users\Rasul\DeepLearning\AITU_template_extract\` for reference.

### ⚠️ Minor inconsistencies

- "Section 1.5 reports the first measurement" — true but the 0.012 number appears in Sections 1.5, 3.5.1, 3.7.2, 4.7.4, Conclusion. Some duplication is OK but should be cross-referenced.
- "approximately 100 source images" — exact count is 101 (16 v1 + 28 v2 + 19 v3 train + 5 v1 + 5 v2 + 5 v3 val = 78 used in M14 / +remainder). Should be precise: "78 annotated images" or break out by phase.
- "approximately 8 700 polygon annotations" — exact: 4733 train + 726 val + tiling produces 8760 train tiles + 755 val polygons. Mix of "source-image" vs "tile-level" polygons throughout, need to disambiguate consistently.
- Section 3.5.2 says "training trajectory NEON → v4 → v3" — but earlier sections (2.6.3) describe v4 as Roboflow-only. Need consistent narrative about what v4 was.

---

## Rewrite roadmap

**Order chosen to minimise re-work:** start with abstract and conclusion (top + bottom of thesis), since they synthesise everything below. Then update Chapter 3 (the main empirical contribution). Chapter 2 (methodology) needs only minor edits. Chapter 1 (lit review) is essentially complete.

### Phase A — Numbers consolidation (priority, no new content)

**A1.** Decide production weights: keep exp1 (yolov8m, 0.308 lucky) OR promote v4_x_clean (yolov8x, 0.315 measured). Recommend: **promote v4_x_clean to production** because (a) it's our measured best, (b) it's also the best on out-of-distribution v3-val (0.313 vs exp1's 0.287), (c) it eliminates the "lucky run" awkwardness. Story becomes: "the systematic ablation discovered that yolov8x-seg with Ultralytics' default hyperparameters outperforms our hand-tuned configurations" — cleaner narrative.

**A2.** Update abstract — headline numbers, scope (28 experiments), product name.

**A3.** Update conclusion — final numbers, contributions list (variance check, v4 sweep, cross-YOLO ensemble).

### Phase B — Chapter 3 expansion (largest delta)

**B1.** Section 3.3.7 — extend the 16-experiment ablation to 28 experiments. Add:
- 3.3.7.4 v4 clean-defaults sweep (Round 4): table of n/s/m/l/x with defaults, finding that x with defaults beats all hand-tuned configs.
- 3.3.7.5 Multi-replicate variance check (Round 5): exp1/21/22/23, mean 0.271 ± 0.028.
- 3.3.7.6 Random-chain control (exp17/18): isolates distribution drift as the killer.

**B2.** Section 3.7.5 NEW — Cross-YOLO ensemble. Reuse the test-image figure (asdf_top4_ensemble.png). Explain IoU≥0.5 clustering + vote_2 voting. Note that quantitative M14 eval still future work; qualitative result is strong (3000 → 790 unified).

**B3.** Section 3.9 — promote the broader mAP-limitation discussion to its own paragraph. Include the 687–819 cross-model spread number from the test image as evidence.

**B4.** Copy `asdf_compare.png` and `asdf_top4_ensemble.png` to `thesis/figures/`. Add LaTeX-style caption references in Chapter 3.

### Phase C — Methodology touch-ups

**C1.** Section 2.4.1 — already correctly mentions both yolov8x-seg (early) and yolov8m-seg (production). Update if production switches to v4_x_clean (Phase A1).

**C2.** Section 2.8 — Weighted Box Fusion. Add a paragraph at the end mentioning the cross-YOLO vote-based ensemble (`backend/models/yolo_ensemble_adapter.py`) as a second ensemble strategy available in the production system.

**C3.** Section 2.11.3 — Detection display modes — already mentions four modes. Verify match with current UI (which has hierarchical model picker as of commit f01a18b, modal-based as of 6d0c341, error boundary cd037b4, scan modal fix bf624cd, adapter fixes f3fbb20).

### Phase D — Frontmatter & references

**D1.** Verify `07_references.md` has all 31 entries and matches all `[@Cite]` mentions in body. Generate the BibTeX `thesisbiblio.bib` for LaTeX.

**D2.** Update `01_abstract.md` per Phase A2.

**D3.** Optional: add a paragraph in `02_intro.md` mentioning the April → May progression (PPTX historical anchor).

### Phase E — LaTeX build

**E1.** Migrate markdown chapters to LaTeX per AITU template structure. The build pipeline lives in `thesis/build_latex.py`. Render and check PDF compiles cleanly.

**E2.** Verify figures resolve in LaTeX (relative paths to `figures/`).

**E3.** Cross-reference labels (table, figure, equation, citation) all resolve.

### Phase F — Final pass

**F1.** Spell-check + grammar across all chapters.

**F2.** Page count check (AITU requires 30–40 pages main body).

**F3.** Plagiarism self-check via `Antiplagiat.ru` or similar (≥70% unique per AITU requirement per memory).

**F4.** Final read-through.

---

## Suggested execution order (overnight goal)

1. **PHASE A (numbers consolidation)** — 1 h
   - Decide production weights swap (or document the dual)
   - Rewrite abstract
   - Rewrite conclusion
2. **PHASE B (Chapter 3 expansion)** — 2 h
   - Section 3.3.7 expansion
   - Section 3.7.5 new
   - Section 3.9 expansion
   - Copy figures
3. **PHASE C (methodology touch-ups)** — 0.5 h
4. **PHASE D (frontmatter & refs)** — 0.5 h
5. **PHASE E (LaTeX build)** — 1 h
6. **PHASE F (final pass)** — 0.5 h

**Total estimate: ~5.5 hours.** Achievable autonomously if I commit after each phase so user can review pieces tomorrow morning.

---

## Notes for any successor session

- **Trust hierarchy:** the user has explicitly granted me ownership across all sections, overriding the memory note `thesis_ownership_split.md`. Mention this if Anuar/Berik flag changes to DeepForest/Mask R-CNN sections — those edits are documented in this plan.
- **Source of truth for numbers:** `results/v3_experiments.json`, `results/v4_clean_modelsweep.json`, `results/v5_unified_eval.json`, `results/yolo_mergedval_eval.json`. Run `python results/summarize_v3.py` for live leaderboard.
- **Source of truth for code:** all commits on branch `design-rethink`. Latest at time of writing: `f3fbb20` (adapter fixes), `bf624cd` (scan modal fix), `cd037b4` (error boundary), `6d0c341` (centered model modal), `f01a18b` (hierarchical picker + v4 variants).
- **AITU formal:** 30–40 pages main body (IT variant), ≥20 references, plagiarism ≥70% unique, **English language**, LaTeX, defense window 1–6 June 2026.
