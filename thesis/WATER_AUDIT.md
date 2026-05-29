# Water audit — what to cut to hit 30-40 pages

Read-through completed 2026-05-19. The thesis is **121 pages PDF for a 30-40
AITU-bachelor budget**. Approximately 60-65 % of the body text is water by my
rough estimate. Below is a concrete cut-list, ranked by impact (estimated
pages removed) and risk (impact on substance).

## TL;DR — three cuts that recover ~40 pages

1. **Collapse Chapter 3.3 from 12 sub-sections into 1 table + 3 paragraphs.** ≈ −15 pages, no substance loss.
2. **Cut Chapter 2 boilerplate** (per-branch architecture/loss math, REST endpoints table, three duplicate architecture overviews). ≈ −12 pages.
3. **Compress Chapter 1 paradigm-by-paradigm survey** into a single comparative paragraph + Table 1.2. ≈ −10 pages.

Total: **~37 pages** cut → fits AITU budget cleanly.

---

## Chapter-by-chapter audit

### Chapter 1 — Literature review (28 KB markdown / ≈ 20 PDF pages)

**Status:** Most water of any "essential" chapter. Has 5 paradigm sub-sections each describing 3-5 papers extensively, when commission only cares about which methods we picked and why.

**Worst offenders:**

- **§ 1.1 (urban-tree inventory as applied problem)** — 5 paragraphs explaining what tree inventories are and why municipalities need them. **Reader knows.** Cut to 1 paragraph linking to Zelenstroy use-case.
- **§ 1.2 (data sources)** — Three classes of alternatives (LiDAR / UAV / satellite) described, but we only use satellite. The LiDAR + UAV paragraphs justify negative choices nobody questions. Cut to: "We use satellite imagery because UAV and LiDAR are not available for Astana at our scale. Table 1.1 lists data sources covered in the surveyed literature." Done.
- **§ 1.3.1 — 1.3.5** — Each paradigm has its own multi-paragraph survey. The 5 sub-sections currently span ~10 pages. **Replace with one comparative table** (method family / best representative work / reported result / why chosen-or-not in this work) and 1 paragraph per family explaining the project's choice. **−6-7 pages.**
- **§ 1.3.3 (semantic segmentation)** — We don't use semantic segmentation. Currently 1 page describing U-Net, DeepLab, HRNet variants. Cut to 2 sentences: "Semantic segmentation does not produce per-tree instance identities and was not selected. The closest precedent on Astana morphology is [@SofiaDeepForest2024] (Section 1.3.4 below)."
- **§ 1.4 (Table 1.2)** — Currently has 14 rows of literature baselines plus 5 rows of our own results plus 3 observation paragraphs. The 5 own-result rows belong in Chapter 3, not Chapter 1. Move them. **−3-4 rows in Ch 1, +0 rows elsewhere (already in Ch 3 Table 3.5).** Two observation paragraphs (the "best numbers are very high" + "all best results are non-Central-Asian") can be 1 paragraph.
- **§ 1.5 (geographic generalisation gap)** — 4 explanatory bullets about why the gap exists (floristic / morphology / acquisition / annotation). Currently each is a paragraph. Could be one paragraph with the four causes listed.
- **§ 1.6 (problem statement)** — Formal `Given / Produce / Subject to` block. Currently includes an `s_i ∈ [0, 1]` confidence definition and a polygon-mask definition that are also in 01c_definitions.md. Single source of truth.

**Recommended target:** Chapter 1 → **8-10 PDF pages** (from ~20). Keep Table 1.2 (lit baselines), keep §1.5 first paragraph (the 0.012 number anchor), drop everything else to summary form.

### Chapter 2 — Methodology (63 KB markdown / ≈ 35 PDF pages)

**Status:** Heaviest chapter, large fraction water. Mostly because each model branch (YOLO, Mask R-CNN, DeepForest, SAM 2) has its own architecture + training data + procedure + loss sub-sections, and most of the content is well-known framework details a CS commission already knows.

**Worst offenders:**

- **§ 2.1 (system architecture)** — Currently has: prose overview, 6-step data flow, adapter-pattern paragraph, Table 2.1 technology stack, **another paragraph repeating the three-tier separation**, and Figure 2.1 ASCII architecture diagram. **Three overlapping descriptions of the same architecture.** Pick one: keep Figure 2.1 + Table 2.1, cut the prose to 2 paragraphs.
- **§ 2.2 (image input categories)** — Three input paths (regular images / GeoTIFF / map capture) each get a multi-paragraph description. The map-capture path has a 4-step list AND a separate paragraph AND Table 2.1a tile providers. Collapse to ~6 sentences + Table 2.1a.
- **§ 2.3 (tiled processing)** — Currently 3 paragraphs justifying why we tile. The justification is well-known and can be 1 paragraph: "Image dimensions exceed network input; we tile at 640+128 overlap and apply global NMS." Done.
- **§ 2.4 (YOLO branch)** — 5 sub-sections (architecture / why-instance-seg / training data / procedure / loss function).
  - **2.4.5 (loss function)** has detailed CIoU equation derivation. **Reader knows what CIoU is.** Cut to: "We use the YOLOv8 default loss formulation: CIoU box regression + BCE classification + DFL + BCE mask (loss weights 7.5 / 0.5 / 1.5 / 1.0)." **−1 page.**
  - **2.4.4 (training procedure)** has Table 2.2 with 14 hyper-parameters. Useful for reproducibility. Keep.
  - **2.4.3 (training data)** describes v1 → v2 → v3 evolution which is also in Chapter 3.2. **Duplicate. Cut Chapter 2 version** and link to Section 3.2.
- **§ 2.5 (Mask R-CNN)** — Mirror structure of 2.4 (architecture / training data / procedure / adapter). The architecture paragraph re-explains Mask R-CNN basics that the reader can read from Lv 2023 or He 2017 directly. Cut to: "Standard `maskrcnn_resnet50_fpn_v2` from torchvision, with the two RoI heads replaced for `num_classes = 2` (background + tree). Hyper-parameters in Table 2.3." **−1.5 pages.**
- **§ 2.6 (DeepForest)** — Similar architectural re-explanation of RetinaNet + focal loss. Same treatment: keep paragraph about why we picked DeepForest and the Table 2.3a hyper-parameters; cut the architecture re-derivation.
- **§ 2.7 (SAM 2)** — Three paragraphs explaining what SAM and SAM 2 are. Cut to one. The four-step pipeline list is essential, keep it.
- **§ 2.8 (Ensemble strategies)** — Now has two sub-sections.
  - 2.8.1 WBF — long mathematical derivation. Equations can stay; the "5-step procedure" list duplicates the math. Pick one.
  - 2.8.2 Cross-YOLO vote — just added in Phase C, length is OK.
- **§ 2.9 (Geographic conversion)** — Four modes each with a math equation. The first two (GeoTIFF affine, bilinear) are commodity GIS math. Could be summarised in a table; equations only for the non-trivial one (two-corner axis-aligned interpolation).
- **§ 2.10 (Persistent storage + export)** — Three tables-cascade-delete-rationale. The cascade-delete behaviour is one sentence not a paragraph.
- **§ 2.11 (Frontend application)** — Four sub-sections. Most is description of the UI ("the user clicks here, then clicks here..."). Cut to one page with one figure showing the two view modes. Table 2.4 REST endpoints (15 endpoints with descriptions) is fine for an appendix; **move out of Chapter 2 main body**.
- **§ 2.12 (Summary)** — Paragraph that says "this chapter presented" and recaps every section. **Pure water. Delete entirely.** Chapter 3 introduction already references back to Chapter 2 sections by number.

**Recommended target:** Chapter 2 → **15-18 PDF pages** (from ~35).

### Chapter 3 — Experiments and results (80 KB markdown / ≈ 45 PDF pages)

**Status:** Largest chapter, biggest signal-to-noise issue. **Section 3.3 alone has 12 sub-sections after my Phase B expansion** — each round of the ablation got its own table + paragraph.

**Worst offenders:**

- **§ 3.1 (hardware)** — Table 3.0 listing three workstations is the entire useful payload. Two paragraphs of narrative about which workstation did what and why VRAM was the bottleneck can be 2 sentences after the table.
- **§ 3.2 (dataset)** — Five sub-sections (3.2.1 strategy / 3.2.2 v1 / 3.2.3 v2 / 3.2.4 v3+M14 / 3.2.5 pipeline). Each has a stat table. Could consolidate: **one master table** (v1 / v2 / v3 / merged with columns source-imgs / train-tiles / val-tiles / polygons), one paragraph on the model-in-the-loop pre-labelling workflow, one paragraph on M14 construction. **−3 pages.**
- **§ 3.3 (YOLO results — 12 sub-sections!) — biggest cut opportunity.** Currently:
  - 3.3.1 v1 training run (procedural narrative)
  - 3.3.2 Training loss (interpretation of each loss component)
  - 3.3.3 v1 validation metrics (Table 3.3 + 3 paragraphs of mitigating factors)
  - 3.3.4 Qualitative analysis (Figure 3.2 + 3 observations)
  - 3.3.5 v2 era (Table 3.3a + long narrative about v2-fromscratch vs v2-finetune)
  - 3.3.6 v3 first attempts (Table 3.3b + 3 lesson paragraphs)
  - 3.3.7 16-exp ablation (Tables 3.3c / 3.3d / 3.3e — three sub-rounds)
  - 3.3.8 v4 sweep
  - 3.3.9 variance check
  - 3.3.10 random chain
  - 3.3.11 final production + Table 3.3i
  - 3.3.12 OOD evaluation

  **Recommended consolidation:** Replace 3.3.1 – 3.3.7 with a single "Pre-ablation phase (v1 / v2 era)" sub-section that shows the chain in one table (5 rows: v1 → v2-fs → v2-ft → run1 → run2 → exp1) and 2 paragraphs of narrative. Keep 3.3.8 / 3.3.9 / 3.3.10 (the novel ablation rounds) but trim narrative. Keep 3.3.11 (Table 3.3i full ablation chain — this **is** the contribution) and 3.3.12 (OOD ratio table). **−10-12 pages.**

  - Drop 3.3.1 (v1 training run) entirely — Section 2.4.4 already has the Table 2.2 hyper-parameters; what else is there to say about the v1 run on its own?
  - Drop 3.3.2 (training loss) — three loss components from Ultralytics framework, every reader knows what `box_loss` is.
  - 3.3.3 (v1 validation) — the "3 mitigating factors" discussion is defensive over-elaboration. Could be 1 sentence: "v1 mAP@50 0.478 should be read as 0.48 ± 0.10 because of the 4-tile validation set size."
  - 3.3.4 (qualitative) — keep the figure, cut the 3 observations to 2 sentences.
  - 3.3.5 (v2 era) — long discussion about the methodological subtlety of `v2-fromscratch` vs `v2-finetune`. The pre-ablation table can show both with one Δ-comparison sentence; the methodological subtlety belongs in Section 3.3.10 (random chain) which already exists.
- **§ 3.4 (Mask R-CNN)** — 4 sub-sections. 3.4.3 (qualitative) is one paragraph that mostly says "see the prediction screenshots", cut. 3.4.4 (comparison with YOLO) is 4-row table that overlaps with Table 3.5 in Section 3.7, cut. **Keep 3.4.1 (setup) + 3.4.2 (results Tables 3.4a/b).** **−1 page.**
- **§ 3.5 (DeepForest)** — Three sub-sections. 3.5.3 (cross-comparison with YOLO Table 3.4) is again overlap with Table 3.5. Cut. **Keep 3.5.1 (NEON baseline 0.012) + 3.5.2 (v3 fine-tune Table 3.6a/b).** **−1 page.**
- **§ 3.6 (SAM 2)** — Two paragraphs of background + one quantitative result + one qualitative observation. Cut to 1 paragraph: "SAM 2 used `sam2.1-hiera-base-plus` post-processor on DeepForest boxes; M14 Mask mAP@50 = 0.134, Box unchanged."
- **§ 3.7 (cross-model comparison)** — 5 sub-sections.
  - 3.7.1 M14 construction — keep (this is the methodological centrepiece).
  - 3.7.2 Cross-model Table 3.5 — keep.
  - 3.7.3 WBF ensemble status — currently 2 paragraphs saying "not evaluated yet". Could be 1 sentence in Limitations.
  - 3.7.4 Cross-YOLO ensemble — recently added, keep but trim narrative (Algorithm description can be 3 sentences instead of 1 paragraph; we don't need to teach union-find).
  - 3.7.5 Literature comparison — Table 3.6 + 3 observations. Could be 2 observations.
- **§ 3.8 (integrated pipeline)** — 4 sub-sections.
  - 3.8.1 production configuration — useful, keep.
  - 3.8.2 end-to-end demonstration — narrative of "the user does X, then Y" — could be 3 sentences pointing to UI screenshot.
  - 3.8.3 city-map view — the SQLite + cluster + 50 000 cap details belong in Chapter 2, not duplicated here. Move or cut.
  - 3.8.4 export formats — full JSON / CSV / HTML examples. Bulky. Cut to: "Examples in `backend/export.py`. GeoJSON FeatureCollection, CSV one-row-per-detection, standalone HTML with embedded Leaflet."
- **§ 3.9 (limitations)** — 8 items now. Items 1-7 are concise. Item 0 (mAP-limitation, just added in Phase B) is 1 long paragraph but justifiably so. Item 6 (train/serve domain shift) repeats Section 3.2.1. Cut item 6.

**Recommended target:** Chapter 3 → **15-18 PDF pages** (from ~45).

### Conclusion — 13 KB markdown / ≈ 8 PDF pages

**Status:** Substantial but on-target for a bachelor thesis. Three sub-sections (objectives / contributions / limitations-and-future-work) plus closing remark. The objectives list (1-6 with detailed re-statements) duplicates the abstract and introduction. **Cut to 1 paragraph saying "the six objectives are addressed in Sections 1.6, 3.2, 3.3, 3.4-3.7, Chapter 2, 3.7-3.8 respectively."** **−2 pages.**

The contributions list (8 items, each a paragraph) is on-target. Future work list (7 items) is on-target.

**Recommended target:** Conclusion → **4-5 PDF pages** (from ~8).

### Appendices, definitions, frontmatter

- `01c_definitions.md` — currently 16 terms. Mostly OK. Could drop "Bounding box" (universal CS term) and "WGS-84" (standard).
- `07_references.md` — should have ≥ 20 refs per AITU. Currently 35-36 entries (some duplicate citation keys). **Verify and dedupe.**

---

## Specific lines that scream "water"

These are signature water phrases I'd remove on sight throughout the body:

- "It is worth noting that..." — drop, just state the thing.
- "The reader will note that..." — drop.
- "As mentioned above..." — drop, the structure should make this clear.
- "Two observations follow from this table." / "Three observations follow." — too verbose. Just state the observations in numbered list directly.
- "The principal observations are the following." — same.
- "This is consistent with the literature finding that..." — keep the citation but cut the wrapper.
- "Empirically, [some result] [some Δ comparison]" — keep the data, cut "empirically".
- "Beyond the immediate scope of this work..." — drop, just say what we did or didn't do.

---

## Recommended cut order (decreasing risk)

1. **§ 2.12 Chapter 2 Summary paragraph** — pure recap, zero risk, delete.
2. **§ 3.3.1 + § 3.3.2** (v1 training procedural / loss interpretation) — already in Ch 2.4.4, low risk to drop.
3. **§ 3.4.3 / § 3.4.4 / § 3.5.3** (qualitative overlaps with Section 3.7 cross-model table) — straight duplicates, drop.
4. **§ 1.3.1 – 1.3.5 paradigm sub-sections** — compress to comparative paragraph + Table 1.2 (already exists). Moderate risk: removing literature survey could hurt "Topic coverage" (40-point criterion) — but the SAME literature can be referenced inline in Sections 2.4-2.7 instead, so net no signal loss.
5. **§ 2.4.5 / § 2.5 architecture re-derivation / § 2.6 / § 2.7 architecture re-explanation** — known framework details, low-moderate risk.
6. **§ 2.11 frontend description** — move REST endpoints table to appendix, keep one view-mode paragraph + Figure 2.X.
7. **§ 3.3.5 – 3.3.7 collapsed into single "Pre-ablation" sub-section** — biggest cut, moderate risk because it removes detail from the methodological story. **Mitigation:** keep a single comprehensive Table 3.3i (already exists) and a 2-paragraph narrative.

## What I will NOT cut

- Table 1.2 literature baselines (anchors the contribution claim).
- Table 3.3i full YOLO ablation chain (the central diploma result).
- Table 3.5 cross-model M14 ablation (the methodological centrepiece).
- Section 3.7.1 M14 construction rationale (defends the validation methodology to the commission).
- Section 3.9 limitations item 0 (mAP-limitation discussion — novel methodological observation worth defending).
- Multi-replicate variance section 3.3.9 (honest empirical contribution).
- Visual comparison figures 3.7a / 3.7b (cross-checkpoint qualitative evidence).
- Anything in the Conclusion's "Scientific and engineering contributions" list (this is the 40-point Topic coverage payload).

---

## Verdict

The current thesis is **substantially over-written** for a bachelor IT diploma. The over-writing pattern is consistent: each section explains itself, references back to sections that themselves explain themselves, and re-derives commodity framework details. The empirical contribution (the 23-experiment ablation + M14 cross-model evaluation + variance check + cross-YOLO ensemble) is solid and doesn't need re-stating; it needs **one clean place per chapter** where it lives.

If you want, I can execute the cuts and produce a slimmed version. Recommended: leave the current 121-page draft as `thesis/` snapshot under git tag `v1-defense-draft`, then create a slimmed `thesis/40p/` variant for submission. That way both exist if commission asks for the long version.
