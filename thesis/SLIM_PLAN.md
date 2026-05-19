# Slim thesis plan — 40-50 page Diploma PROJECT for Bachelor IT (AITU)

## AITU Bachelor IT criteria — what the commission scores

Per official `Критерии_оценивания_дипломной_работы.docx` (table 1 — Bachelor IT):

| Criterion | Points | What |
|---|---|---|
| **Topic coverage** | **40** | (a) structure matches title + clear aim/objectives/methods — 5 pts; (b) detailed critical analysis of peer-reviewed literature, last 5 years — 10 pts; (c) **mathematical support: models & methods — 10 pts**; (d) **information system developed + architecture + DB + software + hardware described — 10 pts**; (e) aim & objectives achieved + comparison with known solutions + conclusions — 5 pts |
| Presentation & report | 25 | Coherent aim/objectives/methods + math + arch + DB + SW + HW + comparison + own contribution + quality slides |
| Q&A | 25 | Free command of material |
| Note quality | 10 | Single style + ≥20 sources + methodical guide compliance |

**Type:** Diploma **PROJECT** (40-50 pages) — we built a system, not a research-only work. Slide 4 of official PPTX confirms: project = 40-50 pages; research-work = 30-40.

**Structure (per AITU PPTX slide 4):**
1. Introduction
2. Chapter 1: Subject area analysis + problem statement
3. Chapter 2: Description of models & methods for solving the task
4. Chapter 3: Testing & comparison of results
5. Conclusion (Выводы)
6. References (≥ 20)
7. Appendices

**Formal:** English, LaTeX, ≥20 academic sources (≤30% internet), tables/formulas as tables/formulas not images, plagiarism ≥70% unique.

---

## What's currently UNDER-emphasised in long thesis

The 10-point criterion **"information system developed + architecture + DB + software + hardware described"** wants the APPLICATION featured prominently. In our long thesis, application features are scattered across Chapter 2 sections 2.1, 2.2, 2.10, 2.11 and briefly in Chapter 3.8. The slim version must FEATURE them.

### Application features built (full inventory)

| # | Feature | Currently mentioned in |
|---|---|---|
| 1 | FastAPI backend with adapter pattern | § 2.1 |
| 2 | React 18 + Leaflet frontend, no build step | § 2.11 |
| 3 | 4 model branches: YOLO, DeepForest, Mask R-CNN, DF+SAM 2 | § 2.4 — 2.7 |
| 4 | WBF ensemble (YOLO + DeepForest) | § 2.8.1 |
| 5 | **Cross-YOLO ensemble** (4× vote_2 IoU-merge) | § 2.8.2 (new) |
| 6 | **8 YOLO checkpoint variants** registered via `ModelKind` enum | § 3.8.1 |
| 7 | **Hierarchical model picker** (family → variant) | § 3.8.1 |
| 8 | **Centred per-action model picker modal** | Conclusion |
| 9 | **Dark mode default** | Conclusion |
| 10 | **Auto-Zoom Region Scan** (subdivides bbox at z=19) | § 2.2 + § 2.11.2 |
| 11 | **Polygon-based scan** (free-shaped, point-in-polygon filter) | § 2.11.2 |
| 12 | **Streaming NDJSON progress** (`/api/scan_region/stream`) | § 2.11.2 |
| 13 | **Two tile providers**: ESRI + Google Satellite | § 2.2 + Table 2.1a |
| 14 | Four geographic conversion modes | § 2.9 |
| 15 | SQLite persistence: snapshots / runs / detections / scan_sessions tables | § 2.10 |
| 16 | `ON DELETE CASCADE` schema (single-statement cleanup) | § 2.10 |
| 17 | **City-map view** (aggregate visualisation) | § 2.11.1 |
| 18 | Detection display modes (Point/Box/Polygon/**Heat-map**) | § 2.11.3 |
| 19 | **Heat-map** via `leaflet.heat` plugin | § 2.11.3 |
| 20 | Three exporters: GeoJSON / CSV / standalone HTML | § 2.10 |

**For slim version:** create a dedicated Section 2.10 "Application architecture and frontend workflows" that consolidates features 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20 with **one screenshot per major flow**. Database schema gets its own diagram (criterion 1d). REST API as a compact table (≤ 10 endpoints).

---

## Page budget for slim version

| Component | Pages | Notes |
|---|---|---|
| Frontmatter (title, declaration, abstract, definitions) | 4 | Abstract in EN; RU+KZ versions go in appendix per AITU PPTX slide 33 |
| Introduction | 3-4 | Relevance, object/subject, aim, objectives, methods, novelty, structure |
| Chapter 1: Analysis + problem statement | 6-8 | Lit review in **table form** + gap analysis + formal problem statement |
| Chapter 2: Models & methods | 14-16 | System architecture diagram + 4 model branches (3 pages each) + ensemble + geo + **application section** with DB schema + UI screenshots + REST table |
| Chapter 3: Testing & results | 10-12 | Hardware + dataset table + cross-model M14 + ablation summary + qualitative figures + limitations |
| Conclusion | 3-4 | Per-objective recap + contributions + future work |
| References | 2-3 | ≥ 20 |
| Appendices | 4-6 | RU/KZ abstracts + extra tables + per-checkpoint training curves |
| **Total** | **46-57** | within the 40-50 target with appendix flexibility |

---

## Cut decisions (vs long source_long)

**Drop entirely:**
- § 1.1 (3-paragraph "what is urban inventory" — reader knows)
- § 1.2 sub-paragraphs about LiDAR and UAV (we don't use them)
- § 1.3.1-1.3.5 paradigm-by-paradigm survey → **collapse into one comparative table** + one paragraph per family
- § 1.3.3 semantic segmentation specifically — we don't use it
- § 2.1 second paragraph + Table 2.1 stack overlap with Figure 2.1 (pick one)
- § 2.4.5 detailed CIoU equation derivation
- § 2.5/2.6 architectural re-derivation paragraphs
- § 2.7 SAM history paragraphs
- § 2.8 WBF detailed 5-step procedure (math equations are fine)
- § 2.12 chapter summary
- § 3.1 narrative around Table 3.0
- § 3.2 sub-sections 3.2.2 / 3.2.3 / 3.2.4 → consolidate into one master table
- § 3.3.1 v1 procedural narrative
- § 3.3.2 training loss interpretation
- § 3.3.3 v1 mitigating-factors paragraphs
- § 3.3.5 long v2-fromscratch vs v2-finetune discussion → one row in master table
- § 3.3.6 v3 attempts narrative
- § 3.3.7 round-by-round descriptions (keep Tables 3.3c / 3.3d / 3.3e but tighten prose)
- § 3.4.3 / 3.4.4 / 3.5.3 (duplicates of Section 3.7)
- § 3.6 SAM2 background paragraphs
- § 3.7.3 WBF status (collapse to one sentence in Limitations)
- § 3.8.2 narrative of "user does X, then Y"
- § 3.8.4 full JSON/CSV examples (just say what the exports look like)
- Conclusion's per-objective re-statement (already in abstract + intro)

**Promote / expand:**
- § 2.10 new — Application architecture with **database schema diagram**, REST table (10 endpoints), key UI screenshots
- § 3.7.4 cross-YOLO ensemble — keep, this is novel
- § 3.9 item 0 mAP limitation — keep, this is novel
- Section in Conclusion about contributions — keep the 8-item list

---

## File mapping

```
thesis/
├── source_long/        ← preserved as reference / source of truth
│   ├── 00_title.md
│   ├── 01_abstract.md
│   ├── 01b_declaration.md
│   ├── 01c_definitions.md
│   ├── 02_intro.md
│   ├── 03_chapter1.md     (28 KB)
│   ├── 04_chapter2.md     (63 KB)
│   ├── 05_chapter3.md     (80 KB)
│   ├── 06_conclusion.md
│   └── 07_references.md
└── (root — slim new)    ← write fresh, target 40-50 PDF pages
    ├── 00_title.md         (1 page, reuse from source_long)
    ├── 01_abstract.md      (1-2 page, EN only here; RU/KZ in appendix)
    ├── 01b_declaration.md  (boilerplate from source_long)
    ├── 01c_definitions.md  (trim to 8 most-used terms)
    ├── 02_intro.md         (3-4 pages)
    ├── 03_chapter1.md      (6-8 pages, table-driven lit review)
    ├── 04_chapter2.md      (14-16 pages, application featured)
    ├── 05_chapter3.md      (10-12 pages, single ablation table)
    ├── 06_conclusion.md    (3-4 pages)
    ├── 07_references.md    (verify ≥ 20)
    └── appendices/
        ├── A_abstract_ru.md
        ├── A_abstract_kz.md
        └── B_training_curves.md (per-checkpoint curves)
```

---

## Execution order

1. **Audit references** — verify ≥ 20, dedupe `@Lv2023`/`@LvMCAN2023`, `@Martins2021`/`@Martins2021Species`, `@SAM2023`/`@SAM2_2024`
2. **Write 00_title.md, 01b_declaration.md, 01c_definitions.md** — reuse / trim from source_long
3. **Write 02_intro.md** — 4 pages, table of objectives mapping to sections
4. **Write 03_chapter1.md** — section by section: 1.1 problem area (1p), 1.2 data sources (1p), 1.3 **lit review TABLE** (2p), 1.4 gap analysis (1p), 1.5 problem statement (1p)
5. **Write 04_chapter2.md** — biggest job. § 2.1 system overview + figure (1p), § 2.2 image input + tiling (1p), § 2.3-2.6 four model branches at 2p each (8p), § 2.7 ensembles (2p), § 2.8 geo (1p), § 2.9 persistence (1p), § 2.10 **application** with DB schema diagram + REST table + UI screenshots (2-3p)
6. **Write 05_chapter3.md** — § 3.1 hardware/dataset (1p), § 3.2 single master ablation table (2p), § 3.3 cross-model M14 with v4 hero (2p), § 3.4 per-branch details (3p), § 3.5 cross-YOLO ensemble (1p), § 3.6 OOD eval (1p), § 3.7 limitations (1p)
7. **Write 06_conclusion.md** — 3-4 pages
8. **Write 01_abstract.md** — 1-2 pages, condensed from source_long
9. **Build LaTeX, verify page count**
10. **Iterate** until 40-50 pages
11. **Commit + push**

Estimated wall time: 4-6 hours.
