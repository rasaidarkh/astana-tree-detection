# Mask R-CNN branch — instance segmentation

## Why this branch

Two-stage instance segmenters (Faster R-CNN + mask head) are the canonical
baseline for tree-crown segmentation. The literature reference closest to our
domain — Lv et al., 2023 (MCAN, UAV RGB, Zhejiang) — reports Det AP 92.40% and
Seg AP 97.70% on dense urban canopy using a Mask R-CNN-family architecture.

For our setting (44 train / 10 val images, low-resolution satellite, ~20–40 px
crowns) those headline numbers are unreachable. The realistic target —
calibrated against the team's YOLOv8-seg v2-finetune on the same merged val —
is **Mask mAP@50 ≥ 0.45**, beating the YOLO branch on mask quality while
accepting slower inference.

The thesis already cites Mask R-CNN as the canonical two-stage detector
(Chapter 1 §three), but it was not implemented; this branch closes that gap
and provides the fourth comparison row for the ablation table.

## Architecture

- Backbone: torchvision `maskrcnn_resnet50_fpn_v2` (COCO V1 weights)
- Box predictor: `FastRCNNPredictor(in_features, num_classes=2)`
- Mask predictor: `MaskRCNNPredictor(in_features_mask, dim_reduced=256, num_classes=2)`
- `num_classes = 2` → {background, tree (`Дерево`, COCO category_id=1)}
- `MaskRCNNAdapter.build_model(num_classes)` is the single source of truth for
  the architecture — used at training time (head replacement) and at inference
  time (architecture reconstruction before loading state_dict).

## Dataset

Direct ingest of `annotations_merged/instances_{Train,Validation}.json` (COCO
1.0). `CocoMaskRCNNDataset` reads `H/W` from the COCO record, **not** from the
file on disk — this guards against the one hand-cropped image
(`Снимок экрана 2026-05-10 102326.png`, 1613×862) where the COCO height
differs from the original CVAT export.

Image lookup spans both photo roots
(`yolov train dataset/фотографии/` for v1 and
`yolov train dataset/новые фотографии/` for v2); first-existing wins.

### v3 dataset (May 2026 — Google Maps tile batch)

A third batch of images is being annotated from the production deployment
imagery source (Google Maps satellite tiles), addressing domain shift between
v1/v2 training data (Earth Pro) and runtime. **Read
[`yolov train dataset/v3 annotations/README.md`](../yolov%20train%20dataset/v3%20annotations/README.md)
for the full plan and per-team-member instructions** — including a recommended
from-scratch retrain path for Mask R-CNN against merged v2+v3 (current
`weights/maskrcnn_astana.pt` is v1+v2 only and not appropriate for the new
imagery source).

## Quickstart

```bash
.venv\Scripts\activate

# Train (default: 50 epochs, SGD, mixed precision, batch 2)
python -m ml.train_maskrcnn --epochs 50 --batch-size 2 --lr 0.005

# Evaluate
python -m ml.eval_maskrcnn --checkpoint weights/maskrcnn_astana.pt
```

Outputs:

| Path | Contents |
|---|---|
| `weights/maskrcnn_astana.pt` | best (by `mask_map_50`) state_dict |
| `weights/maskrcnn_astana_last.pt` | last epoch state_dict |
| `lightning_logs/maskrcnn_v0/metrics.csv` | per-epoch metrics |
| `results/maskrcnn_eval/metrics.json` | full COCOeval Box+Segm + P/R at conf |
| `results/maskrcnn_eval/predictions.json` | COCO-format predictions |
| `results/maskrcnn_eval/predictions/*.png` | 5 example overlays |
| `results/maskrcnn_eval/comparison_table.md` | vs YOLO baseline |

## Hyperparameters

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | SGD, momentum 0.9, weight_decay 5e-4 | torchvision Mask R-CNN canonical |
| LR | 0.005 | small batch + small dataset |
| Scheduler | StepLR, step 10, γ 0.5 | gentle decay over 50 epochs |
| Batch size | 2 | 8 GB VRAM ceiling at ~1700 × 1100 |
| Mixed precision | `torch.amp.autocast("cuda")` | ~30 % memory & speed win |
| Epochs | 50 | plateau expected ~25–35 on small val |
| `num_workers` | 0 (Windows default) | avoids fork-pickle artefacts; bump to 2 on Linux |

## Expected metrics (literature anchors)

| Source | Det AP / Box mAP@50 | Seg AP / Mask mAP@50 |
|---|---|---|
| Lv et al. 2023 (MCAN, UAV) | 92.40 % | 97.70 % |
| Sun 2025 (YOLOv8 Wellington) | best of MaskRCNN / YOLOv5 / SOLOv2 | — |
| **Our target (small dataset, satellite)** | **≥ 0.40** | **≥ 0.45** |

## Comparison vs team baselines (YOLOv8-seg v2-finetune, merged val)

| Metric | Mask R-CNN target | YOLOv8-seg (team) |
|---|---|---|
| Box mAP@50 | ≥ 0.40 | **0.372** |
| Mask mAP@50 | ≥ 0.45 | **0.331** |
| Box Precision | TBD | 0.425 |
| Box Recall | TBD | 0.391 |

Hypothesis (informed by Sun 2025 and Lv 2023): two-stage Mask R-CNN should
beat one-stage YOLOv8-seg on mask quality but trail it on inference latency by
~3–5×. That trade-off is the thesis-relevant story for the Discussion.

## Notes & gotchas

- **Mixed precision is on by default** on CUDA; if numerical issues arise, set
  `use_amp = False` in `train_maskrcnn.py`.
- **`num_workers=0` is the Windows default** — multiprocessing DataLoader on
  Windows can trigger pickling artefacts with pycocotools handles.
- **H/W from COCO json, not from disk** — `CocoMaskRCNNDataset` enforces this
  so masks stay aligned even for the hand-cropped val image.
- **Polygon round-trip in eval**: the backend adapter returns simplified
  polygons (`approxPolyDP eps=1.5 px`) for frontend rendering. `eval_maskrcnn`
  rasterises those back to masks before COCO RLE encoding. The simplification
  loss should be ≪ 0.01 AP at this resolution; if you need stricter eval,
  bypass the adapter and run the underlying torchvision model directly.
- **Class indices**: torchvision Mask R-CNN reserves class 0 for background;
  `Дерево` is class 1. All inference outputs have `labels == 1`.
