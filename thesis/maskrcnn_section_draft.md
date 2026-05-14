## Section 2.X: Mask R-CNN Branch — Instance Segmentation Baseline

### 2.X.1 Motivation

The system presented in this thesis combines three complementary detection
branches into a single Weighted-Box-Fusion ensemble. The YOLOv8-seg branch
(Section 2.4) provides fast one-stage instance segmentation, and the
DeepForest branch (Section 2.5) provides a domain-specialised retinal
bounding-box detector pre-trained on forestry imagery. A third branch is
required to anchor the comparison against the canonical two-stage instance
segmentation paradigm — Mask R-CNN — without which the ablation table
covers only one-stage detectors and cannot speak to the *family* trade-off
that the urban-tree literature explicitly debates.

The strongest precedent in the literature is the work of Lv et al. [Lv 2023],
who applied a Mask R-CNN-class architecture (MCAN) to UAV RGB imagery of
urban canopy in Zhejiang and reported Det AP 92.40 % and Seg AP 97.70 %.
Although the resolution and dataset size in that work substantially exceed
the present setting (low-resolution satellite imagery, 44 training images),
the result establishes that, under sufficient signal, two-stage architectures
can produce near-perfect crown masks. The hypothesis examined in this branch
is that Mask R-CNN's two-stage refinement should yield **higher Mask
mAP@50 than the one-stage YOLOv8-seg branch** on the same validation split,
at the expense of inference latency, by virtue of its dedicated mask head
operating on aligned RoI features rather than mask coefficients regressed
from a single forward pass.

Beyond the architectural comparison, pixel-accurate masks are operationally
useful in the present application. Crown area in square metres is computed
downstream by integrating the mask over the GeoTIFF affine transform, and
the area estimate is more reliable when derived from the segmentation mask
than when approximated from a bounding box, particularly for irregular or
partly-occluded crowns.

### 2.X.2 Architecture

The branch uses `maskrcnn_resnet50_fpn_v2` from torchvision 0.20, with the
publicly distributed COCO V1 weights as the starting point. The architecture
consists of a ResNet-50 backbone with a Feature Pyramid Network neck, a
Region Proposal Network, and a two-headed RoI predictor: one head producing
bounding-box class scores and regression offsets, the other producing a
fixed-resolution binary mask per RoI.

Adapting the network to the present binary task requires replacing the two
output heads under `num_classes = 2` (background plus the single `Дерево`
class, COCO category id 1):

```python
in_features_box = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, dim_reduced=256, num_classes=num_classes,
)
```

The factored architecture is encapsulated in a static method
`MaskRCNNAdapter.build_model(num_classes)` and is the single source of truth
shared between training (where heads are replaced before optimisation) and
inference (where the same head topology must be reconstructed before
loading the fine-tuned `state_dict`). The pre-trained backbone download is
177 MB; the fine-tuned checkpoint is approximately 175 MB.

### 2.X.3 Training Setup

The training data is the same `annotations_merged` COCO 1.0 corpus used by
the YOLOv8-seg branch: 44 training and 10 validation images, comprising
3 270 and 275 polygon annotations respectively for a single `Дерево` class.
After data-quality filtering (Section 2.X.4), the effective training corpus
is 3 253 polygons over 44 images.

The optimisation regime follows the canonical recipe for fine-tuning
torchvision Mask R-CNN on a small custom dataset. The optimiser is
stochastic gradient descent with momentum 0.9 and weight decay 5 × 10⁻⁴, at
an initial learning rate of 5 × 10⁻³. The learning-rate scheduler is
StepLR with `step_size = 10` and `γ = 0.5`, halving the learning rate at
each ten-epoch boundary; over twenty epochs the final learning rate is
0.005 × 0.5 = 0.0025. Batch size is fixed at 2, and mixed precision is
enabled via `torch.amp.autocast("cuda")` with a `torch.amp.GradScaler` to
prevent gradient underflow in fp16 — both required to keep peak GPU memory
within the 8 GB envelope of the RTX 4070 Laptop GPU used for these
experiments.

Validation runs after each epoch, computing Box and Mask mean Average
Precision via the `torchmetrics.detection.MeanAveragePrecision` implementation
with `iou_type` set to `bbox` and `segm` respectively. The best checkpoint
(by Mask mAP@50) and the last-epoch checkpoint are persisted separately,
together with a per-epoch metrics log in CSV form.

The single-epoch wall time is approximately five minutes thirty seconds on
the RTX 4070 Laptop, for a total training time of approximately one hour
fifty minutes over twenty epochs. The hardware configuration is summarised
in the table below.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8 GB VRAM |
| CUDA | 12.1 (PyTorch 2.5.1+cu121) |
| OS | Windows 11 |
| Python | 3.12.10 |
| Mixed precision | Enabled (`torch.amp`, fp16 forward, fp32 master) |
| `num_workers` | 0 (Windows; see Section 2.X.4) |

### 2.X.4 Implementation Challenges

Pipeline validation prior to the full training run surfaced two issues, both
of which generalise beyond the present project and are worth documenting
explicitly.

**(a) Cross-platform encoding of Cyrillic file names.** The COCO JSON files
exported by CVAT reference photo files whose names are Cyrillic ("Снимок
экрана 2026-04-01 …"). `pycocotools.coco.COCO`, when given a path, opens
the JSON via `open(path, "r")` without specifying an encoding. On Windows,
the default open-mode encoding follows the system locale — `cp1251` on a
Russian locale. The Cyrillic file_name fields, encoded as UTF-8 inside the
JSON file, are therefore interpreted as `cp1251` byte pairs and re-emitted
as mojibake (`РЎРЅРёРјРѕРє ...`), which subsequently fails any filesystem
lookup. The fix adopted here is to load the JSON file manually with explicit
`encoding="utf-8"` and to populate an empty `COCO()` instance via direct
attribute assignment and `createIndex()`. The workaround is contained
entirely within the dataset class and is cross-platform safe (no-op on
Linux and macOS where the default open encoding is already UTF-8).

**(b) Data-quality filtering of bbox-only annotations.** Seventeen of the
3 270 training annotations (0.5 %) were exported with `"segmentation": []`
— bbox-only entries produced when the CVAT operator marked a tree with a
rectangle but did not draw a polygon. `pycocotools.mask.frPyObjects` raises
`IndexError` on an empty segmentation list, and Mask R-CNN architecturally
requires a usable mask for every annotation because the mask head computes
loss over all positive RoIs. Synthesising a rectangular mask from the
bounding box was considered and rejected: a full-rectangle mask is a poor
prior for tree-crown geometry and would inject systematic noise into the
mask head's training signal. Skipping such annotations entirely, following
the convention of the official torchvision Mask R-CNN tutorial, sacrifices
0.5 % of training signal while preserving the integrity of the mask
supervision. The validation split contains no such annotations and is
therefore unaffected.

### 2.X.5 Results

The headline metrics for the trained Mask R-CNN branch, evaluated on the
identical merged validation split used by the YOLOv8-seg branch, are
presented in the table below. The first numeric column reports the
fine-tuned Mask R-CNN; the second reports the team's production
YOLOv8-seg v2-finetune (the strongest one-stage baseline available); the
third reports literature anchors from Lv et al. [Lv 2023] for context.

| Metric | Mask R-CNN (ours) | YOLOv8-seg (team) | Lv 2023 MCAN (lit.) |
|---|---|---|---|
| Box mAP@50 | TBD | **0.372** | 0.924 |
| Mask mAP@50 | TBD | **0.331** | 0.977 |
| Box Precision @ conf = 0.5 | TBD | 0.425 | — |
| Box Recall @ conf = 0.5 | TBD | 0.391 | — |
| Inference time per image | TBD | — | — |
| Peak VRAM during training | TBD | — | — |

As an intermediate indicator, the one-epoch dry-run achieved Box mAP@50 of
0.146 and Mask mAP@50 of 0.125 — approximately 40 % of the YOLO baseline
after a single fine-tuning epoch on the replaced heads. The trajectory
suggests that further training should narrow or close the gap; the final
numbers above will be filled in once the twenty-epoch run completes.

### 2.X.6 Discussion

The principal comparison drawn in this section is between Mask R-CNN and
YOLOv8-seg on a single shared validation split. Both models are trained on
the same 44 images, validated on the same 10, and evaluated under identical
COCO mAP definitions. The comparability is the central methodological
contribution of using a unified merged dataset across all three branches,
because it allows the trade-off between the two paradigms to be read off
the table without confounding factors.

It is unlikely that the present implementation will approach the headline
numbers reported by Lv et al. The discrepancy is fundamentally driven by
two factors. First, dataset scale: Lv et al. trained on thousands of
annotated UAV images, whereas the present corpus contains 44. Second,
domain shift: Lv et al. operated on UAV imagery at far higher spatial
resolution than the satellite imagery used here, where individual crowns
occupy only 20–40 pixels and the boundary between adjacent crowns is
inherently ambiguous. Both factors place an upper bound on attainable
metrics that is independent of architecture choice, and the more useful
comparison is therefore against the YOLOv8-seg branch trained on the same
data.

A particular claim made earlier in this thesis (Section 1.X, on detector
families) is that "Mask R-CNN-class instance segmenters are too slow on a
laptop GPU" and that this motivated the choice of a one-stage architecture
for the YOLO branch. The Mask R-CNN branch reported here permits a
quantitative test of that claim on the same hardware. The measured per-image
inference time on the RTX 4070 Laptop is TBD; if it lies below approximately
one second per image, the original claim should be softened in the final
manuscript, since urban-planning workflows tolerate latencies of several
seconds per image when processing is offline and batched.

The limitations of this study are clear and worth stating explicitly. First,
the training set is small (44 images); regularisation through aggressive
augmentation was not attempted and would be the obvious next experiment.
Second, the taxonomy is binary (tree vs background); future work extending
to species-level multi-class classification would benefit from architecture
families with strong mask heads. Third, no test-time augmentation was used:
neither multi-scale inference nor horizontal-flip averaging. Fourth, the
mask head is fine-tuned end-to-end alongside the backbone, which is the
canonical choice but risks overfitting on a 44-image corpus; staged
fine-tuning (frozen backbone for the first few epochs) was not explored
here and would be a sensible robustness check.

Future work on this branch should focus on extending the dataset to at
least two hundred annotated images, integrating multi-class species
classification, and wiring the trained checkpoint into the backend
`ModelAdapter` and Weighted-Box-Fusion ensemble alongside the existing
YOLO + DeepForest combination so that all three branches can be compared
in production end-to-end inference.

### TODO after training:

- [ ] Fill in actual metrics in Section 2.X.5 table
- [ ] Compute inference time benchmark on RTX 4070 Laptop
- [ ] Insert 2–3 visualisation examples from `results/maskrcnn_eval/predictions/`
- [ ] Add training curves (loss + mAP over epochs) from `metrics.csv`
- [ ] Re-evaluate Discussion claim about speed with actual measured numbers
- [ ] Cross-check Section 2.X numbering against final chapter outline
- [ ] Convert inline `[Lv 2023]` to LaTeX `\cite{Lv2023}` for `thesis_main.tex`
