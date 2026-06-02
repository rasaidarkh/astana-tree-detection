"""Generate 3 charts for the YOLOv8-seg slide, in the EXACT house style shared
with Mask R-CNN (Berik) and DeepForest (Anuar) so all three model slides match.

Style mirrors charts_maskrcnn.py / charts_deepforest.py: figsize (9,4.5),
palette C_BLUE/C_ORANGE/C_GREEN/C_GRAY/C_LIGHT, white grid, no top/right spine,
title 13pt bold #222, dpi 150.

Real data:
  - per-epoch loss + val mAP from runs/segment/v4_x_clean/results.csv (39 epochs,
    best Box mAP@50 at epoch 37; Ultralytics patience-50 early-stop @ 200-epoch cap)
  - M14 numbers: champion v4_x_clean = Box 0.315 / Mask 0.289 (results/v4_clean_modelsweep.json,
    yolo_satellite eval); v1-from-scratch baseline = Box 0.131 (results/yolo_mergedval_eval.json)

Run:  venv/Scripts/python.exe thesis/charts_yolo_v3.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = Path(os.path.join(HERE, "figures"))
OUT.mkdir(exist_ok=True)

# ---- shared house palette (identical to teammates') ----
C_BLUE   = "#4178C9"
C_ORANGE = "#E87722"
C_GREEN  = "#27AE60"
C_GRAY   = "#BDC3C7"
C_LIGHT  = "#F4F6F9"

# ---- real per-epoch series (v4_x_clean, 39 epochs) ----
EPOCHS = list(range(1, 40))
# total loss = box+seg+cls+dfl; val rebased to where YOLO first logs a real val pass (ep 9)
TRAIN_LOSS = [12.35,10.30,10.12,9.99,9.95,9.96,9.60,9.29,9.26,9.31,9.18,8.83,8.77,8.85,
              8.64,8.66,8.46,8.32,8.22,8.14,8.05,8.03,8.09,7.87,7.90,7.77,7.70,7.61,7.89,
              7.76,7.48,7.33,7.35,7.28,7.09,7.09,6.96,7.00,6.98]
VAL_LOSS   = [None]*8 + [13.73,11.89,11.72,10.48,10.25,10.58,9.78,9.67,9.73,9.74,9.63,9.48,
              9.48,9.65,9.47,9.36,9.38,9.41,9.59,9.40,9.48,9.69,9.31,9.36,9.39,9.36,9.40,
              9.32,9.27,9.25,9.31]
BOX_MAP50  = [0.032,0.000,0.000,0.000,0.000,0.000,0.000,0.009,0.052,0.123,0.181,0.167,0.218,
              0.110,0.227,0.250,0.244,0.241,0.285,0.244,0.285,0.238,0.295,0.300,0.301,0.310,
              0.295,0.292,0.271,0.226,0.279,0.289,0.270,0.294,0.307,0.302,0.312,0.311,0.305]
MASK_MAP50 = [0.011,0.000,0.000,0.000,0.000,0.000,0.000,0.005,0.044,0.109,0.165,0.160,0.218,
              0.102,0.195,0.214,0.208,0.216,0.264,0.222,0.260,0.215,0.264,0.256,0.275,0.297,
              0.256,0.271,0.260,0.216,0.248,0.258,0.254,0.260,0.275,0.288,0.288,0.291,0.286]
BEST_EP = 37  # peak Box mAP@50 on this run's own val (0.3116); reported M14 = 0.315

# ---- M14 final numbers (shared cross-model test set) ----
BOX_M14   = 0.315
MASK_M14  = 0.289
BASE_M14  = 0.131   # YOLO v1, from scratch (no COCO init)
FINE_M14  = 0.315   # v4_x_clean, COCO-pretrained + tuned


def style_ax(ax, title="", xlabel="Epoch", ylabel=""):
    ax.set_facecolor(C_LIGHT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#222222", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color="#555555")
    ax.set_ylabel(ylabel, fontsize=10, color="#555555")
    ax.grid(True, color="white", linewidth=1.2, alpha=0.9)


# ── Chart 1: Training & Validation Loss ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")
ax.plot(EPOCHS, TRAIN_LOSS, color=C_BLUE, linewidth=2.5, marker="o",
        markersize=3, label="Train Loss")
vx = [e for e, v in zip(EPOCHS, VAL_LOSS) if v is not None]
vy = [v for v in VAL_LOSS if v is not None]
ax.plot(vx, vy, color=C_ORANGE, linewidth=2.5, marker="s", markersize=3, label="Val Loss")
ax.axvline(BEST_EP, color=C_GREEN, linestyle="--", linewidth=1.5, alpha=0.8)
ax.text(BEST_EP - 0.4, max(TRAIN_LOSS) * 0.96, f"Best\nEpoch {BEST_EP}",
        color=C_GREEN, fontsize=9, va="top", ha="right")
style_ax(ax, title="YOLOv8 — Training & Validation Loss", ylabel="Loss")
ax.set_xticks(range(0, 40, 5))
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
p = OUT / "yolo_loss_v3.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")

# ── Chart 2: Final M14 Metrics Bar ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")
labels = ["Box mAP@50", "Segm mAP@50"]
vals   = [BOX_M14, MASK_M14]
cols   = [C_BLUE, C_ORANGE]
x = np.arange(len(labels))
bars = ax.bar(x, vals, 0.5, color=cols, edgecolor="white")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold",
            color="#222222")
style_ax(ax, title="YOLOv8 — Final Performance on M14 Test Set",
         xlabel="", ylabel="Score")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.40)
plt.tight_layout()
p = OUT / "yolo_m14_v3.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")

# ── Chart 3: Before vs After (baseline vs fine-tuned) ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")
labels = ["Baseline\n(YOLO v1, from scratch)", "Fine-tuned\n(v4, COCO + Astana)"]
vals   = [BASE_M14, FINE_M14]
bars = ax.bar([0, 1], vals, 0.5, color=[C_GRAY, C_BLUE], edgecolor="white")
for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
            f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold",
            color=(C_BLUE if i else "#555555"))
# arrow + delta
pct = round((FINE_M14 - BASE_M14) / BASE_M14 * 100)
ax.annotate("", xy=(1, FINE_M14 - 0.01), xytext=(0, BASE_M14 + 0.01),
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2))
ax.text(0.5, (BASE_M14 + FINE_M14) / 2 + 0.03, f"+{pct}% Box mAP@50",
        ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_GREEN)
style_ax(ax, title="YOLOv8 — Baseline vs Fine-tuned (Box mAP@50, M14)",
         xlabel="", ylabel="Box mAP@50")
ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.40)
plt.tight_layout()
p = OUT / "yolo_before_after_v3.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")

print("\nAll YOLOv8 charts done (house style — matches Mask R-CNN / DeepForest).")
