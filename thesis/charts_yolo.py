"""Generate charts for the YOLOv8-seg slide — SAME house style as
charts_deepforest.py (Anuar) and charts_maskrcnn.py (Berik) so the three
model slides look like one family.

Charts (mirrors Berik's 3-chart pattern + one shared cross-model bar):
  1. yolo_loss.png        — Training + Validation loss vs epoch (v4 champion)
  2. yolo_map.png         — Validation Box/Mask mAP@50 vs epoch (v4 champion)
  3. yolo_comparison.png  — v1 baseline vs v4 champion (Box/Mask mAP@50)
  4. yolo_crossmodel.png  — cross-model comparison on M14 (shared slide)

Data: runs/segment/v4_x_clean/results.csv (39 epochs); bar values from the
thesis (Table 3.2/3.3), verified against results/v4_clean_modelsweep.json.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
CSV = ROOT / "runs" / "segment" / "v4_x_clean" / "results.csv"

# ── shared team palette (identical to charts_deepforest.py / charts_maskrcnn.py) ──
C_BLUE   = "#4178C9"
C_ORANGE = "#E87722"
C_GREEN  = "#27AE60"
C_GRAY   = "#BDC3C7"
C_RED    = "#E74C3C"
C_LIGHT  = "#F4F6F9"


def style_ax(ax, title="", xlabel="Epoch", ylabel=""):
    ax.set_facecolor(C_LIGHT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#222222", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color="#555555")
    ax.set_ylabel(ylabel, fontsize=10, color="#555555")
    ax.grid(True, color="white", linewidth=1.2, alpha=0.9)


df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()
ep = df["epoch"].tolist()
train_loss = (df["train/box_loss"] + df["train/seg_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]).tolist()
val_loss   = (df["val/box_loss"] + df["val/seg_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]).tolist()
box_map = df["metrics/mAP50(B)"].tolist()
mask_map = df["metrics/mAP50(M)"].tolist()
best_i = int(np.argmax(box_map))
best_ep = ep[best_i]
xt = [1, 5, 10, 15, 20, 25, 30, 35, 40]

# ── Chart 1: Training + Validation loss ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("white")
ax.plot(ep, train_loss, color=C_BLUE,   linewidth=2.5, marker="o", markersize=4, label="Train Loss")
ax.plot(ep, val_loss,   color=C_ORANGE, linewidth=2.5, marker="s", markersize=4, label="Val Loss")
ax.axvline(best_ep, color=C_GREEN, linestyle="--", linewidth=1.5, alpha=0.8)
ax.text(best_ep + 0.4, max(train_loss) * 0.97, f"Best\nEpoch {best_ep}", color=C_GREEN, fontsize=9, va="top")
style_ax(ax, title="YOLOv8x-seg (v4 champion) — Training Loss", ylabel="Loss (box+seg+cls+dfl)")
ax.set_xticks(xt)
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
p = OUT / "yolo_loss.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

# ── Chart 2: Validation mAP@50 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("white")
ax.plot(ep, box_map,  color=C_BLUE,   linewidth=2.5, marker="o", markersize=4, label="Box mAP@50")
ax.plot(ep, mask_map, color=C_ORANGE, linewidth=2.5, marker="s", markersize=4, label="Mask mAP@50")
ax.axvline(best_ep, color=C_GREEN, linestyle="--", linewidth=1.5, alpha=0.8)
ax.scatter([best_ep], [box_map[best_i]],  color=C_BLUE,   s=80, zorder=5)
ax.scatter([best_ep], [mask_map[best_i]], color=C_ORANGE, s=80, zorder=5)
ax.text(best_ep + 0.4, box_map[best_i] + 0.004, f"Best: {box_map[best_i]:.3f}", color=C_BLUE, fontsize=9)
style_ax(ax, title="YOLOv8x-seg (v4 champion) — Validation mAP@50", ylabel="mAP@50 (merged-val)")
ax.set_xticks(xt)
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
p = OUT / "yolo_map.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

# ── Chart 3: v1 baseline vs v4 champion ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("white")
metrics  = ["Box mAP@50", "Mask mAP@50"]
baseline = [0.131, 0.134]   # YOLOv8x-seg v1
champion = [0.315, 0.289]   # YOLOv8x-seg v4_x_clean
x = np.arange(len(metrics)); w = 0.35
b1 = ax.bar(x - w/2, baseline, w, label="v1 baseline",  color=C_GRAY, edgecolor="white")
b2 = ax.bar(x + w/2, champion, w, label="v4 champion",  color=C_BLUE, edgecolor="white")
for bar, v in zip(b1, baseline):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#555555")
for bar, v in zip(b2, champion):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C_BLUE, fontweight="bold")
deltas = [f"+{(champion[i]-baseline[i])/baseline[i]*100:.0f}%" for i in range(len(metrics))]
for i, (d, v) in enumerate(zip(deltas, champion)):
    ax.text(i + w/2, v + 0.022, d, ha="center", va="bottom", fontsize=9, color=C_GREEN, fontweight="bold")
style_ax(ax, title="YOLOv8x-seg — v1 Baseline vs v4 Champion", xlabel="Metric", ylabel="Score")
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 0.40)
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
p = OUT / "yolo_comparison.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

# ── Chart 4: cross-model comparison on M14 (shared slide) ─────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")
models = ["NEON\n(off-the-shelf)", "DeepForest\n+SAM2", "Mask R-CNN\nv2+v3", "YOLOv8x-seg\nv4 (ours)"]
box_v  = [0.012, 0.146, 0.166, 0.315]
mask_v = [0.000, 0.134, 0.158, 0.289]
x = np.arange(len(models)); w = 0.38
champ = len(models) - 1
bc = [C_GRAY, C_GRAY, C_GRAY, C_BLUE]
mc = [C_GRAY, C_GRAY, C_GRAY, C_ORANGE]
b1 = ax.bar(x - w/2, box_v,  w, color=bc, edgecolor="white", label="Box mAP@50")
b2 = ax.bar(x + w/2, mask_v, w, color=mc, edgecolor="white", label="Mask mAP@50")
for bar, v in zip(b1, box_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004, f"{v:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold" if bar is b1[champ] else "normal", color="#333333")
for bar, v in zip(b2, mask_v):
    if v <= 0: continue
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004, f"{v:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold" if bar is b2[champ] else "normal", color="#333333")
style_ax(ax, title="Cross-model comparison on M14 (14 images / 702 polygons)", xlabel="", ylabel="mAP@50")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9); ax.set_ylim(0, 0.40)
ax.legend(handles=[mpatches.Patch(color=C_BLUE, label="Box mAP@50 (ours)"),
                   mpatches.Patch(color=C_ORANGE, label="Mask mAP@50 (ours)"),
                   mpatches.Patch(color=C_GRAY, label="Other models")],
          fontsize=9, framealpha=0.9, loc="upper left")
plt.tight_layout()
p = OUT / "yolo_crossmodel.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

print(f"\nBest epoch={best_ep}  Box mAP@50={box_map[best_i]:.3f}  Mask mAP@50={mask_map[best_i]:.3f}")
print("All YOLO charts done.")
