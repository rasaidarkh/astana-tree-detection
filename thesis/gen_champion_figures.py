# -*- coding: utf-8 -*-
"""Champion (v4_x_clean) training-process figures for the defense deck:
   (1) training loss curve, (2) validation mAP@50 curve (Box + Mask).
   Data source: runs/segment/v4_x_clean/results.csv (Ultralytics, 39 epochs).
   Shared style — teammates can reuse the rcParams block on their own logs.
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
CSV = os.path.join(ROOT, "runs", "segment", "v4_x_clean", "results.csv")

# --- shared presentation style (give this block to Berik / Anuar) ---
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 13,
    "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 12,
    "figure.dpi": 160,
})
C_BOX, C_SEG, C_CLS, C_DFL = "#2E5AAC", "#D9661F", "#2E8B57", "#8B5CF6"

df = pd.read_csv(CSV); df.columns = df.columns.str.strip()
ep = df["epoch"]

# (1) Training loss curve
fig, ax = plt.subplots(figsize=(8, 5))
for col, lab, c in [("train/box_loss", "Box loss", C_BOX),
                    ("train/seg_loss", "Seg loss", C_SEG),
                    ("train/cls_loss", "Cls loss", C_CLS),
                    ("train/dfl_loss", "DFL loss", C_DFL)]:
    if col in df:
        ax.plot(ep, df[col], label=lab, lw=2, color=c)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
ax.set_title("YOLOv8x-seg (v4 champion) — Training loss")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
p1 = os.path.join(FIG, "yolo_v4_champion_loss_curve.png")
fig.savefig(p1, bbox_inches="tight"); plt.close(fig); print("Saved", p1)

# (2) Validation mAP@50 curve (Box + Mask)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, df["metrics/mAP50(B)"], label="Box mAP@50", lw=2.4, color=C_BOX)
if "metrics/mAP50(M)" in df:
    ax.plot(ep, df["metrics/mAP50(M)"], label="Mask mAP@50", lw=2.4, color=C_SEG)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mAP@50 (merged-val)")
ax.set_title("YOLOv8x-seg (v4 champion) — Validation mAP@50")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
p2 = os.path.join(FIG, "yolo_v4_champion_map_curve.png")
fig.savefig(p2, bbox_inches="tight"); plt.close(fig); print("Saved", p2)
print("Done.")
