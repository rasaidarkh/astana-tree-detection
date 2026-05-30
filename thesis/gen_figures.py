# -*- coding: utf-8 -*-
"""Generate thesis figures from training CSV files."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
RUNS = os.path.join(os.path.dirname(HERE), "runs", "segment")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_csv(run_name):
    path = os.path.join(RUNS, run_name, "results.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ── Figure 1: YOLO v1 training curves ────────────────────────────────────────
try:
    df1 = load_csv("astana_tiled_x_max")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df1["epoch"], df1["train/box_loss"], label="Train box loss")
    axes[0].plot(df1["epoch"], df1["val/box_loss"],   label="Val box loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("YOLOv8x-seg v1 — Box Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df1["epoch"], df1["metrics/mAP50(B)"],    label="Box mAP@50")
    axes[1].plot(df1["epoch"], df1["metrics/mAP50-95(B)"], label="Box mAP@50:95")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
    axes[1].set_title("YOLOv8x-seg v1 — Validation mAP")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGURES, "yolo_v1_training_curves.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
except Exception as e:
    print(f"SKIP yolo_v1_training_curves: {e}")


# ── Figure 2: YOLO v2-finetune training curves ───────────────────────────────
try:
    df2 = load_csv("astana_tiled_x_v2_finetune")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df2["epoch"], df2["train/box_loss"], label="Train box loss")
    axes[0].plot(df2["epoch"], df2["val/box_loss"],   label="Val box loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("YOLOv8x-seg v2-finetune — Box Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df2["epoch"], df2["metrics/mAP50(B)"],    label="Box mAP@50")
    axes[1].plot(df2["epoch"], df2["metrics/mAP50-95(B)"], label="Box mAP@50:95")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
    axes[1].set_title("YOLOv8x-seg v2-finetune — Validation mAP")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGURES, "yolo_v2_finetune_training_curves.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
except Exception as e:
    print(f"SKIP yolo_v2_finetune_training_curves: {e}")


# ── Figure 3: All four YOLO runs — mAP@50 comparison ────────────────────────
try:
    dv1  = load_csv("astana_tiled_x_max")
    dv2s = load_csv("astana_tiled_x_v2_fromscratch")
    dv2f = load_csv("astana_tiled_x_v2_finetune")
    try:
        dv3 = load_csv("astana_tiled_x_v3_finetune")
    except Exception:
        dv3 = None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dv1["epoch"],  dv1["metrics/mAP50(B)"],  label="v1 (397 ep.)", alpha=0.6)
    ax.plot(dv2s["epoch"], dv2s["metrics/mAP50(B)"], label="v2-fromscratch (204 ep.)", alpha=0.6)
    ax.plot(dv2f["epoch"], dv2f["metrics/mAP50(B)"], label="v2-finetune (173 ep.)", linewidth=1.5)
    if dv3 is not None:
        ax.plot(dv3["epoch"], dv3["metrics/mAP50(B)"], label="v3-finetune (run1, ≈ 133 ep.)", linewidth=2.0, color="#C00000")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Box mAP@50 (training-time val)")
    ax.set_title("YOLOv8x-seg — Box mAP@50 across the four training runs")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, "yolo_all_runs_map50.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
except Exception as e:
    print(f"SKIP yolo_all_runs_map50: {e}")


# ── Figure 4: Cross-model comparison bar chart on M14 (canonical Table 3.5) ─
# All numbers in this block are computed on the same 14-image / 702-polygon
# merged val (M14, see Section 3.7.1). Sources:
#   - YOLO v1/v2-fs/v2-ft/v3-ft: results/yolo_mergedval_eval.json (this work)
#   - Mask R-CNN v2+v3:           results/maskrcnn_14img_eval/metrics.json
#   - DeepForest v3 + SAM 2:      results/df_sam2_14img_eval/metrics.json
try:
    # Canonical Table 3.3 ranking, ending on the FINAL production champion
    # v4_x_clean (0.315 / 0.289) so the chart matches the results table/speech.
    models   = ["NEON DF\n(off-the-shelf)", "YOLO\nv1", "DF v3\n+ SAM 2",
                "Mask\nR-CNN\nv2+v3", "YOLO\nv2-ft", "YOLO\nv3-run1\n(x-seg)",
                "YOLO exp1\n(m-seg)", "YOLO v4\n(x-seg)\nFINAL"]
    box_map  = [0.012, 0.131, 0.146, 0.166, 0.187, 0.287, 0.308, 0.315]
    mask_map = [0.000, 0.134, 0.134, 0.158, 0.185, 0.263, 0.305, 0.289]
    champ    = len(models) - 1  # index of the champion bar to highlight

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.2))
    box_colors  = ["#9DB4D8"] * len(models); box_colors[champ]  = "#2E5AAC"
    mask_colors = ["#F1B488"] * len(models); mask_colors[champ] = "#D9661F"
    bars1 = ax.bar(x - w/2, box_map,  w, label="Box mAP@50",  color=box_colors)
    bars2 = ax.bar(x + w/2, mask_map, w, label="Mask mAP@50", color=mask_colors)

    for i, (bar, val) in enumerate(zip(bars1, box_map)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10 if i == champ else 9,
                fontweight="bold" if i == champ else "normal")
    for i, (bar, val) in enumerate(zip(bars2, mask_map)):
        if val <= 0:
            continue
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10 if i == champ else 9,
                fontweight="bold" if i == champ else "normal")

    # mark the champion column
    ax.annotate("FINAL production", xy=(champ, 0.315), xytext=(champ - 0.4, 0.355),
                fontsize=9, fontweight="bold", color="#2E5AAC",
                ha="center", arrowprops=dict(arrowstyle="->", color="#2E5AAC", lw=1.3))

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel("mAP@50"); ax.set_ylim(0, 0.40)
    ax.set_title("Cross-model comparison on the 14-image M14 validation set (702 polygons)")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, "model_comparison_barchart.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
except Exception as e:
    print(f"SKIP model_comparison_barchart: {e}")


print("Done.")
