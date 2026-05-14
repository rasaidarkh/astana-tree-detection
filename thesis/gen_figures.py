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


# ── Figure 3: All three YOLO runs — mAP@50 comparison ───────────────────────
try:
    dv1  = load_csv("astana_tiled_x_max")
    dv2s = load_csv("astana_tiled_x_v2_fromscratch")
    dv2f = load_csv("astana_tiled_x_v2_finetune")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dv1["epoch"],  dv1["metrics/mAP50(B)"],  label="v1 (397 ep.)")
    ax.plot(dv2s["epoch"], dv2s["metrics/mAP50(B)"], label="v2-fromscratch (204 ep.)")
    ax.plot(dv2f["epoch"], dv2f["metrics/mAP50(B)"], label="v2-finetune (99 ep.)", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Box mAP@50 (validation)")
    ax.set_title("YOLOv8x-seg — Box mAP@50 across three training runs")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, "yolo_all_runs_map50.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
except Exception as e:
    print(f"SKIP yolo_all_runs_map50: {e}")


# ── Figure 4: Model comparison bar chart ─────────────────────────────────────
try:
    models  = ["YOLOv8\nv1", "YOLOv8\nv2-scratch", "YOLOv8\nv2-finetune",
               "Mask\nR-CNN", "DeepForest\n(fine-tuned)", "Ensemble\nYOLO+DF"]
    box_map = [0.265, 0.319, 0.372, 0.241, None, 0.51]
    mask_map = [0.240, 0.288, 0.331, 0.226, None, None]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = [v if v is not None else 0 for v in box_map]
    b2 = [v if v is not None else 0 for v in mask_map]
    bars1 = ax.bar(x - w/2, b1, w, label="Box mAP@50",  color="#4472C4")
    bars2 = ax.bar(x + w/2, b2, w, label="Mask mAP@50", color="#ED7D31")

    for bar, val in zip(bars1, box_map):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, mask_map):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("mAP@50"); ax.set_ylim(0, 0.65)
    ax.set_title("Model Comparison — Box and Mask mAP@50 on Astana v2 Validation Set")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, "model_comparison_barchart.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
except Exception as e:
    print(f"SKIP model_comparison_barchart: {e}")


print("Done.")
