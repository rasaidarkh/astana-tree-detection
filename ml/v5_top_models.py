"""Shared list of top-N YOLO models — reused by v5_unified_eval and v5_visual_compare."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
V3_ARCH = PROJECT_ROOT / "weights" / "v3_runs"
V4_ARCH = PROJECT_ROOT / "weights" / "v4_clean"

# Sorted by historical merged Box mAP@50, top 8.
# Each: (display_name, weights_path, note)
TOP_MODELS = [
    (
        "v4_x_clean (champ)",
        V4_ARCH / "v4_x_clean_v3val0.313_mergedval0.315.pt",
        "yolov8x-seg (71M) Ultralytics defaults — new best 0.315",
    ),
    (
        "exp1_m (tuned)",
        V3_ARCH / "exp1_m_cocostart_v3val0.287_mergedval0.308.pt",
        "yolov8m-seg (27M) v2-proven aug — original 0.308 (lucky run)",
    ),
    (
        "v4_m_clean",
        V4_ARCH / "v4_m_clean_v3val0.267_mergedval0.291.pt",
        "yolov8m-seg (27M) Ultralytics defaults — 0.291",
    ),
    (
        "exp17 chain random",
        V3_ARCH / "exp17_random_chain_3stage_cumulative_v3val0.257_mergedval0.287.pt",
        "yolov8m-seg 3-stage random chain — 0.287",
    ),
    (
        "exp15 v2v3 only",
        V3_ARCH / "exp15_m_v2v3_only_v3val0.291_mergedval0.286.pt",
        "yolov8m-seg drop-v1 train — 0.286, best v3-val (with v4_x)",
    ),
    (
        "exp12 low-lr finish",
        V3_ARCH / "exp12_m_chain_aggressive_lowlr_v3val0.256_mergedval0.286.pt",
        "yolov8m-seg exp1.pt → v3-only lr=0.0001 — 0.286",
    ),
    (
        "v4_s_clean",
        V4_ARCH / "v4_s_clean_v3val0.254_mergedval0.281.pt",
        "yolov8s-seg (12M) Ultralytics defaults — 0.281",
    ),
    (
        "v2-finetune (legacy)",
        PROJECT_ROOT / "weights" / "archive" / "yolo" / "yolo_satellite_v2_finetune.pt",
        "yolov8x-seg v2 production (pre-v3) — old baseline 0.167",
    ),
]


def filter_existing(models):
    """Drop entries whose weights file doesn't exist (gracefully)."""
    out = []
    for name, path, note in models:
        if not path.exists():
            print(f"[!] missing weights for '{name}' at {path} — skipping")
            continue
        out.append((name, path, note))
    return out
