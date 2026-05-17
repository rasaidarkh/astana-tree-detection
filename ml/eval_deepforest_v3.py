"""Evaluate fine-tuned DeepForest on merged val set (v1+v2+v3).

Использование:
    python ml/eval_deepforest_v3.py
    python ml/eval_deepforest_v3.py --checkpoint weights/deepforest_astana_v3.pl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VAL_CSV  = Path("yolov train dataset/v3_deepforest/val.csv")
ROOT_DIR = Path("yolov train dataset/v3_merged/images")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("weights/deepforest_astana_v3.pl"))
    p.add_argument("--val-csv",    type=Path, default=VAL_CSV)
    p.add_argument("--root-dir",   type=Path, default=ROOT_DIR)
    p.add_argument("--iou-threshold", type=float, default=0.4)
    args = p.parse_args()

    for path, name in [(args.checkpoint, "checkpoint"), (args.val_csv, "val-csv"), (args.root_dir, "root-dir")]:
        if not path.exists():
            sys.exit(f"Не найдено --{name}: {path}")

    try:
        from deepforest import main as df_main
    except ImportError:
        sys.exit("deepforest не установлен: pip install deepforest")

    import torch
    torch.set_float32_matmul_precision("high")

    print(f"Загружаю: {args.checkpoint}")
    model = df_main.deepforest()
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    print(f"GPU: {'активен' if torch.cuda.is_available() else 'нет'}")

    print(f"Val CSV: {args.val_csv}")
    print(f"Images:  {args.root_dir}")
    print(f"IoU threshold: {args.iou_threshold}\n")

    import pandas as pd
    import torch as _torch
    from torchmetrics.detection import MeanAveragePrecision

    gt_df = pd.read_csv(str(args.val_csv))
    images = gt_df["image_path"].unique()

    preds_list = []
    targets_list = []

    print(f"Предсказываю на {len(images)} изображениях...")
    for fname in images:
        img_path = str(args.root_dir / fname)

        pred_df = model.predict_image(path=img_path)

        if pred_df is not None and len(pred_df) > 0:
            boxes  = _torch.tensor(pred_df[["xmin","ymin","xmax","ymax"]].values, dtype=_torch.float32)
            scores = _torch.tensor(pred_df["score"].values, dtype=_torch.float32)
            labels = _torch.zeros(len(pred_df), dtype=_torch.long)
        else:
            boxes  = _torch.zeros((0, 4), dtype=_torch.float32)
            scores = _torch.zeros(0, dtype=_torch.float32)
            labels = _torch.zeros(0, dtype=_torch.long)

        gt = gt_df[gt_df["image_path"] == fname]
        gt_boxes  = _torch.tensor(gt[["xmin","ymin","xmax","ymax"]].values, dtype=_torch.float32)
        gt_labels = _torch.zeros(len(gt), dtype=_torch.long)

        preds_list.append({"boxes": boxes, "scores": scores, "labels": labels})
        targets_list.append({"boxes": gt_boxes, "labels": gt_labels})

    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    metric.update(preds_list, targets_list)
    result = metric.compute()

    print("\n" + "=" * 50)
    print(f"mAP@50       : {result['map_50'].item():.4f}")
    print(f"mAP@50:95    : {result['map'].item():.4f}")
    print(f"Precision    : {result['map_per_class'].item() if result['map_per_class'].numel()==1 else 'N/A'}")
    mar_100 = result.get('mar_100')
    if mar_100 is not None:
        print(f"Recall@100   : {mar_100.item():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
