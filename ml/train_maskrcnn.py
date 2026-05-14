"""Train Mask R-CNN (torchvision maskrcnn_resnet50_fpn_v2) on annotations_merged COCO.

  - SGD + StepLR, batch 2, mixed precision (8 GB VRAM friendly)
  - Validation each epoch via torchmetrics MeanAveragePrecision (Box + Segm)
  - Saves best (by mask_map_50) and last checkpoints + metrics CSV

Example:
    python -m ml.train_maskrcnn --epochs 50 --batch-size 2 --lr 0.005

Resume:
    python -m ml.train_maskrcnn --resume weights/maskrcnn_astana_last.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader

# Project root on PYTHONPATH so backend.* and ml.* import regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.maskrcnn_adapter import MaskRCNNAdapter  # noqa: E402
from ml.maskrcnn_dataset import CocoMaskRCNNDataset, collate_fn  # noqa: E402

log = logging.getLogger("astana-tree")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-json",
        default="yolov train dataset/annotations_merged/instances_Train.json",
    )
    p.add_argument(
        "--val-json",
        default="yolov train dataset/annotations_merged/instances_Validation.json",
    )
    p.add_argument(
        "--images-roots",
        nargs="+",
        default=[
            "yolov train dataset/фотографии",
            "yolov train dataset/новые фотографии",
        ],
    )
    p.add_argument("--output", default="weights/maskrcnn_astana.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 default on Windows (avoids pickling artefacts); bump to 2 on Linux",
    )
    p.add_argument("--log-dir", default="lightning_logs/maskrcnn_v0")
    p.add_argument("--resume", default=None, help="Path to state_dict .pt for resume")
    return p.parse_args()


def _validate_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Compute Box and Mask mAP via torchmetrics MeanAveragePrecision."""
    from torchmetrics.detection import MeanAveragePrecision

    model.eval()
    box_metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    mask_metric = MeanAveragePrecision(iou_type="segm", box_format="xyxy")

    with torch.inference_mode():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds_box: list[dict] = []
            preds_mask: list[dict] = []
            gts_box: list[dict] = []
            gts_mask: list[dict] = []
            for out, tgt in zip(outputs, targets):
                preds_box.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                })
                preds_mask.append({
                    "masks": (out["masks"][:, 0] > 0.5).to(torch.uint8).cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                })
                gts_box.append({
                    "boxes": tgt["boxes"],
                    "labels": tgt["labels"],
                })
                gts_mask.append({
                    "masks": tgt["masks"].to(torch.uint8),
                    "labels": tgt["labels"],
                })

            box_metric.update(preds_box, gts_box)
            mask_metric.update(preds_mask, gts_mask)

    box_res = box_metric.compute()
    mask_res = mask_metric.compute()
    return {
        "box_map_50": float(box_res["map_50"].item()),
        "box_map": float(box_res["map"].item()),
        "mask_map_50": float(mask_res["map_50"].item()),
        "mask_map": float(mask_res["map"].item()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = output_path.with_name(output_path.stem + "_last.pt")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = log_dir / "metrics.csv"

    train_ds = CocoMaskRCNNDataset(args.train_json, args.images_roots)
    val_ds = CocoMaskRCNNDataset(args.val_json, args.images_roots)
    log.info("Train: %d images | Val: %d images", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = MaskRCNNAdapter.build_model(num_classes=2)
    model.to(device)

    if args.resume:
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True)
        )
        log.info("Resumed from %s", args.resume)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss_avg", "box_map_50", "mask_map_50", "lr"]
        )

    best_mask_map = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_iters = 0
        for it, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_iters += 1
            if (it + 1) % 10 == 0:
                log.info(
                    "Epoch %d | iter %d/%d | loss %.4f",
                    epoch, it + 1, len(train_loader), float(loss.item()),
                )

        scheduler.step()
        train_loss_avg = running_loss / max(n_iters, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        val = _validate_one_epoch(model, val_loader, device)
        log.info(
            "Epoch %d val | box_map_50=%.4f box_map=%.4f | mask_map_50=%.4f mask_map=%.4f",
            epoch, val["box_map_50"], val["box_map"], val["mask_map_50"], val["mask_map"],
        )

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                round(train_loss_avg, 4),
                round(val["box_map_50"], 4),
                round(val["mask_map_50"], 4),
                current_lr,
            ])

        torch.save(model.state_dict(), last_path)
        if val["mask_map_50"] > best_mask_map:
            best_mask_map = val["mask_map_50"]
            torch.save(model.state_dict(), output_path)
            log.info(
                "New best mask_map_50=%.4f -> saved to %s",
                best_mask_map, output_path,
            )

    log.info("=" * 60)
    log.info("Training done. Best mask_map_50=%.4f at %s", best_mask_map, output_path)
    log.info("Last checkpoint at %s", last_path)
    log.info("Metrics CSV at %s", metrics_csv)


if __name__ == "__main__":
    main()
