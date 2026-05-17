"""Fine-tune DeepForest от существующего checkpoint на merged v1+v2+v3.

Использование:
    python ml/train_deepforest.py
    python ml/train_deepforest.py --epochs 50 --batch 4 --lr 0.0001
    python ml/train_deepforest.py --checkpoint weights/deepforest_astana.pl --output weights/deepforest_astana_v3.pl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_CSV = Path("yolov train dataset/v3_deepforest/train.csv")
VAL_CSV   = Path("yolov train dataset/v3_deepforest/val.csv")
ROOT_DIR  = Path("yolov train dataset/v3_merged/images")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("weights/deepforest_astana.pl"))
    p.add_argument("--output",     type=Path, default=Path("weights/deepforest_astana_v3.pl"))
    p.add_argument("--train-csv",  type=Path, default=TRAIN_CSV)
    p.add_argument("--val-csv",    type=Path, default=VAL_CSV)
    p.add_argument("--root-dir",   type=Path, default=ROOT_DIR)
    p.add_argument("--epochs",     type=int,  default=30)
    p.add_argument("--batch",      type=int,  default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--workers",    type=int,  default=2)
    args = p.parse_args()

    for path, name in [(args.train_csv, "train-csv"), (args.val_csv, "val-csv"), (args.root_dir, "root-dir")]:
        if not path.exists():
            sys.exit(f"Не найдено --{name}: {path}\nЗапусти сначала шаги подготовки данных.")

    try:
        from deepforest import main as df_main
    except ImportError:
        sys.exit("deepforest не установлен: pip install deepforest")

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint не найден: {args.checkpoint}")

    # Создаём модель с нашим конфигом ПЕРЕД загрузкой весов.
    # load_from_checkpoint нельзя использовать — чекпоинт содержит
    # старые пути к CSV которых нет на этой машине, и setup_metrics
    # падает при восстановлении hparams.
    model = df_main.deepforest()
    model.config["train"]["csv_file"]        = str(args.train_csv)
    model.config["train"]["root_dir"]        = str(args.root_dir)
    model.config["validation"]["csv_file"]   = str(args.val_csv)
    model.config["validation"]["root_dir"]   = str(args.root_dir)
    model.config["train"]["epochs"]          = args.epochs
    model.config["train"]["batch_size"]      = args.batch
    model.config["train"]["lr"]              = args.lr
    model.config["workers"]                  = args.workers

    print(f"Загружаю веса из checkpoint: {args.checkpoint}")
    import torch
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")

    print(f"Train: {len(open(args.train_csv, encoding='utf-8').readlines()) - 1} bboxes / {args.train_csv}")
    print(f"Val:   {len(open(args.val_csv, encoding='utf-8').readlines()) - 1} bboxes / {args.val_csv}")
    print(f"Epochs={args.epochs}, batch={args.batch}, lr={args.lr}\n")

    # create_trainer() инициализирует iou_metric, mAP_metric и т.д.
    model.create_trainer()

    import pytorch_lightning as pl
    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=5,
    )
    trainer.fit(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    print(f"\nСохранено → {args.output}")
    print(f"Чтобы backend подхватил:")
    print(f'  copy "{args.output}" weights\\deepforest_astana.pl')


if __name__ == "__main__":
    main()
