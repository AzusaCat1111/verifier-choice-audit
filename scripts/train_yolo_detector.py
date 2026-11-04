#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on a YOLO-format dataset.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="train_img1024_e100")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    args.project.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.project / f"{args.name}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = YOLO(str(args.weights))
    result = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    print(result)
    print(args.project / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
