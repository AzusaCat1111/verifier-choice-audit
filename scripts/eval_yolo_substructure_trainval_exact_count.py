#!/usr/bin/env python3
"""Compute exact count sanity for YOLO substructure train/val splits."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


TASK_CLASS_ID = {
    "turbine_substructure_count": 0,
    "clover_leaf_substructure_count": 1,
    "dresser_substructure_count": 2,
    "power_strip_substructure_count": 3,
}


def read_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def label_path_for_image(image_path: Path) -> Path:
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def task_from_image(image_path: Path) -> str:
    for task in TASK_CLASS_ID:
        if image_path.name.startswith(f"{task}__"):
            return task
    raise ValueError(f"Cannot infer task from image name: {image_path.name}")


def count_gt(label_path: Path, class_id: int) -> int:
    count = 0
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        if int(line.split()[0]) == class_id:
            count += 1
    return count


def count_pred(model: YOLO, image_path: Path, class_id: int, *, conf: float, imgsz: int, device: str) -> int:
    pred = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if pred.boxes is None:
        return 0
    return sum(1 for cls in pred.boxes.cls.detach().cpu().tolist() if int(cls) == class_id)


def summarize(rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    split_rows = [row for row in rows if row["split"] == split]
    groups = [("overall", split_rows)]
    for task in sorted(TASK_CLASS_ID):
        groups.append((task, [row for row in split_rows if row["task_name"] == task]))
    for name, group in groups:
        exact = sum(1 for row in group if row["exact_match"])
        diffs = Counter(int(row["pred_count"]) - int(row["gt_count"]) for row in group)
        out.append(
            {
                "split": split,
                "task_name": name,
                "n": len(group),
                "exact_count": exact / len(group) if group else "",
                "diff_counts": dict(sorted(diffs.items())),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    rows: list[dict[str, object]] = []
    for split, list_path in [("train", args.train_list), ("val", args.val_list)]:
        for image_path in read_list(list_path):
            task = task_from_image(image_path)
            class_id = TASK_CLASS_ID[task]
            gt = count_gt(label_path_for_image(image_path), class_id)
            pred = count_pred(model, image_path, class_id, conf=args.conf, imgsz=args.imgsz, device=args.device)
            rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "task_name": task,
                    "class_id": class_id,
                    "gt_count": gt,
                    "pred_count": pred,
                    "exact_match": int(gt == pred),
                }
            )

    with (args.out_dir / "trainval_exact_count_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, "train") + summarize(rows, "val")
    with (args.out_dir / "trainval_exact_count_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote exact-count sanity to {args.out_dir}")


if __name__ == "__main__":
    main()
