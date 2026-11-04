#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


TASK_CLASS_ID = {
    "cell_phone_object_count": 0,
    "chair_object_count": 1,
    "traffic_light_object_count": 2,
}

TASK_CLASS_NAME = {
    "cell_phone_object_count": "Cellphone",
    "chair_object_count": "chair",
    "traffic_light_object_count": "traffic light",
}


def remap_label(src: Path, dst: Path, source_class_id: int) -> int:
    kept = 0
    lines: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if int(parts[0]) != source_class_id:
            continue
        lines.append(" ".join(["0", *parts[1:]]))
        kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-class YOLO datasets for each object task from task26 export.")
    parser.add_argument("--manifest", type=Path, default=Path("dataset/detector/yolo/object_manifest.jsonl"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/yolo/per_task_datasets"),
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary: dict[str, Counter[str]] = {}
    for task_name, source_class_id in TASK_CLASS_ID.items():
        task_dir = args.out_root / task_name
        dataset_dir = task_dir / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        counts: Counter[str] = Counter()
        manifest_rows: list[dict[str, object]] = []
        for row in rows:
            if row["task_name"] != task_name:
                continue
            split = row["split"]
            image_src = Path(row["dataset_image"])
            label_src = Path(row["dataset_label"])
            image_dst = dataset_dir / "images" / split / image_src.name
            label_dst = dataset_dir / "labels" / split / f"{image_src.stem}.txt"
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_src, image_dst)
            boxes = remap_label(label_src, label_dst, source_class_id)
            counts[f"{split}_images"] += 1
            counts[f"{split}_boxes"] += boxes
            manifest_rows.append(
                {
                    **row,
                    "task_dataset_image": str(image_dst),
                    "task_dataset_label": str(label_dst),
                    "task_class_name": TASK_CLASS_NAME[task_name],
                    "remapped_class_id": 0,
                    "remapped_box_count": boxes,
                }
            )
        if not manifest_rows:
            raise SystemExit(f"No rows for {task_name}")
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    f"path: {dataset_dir.resolve()}",
                    "train: images/train",
                    "val: images/val",
                    "names:",
                    f"  0: {TASK_CLASS_NAME[task_name]}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for item in manifest_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary[task_name] = counts

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(
        json.dumps({task: dict(counts) for task, counts in summary.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for task, counts in summary.items():
        print(task, dict(counts))


if __name__ == "__main__":
    main()
