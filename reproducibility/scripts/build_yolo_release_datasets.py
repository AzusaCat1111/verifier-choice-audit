#!/usr/bin/env python3
"""Materialize trainable joint YOLO datasets from the released support manifests.

The release keeps generated images in ``dataset/release`` and support labels in
``dataset/detector/yolo`` to avoid duplicating images. Ultralytics expects
parallel ``images/<split>/<stem>`` and ``labels/<split>/<stem>.txt`` paths.
This helper creates that layout under a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


FAMILIES = ("object", "substructure")


def dataset_path(dataset_root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    if normalized.startswith("dataset/"):
        normalized = normalized[len("dataset/") :]
    path = (dataset_root / normalized).resolve()
    path.relative_to(dataset_root)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def validate_label(path: Path, class_ids: set[int]) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(parts[0])
        if class_id not in class_ids:
            raise ValueError(f"{path}:{line_number}: unknown class {class_id}")
        values = [float(value) for value in parts[1:]]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        boxes += 1
    return boxes


def place_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(mode)


def build_family(
    dataset_root: Path,
    out_root: Path,
    family: str,
    image_mode: str,
) -> dict[str, Any]:
    family_root = out_root / family
    if family_root.exists() and any(family_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty family output: {family_root}"
        )
    family_root.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / f"detector/yolo/{family}_manifest.jsonl"
    schema_path = dataset_root / f"detector/yolo/{family}_label_schema.json"
    if not manifest_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError(f"Missing manifest or schema for {family}")
    rows = read_jsonl(manifest_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    classes_raw = schema.get("classes")
    if not isinstance(classes_raw, dict) or not classes_raw:
        raise ValueError(f"{schema_path}: classes must be a non-empty object")
    classes = {int(key): str(value) for key, value in classes_raw.items()}
    expected_ids = set(range(len(classes)))
    if set(classes) != expected_ids:
        raise ValueError(
            f"{schema_path}: class IDs must be contiguous from zero; found {sorted(classes)}"
        )

    counts: Counter[str] = Counter()
    output_manifest: list[dict[str, Any]] = []
    seen_destinations: set[tuple[str, str]] = set()
    for row in rows:
        split = str(row.get("split") or "")
        if split not in {"train", "val"}:
            raise ValueError(f"Unexpected split in {manifest_path}: {split!r}")
        image_source = dataset_path(dataset_root, str(row.get("dataset_image") or ""))
        label_source = dataset_path(dataset_root, str(row.get("dataset_label") or ""))
        if not image_source.is_file() or not label_source.is_file():
            raise FileNotFoundError(
                f"Missing image/label pair: {image_source}, {label_source}"
            )
        key = (split, image_source.stem)
        if key in seen_destinations:
            raise ValueError(f"Duplicate destination stem: {key}")
        seen_destinations.add(key)

        image_destination = (
            family_root / "images" / split / image_source.name
        )
        label_destination = (
            family_root / "labels" / split / f"{image_source.stem}.txt"
        )
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        place_image(image_source, image_destination, image_mode)
        shutil.copy2(label_source, label_destination)
        boxes = validate_label(label_destination, expected_ids)
        expected_boxes = int(row.get("box_count"))
        if boxes != expected_boxes:
            raise ValueError(
                f"{label_source}: manifest box_count={expected_boxes}, labels={boxes}"
            )
        counts[f"{split}_images"] += 1
        counts[f"{split}_boxes"] += boxes
        output_manifest.append(
            {
                "family": family,
                "split": split,
                "image_id": row.get("image_id"),
                "task_name": row.get("task_name"),
                "target_count": row.get("target_count"),
                "observed_count": row.get("observed_count"),
                "source_image": str(image_source),
                "source_label": str(label_source),
                "output_image": str(image_destination),
                "output_label": str(label_destination),
                "box_count": boxes,
            }
        )

    yaml_lines = [
        f"path: {family_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {class_id}: {classes[class_id]}" for class_id in sorted(classes))
    (family_root / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )
    with (family_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "family": family,
        "classes": classes,
        "source_manifest_rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "data_yaml": str(family_root / "data.yaml"),
        "image_mode": image_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
    )
    parser.add_argument(
        "--image-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="Symlink avoids duplicating released PNG files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_root = args.out_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if out_root == dataset_root or out_root.is_relative_to(dataset_root):
        raise SystemExit("Output must not be inside the read-only dataset root.")
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        build_family(dataset_root, out_root, family, args.image_mode)
        for family in args.families
    ]
    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps({"families": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
