#!/usr/bin/env python3
"""Preserve context-specific Human-2 records for broad and traffic audits.

The released canonical Human-2 CSV has one row per image. Forty images occur in
both the broad-280 and traffic-360 audits, and three of those images have
different Human-2 counts across audit contexts. A one-row-per-image table cannot
represent both observations. This script emits separate context tables and a
640-row long-form table without changing either source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


FIELDS = [
    "image_id",
    "audit_context",
    "generator_model",
    "task_name",
    "task_type",
    "prompt_id",
    "target_count",
    "image_path",
    "human1_validity_label",
    "human1_count",
    "human2_validity_label",
    "human2_count",
    "human2_status",
    "human1_benchmark_success",
    "human2_benchmark_success",
    "source_record",
    "source_audit_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def integer_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(int(float(text)))


def success(validity: str, count: str, target: str) -> bool:
    return (
        validity == "valid"
        and integer_text(count) != ""
        and integer_text(count) == integer_text(target)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--broad-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    broad_manifest = args.broad_manifest.resolve()
    out_dir = args.out_dir.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Missing dataset root: {dataset_root}")
    if not broad_manifest.is_file():
        raise SystemExit(f"Missing broad-audit manifest: {broad_manifest}")
    if out_dir == dataset_root or out_dir.is_relative_to(dataset_root):
        raise SystemExit("Output must not be inside the read-only dataset root.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    human1_rows = read_csv(dataset_root / "annotations/human1_annotations.csv")
    canonical_h2_rows = read_csv(
        dataset_root / "annotations/human2_annotations.csv"
    )
    broad_rows_source = read_csv(broad_manifest)
    human1 = {row["image_id"]: row for row in human1_rows}
    canonical_h2 = {row["image_id"]: row for row in canonical_h2_rows}

    broad_rows: list[dict[str, Any]] = []
    for source in broad_rows_source:
        image_id = source["image_id"]
        h1 = human1.get(image_id)
        if h1 is None:
            raise ValueError(f"Broad record missing from Human-1: {image_id}")
        h2_validity = source["human2_validity"]
        h2_count = integer_text(source["human2_count"])
        target = integer_text(h1["target_count"])
        broad_rows.append(
            {
                "image_id": image_id,
                "audit_context": "broad",
                "generator_model": h1["generator_model"],
                "task_name": h1["task_name"],
                "task_type": h1["task_type"],
                "prompt_id": h1["prompt_id"],
                "target_count": target,
                "image_path": h1["image_path"],
                "human1_validity_label": h1["validity_label"],
                "human1_count": integer_text(h1["human_count"]),
                "human2_validity_label": h2_validity,
                "human2_count": h2_count,
                "human2_status": "completed",
                "human1_benchmark_success": h1["benchmark_success"],
                "human2_benchmark_success": success(
                    h2_validity, h2_count, target
                ),
                "source_record": broad_manifest.name,
                "source_audit_id": source.get("audit_id", ""),
            }
        )

    traffic_rows: list[dict[str, Any]] = []
    for source in canonical_h2_rows:
        contexts = set(source["human2_audit_subset"].split(";"))
        if "traffic_test" not in contexts:
            continue
        h2_validity = source["human2_validity_label"]
        h2_count = integer_text(source["human2_count"])
        target = integer_text(source["target_count"])
        traffic_rows.append(
            {
                "image_id": source["image_id"],
                "audit_context": "traffic_test",
                "generator_model": source["generator_model"],
                "task_name": source["task_name"],
                "task_type": source["task_type"],
                "prompt_id": source["prompt_id"],
                "target_count": target,
                "image_path": source["image_path"],
                "human1_validity_label": source["human1_validity_label"],
                "human1_count": integer_text(source["human1_count"]),
                "human2_validity_label": h2_validity,
                "human2_count": h2_count,
                "human2_status": source["human2_status"],
                "human1_benchmark_success": source["human1_benchmark_success"],
                "human2_benchmark_success": success(
                    h2_validity, h2_count, target
                ),
                "source_record": "human2_annotations.csv:traffic_test",
                "source_audit_id": "",
            }
        )

    if len(broad_rows) != 280 or len(traffic_rows) != 360:
        raise ValueError(
            f"Expected broad=280 and traffic=360; "
            f"found broad={len(broad_rows)}, traffic={len(traffic_rows)}"
        )
    long_rows = sorted(
        broad_rows + traffic_rows,
        key=lambda row: (row["audit_context"], row["image_id"]),
    )
    if len({(row["audit_context"], row["image_id"]) for row in long_rows}) != 640:
        raise ValueError("Context/image key is not unique")

    broad_by_id = {row["image_id"]: row for row in broad_rows}
    traffic_by_id = {row["image_id"]: row for row in traffic_rows}
    overlap = sorted(set(broad_by_id) & set(traffic_by_id))
    context_differences = []
    for image_id in overlap:
        broad = broad_by_id[image_id]
        traffic = traffic_by_id[image_id]
        if (
            broad["human2_validity_label"] != traffic["human2_validity_label"]
            or broad["human2_count"] != traffic["human2_count"]
        ):
            context_differences.append(
                {
                    "image_id": image_id,
                    "broad_validity": broad["human2_validity_label"],
                    "broad_count": broad["human2_count"],
                    "traffic_validity": traffic["human2_validity_label"],
                    "traffic_count": traffic["human2_count"],
                }
            )

    both_valid = [
        row
        for row in broad_rows
        if row["human1_validity_label"] == "valid"
        and row["human2_validity_label"] == "valid"
    ]
    count_agreements = sum(
        row["human1_count"] == row["human2_count"] for row in both_valid
    )
    if len(overlap) != 40 or len(context_differences) != 3:
        raise ValueError(
            f"Expected overlap=40 and context differences=3; "
            f"found overlap={len(overlap)}, differences={len(context_differences)}"
        )
    if len(both_valid) != 248 or count_agreements != 244:
        raise ValueError(
            f"Broad Table-2 reproduction mismatch: "
            f"{count_agreements}/{len(both_valid)}"
        )

    write_csv(out_dir / "human2_broad_annotations.csv", broad_rows)
    write_csv(out_dir / "human2_traffic_annotations.csv", traffic_rows)
    write_csv(out_dir / "human2_annotations_context_long.csv", long_rows)
    summary = {
        "broad_rows": len(broad_rows),
        "traffic_rows": len(traffic_rows),
        "long_rows": len(long_rows),
        "unique_images": len({row["image_id"] for row in long_rows}),
        "overlap_images": len(overlap),
        "context_specific_label_differences": context_differences,
        "broad_both_valid": len(both_valid),
        "broad_count_agreements": count_agreements,
        "broad_count_agreement_percent": 100 * count_agreements / len(both_valid),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
