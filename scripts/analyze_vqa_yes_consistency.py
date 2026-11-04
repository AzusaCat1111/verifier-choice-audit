#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(num: int, den: int) -> str:
    return "-" if den == 0 else f"{100.0 * num / den:.2f}%"


def image_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("generator_model") or ""),
        str(row.get("task_name") or ""),
        str(row.get("prompt_id") or ""),
        str(row.get("image_path") or row.get("image_rel_path") or row.get("annotation_path") or ""),
    )


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("status") == "ok" and row.get("prediction_answer") in {"yes", "no"}]
    yes_rows = [row for row in usable if row.get("prediction_answer") == "yes"]
    expected_yes_rows = [row for row in usable if row.get("expected_answer") == "yes"]
    candidate_counts = sorted(row.get("candidate_count") for row in usable)
    yes_candidate_counts = sorted(row.get("candidate_count") for row in yes_rows)
    expected_yes_candidate_counts = sorted(row.get("candidate_count") for row in expected_yes_rows)
    human_counts = sorted({row.get("human_count") for row in usable})
    target_counts = sorted({row.get("target_count") for row in usable})
    first = rows[0]
    return {
        "generator_model": first.get("generator_model"),
        "task_name": first.get("task_name"),
        "task_type": first.get("task_type"),
        "prompt_id": first.get("prompt_id"),
        "target_count": ",".join(str(value) for value in target_counts),
        "human_count": ",".join(str(value) for value in human_counts),
        "image_id": first.get("image_id") or Path(str(first.get("image_path") or "")).stem,
        "image_path": first.get("image_path"),
        "image_rel_path": first.get("image_rel_path"),
        "annotation_path": first.get("annotation_path"),
        "n_queries": len(rows),
        "n_usable_queries": len(usable),
        "yes_count": len(yes_rows),
        "candidate_counts": ",".join(str(value) for value in candidate_counts),
        "yes_candidate_counts": ",".join(str(value) for value in yes_candidate_counts),
        "expected_yes_candidate_counts": ",".join(str(value) for value in expected_yes_candidate_counts),
        "consistent_single_yes": int(len(yes_rows) == 1),
        "zero_yes": int(len(yes_rows) == 0),
        "multiple_yes": int(len(yes_rows) > 1),
        "correct_yes_set": int(yes_candidate_counts == expected_yes_candidate_counts),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "| " + " | ".join("---" for _key, _label in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _label in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze image-level yes/no consistency for VQA counting.")
    parser.add_argument("--prediction", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.prediction:
        rows.extend(read_jsonl(path))

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[image_key(row)].append(row)

    per_image = [summarize_group(group_rows) for _key, group_rows in grouped.items()]
    per_image.sort(key=lambda row: (str(row["task_name"]), str(row["prompt_id"]), str(row["image_path"])))

    summary_rows: list[dict[str, Any]] = []
    for task_name in ["__overall__", *sorted({str(row["task_name"]) for row in per_image})]:
        task_rows = per_image if task_name == "__overall__" else [row for row in per_image if row["task_name"] == task_name]
        total = len(task_rows)
        zero_yes = sum(int(row["zero_yes"]) for row in task_rows)
        one_yes = sum(1 for row in task_rows if int(row["yes_count"]) == 1)
        multiple_yes = sum(int(row["multiple_yes"]) for row in task_rows)
        correct_yes_set = sum(int(row["correct_yes_set"]) for row in task_rows)
        yes_count_dist = dict(sorted(Counter(int(row["yes_count"]) for row in task_rows).items()))
        summary_rows.append(
            {
                "task_name": task_name,
                "n_images": total,
                "image_level_accuracy": correct_yes_set / total if total else None,
                "image_level_accuracy_pct": fmt_pct(correct_yes_set, total),
                "zero_yes": zero_yes,
                "zero_yes_rate": fmt_pct(zero_yes, total),
                "one_yes": one_yes,
                "one_yes_rate": fmt_pct(one_yes, total),
                "multiple_yes": multiple_yes,
                "multiple_yes_rate": fmt_pct(multiple_yes, total),
                "correct_yes_set": correct_yes_set,
                "correct_yes_set_rate": fmt_pct(correct_yes_set, total),
                "yes_count_distribution": yes_count_dist,
            }
        )

    zero_yes_examples = [row for row in per_image if int(row["zero_yes"])]
    multiple_yes_examples = [row for row in per_image if int(row["multiple_yes"])]
    inconsistent_examples = [row for row in per_image if not int(row["correct_yes_set"])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_image_yes_counts.csv", per_image)
    write_csv(args.output_dir / "summary_by_task.csv", summary_rows)
    write_csv(args.output_dir / "zero_yes_examples.csv", zero_yes_examples)
    write_csv(args.output_dir / "multiple_yes_examples.csv", multiple_yes_examples)
    write_csv(args.output_dir / "inconsistent_yes_set_examples.csv", inconsistent_examples)

    lines = [
        "# VQA Image-Level Yes Consistency",
        "",
        f"- Prediction files: `{', '.join(path.as_posix() for path in args.prediction)}`",
        f"- Total images: `{len(per_image)}`",
        "",
        "## Summary",
        "",
        markdown_table(
            summary_rows,
            [
                ("task_name", "task"),
                ("n_images", "images"),
                ("image_level_accuracy_pct", "image acc"),
                ("zero_yes", "0 yes"),
                ("zero_yes_rate", "0 yes rate"),
                ("one_yes", "1 yes"),
                ("one_yes_rate", "1 yes rate"),
                ("multiple_yes", ">1 yes"),
                ("multiple_yes_rate", ">1 yes rate"),
                ("correct_yes_set", "correct images"),
                ("yes_count_distribution", "yes-count dist"),
            ],
        ),
        "",
        "## Example Files",
        "",
        f"- Per-image counts: `{args.output_dir / 'per_image_yes_counts.csv'}`",
        f"- Zero-yes examples: `{args.output_dir / 'zero_yes_examples.csv'}`",
        f"- Multiple-yes examples: `{args.output_dir / 'multiple_yes_examples.csv'}`",
        f"- Inconsistent yes-set examples: `{args.output_dir / 'inconsistent_yes_set_examples.csv'}`",
        "",
        "## Multiple-Yes Examples",
        "",
        markdown_table(
            multiple_yes_examples[: args.max_examples],
            [
                ("task_name", "task"),
                ("prompt_id", "prompt"),
                ("image_id", "image"),
                ("human_count", "human"),
                ("yes_candidate_counts", "yes k"),
                ("expected_yes_candidate_counts", "gold yes k"),
            ],
        ),
        "",
        "## Zero-Yes Examples",
        "",
        markdown_table(
            zero_yes_examples[: args.max_examples],
            [
                ("task_name", "task"),
                ("prompt_id", "prompt"),
                ("image_id", "image"),
                ("human_count", "human"),
                ("candidate_counts", "asked k"),
                ("expected_yes_candidate_counts", "gold yes k"),
            ],
        ),
        "",
    ]
    (args.output_dir / "yes_consistency_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "yes_consistency_report.md")


if __name__ == "__main__":
    main()
