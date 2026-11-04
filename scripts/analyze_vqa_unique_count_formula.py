#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def image_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("generator_model") or ""),
        str(row.get("task_name") or ""),
        str(row.get("prompt_id") or ""),
        str(row.get("image_path") or row.get("image_rel_path") or row.get("annotation_path") or ""),
    )


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def metric_row(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("status") == "ok" and row.get("prediction_answer") in {"yes", "no"}]
    tp = sum(1 for row in usable if row.get("expected_answer") == "yes" and row.get("prediction_answer") == "yes")
    fn = sum(1 for row in usable if row.get("expected_answer") == "yes" and row.get("prediction_answer") == "no")
    fp = sum(1 for row in usable if row.get("expected_answer") == "no" and row.get("prediction_answer") == "yes")
    tn = sum(1 for row in usable if row.get("expected_answer") == "no" and row.get("prediction_answer") == "no")
    r = safe_div(tp, tp + fn)
    p = safe_div(fp, fp + tn)

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[image_key(row)].append(row)

    image_count = 0
    correct_images = 0
    predicted_probs: list[float] = []
    false_candidate_counts: list[int] = []
    incomplete_images = 0
    for group_rows in buckets.values():
        group_usable = [
            row
            for row in group_rows
            if row.get("status") == "ok" and row.get("prediction_answer") in {"yes", "no"}
        ]
        if len(group_usable) != len(group_rows):
            incomplete_images += 1
        yes_candidates = sorted(row.get("candidate_count") for row in group_usable if row.get("prediction_answer") == "yes")
        gold_yes_candidates = sorted(row.get("candidate_count") for row in group_usable if row.get("expected_answer") == "yes")
        false_m = sum(1 for row in group_usable if row.get("expected_answer") == "no")
        image_count += 1
        correct_images += int(yes_candidates == gold_yes_candidates)
        false_candidate_counts.append(false_m)
        if not math.isnan(r) and not math.isnan(p):
            predicted_probs.append(r * ((1.0 - p) ** false_m))

    unique_m = sorted(set(false_candidate_counts))
    return {
        "group": name,
        "queries": len(rows),
        "usable": len(usable),
        "images": image_count,
        "incomplete_images": incomplete_images,
        "true_yes_recall_r": r,
        "false_positive_rate_p": p,
        "false_k_m": ",".join(str(value) for value in unique_m),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "observed_unique_correct": safe_div(correct_images, image_count),
        "formula_predicted_unique_correct": safe_div(sum(predicted_probs), len(predicted_probs)),
        "abs_error": abs(
            safe_div(correct_images, image_count)
            - safe_div(sum(predicted_probs), len(predicted_probs))
        ),
        "correct_images": correct_images,
    }


def fmt_pct(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value * 100:.2f}%"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if key in {
                "true_yes_recall_r",
                "false_positive_rate_p",
                "observed_unique_correct",
                "formula_predicted_unique_correct",
                "abs_error",
            }:
                values.append(fmt_pct(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze_model(model_name: str, paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_rows(paths)
    task_names = sorted({str(row.get("task_name")) for row in rows})
    task_rows = [
        {"model": model_name, **metric_row(task, [row for row in rows if str(row.get("task_name")) == task])}
        for task in task_names
    ]
    overall_observed_num = sum(row["observed_unique_correct"] * row["images"] for row in task_rows)
    overall_formula_num = sum(row["formula_predicted_unique_correct"] * row["images"] for row in task_rows)
    images = sum(row["images"] for row in task_rows)
    overall = {
        "model": model_name,
        "group": "__overall_task_weighted__",
        "queries": sum(row["queries"] for row in task_rows),
        "usable": sum(row["usable"] for row in task_rows),
        "images": images,
        "incomplete_images": sum(row["incomplete_images"] for row in task_rows),
        "true_yes_recall_r": safe_div(
            sum(row["tp"] for row in task_rows),
            sum(row["tp"] + row["fn"] for row in task_rows),
        ),
        "false_positive_rate_p": safe_div(
            sum(row["fp"] for row in task_rows),
            sum(row["fp"] + row["tn"] for row in task_rows),
        ),
        "false_k_m": "task-specific",
        "observed_unique_correct": safe_div(overall_observed_num, images),
        "formula_predicted_unique_correct": safe_div(overall_formula_num, images),
        "abs_error": abs(safe_div(overall_observed_num, images) - safe_div(overall_formula_num, images)),
        "correct_images": sum(row["correct_images"] for row in task_rows),
        "tp": sum(row["tp"] for row in task_rows),
        "fn": sum(row["fn"] for row in task_rows),
        "fp": sum(row["fp"] for row in task_rows),
        "tn": sum(row["tn"] for row in task_rows),
    }
    return overall, task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check r(1-p)^m approximation for VQA count recovery.")
    parser.add_argument(
        "--model",
        action="append",
        nargs="+",
        metavar=("NAME", "PREDICTION"),
        required=True,
        help="Model name followed by one or more prediction jsonl files. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    for spec in args.model:
        model_name = spec[0]
        paths = [Path(value) for value in spec[1:]]
        overall, task_rows = analyze_model(model_name, paths)
        overall_rows.append(overall)
        per_task_rows.extend(task_rows)

    columns = [
        ("model", "model"),
        ("group", "group"),
        ("images", "images"),
        ("true_yes_recall_r", "r"),
        ("false_positive_rate_p", "p"),
        ("false_k_m", "m"),
        ("observed_unique_correct", "observed unique"),
        ("formula_predicted_unique_correct", "formula"),
        ("abs_error", "abs err"),
        ("incomplete_images", "incomplete images"),
    ]
    report = "\n".join(
        [
            "# VQA Unique-Count Formula Check",
            "",
            "Formula checked: `P(unique correct) ~= r * (1 - p)^m`, where `r` is yes recall on the true count, `p` is false-positive rate over false counts, and `m` is the number of false count queries for an image.",
            "",
            "The overall row is task-weighted: formula predictions are computed per task, then averaged by image count. This avoids mixing tasks with different candidate-set sizes.",
            "",
            "## Overall",
            "",
            markdown_table(overall_rows, columns),
            "",
            "## Per Task",
            "",
            markdown_table(per_task_rows, columns),
            "",
        ]
    )
    (output_dir / "formula_check_report.md").write_text(report, encoding="utf-8")

    for name, rows in [("overall.csv", overall_rows), ("per_task.csv", per_task_rows)]:
        with (output_dir / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)

    print(output_dir / "formula_check_report.md")


if __name__ == "__main__":
    main()
