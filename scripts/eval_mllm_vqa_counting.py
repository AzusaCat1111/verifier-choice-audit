#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("status") == "ok" and row.get("prediction_answer") in {"yes", "no"}]
    tp = sum(1 for row in usable if row.get("expected_answer") == "yes" and row.get("prediction_answer") == "yes")
    tn = sum(1 for row in usable if row.get("expected_answer") == "no" and row.get("prediction_answer") == "no")
    fp = sum(1 for row in usable if row.get("expected_answer") == "no" and row.get("prediction_answer") == "yes")
    fn = sum(1 for row in usable if row.get("expected_answer") == "yes" and row.get("prediction_answer") == "no")
    accuracy = safe_div(tp + tn, len(usable))
    yes_precision = safe_div(tp, tp + fp)
    yes_recall = safe_div(tp, tp + fn)
    no_precision = safe_div(tn, tn + fn)
    no_recall = safe_div(tn, tn + fp)
    yes_f1 = f1(yes_precision, yes_recall)
    no_f1 = f1(no_precision, no_recall)
    recalls = [value for value in (yes_recall, no_recall) if value is not None]
    f1s = [value for value in (yes_f1, no_f1) if value is not None]
    return {
        "n_total": len(rows),
        "n_usable": len(usable),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "expected_answer_distribution": dict(Counter(row.get("expected_answer") for row in usable)),
        "prediction_answer_distribution": dict(Counter(row.get("prediction_answer") for row in usable)),
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "yes_precision": yes_precision,
        "yes_recall": yes_recall,
        "yes_f1": yes_f1,
        "no_precision": no_precision,
        "no_recall": no_recall,
        "no_f1": no_f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_truncation_suspected": sum(1 for row in rows if row.get("truncation_suspected")),
    }


def grouped_summaries(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_tuple in sorted(buckets):
        group = {key: value for key, value in zip(keys, key_tuple, strict=True)}
        output.append({**group, **metric_summary(buckets[key_tuple])})
    return output


def image_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("generator_model") or ""),
        str(row.get("task_name") or ""),
        str(row.get("prompt_id") or ""),
        str(row.get("image_path") or row.get("image_rel_path") or row.get("annotation_path") or ""),
    )


def image_level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[image_key(row)].append(row)
    image_rows: list[dict[str, Any]] = []
    for group_rows in buckets.values():
        usable = [row for row in group_rows if row.get("status") == "ok" and row.get("prediction_answer") in {"yes", "no"}]
        yes_candidates = sorted(row.get("candidate_count") for row in usable if row.get("prediction_answer") == "yes")
        gold_yes_candidates = sorted(row.get("candidate_count") for row in usable if row.get("expected_answer") == "yes")
        first = group_rows[0]
        image_rows.append(
            {
                "task_name": first.get("task_name"),
                "prompt_id": first.get("prompt_id"),
                "image_path": first.get("image_path"),
                "yes_count": len(yes_candidates),
                "correct_yes_set": int(yes_candidates == gold_yes_candidates),
                "zero_yes": int(len(yes_candidates) == 0),
                "multiple_yes": int(len(yes_candidates) > 1),
            }
        )

    total = len(image_rows)
    correct = sum(row["correct_yes_set"] for row in image_rows)
    zero_yes = sum(row["zero_yes"] for row in image_rows)
    multiple_yes = sum(row["multiple_yes"] for row in image_rows)
    by_task: dict[str, dict[str, Any]] = {}
    for task_name in sorted({str(row["task_name"]) for row in image_rows}):
        task_rows = [row for row in image_rows if row["task_name"] == task_name]
        task_total = len(task_rows)
        by_task[task_name] = {
            "n_images": task_total,
            "image_level_accuracy": safe_div(sum(row["correct_yes_set"] for row in task_rows), task_total),
            "zero_yes_rate": safe_div(sum(row["zero_yes"] for row in task_rows), task_total),
            "multiple_yes_rate": safe_div(sum(row["multiple_yes"] for row in task_rows), task_total),
        }
    return {
        "n_images": total,
        "image_level_accuracy": safe_div(correct, total),
        "correct_images": correct,
        "zero_yes_images": zero_yes,
        "zero_yes_rate": safe_div(zero_yes, total),
        "multiple_yes_images": multiple_yes,
        "multiple_yes_rate": safe_div(multiple_yes, total),
        "by_task": by_task,
    }


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "| " + " | ".join("---" for _key, _label in columns) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for key, _label in columns:
            value = row.get(key)
            if key in {
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "image_level_accuracy",
                "zero_yes_rate",
                "multiple_yes_rate",
                "yes_precision",
                "yes_recall",
                "yes_f1",
                "no_precision",
                "no_recall",
                "no_f1",
            }:
                cells.append(fmt_pct(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and evaluate yes/no VQA counting predictions.")
    parser.add_argument("--prediction", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.prediction:
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: int(row.get("item_id") or 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)

    overall = metric_summary(rows)
    by_task = grouped_summaries(rows, ["task_name"])
    by_task_prompt = grouped_summaries(rows, ["task_name", "prompt_id"])
    by_task_candidate = grouped_summaries(rows, ["task_name", "candidate_count"])
    by_cell = grouped_summaries(rows, ["task_name", "prompt_id", "target_count", "candidate_count"])
    image_level = image_level_summary(rows)

    write_csv(args.output_dir / "by_task.csv", by_task)
    write_csv(args.output_dir / "by_task_prompt.csv", by_task_prompt)
    write_csv(args.output_dir / "by_task_candidate.csv", by_task_candidate)
    write_csv(args.output_dir / "by_cell.csv", by_cell)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "overall": overall,
                "by_task": by_task,
                "by_task_prompt": by_task_prompt,
                "by_task_candidate": by_task_candidate,
                "by_cell": by_cell,
                "image_level": image_level,
                "prediction_files": [path.as_posix() for path in args.prediction],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# MLLM VQA Counting Evaluation",
        "",
        "## Overall",
        "",
        markdown_table(
            [overall],
            [
                ("n_usable", "usable"),
                ("accuracy", "acc"),
                ("balanced_accuracy", "balanced acc"),
                ("macro_f1", "macro-F1"),
                ("yes_recall", "yes recall"),
                ("no_recall", "no recall"),
                ("tp", "TP"),
                ("tn", "TN"),
                ("fp", "FP"),
                ("fn", "FN"),
            ],
        ),
        "",
        "## Image-Level Count Accuracy",
        "",
        markdown_table(
            [
                {
                    "n_images": image_level["n_images"],
                    "image_level_accuracy": image_level["image_level_accuracy"],
                    "correct_images": image_level["correct_images"],
                    "zero_yes_rate": image_level["zero_yes_rate"],
                    "multiple_yes_rate": image_level["multiple_yes_rate"],
                }
            ],
            [
                ("n_images", "images"),
                ("image_level_accuracy", "image acc"),
                ("correct_images", "correct images"),
                ("zero_yes_rate", "0 yes rate"),
                ("multiple_yes_rate", ">1 yes rate"),
            ],
        ),
        "",
        "## Image-Level By Task",
        "",
        markdown_table(
            [
                {
                    "task_name": task_name,
                    **task_summary,
                }
                for task_name, task_summary in image_level["by_task"].items()
            ],
            [
                ("task_name", "task"),
                ("n_images", "images"),
                ("image_level_accuracy", "image acc"),
                ("zero_yes_rate", "0 yes rate"),
                ("multiple_yes_rate", ">1 yes rate"),
            ],
        ),
        "",
        "## By Task",
        "",
        markdown_table(
            by_task,
            [
                ("task_name", "task"),
                ("n_usable", "usable"),
                ("accuracy", "acc"),
                ("balanced_accuracy", "balanced acc"),
                ("macro_f1", "macro-F1"),
                ("yes_recall", "yes recall"),
                ("no_recall", "no recall"),
                ("prediction_answer_distribution", "pred dist"),
                ("expected_answer_distribution", "gold dist"),
            ],
        ),
        "",
        "## By Task And Candidate K",
        "",
        markdown_table(
            by_task_candidate,
            [
                ("task_name", "task"),
                ("candidate_count", "ask k"),
                ("n_usable", "usable"),
                ("accuracy", "acc"),
                ("balanced_accuracy", "balanced acc"),
                ("yes_recall", "yes recall"),
                ("no_recall", "no recall"),
            ],
        ),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
