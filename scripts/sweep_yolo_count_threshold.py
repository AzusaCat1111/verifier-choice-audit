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


def metric_summary(rows: list[dict[str, Any]], conf: float) -> dict[str, Any]:
    preds: list[tuple[int, int]] = []
    for row in rows:
        human_count = row.get("human_count")
        if human_count is None:
            continue
        prediction_count = sum(1 for box in row.get("boxes", []) if float(box["confidence"]) >= conf)
        preds.append((prediction_count, int(human_count)))

    exact = sum(1 for pred, human in preds if pred == human)
    within_one = sum(1 for pred, human in preds if abs(pred - human) <= 1)
    abs_errors = [abs(pred - human) for pred, human in preds]
    signed_errors = [pred - human for pred, human in preds]
    return {
        "n": len(preds),
        "conf": conf,
        "exact_correct": exact,
        "exact_match_accuracy": exact / len(preds) if preds else None,
        "within_1_correct": within_one,
        "within_1_accuracy": within_one / len(preds) if preds else None,
        "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "mean_signed_error": sum(signed_errors) / len(signed_errors) if signed_errors else None,
        "prediction_distribution": dict(sorted(Counter(pred for pred, _human in preds).items())),
        "human_distribution": dict(sorted(Counter(human for _pred, human in preds).items())),
    }


def grouped_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    return buckets


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "| " + " | ".join("---" for _key, _label in columns) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for key, _label in columns:
            value = row.get(key)
            if key in {"exact_match_accuracy", "within_1_accuracy"}:
                cells.append(format_pct(value))
            elif key in {"mean_abs_error", "mean_signed_error"}:
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_thresholds(value: str) -> list[float]:
    thresholds: list[float] = []
    for part in value.split(","):
        if part.strip():
            thresholds.append(float(part))
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds over saved YOLO count predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    rows = read_jsonl(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = [metric_summary(rows, conf) for conf in thresholds]
    by_task_rows: list[dict[str, Any]] = []
    for (task_name,), task_rows in sorted(grouped_rows(rows, ["task_name"]).items()):
        for conf in thresholds:
            by_task_rows.append({"task_name": task_name, **metric_summary(task_rows, conf)})

    by_cell_rows: list[dict[str, Any]] = []
    for (task_name, prompt_id, target_count), cell_rows in sorted(
        grouped_rows(rows, ["task_name", "prompt_id", "target_count"]).items()
    ):
        for conf in thresholds:
            by_cell_rows.append(
                {
                    "task_name": task_name,
                    "prompt_id": prompt_id,
                    "target_count": target_count,
                    **metric_summary(cell_rows, conf),
                }
            )

    write_csv(args.output_dir / "threshold_sweep_overall.csv", overall_rows)
    write_csv(args.output_dir / "threshold_sweep_by_task.csv", by_task_rows)
    write_csv(args.output_dir / "threshold_sweep_by_cell.csv", by_cell_rows)

    best_overall = max(overall_rows, key=lambda row: row["exact_match_accuracy"] or -1)
    best_by_task = []
    for (task_name,), task_rows in sorted(grouped_rows(by_task_rows, ["task_name"]).items()):
        best_by_task.append(max(task_rows, key=lambda row: row["exact_match_accuracy"] or -1))

    lines = [
        "# YOLO Count Confidence Sweep",
        "",
        f"- Predictions: `{args.predictions}`",
        "",
        "## Best Overall",
        "",
        markdown_table(
            [best_overall],
            [
                ("conf", "conf"),
                ("n", "n"),
                ("exact_match_accuracy", "exact"),
                ("within_1_accuracy", "within-1"),
                ("mean_abs_error", "MAE"),
                ("mean_signed_error", "bias"),
            ],
        ),
        "",
        "## Best By Task",
        "",
        markdown_table(
            best_by_task,
            [
                ("task_name", "task"),
                ("conf", "conf"),
                ("n", "n"),
                ("exact_match_accuracy", "exact"),
                ("within_1_accuracy", "within-1"),
                ("mean_abs_error", "MAE"),
                ("mean_signed_error", "bias"),
            ],
        ),
        "",
    ]
    (args.output_dir / "threshold_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_dir / 'threshold_sweep_report.md'}")


if __name__ == "__main__":
    main()
