#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def cohen_kappa(human: list[int], auto: list[int]) -> float | None:
    if len(human) != len(auto) or not human:
        return None
    tp = fp = fn = tn = 0
    for h, a in zip(human, auto, strict=True):
        if h and a:
            tp += 1
        elif not h and a:
            fp += 1
        elif h and not a:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    observed = (tp + tn) / n
    human_pos = tp + fn
    human_neg = fp + tn
    auto_pos = tp + fp
    auto_neg = fn + tn
    expected = ((human_pos * auto_pos) + (human_neg * auto_neg)) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "run",
        "generator_model",
        "task_name",
        "validity_label",
        "annotation_available",
        "human_success_on_prediction_subset",
        "auto_pass_target_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df[df["annotation_available"].astype(bool)].copy()
    df["human_pass_full"] = df["human_success_on_prediction_subset"].astype(int)
    df["auto_pass"] = df["auto_pass_target_count"].astype(int)
    df["is_valid"] = df["validity_label"].astype(str).eq("valid")
    return df


def summarize_group(rows: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = rows.groupby(keys, dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        valid = group[group["is_valid"]].copy()
        n_full = len(group)
        n_valid = len(valid)
        full_h = float(group["human_pass_full"].mean()) if n_full else float("nan")
        full_a = float(group["auto_pass"].mean()) if n_full else float("nan")
        valid_h = float(valid["human_pass_full"].mean()) if n_valid else float("nan")
        valid_a = float(valid["auto_pass"].mean()) if n_valid else float("nan")
        full_gap = full_a - full_h
        valid_gap = valid_a - valid_h
        kappa_full = cohen_kappa(group["human_pass_full"].astype(int).tolist(), group["auto_pass"].astype(int).tolist())
        kappa_valid = cohen_kappa(valid["human_pass_full"].astype(int).tolist(), valid["auto_pass"].astype(int).tolist())
        payload = {key: value for key, value in zip(keys, group_key, strict=True)}
        payload.update(
            {
                "n_full": n_full,
                "n_valid": n_valid,
                "valid_rate": pct(n_valid / n_full) if n_full else "-",
                "human_full": pct(full_h),
                "auto_full": pct(full_a),
                "gap_full_pp": pp(full_gap),
                "kappa_full": "-" if kappa_full is None else f"{kappa_full:.3f}",
                "human_valid_only": pct(valid_h),
                "auto_valid_only": pct(valid_a),
                "gap_valid_only_pp": pp(valid_gap),
                "kappa_valid_only": "-" if kappa_valid is None else f"{kappa_valid:.3f}",
                "human_shift_pp": pp(valid_h - full_h),
                "auto_shift_pp": pp(valid_a - full_a),
                "gap_shift_pp": pp(valid_gap - full_gap),
            }
        )
        out.append(payload)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    rows = rows if max_rows is None else rows[:max_rows]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full-denominator and valid-only human/auto scores.")
    parser.add_argument(
        "--scored-rows",
        default="outputs/mllm_evaluation/scored_rows.csv",
    )
    parser.add_argument("--out-dir", default="outputs/valid_only_score_shift")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = load_rows(Path(args.scored_rows))

    by_dataset = summarize_group(rows, ["run", "generator_model"])
    by_task = summarize_group(rows, ["run", "generator_model", "task_name"])
    overall = summarize_group(rows, ["run"])

    write_csv(out_dir / "valid_only_shift_by_run_dataset.csv", by_dataset)
    write_csv(out_dir / "valid_only_shift_by_run_dataset_task.csv", by_task)
    write_csv(out_dir / "valid_only_shift_by_run.csv", overall)

    # Compact task rows with largest absolute gap change.
    largest_gap_change = sorted(
        by_task,
        key=lambda row: abs(float(str(row["gap_shift_pp"]).replace("+", ""))),
        reverse=True,
    )
    largest_auto_shift = sorted(
        by_task,
        key=lambda row: abs(float(str(row["auto_shift_pp"]).replace("+", ""))),
        reverse=True,
    )

    dataset_cols = [
        "run",
        "generator_model",
        "n_full",
        "n_valid",
        "valid_rate",
        "human_full",
        "auto_full",
        "gap_full_pp",
        "human_valid_only",
        "auto_valid_only",
        "gap_valid_only_pp",
        "gap_shift_pp",
    ]
    task_cols = [
        "run",
        "generator_model",
        "task_name",
        "n_full",
        "n_valid",
        "valid_rate",
        "human_full",
        "auto_full",
        "gap_full_pp",
        "human_valid_only",
        "auto_valid_only",
        "gap_valid_only_pp",
        "gap_shift_pp",
    ]
    report = [
        "# Valid-Only Score Shift Check",
        "",
        "This report recomputes both human benchmark score and automatic target-pass score after filtering to human-valid/countable images only. It does not overwrite the main full-denominator benchmark report.",
        "",
        "Interpretation: if a large full-denominator gap disappears here, the previous gap was mainly a validity/rejection problem rather than a count-readout problem on valid images.",
        "",
        "## By Run And Dataset",
        "",
        markdown_table(by_dataset, dataset_cols),
        "",
        "## Largest Task-Level Gap Changes",
        "",
        markdown_table(largest_gap_change, task_cols, max_rows=25),
        "",
        "## Largest Task-Level Auto Score Shifts",
        "",
        markdown_table(largest_auto_shift, task_cols, max_rows=25),
        "",
        "## Output Files",
        "",
        "- `valid_only_shift_by_run.csv`",
        "- `valid_only_shift_by_run_dataset.csv`",
        "- `valid_only_shift_by_run_dataset_task.csv`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "report.md")


if __name__ == "__main__":
    main()
