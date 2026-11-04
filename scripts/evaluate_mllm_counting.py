#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


COUNT_METRICS = [
    "count_fidelity_valid",
    "count_fidelity_all_invalid_wrong",
    "mae_valid",
    "bias_valid",
    "abs_bias_over_mae_valid",
    "auto_target_pass_valid",
    "auto_target_pass_all",
    "human_benchmark_valid",
    "human_benchmark_all",
    "auto_minus_human_all",
    "agreement_all",
    "kappa_all",
]

VALIDITY_METRICS = [
    "auto_reject_rate_all",
    "human_reject_rate_all",
    "reject_precision",
    "reject_recall",
    "reject_f1",
    "false_reject_rate_valid",
    "auto_pass_rate_on_human_reject",
]

CI_METRICS = COUNT_METRICS + VALIDITY_METRICS


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def norm_validity(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw == "valid":
        return "valid"
    if raw in {"reject", "invalid", "clear_fail", "clearfail", "ambiguous", "unclear"}:
        return "reject"
    return ""


def human_count(annotation: dict[str, Any], task_type: str) -> int | None:
    if annotation.get("validity_label") != "valid":
        return None
    if task_type == "substructure":
        return as_int(annotation.get("observed_part_count"))
    return as_int(annotation.get("observed_count"))


def discover_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    skip_parents: set[str] = set()
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if model_dir.name in skip_parents:
            continue
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if run_dir.name.startswith(("compare_", "eval_", "evaluation_")):
                continue
            if any(run_dir.glob("**/predictions.jsonl")):
                runs.append(run_dir)
    return runs


def infer_protocol(run_name: str) -> str:
    if run_name.startswith("zero_count_"):
        return "zero_count"
    if run_name.startswith("few_shot_count_"):
        return "few_shot_count"
    if run_name.startswith("zero_validity_"):
        return "zero_validity"
    if run_name.startswith("few_shot_validity_"):
        return "few_shot_validity"
    if "few_shot_validity" in run_name:
        return "few_shot_validity"
    if "few_shot_count" in run_name:
        return "few_shot_count"
    if "zero_validity" in run_name:
        return "zero_validity"
    if "zero_count" in run_name:
        return "zero_count"
    return run_name.split("_", 1)[0]


def infer_model(run_dir: Path) -> str:
    return run_dir.parent.name


def prediction_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("**/predictions.jsonl"))


def read_run_spec(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    runs = payload if isinstance(payload, list) else payload.get("runs")
    if not isinstance(runs, list):
        raise SystemExit(f"Run spec must be a JSON list or contain a 'runs' list: {path}")
    normalized: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise SystemExit(f"Run spec entry {index} is not an object")
        run_dirs = run.get("run_dirs", run.get("run_dir"))
        if isinstance(run_dirs, str):
            run_dirs = [run_dirs]
        if not isinstance(run_dirs, list) or not run_dirs:
            raise SystemExit(f"Run spec entry {index} must provide run_dir or run_dirs")
        model_name = str(run.get("model_name") or "")
        protocol = str(run.get("protocol") or "")
        if not model_name or not protocol:
            raise SystemExit(f"Run spec entry {index} must provide model_name and protocol")
        run_id = str(run.get("run_id") or f"{model_name}/{protocol}")
        normalized.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "protocol": protocol,
                "run_dirs": [Path(str(run_dir)) for run_dir in run_dirs],
                "note": str(run.get("note") or ""),
            }
        )
    return normalized


def load_scored_rows(
    run_dir: Path,
    *,
    run_id_override: str | None = None,
    model_name_override: str | None = None,
    protocol_override: str | None = None,
) -> list[dict[str, Any]]:
    model_name = model_name_override or infer_model(run_dir)
    protocol = protocol_override or infer_protocol(run_dir.name)
    run_id = run_id_override or f"{model_name}/{run_dir.name}"
    requires_pred_valid = protocol in {"zero_validity", "few_shot_validity"}
    rows: list[dict[str, Any]] = []
    for pred_path in prediction_paths(run_dir):
        shard_name = pred_path.parent.name
        for pred in read_jsonl(pred_path):
            annotation_path = Path(str(pred["annotation_path"]))
            annotation = read_json(annotation_path)
            task_type = str(pred.get("task_type") or annotation.get("task_type") or "")
            h_count = human_count(annotation, task_type)
            target = as_int(pred.get("target_count"))
            p_count = as_int(pred.get("prediction_count"))
            pred_validity = norm_validity(pred.get("prediction_validity"))
            human_validity_raw = str(annotation.get("validity_label") or "")
            human_valid = human_validity_raw == "valid" and h_count is not None
            human_reject = not human_valid
            pred_ok = pred.get("status") == "ok" and p_count is not None
            auto_valid_gate = (pred_validity == "valid") if requires_pred_valid else True
            auto_pass = bool(pred_ok and auto_valid_gate and target is not None and p_count == target)
            count_correct = bool(human_valid and pred_ok and auto_valid_gate and p_count == h_count)
            human_success = bool(human_valid and target is not None and h_count == target)
            auto_reject = bool(requires_pred_valid and pred_validity == "reject")

            rows.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "protocol": protocol,
                    "run_dir": run_dir.as_posix(),
                    "shard": shard_name,
                    "item_id": str(pred.get("item_id") or ""),
                    "image_rel_path": str(pred.get("image_rel_path") or ""),
                    "generator_model": str(pred.get("generator_model") or ""),
                    "task_name": str(pred.get("task_name") or ""),
                    "task_type": task_type,
                    "prompt_id": str(pred.get("prompt_id") or ""),
                    "cell_id": str(pred.get("cell_id") or ""),
                    "target_count": target,
                    "human_validity": human_validity_raw,
                    "human_valid": human_valid,
                    "human_reject": human_reject,
                    "human_count": h_count,
                    "prediction_status": str(pred.get("status") or ""),
                    "prediction_count": p_count,
                    "prediction_validity": pred_validity,
                    "prediction_reject": auto_reject,
                    "prediction_ok": pred_ok,
                    "count_correct": count_correct,
                    "auto_target_pass": auto_pass,
                    "human_benchmark_success": human_success,
                    "agreement_all": auto_pass == human_success,
                    "truncation_suspected": bool(pred.get("truncation_suspected")),
                    "annotation_path": annotation_path.as_posix(),
                }
            )
    return rows


def ratio(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    n = tp + fp + fn + tn
    if n == 0:
        return None
    observed = (tp + tn) / n
    auto_pos = tp + fp
    auto_neg = fn + tn
    human_pos = tp + fn
    human_neg = fp + tn
    expected = ((auto_pos * human_pos) + (auto_neg * human_neg)) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_all = len(rows)
    valid_rows = [row for row in rows if row["human_valid"]]
    human_reject_rows = [row for row in rows if row["human_reject"]]
    count_correct_valid = sum(1 for row in valid_rows if row["count_correct"])
    count_correct_all = sum(1 for row in rows if row["count_correct"])
    error_rows = [
        row
        for row in valid_rows
        if row["prediction_ok"]
        and row["prediction_count"] is not None
        and row["human_count"] is not None
    ]
    signed_errors = [int(row["prediction_count"]) - int(row["human_count"]) for row in error_rows]
    abs_errors = [abs(err) for err in signed_errors]
    mae_valid = (sum(abs_errors) / len(abs_errors)) if abs_errors else None
    bias_valid = (sum(signed_errors) / len(signed_errors)) if signed_errors else None
    abs_bias_over_mae = (
        abs(bias_valid) / mae_valid
        if mae_valid not in {None, 0} and bias_valid is not None
        else None
    )
    auto_valid = sum(1 for row in valid_rows if row["auto_target_pass"])
    auto_all = sum(1 for row in rows if row["auto_target_pass"])
    human_valid_success = sum(1 for row in valid_rows if row["human_benchmark_success"])
    human_all = sum(1 for row in rows if row["human_benchmark_success"])
    agree_all = sum(1 for row in rows if row["agreement_all"])
    tp = sum(1 for row in rows if row["auto_target_pass"] and row["human_benchmark_success"])
    fp = sum(1 for row in rows if row["auto_target_pass"] and not row["human_benchmark_success"])
    fn = sum(1 for row in rows if (not row["auto_target_pass"]) and row["human_benchmark_success"])
    tn = n_all - tp - fp - fn

    pred_status_counts = Counter(str(row["prediction_status"]) for row in rows)
    human_validity_counts = Counter(str(row["human_validity"]) for row in rows)
    pred_validity_counts = Counter(str(row["prediction_validity"]) for row in rows)
    pred_reject = sum(1 for row in rows if row["prediction_reject"])
    reject_tp = sum(1 for row in rows if row["prediction_reject"] and row["human_reject"])
    valid_false_reject = sum(1 for row in valid_rows if row["prediction_reject"])
    auto_pass_on_human_reject = sum(1 for row in human_reject_rows if row["auto_target_pass"])
    reject_precision = ratio(reject_tp, pred_reject)
    reject_recall = ratio(reject_tp, len(human_reject_rows))
    reject_f1 = (
        2 * reject_precision * reject_recall / (reject_precision + reject_recall)
        if reject_precision is not None
        and reject_recall is not None
        and (reject_precision + reject_recall) > 0
        else None
    )

    human_all_rate = ratio(human_all, n_all)
    auto_all_rate = ratio(auto_all, n_all)
    return {
        "n_all": n_all,
        "n_valid": len(valid_rows),
        "n_human_reject": len(human_reject_rows),
        "n_prediction_ok": sum(1 for row in rows if row["prediction_ok"]),
        "n_truncation_suspected": sum(1 for row in rows if row["truncation_suspected"]),
        "prediction_status_counts": json.dumps(dict(sorted(pred_status_counts.items())), sort_keys=True),
        "human_validity_counts": json.dumps(dict(sorted(human_validity_counts.items())), sort_keys=True),
        "prediction_validity_counts": json.dumps(dict(sorted(pred_validity_counts.items())), sort_keys=True),
        "prediction_ok_rate": ratio(sum(1 for row in rows if row["prediction_ok"]), n_all),
        "valid_rate": ratio(len(valid_rows), n_all),
        "count_fidelity_valid": ratio(count_correct_valid, len(valid_rows)),
        "count_fidelity_all_invalid_wrong": ratio(count_correct_all, n_all),
        "mae_valid": mae_valid,
        "bias_valid": bias_valid,
        "abs_bias_over_mae_valid": abs_bias_over_mae,
        "auto_target_pass_valid": ratio(auto_valid, len(valid_rows)),
        "auto_target_pass_all": auto_all_rate,
        "human_benchmark_valid": ratio(human_valid_success, len(valid_rows)),
        "human_benchmark_all": human_all_rate,
        "auto_minus_human_all": (
            auto_all_rate - human_all_rate
            if auto_all_rate is not None and human_all_rate is not None
            else None
        ),
        "agreement_all": ratio(agree_all, n_all),
        "kappa_all": kappa(tp, fp, fn, tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auto_reject_rate_all": ratio(pred_reject, n_all),
        "human_reject_rate_all": ratio(len(human_reject_rows), n_all),
        "reject_precision": reject_precision,
        "reject_recall": reject_recall,
        "reject_f1": reject_f1,
        "false_reject_rate_valid": ratio(valid_false_reject, len(valid_rows)),
        "auto_pass_rate_on_human_reject": ratio(auto_pass_on_human_reject, len(human_reject_rows)),
    }


def group_rows(rows: Iterable[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    return buckets


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_ci(
    rows: list[dict[str, Any]],
    n_boot: int,
    seed: int,
    metrics: list[str],
) -> dict[str, tuple[float | None, float | None]]:
    if not rows or n_boot <= 0:
        return {metric: (None, None) for metric in metrics}
    rng = random.Random(seed)
    n = len(rows)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(n_boot):
        boot_rows = [rows[rng.randrange(n)] for _ in range(n)]
        summary = summarize(boot_rows)
        for metric in metrics:
            value = summary.get(metric)
            if value is not None:
                samples[metric].append(float(value))
    return {metric: (percentile(vals, 0.025), percentile(vals, 0.975)) for metric, vals in samples.items()}


def flat_summary(group: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(group)
    for key, value in summary.items():
        if isinstance(value, float):
            if key in {"kappa_all", "mae_valid", "bias_valid", "abs_bias_over_mae_valid"}:
                out[key] = f"{value:.6f}"
            else:
                out[key] = f"{100 * value:.4f}"
        elif value is None:
            out[key] = ""
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_key, group in sorted(group_rows(rows, keys).items()):
        group_payload = {key: value for key, value in zip(keys, group_key, strict=True)}
        output.append(flat_summary(group_payload, summarize(group)))
    return output


def make_ci_rows(rows: list[dict[str, Any]], keys: list[str], n_boot: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, (group_key, group) in enumerate(sorted(group_rows(rows, keys).items())):
        group_payload = {key: value for key, value in zip(keys, group_key, strict=True)}
        ci = bootstrap_ci(group, n_boot=n_boot, seed=seed + idx * 7919, metrics=CI_METRICS)
        row: dict[str, Any] = dict(group_payload)
        row["n_boot"] = n_boot
        for metric, (lo, hi) in ci.items():
            if metric in {"kappa_all", "mae_valid", "bias_valid", "abs_bias_over_mae_valid"}:
                row[f"{metric}_ci_low"] = "" if lo is None else f"{lo:.6f}"
                row[f"{metric}_ci_high"] = "" if hi is None else f"{hi:.6f}"
            else:
                row[f"{metric}_ci_low"] = "" if lo is None else f"{100 * lo:.4f}"
                row[f"{metric}_ci_high"] = "" if hi is None else f"{100 * hi:.4f}"
        output.append(row)
    return output


def pct_cell(value: Any, signed: bool = False) -> str:
    if value in {None, ""}:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if signed:
        return f"{f:+.2f}"
    return f"{f:.2f}"


def float_cell(value: Any, signed: bool = False) -> str:
    if value in {None, ""}:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if signed:
        return f"{f:+.3f}"
    return f"{f:.3f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _, kind in columns:
            val = row.get(key, "")
            if kind == "pct":
                cells.append(pct_cell(val))
            elif kind == "pp":
                cells.append(pct_cell(val, signed=True))
            elif kind == "kappa":
                cells.append("-" if val in {None, ""} else f"{float(val):.3f}")
            elif kind == "float":
                cells.append(float_cell(val))
            elif kind == "sfloat":
                cells.append(float_cell(val, signed=True))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def merge_summary_ci(summary_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    ci_map = {tuple(row[key] for key in keys): row for row in ci_rows}
    merged: list[dict[str, Any]] = []
    for row in summary_rows:
        key = tuple(row[key_name] for key_name in keys)
        out = dict(row)
        out.update({k: v for k, v in ci_map.get(key, {}).items() if k not in keys})
        merged.append(out)
    return merged


def add_ci_text(row: dict[str, Any], metric: str, kappa_metric: bool = False) -> str:
    value = row.get(metric, "")
    lo = row.get(f"{metric}_ci_low", "")
    hi = row.get(f"{metric}_ci_high", "")
    if value in {None, ""}:
        return "-"
    if lo in {None, ""} or hi in {None, ""}:
        return pct_cell(value, signed=(metric == "auto_minus_human_all"))
    if kappa_metric:
        return f"{float(value):.3f} [{float(lo):.3f}, {float(hi):.3f}]"
    if metric in {"mae_valid", "abs_bias_over_mae_valid"}:
        return f"{float_cell(value)} [{float_cell(lo)}, {float_cell(hi)}]"
    if metric == "bias_valid":
        return f"{float_cell(value, signed=True)} [{float_cell(lo, signed=True)}, {float_cell(hi, signed=True)}]"
    sign = metric == "auto_minus_human_all"
    return f"{pct_cell(value, signed=sign)} [{pct_cell(lo, signed=sign)}, {pct_cell(hi, signed=sign)}]"


def write_report(
    out_dir: Path,
    all_rows: list[dict[str, Any]],
    run_summary: list[dict[str, Any]],
    run_ci: list[dict[str, Any]],
    run_gen_summary: list[dict[str, Any]],
    run_gen_ci: list[dict[str, Any]],
    run_task_summary: list[dict[str, Any]],
) -> None:
    run_merged = merge_summary_ci(run_summary, run_ci, ["run_id"])
    run_gen_merged = merge_summary_ci(run_gen_summary, run_gen_ci, ["run_id", "generator_model"])

    compact_rows = []
    for row in run_merged:
        compact_rows.append(
            {
                "model_name": row["model_name"],
                "protocol": row["protocol"],
                "n_all": row["n_all"],
                "ok": row.get("prediction_ok_rate", ""),
                "H_valid": add_ci_text(row, "human_benchmark_valid"),
                "A_valid": add_ci_text(row, "auto_target_pass_valid"),
                "H_all": add_ci_text(row, "human_benchmark_all"),
                "A_all": add_ci_text(row, "auto_target_pass_all"),
                "A-H": add_ci_text(row, "auto_minus_human_all"),
                "count_valid": add_ci_text(row, "count_fidelity_valid"),
                "mae": add_ci_text(row, "mae_valid"),
                "bias": add_ci_text(row, "bias_valid"),
                "bias_over_mae": add_ci_text(row, "abs_bias_over_mae_valid"),
                "agree": add_ci_text(row, "agreement_all"),
                "kappa": add_ci_text(row, "kappa_all", kappa_metric=True),
            }
        )

    by_gen_rows = []
    for row in run_gen_merged:
        by_gen_rows.append(
            {
                "model_name": row["model_name"],
                "protocol": row["protocol"],
                "generator_model": row["generator_model"],
                "n_all": row["n_all"],
                "ok": row.get("prediction_ok_rate", ""),
                "H_valid": add_ci_text(row, "human_benchmark_valid"),
                "A_valid": add_ci_text(row, "auto_target_pass_valid"),
                "H_all": add_ci_text(row, "human_benchmark_all"),
                "A_all": add_ci_text(row, "auto_target_pass_all"),
                "A-H": add_ci_text(row, "auto_minus_human_all"),
                "count_valid": add_ci_text(row, "count_fidelity_valid"),
                "mae": add_ci_text(row, "mae_valid"),
                "bias": add_ci_text(row, "bias_valid"),
                "bias_over_mae": add_ci_text(row, "abs_bias_over_mae_valid"),
                "kappa": add_ci_text(row, "kappa_all", kappa_metric=True),
            }
        )

    top_task_rows = sorted(
        run_task_summary,
        key=lambda row: abs(float(row.get("auto_minus_human_all") or 0.0)),
        reverse=True,
    )[:30]

    lines = [
        "# Task-Guided MLLM Evaluation With Bootstrap CI",
        "",
        f"- Scored predictions: `{len(all_rows)}`",
        f"- Runs: `{len({row['run_id'] for row in all_rows})}`",
        "- CI: image-level nonparametric percentile bootstrap, resampling rows within each reported group.",
        "- Count-only auto pass: predicted count equals prompt target.",
        "- Validity-aware auto pass: predicted validity is `valid` and predicted count equals prompt target; `reject` is a benchmark failure.",
        "- Human all-image benchmark: human validity is `valid` and human count equals prompt target; human invalid/clear-fail stay in denominator.",
        "",
        "## Overall By Run",
        "",
        md_table(
            compact_rows,
            [
                ("model_name", "model", "str"),
                ("protocol", "protocol", "str"),
                ("n_all", "n", "str"),
                ("ok", "ok %", "pct"),
                ("H_valid", "H valid % [CI]", "str"),
                ("A_valid", "A valid % [CI]", "str"),
                ("H_all", "H all % [CI]", "str"),
                ("A_all", "A all % [CI]", "str"),
                ("A-H", "A-H pp [CI]", "str"),
                ("count_valid", "count valid % [CI]", "str"),
                ("mae", "MAE [CI]", "str"),
                ("bias", "bias [CI]", "str"),
                ("bias_over_mae", "|bias|/MAE [CI]", "str"),
                ("agree", "agree % [CI]", "str"),
                ("kappa", "kappa [CI]", "str"),
            ],
        ),
        "",
        "## By Generator",
        "",
        md_table(
            by_gen_rows,
            [
                ("model_name", "model", "str"),
                ("protocol", "protocol", "str"),
                ("generator_model", "generator", "str"),
                ("n_all", "n", "str"),
                ("ok", "ok %", "pct"),
                ("H_valid", "H valid % [CI]", "str"),
                ("A_valid", "A valid % [CI]", "str"),
                ("H_all", "H all % [CI]", "str"),
                ("A_all", "A all % [CI]", "str"),
                ("A-H", "A-H pp [CI]", "str"),
                ("count_valid", "count valid % [CI]", "str"),
                ("mae", "MAE [CI]", "str"),
                ("bias", "bias [CI]", "str"),
                ("bias_over_mae", "|bias|/MAE [CI]", "str"),
                ("kappa", "kappa [CI]", "str"),
            ],
        ),
        "",
        "## Largest Absolute A-H Task Gaps",
        "",
        md_table(
            top_task_rows,
            [
                ("model_name", "model", "str"),
                ("protocol", "protocol", "str"),
                ("generator_model", "generator", "str"),
                ("task_name", "task", "str"),
                ("n_all", "n", "str"),
                ("human_benchmark_all", "H all", "pct"),
                ("auto_target_pass_all", "A all", "pct"),
                ("auto_minus_human_all", "A-H", "pp"),
                ("count_fidelity_valid", "count valid", "pct"),
                ("mae_valid", "MAE", "float"),
                ("bias_valid", "bias", "sfloat"),
                ("abs_bias_over_mae_valid", "|bias|/MAE", "float"),
                ("kappa_all", "kappa", "kappa"),
            ],
        ),
        "",
        "## Output Files",
        "",
        "- `scored_rows.csv`: per-image scored rows.",
        "- `summary_by_run.csv`, `summary_by_run_generator.csv`, `summary_by_run_generator_task.csv`: point estimates.",
        "- `ci_by_run.csv`, `ci_by_run_generator.csv`, `ci_by_run_generator_task.csv`: bootstrap 95% CIs.",
        "- `summary_by_run_with_ci.csv`, `summary_by_run_generator_with_ci.csv`: merged point estimates and CIs.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task-guided MLLM runs with bootstrap CIs.")
    parser.add_argument("--root", default="outputs/mllm_runs")
    parser.add_argument(
        "--run-spec",
        default="",
        help="Optional JSON run spec. Each entry provides run_id, model_name, protocol, and one or more run_dirs.",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--include-scored-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/mllm_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    run_inventory: list[dict[str, Any]] = []

    if args.run_spec:
        run_specs = read_run_spec(Path(args.run_spec))
        for spec in run_specs:
            run_rows: list[dict[str, Any]] = []
            prediction_file_count = 0
            for run_dir in spec["run_dirs"]:
                if not run_dir.exists():
                    raise SystemExit(f"Run dir in spec does not exist: {run_dir}")
                prediction_file_count += len(prediction_paths(run_dir))
                run_rows.extend(
                    load_scored_rows(
                        run_dir,
                        run_id_override=spec["run_id"],
                        model_name_override=spec["model_name"],
                        protocol_override=spec["protocol"],
                    )
                )
            all_rows.extend(run_rows)
            run_inventory.append(
                {
                    "model_name": spec["model_name"],
                    "protocol": spec["protocol"],
                    "run_id": spec["run_id"],
                    "run_dir": ";".join(run_dir.as_posix() for run_dir in spec["run_dirs"]),
                    "n_predictions": len(run_rows),
                    "prediction_files": prediction_file_count,
                    "note": spec["note"],
                }
            )
    else:
        run_dirs = discover_runs(root)
        if not run_dirs:
            raise SystemExit(f"No run directories found under {root}")
        for run_dir in run_dirs:
            rows = load_scored_rows(run_dir)
            all_rows.extend(rows)
            run_inventory.append(
                {
                    "model_name": infer_model(run_dir),
                    "protocol": infer_protocol(run_dir.name),
                    "run_id": f"{infer_model(run_dir)}/{run_dir.name}",
                    "run_dir": run_dir.as_posix(),
                    "n_predictions": len(rows),
                    "prediction_files": len(prediction_paths(run_dir)),
                }
            )

    for row in all_rows:
        row["row_uid"] = f"{row['run_id']}::{row['image_rel_path']}"

    run_summary = make_summary_rows(all_rows, ["run_id", "model_name", "protocol"])
    run_gen_summary = make_summary_rows(all_rows, ["run_id", "model_name", "protocol", "generator_model"])
    run_task_summary = make_summary_rows(
        all_rows,
        ["run_id", "model_name", "protocol", "generator_model", "task_name"],
    )
    run_cell_summary = make_summary_rows(
        all_rows,
        ["run_id", "model_name", "protocol", "generator_model", "task_name", "prompt_id", "target_count"],
    )

    run_ci = make_ci_rows(all_rows, ["run_id"], args.n_boot, args.seed)
    run_gen_ci = make_ci_rows(all_rows, ["run_id", "generator_model"], args.n_boot, args.seed + 100000)
    run_task_ci = make_ci_rows(
        all_rows,
        ["run_id", "generator_model", "task_name"],
        args.n_boot,
        args.seed + 200000,
    )

    run_with_ci = merge_summary_ci(run_summary, run_ci, ["run_id"])
    run_gen_with_ci = merge_summary_ci(run_gen_summary, run_gen_ci, ["run_id", "generator_model"])

    write_csv(out_dir / "run_inventory.csv", run_inventory)
    if args.include_scored_rows:
        write_csv(out_dir / "scored_rows.csv", all_rows)
    write_csv(out_dir / "summary_by_run.csv", run_summary)
    write_csv(out_dir / "summary_by_run_generator.csv", run_gen_summary)
    write_csv(out_dir / "summary_by_run_generator_task.csv", run_task_summary)
    write_csv(out_dir / "summary_by_cell.csv", run_cell_summary)
    write_csv(out_dir / "ci_by_run.csv", run_ci)
    write_csv(out_dir / "ci_by_run_generator.csv", run_gen_ci)
    write_csv(out_dir / "ci_by_run_generator_task.csv", run_task_ci)
    write_csv(out_dir / "summary_by_run_with_ci.csv", run_with_ci)
    write_csv(out_dir / "summary_by_run_generator_with_ci.csv", run_gen_with_ci)

    write_report(
        out_dir,
        all_rows=all_rows,
        run_summary=run_summary,
        run_ci=run_ci,
        run_gen_summary=run_gen_summary,
        run_gen_ci=run_gen_ci,
        run_task_summary=run_task_summary,
    )
    print((out_dir / "report.md").as_posix())


if __name__ == "__main__":
    main()
