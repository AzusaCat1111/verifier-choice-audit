#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO


TASK_CLASS_ID = {
    "cell_phone_object_count": 0,
    "chair_object_count": 1,
    "traffic_light_object_count": 2,
}


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def cohen_kappa_binary(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return math.nan
    n = len(y_true)
    observed = sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True)) / n
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = sum((true_counts[c] / n) * (pred_counts[c] / n) for c in (0, 1))
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else math.nan


def load_calibration_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_name = item["task_name"]
            if task_name not in TASK_CLASS_ID:
                continue
            rows.append(
                {
                    "image_path": str(Path(item["dataset_image"]).resolve()),
                    "image_rel_path": item["dataset_image"],
                    "generator_model": "qwenimage2512",
                    "task_name": task_name,
                    "prompt_id": item["prompt_id"],
                    "cell_id": f"{item['task_name']}__{item['prompt_id']}__{item['count_dir']}",
                    "target_count": int(item["target_count"]),
                    "human_validity": item["validity_label"],
                    "human_valid": item["validity_label"] == "valid",
                    "human_count": int(item["observed_count"]),
                    "human_benchmark_success": item["validity_label"] == "valid"
                    and int(item["observed_count"]) == int(item["target_count"]),
                    "split": item["split"],
                }
            )
    return rows


def resolve_image_path(repo_root: Path, image_rel_path: str) -> Path:
    rel = Path(image_rel_path)
    candidates = [repo_root / rel, repo_root / "release" / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for {image_rel_path}")


def load_test_rows(csv_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["task_name"]
            if task_name not in TASK_CLASS_ID:
                continue
            image_rel_path = row["image_rel_path"]
            if image_rel_path in seen:
                continue
            seen.add(image_rel_path)
            target_count = parse_int_or_none(row["target_count"])
            human_count = parse_int_or_none(row.get("human_count"))
            if target_count is None:
                raise ValueError(f"Missing target_count for {image_rel_path}")
            rows.append(
                {
                    "image_rel_path": image_rel_path,
                    "image_path": str(resolve_image_path(repo_root, image_rel_path)),
                    "annotation_path": row["annotation_path"],
                    "generator_model": row["generator_model"],
                    "task_name": task_name,
                    "task_type": row["task_type"],
                    "prompt_id": row["prompt_id"],
                    "cell_id": row["cell_id"],
                    "target_count": target_count,
                    "human_validity": row["human_validity"],
                    "human_valid": parse_bool(row.get("human_valid")),
                    "human_reject": parse_bool(row.get("human_reject")),
                    "human_count": human_count,
                    "human_benchmark_success": parse_bool(row.get("human_benchmark_success")),
                }
            )
    return rows


def predict_target_confidences(
    model: YOLO,
    rows: list[dict[str, Any]],
    *,
    imgsz: int,
    batch: int,
    device: str,
    chunk_size: int,
    min_conf: float,
) -> list[list[float]]:
    out: list[list[float]] = []
    total = len(rows)
    for start in range(0, total, chunk_size):
        chunk = rows[start : start + chunk_size]
        results = model.predict(
            source=[row["image_path"] for row in chunk],
            conf=min_conf,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )
        for row, result in zip(chunk, results, strict=True):
            target_class = TASK_CLASS_ID[row["task_name"]]
            confs: list[float] = []
            if result.boxes is not None:
                cls_values = result.boxes.cls.detach().cpu().tolist()
                conf_values = result.boxes.conf.detach().cpu().tolist()
                for cls_value, conf_value in zip(cls_values, conf_values, strict=True):
                    if int(cls_value) == target_class:
                        confs.append(float(conf_value))
            out.append(confs)
        print(f"processed {min(start + chunk_size, total)}/{total}", flush=True)
    return out


def count_at(confidences: list[float], threshold: float) -> int:
    return sum(1 for conf in confidences if conf >= threshold)


def summarize_binary(rows: list[dict[str, Any]], pred_counts: list[int]) -> dict[str, Any]:
    n = len(rows)
    human_accept = [int(bool(row["human_benchmark_success"])) for row in rows]
    auto_accept = [int(pred == int(row["target_count"])) for row, pred in zip(rows, pred_counts, strict=True)]
    agreement = sum(int(h == a) for h, a in zip(human_accept, auto_accept, strict=True))
    valid_indices = [idx for idx, row in enumerate(rows) if bool(row["human_valid"]) and row["human_count"] is not None]
    if valid_indices:
        diffs = [pred_counts[idx] - int(rows[idx]["human_count"]) for idx in valid_indices]
        count_fidelity = sum(int(diff == 0) for diff in diffs) / len(diffs)
        mae = sum(abs(diff) for diff in diffs) / len(diffs)
        bias = sum(diffs) / len(diffs)
    else:
        count_fidelity = math.nan
        mae = math.nan
        bias = math.nan
    human_reject = [1 - value for value in human_accept]
    auto_reject = [1 - value for value in auto_accept]
    tp = sum(int(a == 1 and h == 1) for a, h in zip(auto_accept, human_accept, strict=True))
    fp = sum(int(a == 1 and h == 0) for a, h in zip(auto_accept, human_accept, strict=True))
    fn = sum(int(a == 0 and h == 1) for a, h in zip(auto_accept, human_accept, strict=True))
    tn = sum(int(a == 0 and h == 0) for a, h in zip(auto_accept, human_accept, strict=True))
    return {
        "n": n,
        "human_accept_rate": sum(human_accept) / n if n else math.nan,
        "auto_accept_rate": sum(auto_accept) / n if n else math.nan,
        "auto_reject_rate": sum(auto_reject) / n if n else math.nan,
        "score_gap_a_minus_h_pp": ((sum(auto_accept) - sum(human_accept)) / n * 100) if n else math.nan,
        "accept_reject_agreement": agreement / n if n else math.nan,
        "accept_reject_kappa": cohen_kappa_binary(human_accept, auto_accept),
        "reject_kappa_same": cohen_kappa_binary(human_reject, auto_reject),
        "count_fidelity_valid": count_fidelity,
        "mae_valid": mae,
        "bias_valid": bias,
        "tp_accept": tp,
        "fp_accept": fp,
        "fn_accept": fn,
        "tn_accept": tn,
    }


def choose_conf_by_task(
    rows: list[dict[str, Any]],
    confidences: list[list[float]],
    conf_grid: list[float],
) -> tuple[dict[str, float], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for task_name in sorted(TASK_CLASS_ID):
        indices = [idx for idx, row in enumerate(rows) if row["task_name"] == task_name]
        task_records = []
        for conf in conf_grid:
            pred_counts = [count_at(confidences[idx], conf) for idx in indices]
            gt_counts = [int(rows[idx]["human_count"]) for idx in indices]
            diffs = [pred - gt for pred, gt in zip(pred_counts, gt_counts, strict=True)]
            exact = sum(int(diff == 0) for diff in diffs) / len(diffs)
            mae = sum(abs(diff) for diff in diffs) / len(diffs)
            bias = sum(diffs) / len(diffs)
            human_accept = [int(bool(rows[idx]["human_benchmark_success"])) for idx in indices]
            auto_accept = [int(pred == int(rows[idx]["target_count"])) for idx, pred in zip(indices, pred_counts, strict=True)]
            record = {
                "task_name": task_name,
                "conf": conf,
                "n": len(indices),
                "exact_count_rate": exact,
                "mae": mae,
                "bias": bias,
                "accept_reject_agreement": sum(int(h == a) for h, a in zip(human_accept, auto_accept, strict=True)) / len(indices),
                "accept_reject_kappa": cohen_kappa_binary(human_accept, auto_accept),
            }
            task_records.append(record)
            records.append(record)
        best = max(
            task_records,
            key=lambda row: (
                row["exact_count_rate"],
                -row["mae"],
                -abs(row["bias"]),
                row["conf"],
            ),
        )
        selected[task_name] = float(best["conf"])
    return selected, pd.DataFrame.from_records(records)


def score_rows_with_task_conf(
    rows: list[dict[str, Any]],
    confidences: list[list[float]],
    selected_conf: dict[str, float],
    *,
    model_name: str,
    weights: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    pred_counts = [count_at(confs, selected_conf[row["task_name"]]) for row, confs in zip(rows, confidences, strict=True)]
    for row, pred_count in zip(rows, pred_counts, strict=True):
        target = int(row["target_count"])
        human_count = row["human_count"]
        human_valid = bool(row["human_valid"])
        human_accept = bool(row["human_benchmark_success"])
        auto_accept = pred_count == target
        records.append(
            {
                "run_id": f"{model_name}/adaptive_conf_by_task",
                "model_name": model_name,
                "protocol": "adaptive_conf_by_task_trainval",
                "weights": str(weights),
                "conf_threshold": selected_conf[row["task_name"]],
                "image_rel_path": row["image_rel_path"],
                "image_path": row["image_path"],
                "annotation_path": row.get("annotation_path", ""),
                "generator_model": row["generator_model"],
                "task_name": row["task_name"],
                "task_type": row.get("task_type", "object_count"),
                "prompt_id": row["prompt_id"],
                "cell_id": row["cell_id"],
                "target_count": target,
                "human_validity": row["human_validity"],
                "human_valid": human_valid,
                "human_count": human_count,
                "prediction_count": pred_count,
                "auto_accept": auto_accept,
                "auto_reject_count_ne_target": not auto_accept,
                "human_accept": human_accept,
                "human_reject": not human_accept,
                "auto_target_pass": auto_accept,
                "human_benchmark_success": human_accept,
                "agreement_all": auto_accept == human_accept,
                "count_correct": human_valid and human_count is not None and pred_count == int(human_count),
            }
        )
    return pd.DataFrame.from_records(records)


def build_summary(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [((), scored)] if not by else scored.groupby(by, dropna=False, sort=True)
    for keys, group in groups:
        if by and not isinstance(keys, tuple):
            keys = (keys,)
        record = {key: value for key, value in zip(by, keys, strict=True)}
        summary = summarize_binary(group.to_dict("records"), group["prediction_count"].astype(int).tolist())
        record.update(summary)
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def write_report(out_dir: Path, *, selected_conf_df: pd.DataFrame, summaries: dict[str, pd.DataFrame], model_name: str) -> None:
    def fmt(df: pd.DataFrame) -> str:
        display = df.copy()
        for col in [
            "exact_count_rate",
            "human_accept_rate",
            "auto_accept_rate",
            "auto_reject_rate",
            "accept_reject_agreement",
            "count_fidelity_valid",
        ]:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{100 * x:.2f}%" if pd.notna(x) else "-")
        for col in ["conf", "mae", "bias", "score_gap_a_minus_h_pp", "accept_reject_kappa", "reject_kappa_same", "mae_valid", "bias_valid"]:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        return display.to_markdown(index=False)

    lines = [
        f"# {model_name} Adaptive Confidence Object Evaluation",
        "",
        "- Confidence is selected on task26 train+val only, then fixed for MLLM split test.",
        "- Calibration objective: exact detector count against labeled/observed train+val count.",
        "- Test accept/reject label: accept iff `detector_count == prompt_target`; reject iff `detector_count != prompt_target`.",
        "- Human accept/reject label: accept iff `human_valid && human_count == prompt_target`.",
        "",
        "## Selected Confidence",
        "",
        fmt(selected_conf_df),
        "",
        "## Test Overall",
        "",
        fmt(summaries["overall"]),
        "",
        "## Test By Generator",
        "",
        fmt(summaries["by_generator"]),
        "",
        "## Test By Task",
        "",
        fmt(summaries["by_task"]),
        "",
        "## Test By Generator And Task",
        "",
        fmt(summaries["by_generator_task"]),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate YOLO object detector confidence on train+val and evaluate on MLLM object split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("dataset/detector/yolo/object_manifest.jsonl"))
    parser.add_argument(
        "--mllm-scored-rows",
        type=Path,
        default=Path("outputs/mllm_evaluation/scored_rows.csv"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-conf", type=float, default=0.001)
    parser.add_argument("--conf-min", type=float, default=0.01)
    parser.add_argument("--conf-max", type=float, default=0.99)
    parser.add_argument("--conf-step", type=float, default=0.01)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conf_grid = []
    value = args.conf_min
    while value <= args.conf_max + 1e-12:
        conf_grid.append(round(value, 4))
        value += args.conf_step

    model = YOLO(str(args.weights))

    calibration_rows = load_calibration_rows(args.manifest)
    print(f"Calibrating on {len(calibration_rows)} train+val images", flush=True)
    calibration_confidences = predict_target_confidences(
        model,
        calibration_rows,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        chunk_size=args.chunk_size,
        min_conf=args.min_conf,
    )
    selected_conf, calibration_curve = choose_conf_by_task(calibration_rows, calibration_confidences, conf_grid)
    selected_rows = []
    for task_name, conf in selected_conf.items():
        row = calibration_curve[(calibration_curve["task_name"] == task_name) & (calibration_curve["conf"] == conf)].iloc[0].to_dict()
        selected_rows.append(row)
    selected_conf_df = pd.DataFrame.from_records(selected_rows)

    test_rows = load_test_rows(args.mllm_scored_rows, args.repo_root.resolve())
    print(f"Evaluating on {len(test_rows)} MLLM split images", flush=True)
    test_confidences = predict_target_confidences(
        model,
        test_rows,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        chunk_size=args.chunk_size,
        min_conf=args.min_conf,
    )
    scored = score_rows_with_task_conf(test_rows, test_confidences, selected_conf, model_name=args.model_name, weights=args.weights)
    summaries = {
        "overall": build_summary(scored, []),
        "by_generator": build_summary(scored, ["generator_model"]),
        "by_task": build_summary(scored, ["task_name"]),
        "by_generator_task": build_summary(scored, ["generator_model", "task_name"]),
        "by_cell": build_summary(scored, ["generator_model", "task_name", "cell_id"]),
    }

    calibration_curve.to_csv(args.out_dir / "calibration_curve.csv", index=False)
    selected_conf_df.to_csv(args.out_dir / "selected_conf_by_task.csv", index=False)
    scored.to_csv(args.out_dir / "scored_rows.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(args.out_dir / f"summary_{name}.csv", index=False)
    write_report(args.out_dir, selected_conf_df=selected_conf_df, summaries=summaries, model_name=args.model_name)
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
