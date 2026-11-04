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


TASKS = [
    "cell_phone_object_count",
    "chair_object_count",
    "traffic_light_object_count",
]


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


def resolve_image_path(repo_root: Path, image_rel_path: str) -> Path:
    rel = Path(image_rel_path)
    for candidate in [repo_root / rel, repo_root / "release" / rel]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(image_rel_path)


def load_task_manifest(path: Path, task_name: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["task_name"] != task_name:
                continue
            rows.append(
                {
                    "image_path": str(Path(item["task_dataset_image"]).resolve()),
                    "image_rel_path": item["task_dataset_image"],
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
                }
            )
    return rows


def load_test_rows(csv_path: Path, repo_root: Path, task_name: str) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["task_name"] != task_name:
                continue
            if row["image_rel_path"] in seen:
                continue
            seen.add(row["image_rel_path"])
            target = parse_int_or_none(row["target_count"])
            human_count = parse_int_or_none(row.get("human_count"))
            rows.append(
                {
                    "image_rel_path": row["image_rel_path"],
                    "image_path": str(resolve_image_path(repo_root, row["image_rel_path"])),
                    "annotation_path": row["annotation_path"],
                    "generator_model": row["generator_model"],
                    "task_name": row["task_name"],
                    "task_type": row["task_type"],
                    "prompt_id": row["prompt_id"],
                    "cell_id": row["cell_id"],
                    "target_count": target,
                    "human_validity": row["human_validity"],
                    "human_valid": parse_bool(row.get("human_valid")),
                    "human_reject": parse_bool(row.get("human_reject")),
                    "human_count": human_count,
                    "human_benchmark_success": parse_bool(row.get("human_benchmark_success")),
                }
            )
    return rows


def predict_confidences(model: YOLO, rows: list[dict[str, Any]], *, min_conf: float, imgsz: int, batch: int, device: str) -> list[list[float]]:
    out = []
    for start in range(0, len(rows), 32):
        chunk = rows[start : start + 32]
        results = model.predict(
            source=[row["image_path"] for row in chunk],
            conf=min_conf,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )
        for result in results:
            confs = []
            if result.boxes is not None:
                cls_values = result.boxes.cls.detach().cpu().tolist()
                conf_values = result.boxes.conf.detach().cpu().tolist()
                confs = [float(conf) for cls, conf in zip(cls_values, conf_values, strict=True) if int(cls) == 0]
            out.append(confs)
        print(f"processed {min(start + 32, len(rows))}/{len(rows)}", flush=True)
    return out


def count_at(confs: list[float], conf: float) -> int:
    return sum(1 for value in confs if value >= conf)


def choose_conf(rows: list[dict[str, Any]], confs: list[list[float]], grid: list[float]) -> tuple[float, pd.DataFrame]:
    records = []
    for conf in grid:
        preds = [count_at(values, conf) for values in confs]
        diffs = [pred - int(row["human_count"]) for row, pred in zip(rows, preds, strict=True)]
        human_accept = [int(row["human_benchmark_success"]) for row in rows]
        auto_accept = [int(pred == int(row["target_count"])) for row, pred in zip(rows, preds, strict=True)]
        records.append(
            {
                "conf": conf,
                "n": len(rows),
                "exact_count_rate": sum(diff == 0 for diff in diffs) / len(diffs),
                "mae": sum(abs(diff) for diff in diffs) / len(diffs),
                "bias": sum(diffs) / len(diffs),
                "accept_reject_agreement": sum(h == a for h, a in zip(human_accept, auto_accept, strict=True)) / len(rows),
                "accept_reject_kappa": cohen_kappa_binary(human_accept, auto_accept),
            }
        )
    curve = pd.DataFrame.from_records(records)
    best = max(records, key=lambda row: (row["exact_count_rate"], -row["mae"], -abs(row["bias"]), row["conf"]))
    return float(best["conf"]), curve


def summarize(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> dict[str, Any]:
        n = len(group)
        human = group["human_benchmark_success"].astype(bool)
        auto = group["auto_target_pass"].astype(bool)
        valid = group[group["human_valid"].astype(bool)]
        if len(valid):
            diff = valid["prediction_count"].astype(float) - valid["human_count"].astype(float)
            count_fidelity = float(valid["count_correct"].astype(bool).mean())
            mae = float(diff.abs().mean())
            bias = float(diff.mean())
        else:
            count_fidelity = math.nan
            mae = math.nan
            bias = math.nan
        return {
            "n": n,
            "human_benchmark_score_all": float(human.mean()),
            "auto_target_pass_all": float(auto.mean()),
            "score_gap_a_minus_h_pp": (float(auto.mean()) - float(human.mean())) * 100,
            "agreement_all": float((human == auto).mean()),
            "cohen_kappa_all": cohen_kappa_binary(human.astype(int).tolist(), auto.astype(int).tolist()),
            "count_fidelity_valid": count_fidelity,
            "mae_valid": mae,
            "bias_valid": bias,
        }

    if not by:
        return pd.DataFrame([one(scored)])
    rows = []
    for keys, group in scored.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {key: value for key, value in zip(by, keys, strict=True)}
        record.update(one(group))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one-class task-specific YOLO object detectors and evaluate.")
    parser.add_argument("--cell-phone-weights", type=Path, required=True)
    parser.add_argument("--chair-weights", type=Path, required=True)
    parser.add_argument("--traffic-light-weights", type=Path, required=True)
    parser.add_argument("--task-manifest-root", type=Path, default=Path("outputs/yolo/per_task_datasets"))
    parser.add_argument(
        "--mllm-scored-rows",
        type=Path,
        default=Path("outputs/mllm_evaluation/scored_rows.csv"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-conf", type=float, default=0.001)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "cell_phone_object_count": args.cell_phone_weights,
        "chair_object_count": args.chair_weights,
        "traffic_light_object_count": args.traffic_light_weights,
    }
    grid = [round(i / 100, 2) for i in range(1, 100)]
    selected = []
    scored_rows = []
    for task_name in TASKS:
        model = YOLO(str(weights[task_name]))
        calib_rows = load_task_manifest(args.task_manifest_root / task_name / "manifest.jsonl", task_name)
        calib_confs = predict_confidences(model, calib_rows, min_conf=args.min_conf, imgsz=args.imgsz, batch=args.batch, device=args.device)
        conf, curve = choose_conf(calib_rows, calib_confs, grid)
        curve.insert(0, "task_name", task_name)
        curve.to_csv(args.out_dir / f"calibration_curve_{task_name}.csv", index=False)
        selected.append({"task_name": task_name, "conf": conf, **curve[curve["conf"].eq(conf)].iloc[0].to_dict()})

        test_rows = load_test_rows(args.mllm_scored_rows, args.repo_root.resolve(), task_name)
        test_confs = predict_confidences(model, test_rows, min_conf=args.min_conf, imgsz=args.imgsz, batch=args.batch, device=args.device)
        for row, values in zip(test_rows, test_confs, strict=True):
            pred = count_at(values, conf)
            human_count = row["human_count"]
            human_valid = bool(row["human_valid"])
            scored_rows.append(
                {
                    **row,
                    "weights": str(weights[task_name]),
                    "conf_threshold": conf,
                    "prediction_count": pred,
                    "auto_target_pass": pred == int(row["target_count"]),
                    "count_correct": human_valid and human_count is not None and pred == int(human_count),
                    "agreement_all": (pred == int(row["target_count"])) == bool(row["human_benchmark_success"]),
                }
            )

    scored = pd.DataFrame.from_records(scored_rows)
    selected_df = pd.DataFrame.from_records(selected)
    summaries = {
        "overall": summarize(scored, []),
        "by_task": summarize(scored, ["task_name"]),
        "by_generator_task": summarize(scored, ["generator_model", "task_name"]),
    }
    selected_df.to_csv(args.out_dir / "selected_conf_by_task.csv", index=False)
    scored.to_csv(args.out_dir / "scored_rows.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(args.out_dir / f"summary_{name}.csv", index=False)
    lines = [
        "# Task-Specific YOLO11s Adaptive Confidence",
        "",
        "Confidence is selected on each task's task26 train+val set, then fixed on MLLM split test.",
        "",
        "## Selected Confidence",
        "",
        selected_df.to_markdown(index=False),
        "",
        "## Overall",
        "",
        summaries["overall"].to_markdown(index=False),
        "",
        "## By Task",
        "",
        summaries["by_task"].to_markdown(index=False),
        "",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
