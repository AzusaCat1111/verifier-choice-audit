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
    candidates = [repo_root / rel, repo_root / "release" / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for {image_rel_path}")


def load_unique_rows(csv_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task_name"] not in TASKS:
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
                    "task_name": row["task_name"],
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


def predict_task_counts(
    model: YOLO,
    rows: list[dict[str, Any]],
    *,
    conf: float,
    imgsz: int,
    batch: int,
    device: str,
    chunk_size: int,
) -> tuple[list[int], list[list[dict[str, Any]]]]:
    counts: list[int] = []
    boxes_by_image: list[list[dict[str, Any]]] = []
    total = len(rows)
    for start in range(0, total, chunk_size):
        chunk = rows[start : start + chunk_size]
        results = model.predict(
            source=[row["image_path"] for row in chunk],
            conf=conf,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )
        for result in results:
            relevant = 0
            boxes: list[dict[str, Any]] = []
            if result.boxes is not None:
                cls_values = result.boxes.cls.detach().cpu().tolist()
                conf_values = result.boxes.conf.detach().cpu().tolist()
                xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
                for cls_value, conf_value, xyxy_value in zip(cls_values, conf_values, xyxy_values, strict=True):
                    class_id = int(cls_value)
                    if class_id == 0:
                        relevant += 1
                    boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": model.names.get(class_id, str(class_id)),
                            "confidence": float(conf_value),
                            "xyxy": [float(v) for v in xyxy_value],
                            "counted_for_task": class_id == 0,
                        }
                    )
            counts.append(relevant)
            boxes_by_image.append(boxes)
        print(f"processed {min(start + chunk_size, total)}/{total}", flush=True)
    return counts, boxes_by_image


def rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else math.nan


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    n = len(group)
    valid = group[group["human_valid"].astype(bool)]
    human = group["human_benchmark_success"].astype(bool)
    auto = group["auto_target_pass"].astype(bool)
    agreement = human == auto
    human_valid = group["human_valid"].astype(bool)
    human_reject_n = int((~human_valid).sum())
    auto_pass_on_reject = group[(~human_valid) & auto]
    tp = int((auto & human).sum())
    fp = int((auto & ~human).sum())
    fn = int((~auto & human).sum())
    tn = int((~auto & ~human).sum())
    if len(valid):
        diff = valid["prediction_count"].astype(float) - valid["human_count"].astype(float)
        count_fidelity = float(valid["count_correct"].astype(bool).mean())
        mae = float(diff.abs().mean())
        bias = float(diff.mean())
        auto_target_pass_valid = float(valid["auto_target_pass"].astype(bool).mean())
        human_benchmark_valid = float(valid["human_benchmark_success"].astype(bool).mean())
    else:
        count_fidelity = math.nan
        mae = math.nan
        bias = math.nan
        auto_target_pass_valid = math.nan
        human_benchmark_valid = math.nan
    return {
        "n": n,
        "n_human_valid": len(valid),
        "n_human_reject": human_reject_n,
        "human_valid_rate": rate(len(valid), n),
        "human_reject_rate": rate(human_reject_n, n),
        "human_benchmark_score_all": float(human.mean()) if n else math.nan,
        "auto_target_pass_all": float(auto.mean()) if n else math.nan,
        "score_gap_a_minus_h_pp": (float(auto.mean()) - float(human.mean())) * 100 if n else math.nan,
        "agreement_all": float(agreement.mean()) if n else math.nan,
        "cohen_kappa_all": cohen_kappa_binary(human.astype(int).tolist(), auto.astype(int).tolist()),
        "count_fidelity_valid": count_fidelity,
        "auto_target_pass_valid": auto_target_pass_valid,
        "human_benchmark_score_valid": human_benchmark_valid,
        "mae_valid": mae,
        "bias_valid": bias,
        "auto_pass_on_human_reject_rate": rate(len(auto_pass_on_reject), human_reject_n),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_summary(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if not by:
        return pd.DataFrame([{**summarize_group(scored)}])
    records = []
    for keys, group in scored.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {key: value for key, value in zip(by, keys, strict=True)}
        record.update(summarize_group(group))
        records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one YOLO detector per object task on the MLLM object split.")
    parser.add_argument("--cell-phone-weights", type=Path, required=True)
    parser.add_argument("--chair-weights", type=Path, required=True)
    parser.add_argument("--traffic-light-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="yolo11s_task_specific_object")
    parser.add_argument(
        "--mllm-scored-rows",
        type=Path,
        default=Path("outputs/mllm_evaluation/scored_rows.csv"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_unique_rows(args.mllm_scored_rows, args.repo_root.resolve())
    weight_by_task = {
        "cell_phone_object_count": args.cell_phone_weights,
        "chair_object_count": args.chair_weights,
        "traffic_light_object_count": args.traffic_light_weights,
    }
    records: list[dict[str, Any]] = []
    for task_name in TASKS:
        rows = [row for row in all_rows if row["task_name"] == task_name]
        model = YOLO(str(weight_by_task[task_name]))
        counts, boxes_by_image = predict_task_counts(
            model,
            rows,
            conf=args.conf,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            chunk_size=args.chunk_size,
        )
        for row, pred_count, boxes in zip(rows, counts, boxes_by_image, strict=True):
            target = int(row["target_count"])
            human_count = row["human_count"]
            human_valid = bool(row["human_valid"])
            human_pass = bool(row["human_benchmark_success"])
            auto_pass = pred_count == target
            records.append(
                {
                    "run_id": f"{args.model_name}/conf_{args.conf:g}",
                    "model_name": args.model_name,
                    "protocol": f"task_specific_fixed_conf_{args.conf:g}",
                    "weights": str(weight_by_task[task_name]),
                    "conf_threshold": args.conf,
                    "image_rel_path": row["image_rel_path"],
                    "image_path": row["image_path"],
                    "annotation_path": row["annotation_path"],
                    "generator_model": row["generator_model"],
                    "task_name": row["task_name"],
                    "task_type": row["task_type"],
                    "prompt_id": row["prompt_id"],
                    "cell_id": row["cell_id"],
                    "target_count": target,
                    "human_validity": row["human_validity"],
                    "human_valid": human_valid,
                    "human_reject": bool(row["human_reject"]),
                    "human_count": human_count,
                    "prediction_count": pred_count,
                    "prediction_ok": True,
                    "count_correct": human_valid and human_count is not None and pred_count == int(human_count),
                    "auto_target_pass": auto_pass,
                    "human_benchmark_success": human_pass,
                    "agreement_all": auto_pass == human_pass,
                    "num_detected_boxes_target_class": pred_count,
                    "detected_boxes_json": json.dumps(boxes, ensure_ascii=False),
                }
            )
    scored = pd.DataFrame.from_records(records)
    summaries = {
        "overall": build_summary(scored, []),
        "by_generator": build_summary(scored, ["generator_model"]),
        "by_task": build_summary(scored, ["task_name"]),
        "by_generator_task": build_summary(scored, ["generator_model", "task_name"]),
        "by_cell": build_summary(scored, ["generator_model", "task_name", "cell_id"]),
    }
    scored.to_csv(args.out_dir / "scored_rows.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(args.out_dir / f"summary_{name}.csv", index=False)
    lines = [
        "# YOLO11s Task-Specific Object Detectors",
        "",
        f"- Confidence: `{args.conf:g}`",
        "- One one-class detector per task: cell phone, chair, traffic light.",
        "",
        "## Overall",
        "",
        summaries["overall"].to_markdown(index=False),
        "",
        "## By Generator Task",
        "",
        summaries["by_generator_task"].to_markdown(index=False),
        "",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
