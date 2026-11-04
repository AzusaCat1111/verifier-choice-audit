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
    "turbine_substructure_count": 0,
    "clover_leaf_substructure_count": 1,
    "dresser_substructure_count": 2,
    "power_strip_substructure_count": 3,
}

TASK_CLASS_NAME = {
    "turbine_substructure_count": "Turbine Blade",
    "clover_leaf_substructure_count": "Clover Leaflet",
    "dresser_substructure_count": "Dresser Drawer",
    "power_strip_substructure_count": "Powerstrip Socket",
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
    observed = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = sum((true_counts[c] / n) * (pred_counts[c] / n) for c in (0, 1))
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def resolve_image_path(repo_root: Path, image_rel_path: str) -> Path:
    rel = Path(image_rel_path)
    candidates = [
        repo_root / rel,
        repo_root / "release" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for {image_rel_path}")


def load_unique_manifest(csv_path: Path, repo_root: Path) -> list[dict[str, Any]]:
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
            if target_count is None:
                raise ValueError(f"Missing target_count for {image_rel_path}")
            human_count = parse_int_or_none(row.get("human_count"))
            human_valid = parse_bool(row.get("human_valid"))
            human_reject = parse_bool(row.get("human_reject"))
            human_benchmark_success = parse_bool(row.get("human_benchmark_success"))

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
                    "human_valid": human_valid,
                    "human_reject": human_reject,
                    "human_count": human_count,
                    "human_benchmark_success": human_benchmark_success,
                }
            )
    return rows


def predict_counts(
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
        sources = [row["image_path"] for row in chunk]
        results = model.predict(
            source=sources,
            conf=conf,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )
        for row, result in zip(chunk, results, strict=True):
            target_class = TASK_CLASS_ID[row["task_name"]]
            relevant_count = 0
            image_boxes: list[dict[str, Any]] = []
            if result.boxes is not None:
                cls_values = result.boxes.cls.detach().cpu().tolist()
                conf_values = result.boxes.conf.detach().cpu().tolist()
                xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
                for cls_value, conf_value, xyxy_value in zip(cls_values, conf_values, xyxy_values, strict=True):
                    class_id = int(cls_value)
                    if class_id == target_class:
                        relevant_count += 1
                    image_boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": model.names.get(class_id, str(class_id)),
                            "confidence": float(conf_value),
                            "xyxy": [float(v) for v in xyxy_value],
                            "counted_for_task": class_id == target_class,
                        }
                    )
            counts.append(relevant_count)
            boxes_by_image.append(image_boxes)
        print(f"processed {min(start + chunk_size, total)}/{total}", flush=True)
    return counts, boxes_by_image


def make_scored_rows(
    rows: list[dict[str, Any]],
    counts: list[int],
    boxes_by_image: list[list[dict[str, Any]]],
    *,
    conf: float,
    weights: Path,
    model_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row, pred_count, boxes in zip(rows, counts, boxes_by_image, strict=True):
        target = int(row["target_count"])
        human_count = row["human_count"]
        human_valid = bool(row["human_valid"])
        human_benchmark_success = bool(row["human_benchmark_success"])
        auto_target_pass = pred_count == target
        count_correct = human_valid and human_count is not None and pred_count == int(human_count)
        records.append(
            {
                "run_id": f"{model_name}/conf_0.25" if abs(conf - 0.25) < 1e-12 else f"{model_name}/conf_{conf:g}",
                "model_name": model_name,
                "protocol": f"fixed_conf_{conf:g}",
                "weights": str(weights),
                "conf_threshold": conf,
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
                "prediction_status": "ok",
                "prediction_count": pred_count,
                "prediction_validity": "",
                "prediction_reject": False,
                "prediction_ok": True,
                "count_correct": bool(count_correct),
                "auto_target_pass": bool(auto_target_pass),
                "human_benchmark_success": human_benchmark_success,
                "agreement_all": bool(auto_target_pass == human_benchmark_success),
                "target_class_id": TASK_CLASS_ID[row["task_name"]],
                "target_class_name": TASK_CLASS_NAME[row["task_name"]],
                "num_detected_boxes_all_classes": len(boxes),
                "num_detected_boxes_target_class": pred_count,
                "detected_boxes_json": json.dumps(boxes, ensure_ascii=False),
            }
        )
    return pd.DataFrame.from_records(records)


def rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else math.nan


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    n = len(group)
    valid = group[group["human_valid"].astype(bool)]
    human = group["human_benchmark_success"].astype(bool)
    auto = group["auto_target_pass"].astype(bool)
    agreement = (human == auto)

    valid_n = len(valid)
    human_reject_n = int((~group["human_valid"].astype(bool)).sum())
    auto_pass_on_reject = group[(~group["human_valid"].astype(bool)) & group["auto_target_pass"].astype(bool)]

    tp = int((auto & human).sum())
    fp = int((auto & ~human).sum())
    fn = int((~auto & human).sum())
    tn = int((~auto & ~human).sum())

    if valid_n:
        diff = valid["prediction_count"].astype(float) - valid["human_count"].astype(float)
        mae = float(diff.abs().mean())
        bias = float(diff.mean())
        count_fidelity = float(valid["count_correct"].astype(bool).mean())
        auto_target_pass_valid = float(valid["auto_target_pass"].astype(bool).mean())
        human_benchmark_valid = float(valid["human_benchmark_success"].astype(bool).mean())
    else:
        mae = math.nan
        bias = math.nan
        count_fidelity = math.nan
        auto_target_pass_valid = math.nan
        human_benchmark_valid = math.nan

    return {
        "n": n,
        "n_human_valid": valid_n,
        "n_human_reject": human_reject_n,
        "human_valid_rate": rate(valid_n, n),
        "human_reject_rate": rate(human_reject_n, n),
        "prediction_ok_rate": 1.0,
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
        "abs_bias_over_mae": abs(bias) / (mae + 1e-12) if not math.isnan(mae) else math.nan,
        "auto_pass_on_human_reject_rate": rate(len(auto_pass_on_reject), human_reject_n),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_summary(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if by:
        rows = []
        for keys, group in scored.groupby(by, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = {key: value for key, value in zip(by, keys, strict=True)}
            record.update(summarize_group(group))
            rows.append(record)
        return pd.DataFrame(rows)
    return pd.DataFrame([{**summarize_group(scored)}])


def write_report(out_dir: Path, summaries: dict[str, pd.DataFrame], *, conf: float, weights: Path, source_csv: Path, model_name: str) -> None:
    def fmt_table(df: pd.DataFrame) -> str:
        display = df.copy()
        percent_cols = [
            "human_valid_rate",
            "human_reject_rate",
            "prediction_ok_rate",
            "human_benchmark_score_all",
            "auto_target_pass_all",
            "agreement_all",
            "count_fidelity_valid",
            "auto_target_pass_valid",
            "human_benchmark_score_valid",
            "auto_pass_on_human_reject_rate",
        ]
        for col in percent_cols:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{100 * x:.2f}%" if pd.notna(x) else "-")
        for col in ["score_gap_a_minus_h_pp", "mae_valid", "bias_valid", "abs_bias_over_mae", "cohen_kappa_all"]:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        return display.to_markdown(index=False)

    lines = [
        f"# {model_name} Substructure on MLLM Split",
        "",
        f"- Source split: `{source_csv}`",
        f"- Weights: `{weights}`",
        f"- Protocol: count-only detector, fixed confidence `{conf:g}`",
        f"- Supported tasks: `{', '.join(TASK_CLASS_ID)}`",
        "- Auto target pass uses the same benchmark rule as direct-count MLLM: `prediction_count == target_count`.",
        "- Human benchmark pass is copied from the MLLM scored rows: `human_valid && human_count == target_count`.",
        "- Count fidelity / MAE / bias are computed only on human-valid images.",
        "",
        "## Overall",
        "",
        fmt_table(summaries["overall"]),
        "",
        "## By Generator",
        "",
        fmt_table(summaries["by_generator"]),
        "",
        "## By Task",
        "",
        fmt_table(summaries["by_task"]),
        "",
        "## By Generator And Task",
        "",
        fmt_table(summaries["by_generator_task"]),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO substructure detector on the same image split used by MLLM.")
    parser.add_argument(
        "--mllm-scored-rows",
        type=Path,
        default=Path("outputs/mllm_evaluation/scored_rows.csv"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/yolo/substructure/train/weights/best.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/yolo/substructure/evaluation"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--model-name", default="yolo_substructure")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_unique_manifest(args.mllm_scored_rows, repo_root)
    if not rows:
        raise SystemExit("No supported substructure rows found in MLLM scored rows.")

    model = YOLO(str(args.weights))
    counts, boxes_by_image = predict_counts(
        model,
        rows,
        conf=args.conf,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    scored = make_scored_rows(rows, counts, boxes_by_image, conf=args.conf, weights=args.weights, model_name=args.model_name)
    summaries = {
        "overall": build_summary(scored, []),
        "by_generator": build_summary(scored, ["generator_model"]),
        "by_task": build_summary(scored, ["task_name"]),
        "by_generator_task": build_summary(scored, ["generator_model", "task_name"]),
        "by_cell": build_summary(scored, ["generator_model", "task_name", "cell_id"]),
    }

    scored.to_csv(out_dir / "scored_rows.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(out_dir / f"summary_{name}.csv", index=False)
    write_report(out_dir, summaries, conf=args.conf, weights=args.weights, source_csv=args.mllm_scored_rows, model_name=args.model_name)
    print(f"Wrote YOLO MLLM-split evaluation for {len(scored)} images to {out_dir}")


if __name__ == "__main__":
    main()
