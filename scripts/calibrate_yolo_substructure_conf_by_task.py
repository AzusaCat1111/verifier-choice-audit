#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
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


def image_index(path: str) -> int:
    match = re.search(r"_(\d+)\.png$", path)
    return int(match.group(1)) if match else 10**9


def original_basename(path: str) -> str:
    name = Path(path).name
    if "__" in name:
        return name.split("__")[-1]
    return name


def read_used_training_basenames(paths: list[Path]) -> set[str]:
    used: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    used.add(original_basename(text))
    return used


def resolve_image_path(repo_root: Path, image_rel_path: str) -> Path:
    rel = Path(image_rel_path)
    candidates = [repo_root / rel, repo_root / "release" / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for {image_rel_path}")


def load_test_manifest(csv_path: Path, repo_root: Path) -> list[dict[str, Any]]:
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
            rows.append(
                {
                    "split": "test",
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
                    "human_count": parse_int_or_none(row.get("human_count")),
                    "human_benchmark_success": parse_bool(row.get("human_benchmark_success")),
                }
            )
    return rows


def annotation_target(annotation: dict[str, Any]) -> int:
    value = annotation.get("target_c")
    if value is None:
        value = annotation.get("target_k")
    if value is None:
        raise ValueError("Annotation has no target count.")
    return int(value)


def release_image_rel_path(annotation_path: Path, repo_root: Path) -> str:
    image_path = annotation_path.with_suffix(".png")
    return str(image_path.relative_to(repo_root / "release"))


def annotation_to_row(annotation_path: Path, repo_root: Path) -> dict[str, Any]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_rel_path = release_image_rel_path(annotation_path, repo_root)
    generator_model = Path(image_rel_path).parts[0]
    task_name = str(annotation.get("task_id") or annotation_path.parts[-5])
    target_count = annotation_target(annotation)
    prompt_id = str(annotation.get("prompt_id") or annotation_path.parent.name)
    cell_id = f"{generator_model}__{task_name}__{prompt_id}__c{target_count}"
    human_valid = annotation.get("validity_label") == "valid"
    human_count = parse_int_or_none(annotation.get("observed_part_count"))
    return {
        "split": "calibration",
        "image_rel_path": image_rel_path,
        "image_path": str(repo_root / "release" / image_rel_path),
        "annotation_path": str(annotation_path),
        "generator_model": generator_model,
        "task_name": task_name,
        "task_type": str(annotation.get("task_type") or "substructure"),
        "prompt_id": prompt_id,
        "cell_id": cell_id,
        "target_count": target_count,
        "human_validity": str(annotation.get("validity_label") or ""),
        "human_valid": human_valid,
        "human_reject": not human_valid,
        "human_count": human_count,
        "human_benchmark_success": bool(human_valid and human_count == target_count),
    }


def collect_calibration_candidates(
    repo_root: Path,
    *,
    test_image_rel_paths: set[str],
    train_val_basenames: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for generator_dir in sorted((repo_root / "release").iterdir()):
        if not generator_dir.is_dir():
            continue
        for task_name in sorted(TASK_CLASS_ID):
            task_root = generator_dir / "annotation_datasets" / task_name
            if not task_root.exists():
                continue
            for annotation_path in sorted(task_root.rglob("*.json")):
                row = annotation_to_row(annotation_path, repo_root)
                if row["image_rel_path"] in test_image_rel_paths:
                    continue
                if original_basename(row["image_rel_path"]) in train_val_basenames:
                    continue
                if not row["human_valid"] or row["human_count"] is None:
                    continue
                candidates.append(row)
    return candidates


def select_balanced_calibration(candidates: list[dict[str, Any]], per_task: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_task_cell: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_task_cell[row["task_name"]][row["cell_id"]].append(row)

    for task_name in sorted(TASK_CLASS_ID):
        cell_rows = by_task_cell[task_name]
        for rows in cell_rows.values():
            rows.sort(key=lambda row: (image_index(row["image_rel_path"]), row["image_rel_path"]))
        cells = sorted(cell_rows)
        if not cells:
            raise RuntimeError(f"No calibration candidates for {task_name}")
        base = per_task // len(cells)
        remainder = per_task % len(cells)
        cells_by_generator: dict[str, list[str]] = defaultdict(list)
        for cell in cells:
            cells_by_generator[cell.split("__", 1)[0]].append(cell)
        extra_cells: set[str] = set()
        max_group_len = max(len(group_cells) for group_cells in cells_by_generator.values())
        for idx in range(max_group_len):
            for generator in sorted(cells_by_generator):
                group_cells = cells_by_generator[generator]
                if idx < len(group_cells):
                    extra_cells.add(group_cells[idx])
                    if len(extra_cells) == remainder:
                        break
            if len(extra_cells) == remainder:
                break
        task_selected: list[dict[str, Any]] = []
        positions = {cell: 0 for cell in cells}

        for cell in cells:
            quota = base + int(cell in extra_cells)
            take = min(quota, len(cell_rows[cell]))
            task_selected.extend(cell_rows[cell][:take])
            positions[cell] = take

        while len(task_selected) < per_task:
            progressed = False
            for cell in cells:
                pos = positions[cell]
                if pos < len(cell_rows[cell]):
                    task_selected.append(cell_rows[cell][pos])
                    positions[cell] = pos + 1
                    progressed = True
                    if len(task_selected) == per_task:
                        break
            if not progressed:
                break

        if len(task_selected) != per_task:
            raise RuntimeError(f"Only selected {len(task_selected)}/{per_task} calibration rows for {task_name}")
        selected.extend(task_selected)
    return selected


def predict_boxes(
    model: YOLO,
    rows: list[dict[str, Any]],
    *,
    min_conf: float,
    imgsz: int,
    batch: int,
    device: str,
    chunk_size: int,
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    boxes_by_image: dict[str, list[dict[str, Any]]] = {}
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
            boxes: list[dict[str, Any]] = []
            if result.boxes is not None:
                cls_values = result.boxes.cls.detach().cpu().tolist()
                conf_values = result.boxes.conf.detach().cpu().tolist()
                xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
                for cls_value, conf_value, xyxy_value in zip(cls_values, conf_values, xyxy_values, strict=True):
                    class_id = int(cls_value)
                    boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": model.names.get(class_id, str(class_id)),
                            "confidence": float(conf_value),
                            "xyxy": [float(v) for v in xyxy_value],
                        }
                    )
            boxes_by_image[row["image_rel_path"]] = boxes
        print(f"{label}: processed {min(start + chunk_size, total)}/{total}", flush=True)
    return boxes_by_image


def count_for_task(boxes: list[dict[str, Any]], task_name: str, conf: float) -> int:
    class_id = TASK_CLASS_ID[task_name]
    return sum(1 for box in boxes if int(box["class_id"]) == class_id and float(box["confidence"]) >= conf)


def threshold_grid() -> list[float]:
    return [round(v / 100, 2) for v in range(1, 96)]


def score_counts(rows: list[dict[str, Any]], boxes_by_image: dict[str, list[dict[str, Any]]], conf_by_task: dict[str, float]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        conf = float(conf_by_task[row["task_name"]])
        boxes = boxes_by_image[row["image_rel_path"]]
        pred_count = count_for_task(boxes, row["task_name"], conf)
        target = int(row["target_count"])
        human_count = row["human_count"]
        human_valid = bool(row["human_valid"])
        human_benchmark_success = bool(row["human_benchmark_success"])
        auto_target_pass = pred_count == target
        count_correct = bool(human_valid and human_count is not None and pred_count == int(human_count))
        target_class_id = TASK_CLASS_ID[row["task_name"]]
        boxes_with_count_flag = [
            {
                **box,
                "counted_for_task": int(box["class_id"]) == target_class_id and float(box["confidence"]) >= conf,
            }
            for box in boxes
        ]
        records.append(
            {
                **row,
                "conf_threshold": conf,
                "target_class_id": target_class_id,
                "target_class_name": TASK_CLASS_NAME[row["task_name"]],
                "prediction_count": pred_count,
                "prediction_status": "ok",
                "prediction_validity": "",
                "prediction_reject": False,
                "prediction_ok": True,
                "count_correct": count_correct,
                "auto_target_pass": bool(auto_target_pass),
                "agreement_all": bool(auto_target_pass == human_benchmark_success),
                "num_detected_boxes_all_classes": len(boxes),
                "num_detected_boxes_target_class": sum(1 for box in boxes if int(box["class_id"]) == target_class_id),
                "detected_boxes_json": json.dumps(boxes_with_count_flag, ensure_ascii=False),
            }
        )
    return pd.DataFrame.from_records(records)


def calibrate_by_task(calib_rows: list[dict[str, Any]], boxes_by_image: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for task_name in sorted(TASK_CLASS_ID):
        rows = [row for row in calib_rows if row["task_name"] == task_name]
        for conf in threshold_grid():
            diffs = []
            correct = 0
            target_pass = 0
            for row in rows:
                pred = count_for_task(boxes_by_image[row["image_rel_path"]], task_name, conf)
                human_count = int(row["human_count"])
                target = int(row["target_count"])
                diff = pred - human_count
                diffs.append(diff)
                correct += int(pred == human_count)
                target_pass += int(pred == target)
            mae = sum(abs(v) for v in diffs) / len(diffs)
            bias = sum(diffs) / len(diffs)
            records.append(
                {
                    "task_name": task_name,
                    "conf_threshold": conf,
                    "n": len(rows),
                    "count_fidelity": correct / len(rows),
                    "auto_target_pass": target_pass / len(rows),
                    "mae": mae,
                    "bias": bias,
                    "abs_bias": abs(bias),
                    "distance_to_default_025": abs(conf - 0.25),
                }
            )
    sweep = pd.DataFrame(records)
    best = (
        sweep.sort_values(
            ["task_name", "count_fidelity", "mae", "abs_bias", "distance_to_default_025", "conf_threshold"],
            ascending=[True, False, True, True, True, True],
        )
        .groupby("task_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return sweep, best


def rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else math.nan


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    n = len(group)
    valid = group[group["human_valid"].astype(bool)]
    human = group["human_benchmark_success"].astype(bool)
    auto = group["auto_target_pass"].astype(bool)
    agreement = human == auto
    human_reject_n = int((~group["human_valid"].astype(bool)).sum())
    auto_pass_on_reject = group[(~group["human_valid"].astype(bool)) & group["auto_target_pass"].astype(bool)]

    tp = int((auto & human).sum())
    fp = int((auto & ~human).sum())
    fn = int((~auto & human).sum())
    tn = int((~auto & ~human).sum())

    if len(valid):
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
        "n_human_valid": len(valid),
        "n_human_reject": human_reject_n,
        "human_valid_rate": rate(len(valid), n),
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
    rows = []
    for keys, group in scored.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {key: value for key, value in zip(by, keys, strict=True)}
        record.update(summarize_group(group))
        rows.append(record)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, *, best: pd.DataFrame, summaries: dict[str, pd.DataFrame], per_task: int, model_name: str) -> None:
    def fmt_table(df: pd.DataFrame) -> str:
        display = df.copy()
        for col in [
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
            "count_fidelity",
            "auto_target_pass",
        ]:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{100 * x:.2f}%" if pd.notna(x) else "-")
        for col in [
            "score_gap_a_minus_h_pp",
            "cohen_kappa_all",
            "mae_valid",
            "bias_valid",
            "abs_bias_over_mae",
            "conf_threshold",
            "mae",
            "bias",
        ]:
            if col in display.columns:
                display[col] = display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        return display.to_markdown(index=False)

    lines = [
        f"# {model_name} Task-Calibrated Confidence Test",
        "",
        f"Calibration uses `{per_task}` human-valid non-test images per task, balanced across generator/prompt/count cells as evenly as possible.",
        "The selected threshold maximizes calibration count fidelity, then minimizes MAE, absolute bias, and distance to the default 0.25 threshold.",
        "",
        "## Best Confidence By Task",
        "",
        fmt_table(best[["task_name", "conf_threshold", "n", "count_fidelity", "auto_target_pass", "mae", "bias"]]),
        "",
        "## Test Overall",
        "",
        fmt_table(summaries["overall"]),
        "",
        "## Test By Generator",
        "",
        fmt_table(summaries["by_generator"]),
        "",
        "## Test By Task",
        "",
        fmt_table(summaries["by_task"]),
        "",
        "## Test By Generator And Task",
        "",
        fmt_table(summaries["by_generator_task"]),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate YOLO confidence per substructure task, then evaluate on the MLLM split.")
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
    parser.add_argument("--train-list", type=Path, default=Path("dataset/detector/yolo/substructure_train.txt"))
    parser.add_argument("--val-list", type=Path, default=Path("dataset/detector/yolo/substructure_val.txt"))
    parser.add_argument("--per-task-calib", type=int, default=30)
    parser.add_argument("--min-conf", type=float, default=0.001)
    parser.add_argument("--model-name", default="yolo_substructure")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_rows = load_test_manifest(args.mllm_scored_rows, repo_root)
    test_image_rel_paths = {row["image_rel_path"] for row in test_rows}
    train_val_basenames = read_used_training_basenames([args.train_list, args.val_list])
    candidates = collect_calibration_candidates(
        repo_root,
        test_image_rel_paths=test_image_rel_paths,
        train_val_basenames=train_val_basenames,
    )
    calib_rows = select_balanced_calibration(candidates, args.per_task_calib)

    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration_manifest.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "test_manifest.csv", index=False)

    model = YOLO(str(args.weights))
    calib_boxes = predict_boxes(
        model,
        calib_rows,
        min_conf=args.min_conf,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        chunk_size=args.chunk_size,
        label="calibration",
    )
    sweep, best = calibrate_by_task(calib_rows, calib_boxes)
    best_conf_by_task = {str(row.task_name): float(row.conf_threshold) for row in best.itertuples(index=False)}

    test_boxes = predict_boxes(
        model,
        test_rows,
        min_conf=args.min_conf,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        chunk_size=args.chunk_size,
        label="test",
    )

    protocols = {
        "fixed_conf_0.25": {task: 0.25 for task in TASK_CLASS_ID},
        "best_by_task_calib30": best_conf_by_task,
    }
    scored_frames = []
    for protocol, conf_by_task in protocols.items():
        scored = score_counts(test_rows, test_boxes, conf_by_task)
        scored.insert(0, "protocol", protocol)
        scored.insert(0, "model_name", args.model_name)
        scored.insert(0, "run_id", f"{args.model_name}/{protocol}")
        scored_frames.append(scored)
    scored_all = pd.concat(scored_frames, ignore_index=True)

    summary_inputs = {
        "overall": ["protocol"],
        "by_generator": ["protocol", "generator_model"],
        "by_task": ["protocol", "task_name"],
        "by_generator_task": ["protocol", "generator_model", "task_name"],
        "by_cell": ["protocol", "generator_model", "task_name", "cell_id"],
    }
    summaries = {name: build_summary(scored_all, by) for name, by in summary_inputs.items()}

    sweep.to_csv(out_dir / "calibration_sweep.csv", index=False)
    best.to_csv(out_dir / "best_conf_by_task.csv", index=False)
    scored_all.to_csv(out_dir / "test_scored_rows.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(out_dir / f"summary_{name}.csv", index=False)
    write_report(out_dir, best=best, summaries=summaries, per_task=args.per_task_calib, model_name=args.model_name)
    print(f"Wrote task-calibrated YOLO evaluation to {out_dir}")


if __name__ == "__main__":
    main()
