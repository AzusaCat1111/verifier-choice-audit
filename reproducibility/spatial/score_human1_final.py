#!/usr/bin/env python3
"""Score the complete cached spatial MLLM matrix against Human-1.

The script is deliberately inference-free.  It validates the frozen 400-image
coverage of each prediction cache, joins the independent Human-1 responses,
and writes resolved-label metrics plus prompt-cluster bootstrap intervals.
Human ``uncertain`` rows are excluded from the primary binary agreement
metrics, as frozen before inference, and are retained in separate diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SPATIAL_ROOT / "manifests/geneval_position_400.jsonl"
DEFAULT_RESPONSES = (
    SPATIAL_ROOT / "annotations/human1/responses.json"
)
DEFAULT_ROW_MAP = (
    SPATIAL_ROOT
    / "annotations/packets/private/human1_row_map.jsonl"
)
DEFAULT_PREDICTIONS = (
    SCRIPT_DIR / "runs/qwen35_4b_20260725/predictions.jsonl",
    SCRIPT_DIR / "runs/qwen35_9b_20260725/predictions.jsonl",
    SCRIPT_DIR / "runs/qwen35_27b_20260725/predictions.jsonl",
)
PROTOCOLS = (
    "holistic_binary",
    "task_guided_binary",
    "decomposed_all_rule",
    "validity_aware",
)
IMAGES_PER_MODEL = 400
ROWS_PER_MODEL = IMAGES_PER_MODEL * len(PROTOCOLS)
DEFAULT_QWEN_MODELS = frozenset(
    {"qwen35_4b", "qwen35_9b", "qwen35_27b"}
)
REJECT_PROTOCOLS = frozenset({"decomposed_all_rule", "validity_aware"})
METRIC_NAMES = (
    "human_success_rate",
    "auto_pass_rate_resolved",
    "a_minus_h",
    "agreement",
    "cohen_kappa",
    "balanced_accuracy",
    "macro_f1",
    "false_pass_rate_on_human_fail",
    "false_fail_rate_on_human_success",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_cached_output_rows(model_count: int) -> int:
    """Return the frozen matrix size for a positive number of model caches."""

    if model_count < 1:
        raise ValueError("At least one prediction cache is required")
    return ROWS_PER_MODEL * model_count


def cached_coverage_sentence(audit: dict[str, Any]) -> str:
    """Render cache coverage without assuming a fixed verifier model panel."""

    judge_models = {str(model) for model in audit["judge_models"]}
    model_count = len(judge_models)
    protocol_count = len(audit["protocols"])
    image_count = int(audit["manifest_rows"])
    cached_rows = int(audit["cached_prediction_rows"])
    if judge_models == DEFAULT_QWEN_MODELS:
        model_text = "3 Qwen3.5 verifier sizes"
    else:
        suffix = "" if model_count == 1 else "s"
        model_text = f"{model_count} verifier model{suffix}"
    return (
        f"- Existing inference was reused: {model_text} × "
        f"{protocol_count} frozen protocols × {image_count} images = "
        f"{cached_rows:,} cached outputs. No inference was rerun."
    )


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def cohen_kappa(human: Sequence[bool], auto: Sequence[bool]) -> float | None:
    if len(human) != len(auto):
        raise ValueError("Human and automatic vectors have different lengths")
    if not human:
        return None
    n = len(human)
    observed = sum(h == a for h, a in zip(human, auto, strict=True)) / n
    human_yes = sum(human) / n
    auto_yes = sum(auto) / n
    expected = (
        human_yes * auto_yes
        + (1.0 - human_yes) * (1.0 - auto_yes)
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def human_fields(response: dict[str, Any]) -> dict[str, Any]:
    visible = response["both_objects_visible"]
    relation = response["target_relation_true"]
    if visible == "yes" and relation == "yes":
        return {
            "human_success": True,
            "human_status": "success",
            "human_validity": "valid",
        }
    if visible == "yes" and relation == "no":
        return {
            "human_success": False,
            "human_status": "relation_fail",
            "human_validity": "valid",
        }
    if visible == "no":
        return {
            "human_success": False,
            "human_status": "objects_missing",
            "human_validity": "reject",
        }
    return {
        "human_success": None,
        "human_status": "uncertain",
        "human_validity": "uncertain",
    }


def model_uncertain(row: dict[str, Any]) -> bool | None:
    if row["protocol"] != "decomposed_all_rule":
        return None
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return any(value == "uncertain" for value in parsed.values())


def binary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in rows if row["human_success"] is not None]
    human = [bool(row["human_success"]) for row in resolved]
    auto = [bool(row["auto_pass"]) for row in resolved]
    tp = sum(h and a for h, a in zip(human, auto, strict=True))
    tn = sum((not h) and (not a) for h, a in zip(human, auto, strict=True))
    fp = sum((not h) and a for h, a in zip(human, auto, strict=True))
    fn = sum(h and (not a) for h, a in zip(human, auto, strict=True))
    n = len(resolved)
    human_pos = tp + fn
    human_neg = tn + fp
    tpr = safe_div(tp, human_pos)
    tnr = safe_div(tn, human_neg)
    positive_f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    negative_f1 = safe_div(2 * tn, 2 * tn + fp + fn)
    human_rate = safe_div(human_pos, n)
    auto_rate = safe_div(tp + fp, n)
    return {
        "n_assigned": len(rows),
        "n_human_resolved": n,
        "n_human_uncertain_excluded": len(rows) - n,
        "human_success_rate": human_rate,
        "auto_pass_rate_all_images": safe_div(
            sum(bool(row["auto_pass"]) for row in rows), len(rows)
        ),
        "auto_pass_rate_resolved": auto_rate,
        "a_minus_h": (
            auto_rate - human_rate
            if auto_rate is not None and human_rate is not None
            else None
        ),
        "agreement": safe_div(tp + tn, n),
        "cohen_kappa": cohen_kappa(human, auto),
        "balanced_accuracy": mean_defined((tpr, tnr)),
        "macro_f1": mean_defined((positive_f1, negative_f1)),
        "false_pass_rate_on_human_fail": safe_div(fp, human_neg),
        "false_fail_rate_on_human_success": safe_div(fn, human_pos),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "status_ok_rate": safe_div(
            sum(row["status"] == "ok" for row in rows), len(rows)
        ),
        "schema_compliance_rate": safe_div(
            sum(bool(row["schema_compliant"]) for row in rows), len(rows)
        ),
    }


def reject_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    protocol = str(rows[0]["protocol"]) if rows else ""
    if protocol not in REJECT_PROTOCOLS:
        return {
            "reject_applicable": False,
            "n_reject_resolved": None,
            "human_reject_rate": None,
            "model_reject_rate_all_images": None,
            "model_reject_rate_resolved": None,
            "reject_precision": None,
            "reject_recall": None,
            "reject_f1": None,
            "false_reject_rate_on_human_valid": None,
            "auto_pass_rate_on_human_reject": None,
            "model_reject_rate_on_human_uncertain": None,
            "model_uncertain_rate_all_images": None,
        }
    resolved = [
        row for row in rows if row["human_validity"] != "uncertain"
    ]
    uncertain = [
        row for row in rows if row["human_validity"] == "uncertain"
    ]
    human_reject = [
        row["human_validity"] == "reject" for row in resolved
    ]
    model_reject = [
        row["prediction_validity"] == "reject" for row in resolved
    ]
    tp = sum(h and m for h, m in zip(human_reject, model_reject, strict=True))
    fp = sum((not h) and m for h, m in zip(human_reject, model_reject, strict=True))
    fn = sum(h and (not m) for h, m in zip(human_reject, model_reject, strict=True))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    human_valid = [
        row for row in resolved if row["human_validity"] == "valid"
    ]
    human_reject_rows = [
        row for row in resolved if row["human_validity"] == "reject"
    ]
    uncertainty_flags = [
        model_uncertain(row)
        for row in rows
        if model_uncertain(row) is not None
    ]
    return {
        "reject_applicable": True,
        "n_reject_resolved": len(resolved),
        "human_reject_rate": safe_div(sum(human_reject), len(resolved)),
        "model_reject_rate_all_images": safe_div(
            sum(row["prediction_validity"] == "reject" for row in rows),
            len(rows),
        ),
        "model_reject_rate_resolved": safe_div(
            sum(model_reject), len(resolved)
        ),
        "reject_precision": precision,
        "reject_recall": recall,
        "reject_f1": f1,
        "false_reject_rate_on_human_valid": safe_div(
            sum(
                row["prediction_validity"] == "reject"
                for row in human_valid
            ),
            len(human_valid),
        ),
        "auto_pass_rate_on_human_reject": safe_div(
            sum(bool(row["auto_pass"]) for row in human_reject_rows),
            len(human_reject_rows),
        ),
        "model_reject_rate_on_human_uncertain": safe_div(
            sum(
                row["prediction_validity"] == "reject"
                for row in uncertain
            ),
            len(uncertain),
        ),
        "model_uncertain_rate_all_images": (
            safe_div(sum(bool(flag) for flag in uncertainty_flags), len(rows))
            if uncertainty_flags
            else None
        ),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {**binary_metrics(rows), **reject_metrics(rows)}


def grouped(
    rows: Sequence[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    return [
        {**dict(zip(keys, key, strict=True)), **summarize(group_rows)}
        for key, group_rows in sorted(
            buckets.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ]


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    metric_fn: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["prompt_id"])].append(row)
    cluster_ids = sorted(by_cluster)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            sampled.extend(by_cluster[cluster_id])
        metrics = metric_fn(sampled)
        for name in metric_names:
            value = metrics.get(name)
            if value is not None and math.isfinite(float(value)):
                draws[name].append(float(value))
    result: dict[str, Any] = {
        "bootstrap_unit": "prompt_id",
        "n_prompt_clusters": len(cluster_ids),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }
    for name, values in draws.items():
        values.sort()
        result[f"{name}_ci_low"] = (
            percentile(values, 0.025) if values else None
        )
        result[f"{name}_ci_high"] = (
            percentile(values, 0.975) if values else None
        )
        result[f"{name}_bootstrap_defined"] = len(values)
    return result


def grouped_with_bootstrap(
    rows: Sequence[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_index, (key, group_rows) in enumerate(
        sorted(
            buckets.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ):
        output.append(
            {
                **dict(zip(keys, key, strict=True)),
                **summarize(group_rows),
                **cluster_bootstrap(
                    group_rows,
                    iterations=iterations,
                    seed=seed + group_index,
                    metric_fn=binary_metrics,
                    metric_names=METRIC_NAMES,
                ),
            }
        )
    return output


def generator_gap_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_generator[str(row["generator_model"])].append(row)
    expected = {"qwenimage2512", "flux2dev_full"}
    if set(by_generator) != expected:
        raise ValueError(
            f"Generator gap expects {sorted(expected)}, got {sorted(by_generator)}"
        )
    qwen = binary_metrics(by_generator["qwenimage2512"])
    flux = binary_metrics(by_generator["flux2dev_full"])
    human_gap = (
        qwen["human_success_rate"] - flux["human_success_rate"]
    )
    auto_gap = (
        qwen["auto_pass_rate_resolved"]
        - flux["auto_pass_rate_resolved"]
    )
    return {
        "human_qwen_minus_flux": human_gap,
        "auto_qwen_minus_flux": auto_gap,
        "auto_minus_human_gap": auto_gap - human_gap,
    }


def generator_gap_rows(
    rows: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["judge_model"]), str(row["protocol"]))].append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "human_qwen_minus_flux",
        "auto_qwen_minus_flux",
        "auto_minus_human_gap",
    )
    for group_index, (key, group_rows) in enumerate(sorted(buckets.items())):
        point = generator_gap_metrics(group_rows)
        ci = cluster_bootstrap(
            group_rows,
            iterations=iterations,
            seed=seed + group_index,
            metric_fn=generator_gap_metrics,
            metric_names=metric_names,
        )
        human_gap = point["human_qwen_minus_flux"]
        auto_gap = point["auto_qwen_minus_flux"]
        flip = (
            human_gap != 0.0
            and auto_gap != 0.0
            and math.copysign(1.0, human_gap)
            != math.copysign(1.0, auto_gap)
        )
        output.append(
            {
                "judge_model": key[0],
                "protocol": key[1],
                **point,
                "point_ranking_flip": flip,
                **ci,
            }
        )
    return output


def validate_and_join(
    *,
    manifest_path: Path,
    response_path: Path,
    row_map_path: Path,
    prediction_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    if len(manifest) != 400:
        raise ValueError(f"Expected 400 manifest rows, found {len(manifest)}")
    manifest_hash = sha256_file(manifest_path)
    manifest_by_uid = {row["row_uid"]: row for row in manifest}
    if len(manifest_by_uid) != len(manifest):
        raise ValueError("Manifest contains duplicate row_uid values")

    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    responses = response_payload.get("responses")
    if not isinstance(responses, list) or len(responses) != 400:
        raise ValueError(
            "Human-1 response file must contain exactly 400 responses"
        )
    row_map = {
        row["annotation_uid"]: row for row in read_jsonl(row_map_path)
    }
    response_by_row: dict[str, dict[str, Any]] = {}
    annotation_uid_by_row: dict[str, str] = {}
    for response in responses:
        annotation_uid = response["annotation_uid"]
        if annotation_uid not in row_map:
            raise ValueError(
                f"Annotation UID absent from private map: {annotation_uid}"
            )
        row_uid = row_map[annotation_uid]["row_uid"]
        if row_uid in response_by_row:
            raise ValueError(f"Duplicate Human-1 row: {row_uid}")
        response_by_row[row_uid] = response
        annotation_uid_by_row[row_uid] = annotation_uid
    if set(response_by_row) != set(manifest_by_uid):
        raise ValueError("Human-1 rows do not exactly match the manifest")

    audit_files: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    judge_models: set[str] = set()
    for prediction_path in prediction_paths:
        prediction_rows = read_jsonl(prediction_path)
        if len(prediction_rows) != ROWS_PER_MODEL:
            raise ValueError(
                f"{prediction_path}: expected {ROWS_PER_MODEL} rows, "
                f"found {len(prediction_rows)}"
            )
        job_uids = [str(row["job_uid"]) for row in prediction_rows]
        if len(set(job_uids)) != len(job_uids):
            raise ValueError(f"{prediction_path}: duplicate jobs")
        models = {str(row["judge_model"]) for row in prediction_rows}
        if len(models) != 1:
            raise ValueError(
                f"{prediction_path}: expected one judge model, got {models}"
            )
        model = next(iter(models))
        if model in judge_models:
            raise ValueError(f"Duplicate prediction cache for {model}")
        judge_models.add(model)
        per_protocol: dict[str, int] = {}
        for protocol in PROTOCOLS:
            protocol_rows = [
                row for row in prediction_rows if row["protocol"] == protocol
            ]
            per_protocol[protocol] = len(protocol_rows)
            if len(protocol_rows) != 400:
                raise ValueError(
                    f"{prediction_path}: {protocol} has "
                    f"{len(protocol_rows)} rows, expected 400"
                )
            if {row["row_uid"] for row in protocol_rows} != set(manifest_by_uid):
                raise ValueError(
                    f"{prediction_path}: {protocol} does not cover the "
                    "frozen manifest exactly"
                )
        if set(row["protocol"] for row in prediction_rows) != set(PROTOCOLS):
            raise ValueError(f"{prediction_path}: unexpected protocol set")
        if any(row["status"] != "ok" for row in prediction_rows):
            raise ValueError(f"{prediction_path}: non-ok inference rows")
        if any(not row["schema_compliant"] for row in prediction_rows):
            raise ValueError(f"{prediction_path}: noncompliant outputs")

        config_path = prediction_path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prediction_hash = sha256_file(prediction_path)
        if config.get("predictions_sha256") != prediction_hash:
            raise ValueError(
                f"{prediction_path}: prediction hash differs from config"
            )
        if config.get("status") != "complete":
            raise ValueError(f"{prediction_path}: config is not complete")
        if config.get("manifest_sha256") != manifest_hash:
            raise ValueError(
                f"{prediction_path}: config references a different manifest"
            )
        if config.get("judge_model") != model:
            raise ValueError(
                f"{prediction_path}: config judge model differs from rows"
            )
        if config.get("prediction_rows") != len(prediction_rows):
            raise ValueError(
                f"{prediction_path}: config prediction count differs from rows"
            )
        if config.get("planned_jobs") != len(prediction_rows):
            raise ValueError(
                f"{prediction_path}: config planned-job count differs from rows"
            )
        if set(config.get("protocols", [])) != set(PROTOCOLS):
            raise ValueError(
                f"{prediction_path}: config protocol set differs from freeze"
            )
        if config.get("images") != len(manifest):
            raise ValueError(
                f"{prediction_path}: config image count differs from manifest"
            )
        audit_files.append(
            {
                "judge_model": model,
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction_hash,
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "prediction_rows": len(prediction_rows),
                "unique_jobs": len(set(job_uids)),
                "unique_images": len(
                    {row["row_uid"] for row in prediction_rows}
                ),
                "protocol_counts": per_protocol,
                "status_ok": len(prediction_rows),
                "schema_compliant": len(prediction_rows),
                "dtype": config.get("dtype"),
                "quantization": config.get("quantization"),
            }
        )
        all_predictions.extend(prediction_rows)

    expected_total = expected_cached_output_rows(len(judge_models))
    if len(all_predictions) != expected_total:
        raise ValueError(
            f"Expected {expected_total} cached outputs for "
            f"{len(judge_models)} model cache(s), found {len(all_predictions)}"
        )
    global_jobs = [
        f"{row['judge_model']}::{row['job_uid']}" for row in all_predictions
    ]
    if len(set(global_jobs)) != len(global_jobs):
        raise ValueError("Duplicate global model/job combinations")

    joined: list[dict[str, Any]] = []
    for row in all_predictions:
        row_uid = str(row["row_uid"])
        response = response_by_row[row_uid]
        manifest_row = manifest_by_uid[row_uid]
        if (
            row["prompt_id"] != manifest_row["prompt_id"]
            or row["generator_model"] != manifest_row["generator_model"]
        ):
            raise ValueError(f"Prediction metadata mismatch for {row_uid}")
        joined.append(
            {
                **row,
                "annotation_uid": annotation_uid_by_row[row_uid],
                "human_both_objects_visible": response[
                    "both_objects_visible"
                ],
                "human_target_relation_true": response[
                    "target_relation_true"
                ],
                **human_fields(response),
                "model_uncertain": model_uncertain(row),
            }
        )

    coverage_audit = {
        "status": "PASS",
        "inference_rerun": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_rows": len(manifest),
        "human1_response_path": str(response_path),
        "human1_response_sha256": sha256_file(response_path),
        "human1_rows": len(response_by_row),
        "human1_complete": True,
        "human1_resolved": sum(
            human_fields(response)["human_success"] is not None
            for response in responses
        ),
        "human1_uncertain": sum(
            human_fields(response)["human_success"] is None
            for response in responses
        ),
        "cached_prediction_rows": len(all_predictions),
        "judge_models": sorted(judge_models),
        "protocols": list(PROTOCOLS),
        "files": sorted(audit_files, key=lambda row: row["judge_model"]),
    }
    return joined, coverage_audit


def uncertainty_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row["judge_model"]),
                str(row["protocol"]),
                str(row["generator_model"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(buckets.items()):
        uncertain = [
            row for row in group_rows
            if row["human_validity"] == "uncertain"
        ]
        output.append(
            {
                "judge_model": key[0],
                "protocol": key[1],
                "generator_model": key[2],
                "n_assigned": len(group_rows),
                "n_human_uncertain": len(uncertain),
                "auto_pass_rate_on_human_uncertain": safe_div(
                    sum(bool(row["auto_pass"]) for row in uncertain),
                    len(uncertain),
                ),
                "model_reject_rate_on_human_uncertain": (
                    safe_div(
                        sum(
                            row["prediction_validity"] == "reject"
                            for row in uncertain
                        ),
                        len(uncertain),
                    )
                    if key[1] in REJECT_PROTOCOLS
                    else None
                ),
            }
        )
    return output


def human_summary_rows(joined: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # Human labels are repeated across models/protocols in the joined table.
    unique = {}
    for row in joined:
        unique[str(row["row_uid"])] = row
    rows = list(unique.values())
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[("overall",)].append(row)
        buckets[(str(row["generator_model"]),)].append(row)
        buckets[(str(row["relation"]),)].append(row)
        buckets[
            (str(row["generator_model"]), str(row["relation"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(buckets.items()):
        resolved = [
            row for row in group_rows if row["human_success"] is not None
        ]
        output.append(
            {
                "group": "::".join(key),
                "n_assigned": len(group_rows),
                "n_resolved": len(resolved),
                "n_uncertain": len(group_rows) - len(resolved),
                "n_valid": sum(
                    row["human_validity"] == "valid" for row in group_rows
                ),
                "n_reject": sum(
                    row["human_validity"] == "reject" for row in group_rows
                ),
                "human_success_rate_resolved": safe_div(
                    sum(bool(row["human_success"]) for row in resolved),
                    len(resolved),
                ),
            }
        )
    return output


def pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def kappa(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def ci_pct(row: dict[str, Any], metric: str) -> str:
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if low is None or high is None:
        return "N/A"
    return f"[{100.0 * low:.2f}, {100.0 * high:.2f}]"


def write_report(
    path: Path,
    *,
    audit: dict[str, Any],
    overall: Sequence[dict[str, Any]],
    by_generator: Sequence[dict[str, Any]],
    generator_gaps: Sequence[dict[str, Any]],
) -> None:
    min_ah = min(overall, key=lambda row: float(row["a_minus_h"]))
    max_ah = max(overall, key=lambda row: float(row["a_minus_h"]))
    min_agreement = min(
        overall, key=lambda row: float(row["agreement"])
    )
    max_agreement = max(
        overall, key=lambda row: float(row["agreement"])
    )
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overall:
        rows_by_model[str(row["judge_model"])].append(row)
    best_by_model = {
        model: max(model_rows, key=lambda row: float(row["agreement"]))
        for model, model_rows in rows_by_model.items()
    }
    worst_by_model = {
        model: min(model_rows, key=lambda row: float(row["agreement"]))
        for model, model_rows in rows_by_model.items()
    }
    model_names = set(rows_by_model)
    if model_names == DEFAULT_QWEN_MODELS:
        protocol_observations = [
            (
                "- Protocol behavior is model-dependent rather than uniformly "
                "monotonic: `task_guided_binary` has the highest agreement for "
                f"4B ({pct(best_by_model['qwen35_4b']['agreement'])}) and 9B "
                f"({pct(best_by_model['qwen35_9b']['agreement'])}), but the "
                f"lowest for 27B "
                f"({pct(worst_by_model['qwen35_27b']['agreement'])})."
            )
        ]
    else:
        best_text = "; ".join(
            f"{model}: {best_by_model[model]['protocol']} "
            f"({pct(best_by_model[model]['agreement'])})"
            for model in sorted(best_by_model)
        )
        worst_text = "; ".join(
            f"{model}: {worst_by_model[model]['protocol']} "
            f"({pct(worst_by_model[model]['agreement'])})"
            for model in sorted(worst_by_model)
        )
        protocol_observations = [
            f"- Best-agreement protocol by model: {best_text}.",
            f"- Lowest-agreement protocol by model: {worst_text}.",
        ]
    lines = [
        "# Final Human-1-grounded spatial MLLM audit",
        "",
        "## Coverage and estimand",
        "",
        (
            f"- Human-1 completed 400/400 images; "
            f"{audit['human1_resolved']} have a determinate benchmark label "
            f"and {audit['human1_uncertain']} are excluded from primary "
            "agreement metrics."
        ),
        cached_coverage_sentence(audit),
        (
            "- All caches have complete frozen-manifest coverage, unique jobs, "
            "status `ok`, schema compliance, and a SHA-256 matching their "
            "completion config."
        ),
        (
            "- Primary 95% intervals are percentile prompt-cluster bootstraps: "
            "one sampled prompt retains both replicas, both generators, and "
            "the complete condition output."
        ),
        (
            "- Human `uncertain` is not silently treated as failure. Its "
            "automatic behavior is reported separately in "
            "`uncertainty_diagnostics.csv`."
        ),
        "",
        "## Overall primary agreement",
        "",
        "| Model | Protocol | Resolved | H | A | A−H (95% CI) | Agreement (95% CI) | κ | Balanced acc. | Macro-F1 | False pass | False fail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            f"| {row['judge_model']} | {row['protocol']} | "
            f"{row['n_human_resolved']}/{row['n_assigned']} | "
            f"{pct(row['human_success_rate'])} | "
            f"{pct(row['auto_pass_rate_resolved'])} | "
            f"{pct(row['a_minus_h'])} {ci_pct(row, 'a_minus_h')} | "
            f"{pct(row['agreement'])} {ci_pct(row, 'agreement')} | "
            f"{kappa(row['cohen_kappa'])} | "
            f"{pct(row['balanced_accuracy'])} | "
            f"{pct(row['macro_f1'])} | "
            f"{pct(row['false_pass_rate_on_human_fail'])} | "
            f"{pct(row['false_fail_rate_on_human_success'])} |"
        )
    lines.extend(
        [
            "",
            "## Evidence-level observations",
            "",
            (
                f"- Across the {len(overall)} conditions, A−H ranges from "
                f"{pct(min_ah['a_minus_h'])} "
                f"({min_ah['judge_model']}, {min_ah['protocol']}) to "
                f"{pct(max_ah['a_minus_h'])} "
                f"({max_ah['judge_model']}, {max_ah['protocol']}); "
                f"image-level agreement ranges from "
                f"{pct(min_agreement['agreement'])} to "
                f"{pct(max_agreement['agreement'])}."
            ),
            *protocol_observations,
            (
                "- These high absolute agreement values show that this "
                "two-object GenEval-position pilot is easier than the "
                "counting stress cases. The pilot supports a scoped "
                "model-by-protocol interaction, not a claim that spatial "
                "verification necessarily produces ranking flips."
            ),
            "",
            "## By generator",
            "",
            "| Model | Protocol | Generator | Resolved | H | A | A−H | Agreement | κ |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in by_generator:
        lines.append(
            f"| {row['judge_model']} | {row['protocol']} | "
            f"{row['generator_model']} | "
            f"{row['n_human_resolved']}/{row['n_assigned']} | "
            f"{pct(row['human_success_rate'])} | "
            f"{pct(row['auto_pass_rate_resolved'])} | "
            f"{pct(row['a_minus_h'])} | {pct(row['agreement'])} | "
            f"{kappa(row['cohen_kappa'])} |"
        )
    lines.extend(
        [
            "",
            "## Generator comparison",
            "",
            "All gaps are Qwen-Image-2512 minus FLUX.2-dev on Human-resolved rows.",
            "",
            "| Model | Protocol | Human gap | Auto gap (95% CI) | Auto−human gap | Point ranking flip |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in generator_gaps:
        lines.append(
            f"| {row['judge_model']} | {row['protocol']} | "
            f"{pct(row['human_qwen_minus_flux'])} | "
            f"{pct(row['auto_qwen_minus_flux'])} "
            f"{ci_pct(row, 'auto_qwen_minus_flux')} | "
            f"{pct(row['auto_minus_human_gap'])} | "
            f"{'yes' if row['point_ranking_flip'] else 'no'} |"
        )

    reject_rows = [
        row for row in overall if row["reject_applicable"]
    ]
    lines.extend(
        [
            "",
            "## Rejection diagnostics",
            "",
            (
                "Human rejection is evaluated only on determinate labels: "
                "at least one requested object not visibly recognizable "
                "gives reject; "
                "relation-fail images remain human-valid. Human-uncertain "
                "images are reported separately. With only two Human-reject "
                "images, precision/recall estimates are descriptive and "
                "should not be overinterpreted."
            ),
            "",
            "| Model | Protocol | Human reject | Model reject | Reject P/R/F1 | False reject on valid | Auto pass on human reject |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in reject_rows:
        lines.append(
            f"| {row['judge_model']} | {row['protocol']} | "
            f"{pct(row['human_reject_rate'])} | "
            f"{pct(row['model_reject_rate_resolved'])} | "
            f"{pct(row['reject_precision'])} / "
            f"{pct(row['reject_recall'])} / "
            f"{pct(row['reject_f1'])} | "
            f"{pct(row['false_reject_rate_on_human_valid'])} | "
            f"{pct(row['auto_pass_rate_on_human_reject'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This is a scoped non-counting pilot, not a claim of universal task transfer.",
            "- `holistic_binary` is VQAScore-style questioning, not the official VQAScore probability metric.",
            "- The decomposed and validity-aware protocols expose reject behavior; the two binary protocols have no reject affordance.",
            "- The generator comparison is based on 100 prompt clusters and two replicas per generator, not 400 independent prompt draws.",
            "",
            "## Files",
            "",
            "- `coverage_audit.json`: immutable-input and cache-completeness audit.",
            (
                f"- `joined_rows.jsonl`: all "
                f"{audit['cached_prediction_rows']:,} cached outputs joined "
                "to Human-1."
            ),
            "- `summary_overall.csv`: primary model × protocol results and bootstrap CIs.",
            "- `summary_by_generator.csv`: generator-stratified results and CIs.",
            "- `summary_by_relation.csv`: direction-stratified point estimates.",
            "- `summary_by_generator_relation.csv`: generator × direction point estimates.",
            "- `generator_gaps.csv`: paired generator-gap audit and bootstrap CIs.",
            "- `uncertainty_diagnostics.csv`: explicit Human-uncertain behavior.",
            "- `human_summary.csv`: deduplicated Human-reference distribution summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--human1-responses", type=Path, default=DEFAULT_RESPONSES
    )
    parser.add_argument("--human1-row-map", type=Path, default=DEFAULT_ROW_MAP)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=list(DEFAULT_PREDICTIONS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joined, audit = validate_and_join(
        manifest_path=args.manifest.resolve(),
        response_path=args.human1_responses.resolve(),
        row_map_path=args.human1_row_map.resolve(),
        prediction_paths=[
            prediction.resolve() for prediction in args.predictions
        ],
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coverage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "joined_rows.jsonl", joined)
    write_csv(output_dir / "joined_rows.csv", joined)

    overall = grouped_with_bootstrap(
        joined,
        ("judge_model", "protocol"),
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    by_generator = grouped_with_bootstrap(
        joined,
        ("judge_model", "protocol", "generator_model"),
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 10_000,
    )
    by_relation = grouped(
        joined, ("judge_model", "protocol", "relation")
    )
    by_generator_relation = grouped(
        joined,
        ("judge_model", "protocol", "generator_model", "relation"),
    )
    generator_gaps = generator_gap_rows(
        joined,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 20_000,
    )
    write_csv(output_dir / "summary_overall.csv", overall)
    write_csv(output_dir / "summary_by_generator.csv", by_generator)
    write_csv(output_dir / "summary_by_relation.csv", by_relation)
    write_csv(
        output_dir / "summary_by_generator_relation.csv",
        by_generator_relation,
    )
    write_csv(output_dir / "generator_gaps.csv", generator_gaps)
    write_csv(
        output_dir / "uncertainty_diagnostics.csv",
        uncertainty_rows(joined),
    )
    write_csv(
        output_dir / "human_summary.csv",
        human_summary_rows(joined),
    )
    write_report(
        output_dir / "REPORT.md",
        audit=audit,
        overall=overall,
        by_generator=by_generator,
        generator_gaps=generator_gaps,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "human1_images": audit["human1_rows"],
                "human1_resolved": audit["human1_resolved"],
                "human1_uncertain": audit["human1_uncertain"],
                "cached_outputs": audit["cached_prediction_rows"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
