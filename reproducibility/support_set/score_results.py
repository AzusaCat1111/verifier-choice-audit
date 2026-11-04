#!/usr/bin/env python3
"""Score the frozen few-shot ablation with paired clustered bootstrap CIs.

The scorer is deliberately strict:

* every prediction is joined by ``condition_id`` and ``query_id``;
* query metadata and the ordered support identities are checked against the
  frozen design;
* duplicate, unknown, dry-run, and (by default) missing rows are fatal;
* non-compliant outputs remain in every denominator and count as automatic
  failures;
* all conditions and models use the same bootstrap draw for a query universe.

No verifier inference is performed by this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_DESIGN_DIR = HERE / "design"
DEFAULT_OUTPUT_DIR = HERE / "analysis"
DEFAULT_BOOTSTRAPS = 1000
DEFAULT_SEED = 4432509

POINT_METRICS = (
    "compliance_rate",
    "numeric_coverage_human_valid",
    "count_fidelity_human_valid",
    "mae_human_valid",
    "signed_bias_human_valid",
    "first_support_label_match_rate",
    "first_support_label_match_given_numeric",
    "first_support_conflict_anchor_rate",
    "first_support_match_among_wrong_numeric",
    "last_support_label_match_rate",
    "last_support_label_match_given_numeric",
    "A",
    "H",
    "A_minus_H",
    "agreement",
    "kappa",
    "model_reject_rate",
    "reject_precision",
    "reject_recall",
    "reject_f1",
    "false_reject_rate_human_valid",
    "auto_pass_on_human_reject",
)

DELTA_METRICS = POINT_METRICS
PAIR_ONLY_METRICS = (
    "prediction_changed_rate",
    "auto_pass_changed_rate",
    "first_label_changed_rate",
    "prediction_changed_given_first_label_changed",
    "switched_old_to_new_first_given_changed",
    "moved_to_new_first_given_changed",
)

METRIC_DEFINITIONS = {
    "compliance_rate": "Output satisfies the condition schema; rate over all queries.",
    "numeric_coverage_human_valid": (
        "Human-valid queries with a usable protocol-valid numeric prediction, "
        "divided by all human-valid queries."
    ),
    "count_fidelity_human_valid": (
        "Protocol-valid prediction_count equals human_count, with every "
        "human-valid query in the denominator."
    ),
    "mae_human_valid": (
        "Mean absolute prediction_count-human_count error over human-valid "
        "queries with a usable protocol-valid numeric prediction."
    ),
    "signed_bias_human_valid": (
        "Mean prediction_count-human_count over human-valid queries with a "
        "usable protocol-valid numeric prediction."
    ),
    "first_support_label_match_rate": (
        "Usable prediction_count equals the first support count label, with "
        "all label-evaluable queries in the denominator (Panels A/C)."
    ),
    "first_support_label_match_given_numeric": (
        "Prediction_count equals the first support count label, conditional "
        "on a usable numeric prediction (Panels A/C)."
    ),
    "first_support_conflict_anchor_rate": (
        "Among human-valid usable-numeric rows where human_count differs from "
        "the first support label, prediction_count equals that conflicting "
        "first support label (Panels A/C)."
    ),
    "first_support_match_among_wrong_numeric": (
        "Among human-valid usable numeric predictions that disagree with the "
        "human count, prediction_count equals the first support label "
        "(Panels A/C)."
    ),
    "last_support_label_match_rate": (
        "Usable prediction_count equals the last support count label, with "
        "all label-evaluable queries in the denominator (Panels A/C)."
    ),
    "last_support_label_match_given_numeric": (
        "Prediction_count equals the last support count label, conditional "
        "on a usable numeric prediction (Panels A/C)."
    ),
    "A": "Automatic benchmark pass rate over all queries.",
    "H": "Human benchmark pass rate over all queries.",
    "A_minus_H": "A-H in rate units; multiply by 100 for percentage points.",
    "agreement": "Image-level agreement between automatic and human benchmark pass.",
    "kappa": "Cohen's kappa between automatic and human benchmark pass.",
    "model_reject_rate": "Model reject rate over all queries (Panel B).",
    "reject_precision": "Human reject precision of model reject (Panel B).",
    "reject_recall": "Human reject recall of model reject (Panel B).",
    "reject_f1": "F1 for human reject versus model reject (Panel B).",
    "false_reject_rate_human_valid": (
        "Model reject rate among human-valid queries (Panel B)."
    ),
    "auto_pass_on_human_reject": (
        "Automatic benchmark pass rate among human-rejected queries (Panel B)."
    ),
    "prediction_changed_rate": (
        "Paired queries whose compliant/schema-normalized prediction tuple changed."
    ),
    "auto_pass_changed_rate": (
        "Paired queries whose automatic benchmark pass decision changed."
    ),
    "first_label_changed_rate": (
        "Paired queries whose first support count label changed."
    ),
    "prediction_changed_given_first_label_changed": (
        "Prediction tuple changed among pairs whose first support label changed."
    ),
    "switched_old_to_new_first_given_changed": (
        "Among pairs whose first label changed, the forward prediction equals "
        "the old first label and the reverse prediction equals the new first label."
    ),
    "moved_to_new_first_given_changed": (
        "Among pairs whose first label changed, the reverse prediction equals "
        "the new first label while the forward prediction did not."
    ),
}


class ScoringError(RuntimeError):
    """Raised for design or prediction-integrity failures."""


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ScoringError(
                    f"{path}:{line_number}: JSONL row is not an object"
                )
            payload["_source_file"] = path.as_posix()
            payload["_source_line"] = line_number
            rows.append(payload)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strict_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("+", "-")):
            digits = stripped[1:]
        else:
            digits = stripped
        if digits.isdigit() and digits:
            return int(stripped)
    return None


def strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def normalize_prediction_validity(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized if normalized in {"valid", "reject"} else ""


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: Iterable[float | int | bool]) -> float | None:
    materialized = [float(value) for value in values]
    return (
        math.fsum(materialized) / len(materialized)
        if materialized
        else None
    )


def strict_mean(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return math.fsum(float(value) for value in values) / len(values)


def cohen_kappa(reference: Sequence[bool], prediction: Sequence[bool]) -> float | None:
    if len(reference) != len(prediction):
        raise ScoringError("Kappa inputs have different lengths")
    if not reference:
        return None
    observed = mean(a == b for a, b in zip(reference, prediction, strict=True))
    reference_yes = mean(reference)
    prediction_yes = mean(prediction)
    assert observed is not None
    assert reference_yes is not None
    assert prediction_yes is not None
    expected = (
        reference_yes * prediction_yes
        + (1.0 - reference_yes) * (1.0 - prediction_yes)
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def percentile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def load_design(design_dir: Path) -> dict[str, Any]:
    required = (
        "queries.jsonl",
        "conditions.json",
        "support_assignments.json",
        "panel_b_query_ids.json",
        "design_summary.json",
        "FROZEN_SHA256.json",
    )
    for name in required:
        if not (design_dir / name).is_file():
            raise ScoringError(f"Missing frozen design file: {design_dir / name}")

    fingerprints = read_json(design_dir / "FROZEN_SHA256.json")
    expected_hashes = fingerprints.get("files", {})
    hash_errors: list[str] = []
    for name, expected in expected_hashes.items():
        path = design_dir / name
        if not path.is_file():
            hash_errors.append(f"{name}: missing")
            continue
        observed = sha256(path)
        if observed != expected:
            hash_errors.append(
                f"{name}: expected {expected}, observed {observed}"
            )
    if hash_errors:
        raise ScoringError(
            "Frozen design hash validation failed:\n  "
            + "\n  ".join(hash_errors)
        )

    queries = read_jsonl(design_dir / "queries.jsonl")
    # read_jsonl adds provenance keys; these are harmless but not design data.
    for query in queries:
        query.pop("_source_file", None)
        query.pop("_source_line", None)
    conditions = read_json(design_dir / "conditions.json")
    assignments = read_json(design_dir / "support_assignments.json")
    panel_b_query_ids = set(read_json(design_dir / "panel_b_query_ids.json"))
    summary = read_json(design_dir / "design_summary.json")

    query_by_id = {str(row["query_id"]): row for row in queries}
    condition_by_id = {str(row["condition_id"]): row for row in conditions}
    if len(query_by_id) != len(queries):
        raise ScoringError("Frozen design has duplicate query_id values")
    if len(condition_by_id) != len(conditions):
        raise ScoringError("Frozen design has duplicate condition_id values")
    if not panel_b_query_ids.issubset(query_by_id):
        raise ScoringError("Panel-B query list contains unknown query IDs")
    if set(assignments) != set(condition_by_id):
        raise ScoringError(
            "support_assignments and conditions have different condition IDs"
        )

    expected_queries: dict[str, set[str]] = {}
    for condition_id, condition in condition_by_id.items():
        scope = condition["query_scope"]
        if scope == "all_queries":
            ids = set(query_by_id)
        elif scope == "panel_b_intersection":
            ids = set(panel_b_query_ids)
        else:
            raise ScoringError(
                f"{condition_id}: unknown query scope {scope!r}"
            )
        if len(ids) != int(condition["planned_queries"]):
            raise ScoringError(
                f"{condition_id}: planned_queries does not match its scope"
            )
        expected_queries[condition_id] = ids

    return {
        "queries": queries,
        "query_by_id": query_by_id,
        "conditions": conditions,
        "condition_by_id": condition_by_id,
        "assignments": assignments,
        "panel_b_query_ids": panel_b_query_ids,
        "summary": summary,
        "expected_queries": expected_queries,
        "fingerprints": fingerprints,
    }


def prediction_files(run_dirs: Sequence[Path]) -> list[tuple[Path, Path]]:
    discovered: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise ScoringError(f"Run directory does not exist: {run_dir}")
        paths = (
            [run_dir]
            if run_dir.is_file() and run_dir.name == "predictions.jsonl"
            else sorted(run_dir.rglob("predictions.jsonl"))
        )
        if not paths:
            raise ScoringError(f"No predictions.jsonl under {run_dir}")
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append((run_dir, path))
    return discovered


def infer_condition_id(
    row: dict[str, Any],
    path: Path,
    known_conditions: set[str],
) -> str:
    explicit = str(row.get("condition_id") or "")
    if explicit:
        return explicit
    matches = [part for part in path.parts if part in known_conditions]
    if len(matches) == 1:
        return matches[0]
    raise ScoringError(
        f"{path}:{row['_source_line']}: missing unambiguous condition_id"
    )


def infer_model_name(row: dict[str, Any], run_root: Path) -> str:
    for key in ("model_name", "judge_model", "api_model"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    # A separately supplied model run directory is a safe final fallback.
    value = run_root.name.strip()
    if not value:
        raise ScoringError(
            f"{row['_source_file']}:{row['_source_line']}: "
            "cannot infer model name"
        )
    return value


def expected_support_ids(
    condition_id: str,
    query: dict[str, Any],
    assignments: dict[str, Any],
) -> list[str]:
    bank_id = str(query["bank_id"])
    banks = assignments[condition_id]
    if bank_id not in banks:
        raise ScoringError(
            f"{condition_id}/{query['query_id']}: bank {bank_id} "
            "is absent from frozen support assignments"
        )
    return [str(item["image_id"]) for item in banks[bank_id]]


def expected_support_examples(
    condition_id: str,
    query: dict[str, Any],
    assignments: dict[str, Any],
) -> list[dict[str, Any]]:
    bank_id = str(query["bank_id"])
    banks = assignments[condition_id]
    if bank_id not in banks:
        raise ScoringError(
            f"{condition_id}/{query['query_id']}: bank {bank_id} "
            "is absent from frozen support assignments"
        )
    return list(banks[bank_id])


def extract_support_ids(row: dict[str, Any]) -> list[str] | None:
    raw = row.get("support_image_ids")
    if raw is None:
        raw = row.get("support_ids")
    if raw is None and isinstance(row.get("fewshot_examples"), list):
        raw = [
            item.get("image_id")
            for item in row["fewshot_examples"]
            if isinstance(item, dict)
        ]
    if raw is None:
        return None
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw]


def validate_prediction_metadata(
    row: dict[str, Any],
    query: dict[str, Any],
    condition: dict[str, Any],
    assignments: dict[str, Any],
) -> None:
    location = f"{row['_source_file']}:{row['_source_line']}"
    checks = {
        "task_name": query["task_name"],
        "prompt_id": query["prompt_id"],
        "cell_id": query["cell_id"],
        "bank_id": query["bank_id"],
        "target_count": query["target_count"],
        "human_validity": query["human_validity"],
        "human_count": query["human_count"],
        "human_success": query["human_success"],
        "panel": condition["panel"],
        "schema": condition["schema"],
    }
    for key, expected in checks.items():
        if key not in row:
            continue
        observed = row[key]
        if key in {"target_count", "human_count"}:
            observed = strict_int(observed)
        elif key == "human_success":
            observed_bool = strict_bool(observed)
            observed = (
                int(observed_bool) if observed_bool is not None else observed
            )
        if observed != expected:
            raise ScoringError(
                f"{location}: {key} mismatch for {query['query_id']} / "
                f"{condition['condition_id']}: expected {expected!r}, "
                f"observed {observed!r}"
            )

    observed_support = extract_support_ids(row)
    if observed_support is None:
        raise ScoringError(
            f"{location}: missing ordered support_image_ids provenance"
        )
    expected_support = expected_support_ids(
        str(condition["condition_id"]), query, assignments
    )
    if observed_support != expected_support:
        raise ScoringError(
            f"{location}: support identity/order mismatch for "
            f"{condition['condition_id']} / {query['query_id']}; "
            f"expected {expected_support}, observed {observed_support}"
        )


def independently_compliant(
    row: dict[str, Any],
    schema: str,
) -> tuple[bool, int | None, str]:
    status = str(row.get("status", row.get("prediction_status", ""))).lower()
    prediction_count = strict_int(
        row.get("prediction_count", row.get("count"))
    )
    prediction_validity = normalize_prediction_validity(
        row.get("prediction_validity", row.get("validity"))
    )
    if status != "ok":
        return False, prediction_count, prediction_validity
    if schema == "count":
        derived = prediction_count is not None
    elif schema == "validity":
        derived = (
            prediction_validity == "valid" and prediction_count is not None
        ) or (
            prediction_validity == "reject" and prediction_count is None
        )
    else:
        raise ScoringError(f"Unknown prediction schema: {schema}")

    explicit = strict_bool(row.get("schema_compliant"))
    compliant = derived if explicit is None else bool(derived and explicit)
    return compliant, prediction_count, prediction_validity


def score_prediction(
    raw: dict[str, Any],
    model_name: str,
    query: dict[str, Any],
    condition: dict[str, Any],
    support_examples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    schema = str(condition["schema"])
    compliant, prediction_count, prediction_validity = (
        independently_compliant(raw, schema)
    )
    human_count = strict_int(query["human_count"])
    target_count = int(query["target_count"])
    human_valid = (
        query["human_validity"] == "valid" and human_count is not None
    )
    human_reject = not human_valid
    if schema == "validity":
        prediction_reject = compliant and prediction_validity == "reject"
        usable_numeric = (
            compliant
            and prediction_validity == "valid"
            and prediction_count is not None
        )
    else:
        prediction_reject = False
        usable_numeric = compliant and prediction_count is not None

    automatic_pass = bool(
        usable_numeric and prediction_count == target_count
    )
    human_success = bool(query["human_success"])
    count_correct = bool(
        human_valid
        and usable_numeric
        and prediction_count == human_count
    )
    prediction_tuple = json.dumps(
        [
            bool(compliant),
            prediction_validity if schema == "validity" else "",
            prediction_count if usable_numeric else None,
        ],
        separators=(",", ":"),
    )
    support_count_labels = [
        strict_int(example.get("human_count")) for example in support_examples
    ]
    return {
        "model_name": model_name,
        "condition_id": condition["condition_id"],
        "panel": condition["panel"],
        "schema": schema,
        "support_size": condition["support_size"],
        "replicate": condition["replicate"],
        "query_id": query["query_id"],
        "image_id": query["image_id"],
        "task_name": query["task_name"],
        "prompt_id": query["prompt_id"],
        "cell_id": query["cell_id"],
        "cell_sample_index": query["cell_sample_index"],
        "target_count": target_count,
        "human_validity": query["human_validity"],
        "human_valid": human_valid,
        "human_reject": human_reject,
        "human_count": human_count,
        "human_success": human_success,
        "prediction_status": str(
            raw.get("status", raw.get("prediction_status", ""))
        ),
        "schema_compliant": compliant,
        "prediction_count": prediction_count,
        "prediction_validity": prediction_validity,
        "prediction_reject": prediction_reject,
        "usable_numeric": usable_numeric,
        "count_correct": count_correct,
        "automatic_pass": automatic_pass,
        "agreement": automatic_pass == human_success,
        "prediction_tuple": prediction_tuple,
        "support_image_ids": json.dumps(
            [str(example["image_id"]) for example in support_examples],
            separators=(",", ":"),
        ),
        "support_count_labels": json.dumps(
            support_count_labels, separators=(",", ":")
        ),
        "first_support_count": (
            support_count_labels[0] if support_count_labels else None
        ),
        "last_support_count": (
            support_count_labels[-1] if support_count_labels else None
        ),
        "source_file": raw["_source_file"],
        "source_line": raw["_source_line"],
        "is_alias": False,
    }


def load_predictions(
    run_dirs: Sequence[Path],
    design: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, Any],
]:
    query_by_id = design["query_by_id"]
    condition_by_id = design["condition_by_id"]
    inference_conditions = {
        condition_id
        for condition_id, condition in condition_by_id.items()
        if condition["inference_required"]
    }
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    for run_root, path in prediction_files(run_dirs):
        raw_rows = read_jsonl(path)
        source_files.append(
            {
                "run_root": run_root.as_posix(),
                "path": path.as_posix(),
                "sha256": sha256(path),
                "rows": len(raw_rows),
            }
        )
        for raw in raw_rows:
            condition_id = infer_condition_id(
                raw, path, set(condition_by_id)
            )
            query_id = str(raw.get("query_id") or raw.get("item_id") or "")
            model_name = infer_model_name(raw, run_root)
            location = f"{path}:{raw['_source_line']}"
            if condition_id not in inference_conditions:
                raise ScoringError(
                    f"{location}: unknown or non-inference condition "
                    f"{condition_id!r}"
                )
            if query_id not in query_by_id:
                raise ScoringError(
                    f"{location}: unknown query_id {query_id!r}"
                )
            query = query_by_id[query_id]
            condition = condition_by_id[condition_id]
            if query_id not in design["expected_queries"][condition_id]:
                raise ScoringError(
                    f"{location}: query {query_id} is outside "
                    f"{condition_id}'s frozen scope"
                )
            status = str(
                raw.get("status", raw.get("prediction_status", ""))
            ).strip().lower()
            if status in {"dry_run", "planned", "pending"}:
                raise ScoringError(
                    f"{location}: refusing to score status={status!r}"
                )
            validate_prediction_metadata(
                raw, query, condition, design["assignments"]
            )
            key = (model_name, condition_id, query_id)
            if key in records:
                previous = records[key]
                raise ScoringError(
                    f"Duplicate prediction key {key}: "
                    f"{previous['source_file']}:{previous['source_line']} "
                    f"and {location}"
                )
            records[key] = score_prediction(
                raw,
                model_name,
                query,
                condition,
                expected_support_examples(
                    condition_id, query, design["assignments"]
                ),
            )

    models = sorted({key[0] for key in records})
    if not models:
        raise ScoringError("No prediction rows were loaded")
    by_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model_name, condition_id, _), row in records.items():
        by_block[(model_name, condition_id)].append(row)
    for rows in by_block.values():
        rows.sort(key=lambda item: item["query_id"])
    return dict(by_block), {
        "models": models,
        "source_files": source_files,
        "raw_rows": len(records),
    }


def validate_completeness(
    by_block: dict[tuple[str, str], list[dict[str, Any]]],
    design: dict[str, Any],
    allow_incomplete: bool,
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, Any],
]:
    models = sorted({key[0] for key in by_block})
    inference_conditions = [
        item for item in design["conditions"] if item["inference_required"]
    ]
    report_rows: list[dict[str, Any]] = []
    complete: dict[tuple[str, str], list[dict[str, Any]]] = {}
    missing_total = 0
    for model_name in models:
        for condition in inference_conditions:
            condition_id = str(condition["condition_id"])
            rows = by_block.get((model_name, condition_id), [])
            observed = {str(row["query_id"]) for row in rows}
            expected = design["expected_queries"][condition_id]
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            if extra:
                raise ScoringError(
                    f"{model_name}/{condition_id}: unexpected queries {extra}"
                )
            missing_total += len(missing)
            block_complete = not missing and len(rows) == len(expected)
            report_rows.append(
                {
                    "model_name": model_name,
                    "condition_id": condition_id,
                    "expected": len(expected),
                    "observed": len(rows),
                    "missing_count": len(missing),
                    "missing_query_ids": missing,
                    "complete": block_complete,
                }
            )
            if block_complete:
                complete[(model_name, condition_id)] = rows

    if missing_total and not allow_incomplete:
        examples = [
            f"{row['model_name']}/{row['condition_id']}: "
            f"{row['missing_count']} missing"
            for row in report_rows
            if row["missing_count"]
        ][:12]
        raise ScoringError(
            f"Prediction matrix is incomplete ({missing_total} missing rows). "
            "Use --allow-incomplete only for interim scoring; incomplete "
            "condition blocks will then be skipped.\n  "
            + "\n  ".join(examples)
        )

    # Materialize the exact Panel-C alias only from a complete forward block.
    alias = design["condition_by_id"]["C_count_n3_r1_forward"]
    source_condition = str(alias["reuses_condition"])
    for model_name in models:
        source_key = (model_name, source_condition)
        if source_key not in complete:
            continue
        alias_rows: list[dict[str, Any]] = []
        for source_row in complete[source_key]:
            row = dict(source_row)
            row.update(
                {
                    "condition_id": alias["condition_id"],
                    "panel": alias["panel"],
                    "support_size": alias["support_size"],
                    "replicate": alias["replicate"],
                    "is_alias": True,
                }
            )
            alias_rows.append(row)
        complete[(model_name, str(alias["condition_id"]))] = alias_rows

    return complete, {
        "allow_incomplete": allow_incomplete,
        "models": models,
        "missing_rows": missing_total,
        "complete_inference_blocks": sum(
            row["complete"] for row in report_rows
        ),
        "expected_inference_blocks": len(report_rows),
        "blocks": report_rows,
    }


def summarize(rows: Sequence[dict[str, Any]], panel: str) -> dict[str, Any]:
    n = len(rows)
    human_valid = [row for row in rows if row["human_valid"]]
    human_reject = [row for row in rows if row["human_reject"]]
    numeric_valid = [row for row in human_valid if row["usable_numeric"]]
    errors = [
        float(row["prediction_count"]) - float(row["human_count"])
        for row in numeric_valid
    ]
    reference = [bool(row["human_success"]) for row in rows]
    automatic = [bool(row["automatic_pass"]) for row in rows]
    automatic_rate = mean(automatic)
    human_rate = mean(reference)
    first_label_rows = [
        row for row in rows if row["first_support_count"] is not None
    ]
    last_label_rows = [
        row for row in rows if row["last_support_count"] is not None
    ]
    first_numeric_rows = [
        row for row in first_label_rows if row["usable_numeric"]
    ]
    last_numeric_rows = [
        row for row in last_label_rows if row["usable_numeric"]
    ]
    first_conflict_rows = [
        row
        for row in first_numeric_rows
        if row["human_valid"]
        and row["human_count"] != row["first_support_count"]
    ]
    wrong_numeric_rows = [
        row
        for row in first_numeric_rows
        if row["human_valid"]
        and row["prediction_count"] != row["human_count"]
    ]

    result: dict[str, Any] = {
        "n": n,
        "n_human_valid": len(human_valid),
        "n_human_reject": len(human_reject),
        "n_compliant": sum(row["schema_compliant"] for row in rows),
        "n_numeric_human_valid": len(numeric_valid),
        "n_first_support_conflict": (
            len(first_conflict_rows) if panel in {"A", "C"} else None
        ),
        "n_wrong_numeric_human_valid": (
            len(wrong_numeric_rows) if panel in {"A", "C"} else None
        ),
        "compliance_rate": mean(row["schema_compliant"] for row in rows),
        "numeric_coverage_human_valid": safe_ratio(
            len(numeric_valid), len(human_valid)
        ),
        "count_fidelity_human_valid": mean(
            row["count_correct"] for row in human_valid
        ),
        "mae_human_valid": mean(abs(error) for error in errors),
        "signed_bias_human_valid": mean(errors),
        "first_support_label_match_rate": (
            mean(
                row["usable_numeric"]
                and row["prediction_count"] == row["first_support_count"]
                for row in first_label_rows
            )
            if panel in {"A", "C"}
            else None
        ),
        "first_support_label_match_given_numeric": (
            mean(
                row["prediction_count"] == row["first_support_count"]
                for row in first_numeric_rows
            )
            if panel in {"A", "C"}
            else None
        ),
        "first_support_conflict_anchor_rate": (
            mean(
                row["prediction_count"] == row["first_support_count"]
                for row in first_conflict_rows
            )
            if panel in {"A", "C"}
            else None
        ),
        "first_support_match_among_wrong_numeric": (
            mean(
                row["prediction_count"] == row["first_support_count"]
                for row in wrong_numeric_rows
            )
            if panel in {"A", "C"}
            else None
        ),
        "last_support_label_match_rate": (
            mean(
                row["usable_numeric"]
                and row["prediction_count"] == row["last_support_count"]
                for row in last_label_rows
            )
            if panel in {"A", "C"}
            else None
        ),
        "last_support_label_match_given_numeric": (
            mean(
                row["prediction_count"] == row["last_support_count"]
                for row in last_numeric_rows
            )
            if panel in {"A", "C"}
            else None
        ),
        "A": automatic_rate,
        "H": human_rate,
        "A_minus_H": (
            automatic_rate - human_rate
            if automatic_rate is not None and human_rate is not None
            else None
        ),
        "agreement": mean(
            observed == predicted
            for observed, predicted in zip(
                reference, automatic, strict=True
            )
        ),
        "kappa": cohen_kappa(reference, automatic),
        "model_reject_rate": None,
        "reject_precision": None,
        "reject_recall": None,
        "reject_f1": None,
        "false_reject_rate_human_valid": None,
        "auto_pass_on_human_reject": None,
    }
    if panel == "B":
        predicted_reject = [
            bool(row["prediction_reject"]) for row in rows
        ]
        reference_reject = [bool(row["human_reject"]) for row in rows]
        true_positive = sum(
            observed and predicted
            for observed, predicted in zip(
                reference_reject, predicted_reject, strict=True
            )
        )
        false_positive = sum(
            (not observed) and predicted
            for observed, predicted in zip(
                reference_reject, predicted_reject, strict=True
            )
        )
        false_negative = sum(
            observed and (not predicted)
            for observed, predicted in zip(
                reference_reject, predicted_reject, strict=True
            )
        )
        precision = safe_ratio(
            true_positive, true_positive + false_positive
        )
        recall = safe_ratio(
            true_positive, true_positive + false_negative
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0
            else None
        )
        result.update(
            {
                "reject_tp": true_positive,
                "reject_fp": false_positive,
                "reject_fn": false_negative,
                "reject_tn": (
                    n - true_positive - false_positive - false_negative
                ),
                "model_reject_rate": mean(predicted_reject),
                "reject_precision": precision,
                "reject_recall": recall,
                "reject_f1": f1,
                "false_reject_rate_human_valid": mean(
                    row["prediction_reject"] for row in human_valid
                ),
                "auto_pass_on_human_reject": mean(
                    row["automatic_pass"] for row in human_reject
                ),
            }
        )
    return result


def make_clustered_draws(
    queries: Sequence[dict[str, Any]],
    n_bootstraps: int,
    rng: random.Random,
) -> list[list[str]]:
    by_task_cell: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in queries:
        by_task_cell[str(row["task_name"])][str(row["cell_id"])].append(
            str(row["query_id"])
        )
    for cells in by_task_cell.values():
        for query_ids in cells.values():
            query_ids.sort()

    draws: list[list[str]] = []
    for _ in range(n_bootstraps):
        sampled_query_ids: list[str] = []
        for task_name in sorted(by_task_cell):
            cells = by_task_cell[task_name]
            cell_ids = sorted(cells)
            for _ in range(len(cell_ids)):
                sampled_cell = cell_ids[rng.randrange(len(cell_ids))]
                query_ids = cells[sampled_cell]
                for _ in range(len(query_ids)):
                    sampled_query_ids.append(
                        query_ids[rng.randrange(len(query_ids))]
                    )
        draws.append(sampled_query_ids)
    return draws


def rows_by_query(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    mapping = {str(row["query_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ScoringError("Duplicate query rows reached paired scoring")
    return mapping


def resample_rows(
    mapping: dict[str, dict[str, Any]],
    draw: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        return [mapping[query_id] for query_id in draw]
    except KeyError as error:
        raise ScoringError(
            f"Bootstrap query is absent from a supposedly complete block: {error}"
        ) from error


def bootstrap_ci_payload(
    values: dict[str, list[float]],
    n_bootstraps: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for metric, samples in values.items():
        payload[f"{metric}_ci_low"] = percentile(samples, 0.025)
        payload[f"{metric}_ci_high"] = percentile(samples, 0.975)
        payload[f"{metric}_bootstrap_valid"] = len(samples)
        payload[f"{metric}_bootstrap_degenerate"] = (
            n_bootstraps - len(samples)
        )
    return payload


def condition_points(
    complete: dict[tuple[str, str], list[dict[str, Any]]],
    design: dict[str, Any],
    draws_by_scope: dict[str, list[list[str]]],
    n_bootstraps: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (model_name, condition_id), rows in sorted(complete.items()):
        condition = design["condition_by_id"][condition_id]
        panel = str(condition["panel"])
        point = summarize(rows, panel)
        scope = str(condition["query_scope"])
        mapping = rows_by_query(rows)
        bootstrap_values: dict[str, list[float]] = defaultdict(list)
        for draw in draws_by_scope[scope]:
            boot = summarize(resample_rows(mapping, draw), panel)
            for metric in POINT_METRICS:
                value = boot[metric]
                if value is not None and math.isfinite(float(value)):
                    bootstrap_values[metric].append(float(value))
        row: dict[str, Any] = {
            "model_name": model_name,
            "condition_id": condition_id,
            "panel": panel,
            "schema": condition["schema"],
            "support_size": condition["support_size"],
            "replicate": condition["replicate"],
            "query_scope": scope,
            "is_alias": not condition["inference_required"],
            **point,
            **bootstrap_ci_payload(bootstrap_values, n_bootstraps),
        }
        output.append(row)
    return output


def paired_point(
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    panel: str,
) -> dict[str, float | int | None]:
    if len(left_rows) != len(right_rows):
        raise ScoringError("Paired comparison row counts differ")
    left_ids = [str(row["query_id"]) for row in left_rows]
    right_ids = [str(row["query_id"]) for row in right_rows]
    if left_ids != right_ids:
        # Point-estimate inputs are unique blocks that may arrive in different
        # order. Bootstrap inputs intentionally contain duplicate query IDs
        # and already share the exact draw order.
        if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(
            right_ids
        ):
            raise ScoringError(
                "Paired bootstrap rows do not share the same query draw order"
            )
        left_map = rows_by_query(left_rows)
        right_map = rows_by_query(right_rows)
        if set(left_map) != set(right_map):
            raise ScoringError("Paired comparison query sets differ")
        ordered_ids = sorted(left_map)
        left_rows = [left_map[query_id] for query_id in ordered_ids]
        right_rows = [right_map[query_id] for query_id in ordered_ids]
        left_ids = ordered_ids
    left_summary = summarize(left_rows, panel)
    right_summary = summarize(right_rows, panel)
    result: dict[str, float | int | None] = {
        "n_paired": len(left_rows)
    }
    for metric in DELTA_METRICS:
        left = left_summary[metric]
        right = right_summary[metric]
        result[f"delta_{metric}"] = (
            float(right) - float(left)
            if left is not None and right is not None
            else None
        )
    result["prediction_changed_rate"] = mean(
        left["prediction_tuple"] != right["prediction_tuple"]
        for left, right in zip(left_rows, right_rows, strict=True)
    )
    result["auto_pass_changed_rate"] = mean(
        left["automatic_pass"] != right["automatic_pass"]
        for left, right in zip(left_rows, right_rows, strict=True)
    )
    changed_first_label_pairs = [
        (left, right)
        for left, right in zip(left_rows, right_rows, strict=True)
        if left["first_support_count"] is not None
        and right["first_support_count"] is not None
        and left["first_support_count"] != right["first_support_count"]
    ]
    result["first_label_changed_rate"] = mean(
        left["first_support_count"] is not None
        and right["first_support_count"] is not None
        and left["first_support_count"] != right["first_support_count"]
        for left, right in zip(left_rows, right_rows, strict=True)
    )
    if panel == "B":
        result["prediction_changed_given_first_label_changed"] = None
        result["switched_old_to_new_first_given_changed"] = None
        result["moved_to_new_first_given_changed"] = None
    else:
        result["prediction_changed_given_first_label_changed"] = mean(
            left["prediction_tuple"] != right["prediction_tuple"]
            for left, right in changed_first_label_pairs
        )
        result["switched_old_to_new_first_given_changed"] = mean(
            left["usable_numeric"]
            and right["usable_numeric"]
            and left["prediction_count"] == left["first_support_count"]
            and right["prediction_count"] == right["first_support_count"]
            for left, right in changed_first_label_pairs
        )
        result["moved_to_new_first_given_changed"] = mean(
            right["usable_numeric"]
            and right["prediction_count"] == right["first_support_count"]
            and (
                not left["usable_numeric"]
                or left["prediction_count"] != right["first_support_count"]
            )
            for left, right in changed_first_label_pairs
        )
    return result


def replicate_comparison_specs(
    model_name: str,
    complete: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    available = {
        condition_id
        for model, condition_id in complete
        if model == model_name
    }
    specs: list[dict[str, Any]] = []
    for replicate in (1, 2, 3):
        for lower, higher in itertools.combinations((1, 2, 3, 6), 2):
            baseline = f"A_count_n{lower}_r{replicate}"
            candidate = f"A_count_n{higher}_r{replicate}"
            if baseline in available and candidate in available:
                specs.append(
                    {
                        "panel": "A",
                        "comparison_level": "replicate",
                        "comparison": f"n{higher}-n{lower}",
                        "replicate": replicate,
                        "baseline_conditions": [baseline],
                        "candidate_conditions": [candidate],
                    }
                )
    for replicate in (1, 2, 3):
        baseline = f"B_validity_3valid_r{replicate}"
        candidate = f"B_validity_2valid1reject_r{replicate}"
        if baseline in available and candidate in available:
            specs.append(
                {
                    "panel": "B",
                    "comparison_level": "replicate",
                    "comparison": "2valid1reject-3valid",
                    "replicate": replicate,
                    "baseline_conditions": [baseline],
                    "candidate_conditions": [candidate],
                }
            )
    baseline = "C_count_n3_r1_forward"
    candidate = "C_count_n3_r1_reverse"
    if baseline in available and candidate in available:
        specs.append(
            {
                "panel": "C",
                "comparison_level": "single",
                "comparison": "reverse-forward",
                "replicate": 1,
                "baseline_conditions": [baseline],
                "candidate_conditions": [candidate],
            }
        )

    # Mean-over-replicates summaries are averages of three separately paired
    # metric deltas. They never pool three outputs from one image as if those
    # outputs were three independent images.
    for lower, higher in itertools.combinations((1, 2, 3, 6), 2):
        baselines = [
            f"A_count_n{lower}_r{replicate}" for replicate in (1, 2, 3)
        ]
        candidates = [
            f"A_count_n{higher}_r{replicate}" for replicate in (1, 2, 3)
        ]
        if set(baselines + candidates).issubset(available):
            specs.append(
                {
                    "panel": "A",
                    "comparison_level": "mean_over_replicates",
                    "comparison": f"n{higher}-n{lower}",
                    "replicate": "",
                    "baseline_conditions": baselines,
                    "candidate_conditions": candidates,
                }
            )
    baselines = [
        f"B_validity_3valid_r{replicate}" for replicate in (1, 2, 3)
    ]
    candidates = [
        f"B_validity_2valid1reject_r{replicate}"
        for replicate in (1, 2, 3)
    ]
    if set(baselines + candidates).issubset(available):
        specs.append(
            {
                "panel": "B",
                "comparison_level": "mean_over_replicates",
                "comparison": "2valid1reject-3valid",
                "replicate": "",
                "baseline_conditions": baselines,
                "candidate_conditions": candidates,
            }
        )
    return specs


def average_pair_payloads(
    payloads: Sequence[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    if not payloads:
        raise ScoringError("Cannot average zero paired payloads")
    output: dict[str, float | int | None] = {
        "n_paired": int(payloads[0]["n_paired"])
    }
    metric_names = [
        *(f"delta_{metric}" for metric in DELTA_METRICS),
        *PAIR_ONLY_METRICS,
    ]
    for metric in metric_names:
        output[metric] = strict_mean(
            [
                None if payload[metric] is None else float(payload[metric])
                for payload in payloads
            ]
        )
    return output


def paired_deltas(
    complete: dict[tuple[str, str], list[dict[str, Any]]],
    design: dict[str, Any],
    draws_by_scope: dict[str, list[list[str]]],
    n_bootstraps: int,
) -> list[dict[str, Any]]:
    models = sorted({key[0] for key in complete})
    output: list[dict[str, Any]] = []
    for model_name in models:
        for spec in replicate_comparison_specs(model_name, complete):
            panel = str(spec["panel"])
            baselines = list(spec["baseline_conditions"])
            candidates = list(spec["candidate_conditions"])
            if len(baselines) != len(candidates):
                raise ScoringError("Comparison condition lists differ in length")
            scopes = {
                str(design["condition_by_id"][condition_id]["query_scope"])
                for condition_id in baselines + candidates
            }
            if len(scopes) != 1:
                raise ScoringError(
                    f"Comparison crosses query scopes: {spec}"
                )
            scope = scopes.pop()
            pair_payloads = [
                paired_point(
                    complete[(model_name, baseline)],
                    complete[(model_name, candidate)],
                    panel,
                )
                for baseline, candidate in zip(
                    baselines, candidates, strict=True
                )
            ]
            point = (
                pair_payloads[0]
                if len(pair_payloads) == 1
                else average_pair_payloads(pair_payloads)
            )

            mappings = [
                (
                    rows_by_query(complete[(model_name, baseline)]),
                    rows_by_query(complete[(model_name, candidate)]),
                )
                for baseline, candidate in zip(
                    baselines, candidates, strict=True
                )
            ]
            bootstrap_values: dict[str, list[float]] = defaultdict(list)
            for draw in draws_by_scope[scope]:
                boot_pairs = [
                    paired_point(
                        resample_rows(left, draw),
                        resample_rows(right, draw),
                        panel,
                    )
                    for left, right in mappings
                ]
                boot = (
                    boot_pairs[0]
                    if len(boot_pairs) == 1
                    else average_pair_payloads(boot_pairs)
                )
                for metric in (
                    *(f"delta_{name}" for name in DELTA_METRICS),
                    *PAIR_ONLY_METRICS,
                ):
                    value = boot[metric]
                    if value is not None and math.isfinite(float(value)):
                        bootstrap_values[metric].append(float(value))

            output.append(
                {
                    "model_name": model_name,
                    "panel": panel,
                    "comparison_level": spec["comparison_level"],
                    "comparison": spec["comparison"],
                    "replicate": spec["replicate"],
                    "baseline_conditions": json.dumps(
                        baselines, separators=(",", ":")
                    ),
                    "candidate_conditions": json.dumps(
                        candidates, separators=(",", ":")
                    ),
                    "query_scope": scope,
                    **point,
                    **bootstrap_ci_payload(
                        bootstrap_values, n_bootstraps
                    ),
                }
            )
    return output


def task_filtered_draws(
    draws: Sequence[Sequence[str]],
    allowed_query_ids: set[str],
) -> list[list[str]]:
    filtered = [
        [
            query_id
            for query_id in draw
            if query_id in allowed_query_ids
        ]
        for draw in draws
    ]
    if any(not draw for draw in filtered):
        raise ScoringError("A task-filtered bootstrap draw is empty")
    return filtered


def panel_c_order_diagnostics(
    complete: dict[tuple[str, str], list[dict[str, Any]]],
    design: dict[str, Any],
    all_draws: list[list[str]],
    n_bootstraps: int,
) -> list[dict[str, Any]]:
    forward_id = "C_count_n3_r1_forward"
    reverse_id = "C_count_n3_r1_reverse"
    models = sorted({key[0] for key in complete})
    task_names = sorted(
        {str(row["task_name"]) for row in design["queries"]}
    )
    output: list[dict[str, Any]] = []
    for model_name in models:
        if (
            (model_name, forward_id) not in complete
            or (model_name, reverse_id) not in complete
        ):
            continue
        all_forward = complete[(model_name, forward_id)]
        all_reverse = complete[(model_name, reverse_id)]
        for task_name in ["__all__", *task_names]:
            if task_name == "__all__":
                forward = list(all_forward)
                reverse = list(all_reverse)
                draws = all_draws
                group_label = "all"
            else:
                forward = [
                    row
                    for row in all_forward
                    if row["task_name"] == task_name
                ]
                reverse = [
                    row
                    for row in all_reverse
                    if row["task_name"] == task_name
                ]
                allowed = {str(row["query_id"]) for row in forward}
                draws = task_filtered_draws(all_draws, allowed)
                group_label = task_name
            point = paired_point(forward, reverse, "C")
            forward_summary = summarize(forward, "C")
            reverse_summary = summarize(reverse, "C")
            forward_labels = sorted(
                {str(row["support_count_labels"]) for row in forward}
            )
            reverse_labels = sorted(
                {str(row["support_count_labels"]) for row in reverse}
            )
            identity_order_changed_rate = mean(
                left["support_image_ids"] != right["support_image_ids"]
                for left, right in zip(forward, reverse, strict=True)
            )
            label_order_changed_rate = mean(
                left["support_count_labels"]
                != right["support_count_labels"]
                for left, right in zip(forward, reverse, strict=True)
            )
            if label_order_changed_rate == 0:
                intervention_type = (
                    "identity_order_only_label_sequence_palindrome"
                )
            elif label_order_changed_rate == 1:
                intervention_type = "label_and_identity_order_reversed"
            else:
                intervention_type = "mixed_across_tasks"

            forward_map = rows_by_query(forward)
            reverse_map = rows_by_query(reverse)
            bootstrap_values: dict[str, list[float]] = defaultdict(list)
            for draw in draws:
                boot_forward = resample_rows(forward_map, draw)
                boot_reverse = resample_rows(reverse_map, draw)
                paired = paired_point(boot_forward, boot_reverse, "C")
                left_summary = summarize(boot_forward, "C")
                right_summary = summarize(boot_reverse, "C")
                values = {
                    **{
                        metric: paired[metric]
                        for metric in (
                            *(f"delta_{name}" for name in DELTA_METRICS),
                            *PAIR_ONLY_METRICS,
                        )
                    },
                    **{
                        f"forward_{metric}": left_summary[metric]
                        for metric in (
                            "count_fidelity_human_valid",
                            "first_support_label_match_rate",
                            "first_support_conflict_anchor_rate",
                            "first_support_match_among_wrong_numeric",
                            "last_support_label_match_rate",
                        )
                    },
                    **{
                        f"reverse_{metric}": right_summary[metric]
                        for metric in (
                            "count_fidelity_human_valid",
                            "first_support_label_match_rate",
                            "first_support_conflict_anchor_rate",
                            "first_support_match_among_wrong_numeric",
                            "last_support_label_match_rate",
                        )
                    },
                }
                for metric, value in values.items():
                    if value is not None and math.isfinite(float(value)):
                        bootstrap_values[metric].append(float(value))
            output.append(
                {
                    "model_name": model_name,
                    "task_name": group_label,
                    "intervention_type": intervention_type,
                    "forward_support_label_sequences": json.dumps(
                        forward_labels, separators=(",", ":")
                    ),
                    "reverse_support_label_sequences": json.dumps(
                        reverse_labels, separators=(",", ":")
                    ),
                    "identity_order_changed_rate": identity_order_changed_rate,
                    "label_order_changed_rate": label_order_changed_rate,
                    **{
                        f"forward_{metric}": forward_summary[metric]
                        for metric in (
                            "count_fidelity_human_valid",
                            "first_support_label_match_rate",
                            "first_support_label_match_given_numeric",
                            "first_support_conflict_anchor_rate",
                            "first_support_match_among_wrong_numeric",
                            "last_support_label_match_rate",
                            "last_support_label_match_given_numeric",
                        )
                    },
                    **{
                        f"reverse_{metric}": reverse_summary[metric]
                        for metric in (
                            "count_fidelity_human_valid",
                            "first_support_label_match_rate",
                            "first_support_label_match_given_numeric",
                            "first_support_conflict_anchor_rate",
                            "first_support_match_among_wrong_numeric",
                            "last_support_label_match_rate",
                            "last_support_label_match_given_numeric",
                        )
                    },
                    **point,
                    **bootstrap_ci_payload(
                        bootstrap_values, n_bootstraps
                    ),
                }
            )
    return output


def summarize_stability_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    return {
        "n_queries": len(rows),
        "all_replicates_numeric_coverage": mean(
            row["all_replicates_numeric"] for row in rows
        ),
        "any_prediction_disagreement_rate": mean(
            row["prediction_disagreement"] for row in rows
        ),
        "mean_pairwise_prediction_disagreement_rate": mean(
            row["pairwise_prediction_disagreement"] for row in rows
        ),
        "auto_pass_disagreement_rate": mean(
            row["auto_pass_disagreement"] for row in rows
        ),
        "mean_count_range_given_all_numeric": mean(
            row["count_range"]
            for row in rows
            if row["count_range"] is not None
        ),
        "max_count_range_given_all_numeric": (
            max(
                (
                    float(row["count_range"])
                    for row in rows
                    if row["count_range"] is not None
                ),
                default=None,
            )
        ),
    }


def panel_a_stability_diagnostics(
    complete: dict[tuple[str, str], list[dict[str, Any]]],
    design: dict[str, Any],
    all_draws: list[list[str]],
    n_bootstraps: int,
) -> list[dict[str, Any]]:
    models = sorted({key[0] for key in complete})
    task_names = sorted(
        {str(row["task_name"]) for row in design["queries"]}
    )
    output: list[dict[str, Any]] = []
    for model_name in models:
        for support_size in (1, 2, 3, 6):
            condition_ids = [
                f"A_count_n{support_size}_r{replicate}"
                for replicate in (1, 2, 3)
            ]
            if not all(
                (model_name, condition_id) in complete
                for condition_id in condition_ids
            ):
                continue
            mappings = [
                rows_by_query(complete[(model_name, condition_id)])
                for condition_id in condition_ids
            ]
            common = set(mappings[0])
            if any(set(mapping) != common for mapping in mappings[1:]):
                raise ScoringError(
                    f"{model_name}/n{support_size}: replicate query sets differ"
                )
            diagnostic_rows: list[dict[str, Any]] = []
            for query_id in sorted(common):
                replicate_rows = [
                    mapping[query_id] for mapping in mappings
                ]
                tuples = [
                    row["prediction_tuple"] for row in replicate_rows
                ]
                automatic = [
                    bool(row["automatic_pass"]) for row in replicate_rows
                ]
                numeric = [
                    bool(row["usable_numeric"]) for row in replicate_rows
                ]
                counts = [
                    int(row["prediction_count"])
                    for row in replicate_rows
                    if row["usable_numeric"]
                ]
                pairwise = [
                    left != right
                    for left, right in itertools.combinations(tuples, 2)
                ]
                diagnostic_rows.append(
                    {
                        "query_id": query_id,
                        "task_name": replicate_rows[0]["task_name"],
                        "all_replicates_numeric": all(numeric),
                        "prediction_disagreement": len(set(tuples)) > 1,
                        "pairwise_prediction_disagreement": mean(pairwise),
                        "auto_pass_disagreement": len(set(automatic)) > 1,
                        "count_range": (
                            max(counts) - min(counts)
                            if all(numeric)
                            else None
                        ),
                    }
                )
            diagnostic_map = rows_by_query(diagnostic_rows)
            for task_name in ["__all__", *task_names]:
                if task_name == "__all__":
                    subset = diagnostic_rows
                    draws = all_draws
                    group_label = "all"
                else:
                    subset = [
                        row
                        for row in diagnostic_rows
                        if row["task_name"] == task_name
                    ]
                    allowed = {str(row["query_id"]) for row in subset}
                    draws = task_filtered_draws(all_draws, allowed)
                    group_label = task_name
                point = summarize_stability_rows(subset)
                replicate_fidelities: dict[str, float | None] = {}
                for condition_id in condition_ids:
                    condition_subset = [
                        row
                        for row in complete[(model_name, condition_id)]
                        if (
                            group_label == "all"
                            or row["task_name"] == group_label
                        )
                    ]
                    replicate_fidelities[condition_id] = summarize(
                        condition_subset, "A"
                    )["count_fidelity_human_valid"]
                finite_fidelities = [
                    float(value)
                    for value in replicate_fidelities.values()
                    if value is not None
                ]
                fidelity_range = (
                    max(finite_fidelities) - min(finite_fidelities)
                    if len(finite_fidelities) == len(condition_ids)
                    else None
                )
                bootstrap_values: dict[str, list[float]] = defaultdict(list)
                for draw in draws:
                    boot = summarize_stability_rows(
                        resample_rows(diagnostic_map, draw)
                    )
                    for metric, value in boot.items():
                        if metric == "n_queries":
                            continue
                        if value is not None and math.isfinite(float(value)):
                            bootstrap_values[metric].append(float(value))
                    boot_fidelities: list[float | None] = []
                    for condition_id in condition_ids:
                        condition_mapping = rows_by_query(
                            complete[(model_name, condition_id)]
                        )
                        boot_condition = summarize(
                            resample_rows(condition_mapping, draw), "A"
                        )
                        value = boot_condition[
                            "count_fidelity_human_valid"
                        ]
                        boot_fidelities.append(
                            None if value is None else float(value)
                        )
                    boot_range = (
                        max(float(value) for value in boot_fidelities)
                        - min(float(value) for value in boot_fidelities)
                        if all(
                            value is not None for value in boot_fidelities
                        )
                        else None
                    )
                    if boot_range is not None:
                        bootstrap_values[
                            "count_fidelity_range_across_replicates"
                        ].append(boot_range)
                support_sequences = {
                    condition_id: sorted(
                        {
                            row["support_count_labels"]
                            for row in complete[(model_name, condition_id)]
                            if (
                                group_label == "all"
                                or row["task_name"] == group_label
                            )
                        }
                    )
                    for condition_id in condition_ids
                }
                output.append(
                    {
                        "model_name": model_name,
                        "support_size": support_size,
                        "task_name": group_label,
                        "condition_ids": json.dumps(
                            condition_ids, separators=(",", ":")
                        ),
                        "replicate_support_label_sequences": json.dumps(
                            support_sequences, separators=(",", ":")
                        ),
                        "replicate_count_fidelities": json.dumps(
                            replicate_fidelities, separators=(",", ":")
                        ),
                        "count_fidelity_range_across_replicates": (
                            fidelity_range
                        ),
                        **point,
                        **bootstrap_ci_payload(
                            bootstrap_values, n_bootstraps
                        ),
                    }
                )
    return output


def practical_stability_assessment(
    points: Sequence[dict[str, Any]],
    deltas: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    point_index = {
        (str(row["model_name"]), str(row["condition_id"])): row
        for row in points
    }
    delta_index = {
        (
            str(row["model_name"]),
            str(row["panel"]),
            str(row["comparison_level"]),
            str(row["comparison"]),
        ): row
        for row in deltas
    }
    models = sorted({str(row["model_name"]) for row in points})
    output: list[dict[str, Any]] = []
    equivalence_margin = 0.03
    for model_name in models:
        for support_size in (1, 2, 3):
            condition_ids = [
                f"A_count_n{support_size}_r{replicate}"
                for replicate in (1, 2, 3)
            ]
            condition_points_for_size = [
                point_index.get((model_name, condition_id))
                for condition_id in condition_ids
            ]
            contrast = delta_index.get(
                (
                    model_name,
                    "A",
                    "mean_over_replicates",
                    f"n6-n{support_size}",
                )
            )
            if (
                any(row is None for row in condition_points_for_size)
                or contrast is None
            ):
                output.append(
                    {
                        "model_name": model_name,
                        "support_size": support_size,
                        "comparison": f"n6-n{support_size}",
                        "assessable": False,
                        "verdict": "not_assessable_incomplete",
                    }
                )
                continue
            fidelities = [
                row["count_fidelity_human_valid"]
                for row in condition_points_for_size
                if row is not None
            ]
            fidelity_range = (
                max(float(value) for value in fidelities)
                - min(float(value) for value in fidelities)
                if all(value is not None for value in fidelities)
                else None
            )
            point_delta = contrast[
                "delta_count_fidelity_human_valid"
            ]
            ci_low = contrast[
                "delta_count_fidelity_human_valid_ci_low"
            ]
            ci_high = contrast[
                "delta_count_fidelity_human_valid_ci_high"
            ]
            ci_inside = bool(
                ci_low is not None
                and ci_high is not None
                and float(ci_low) >= -equivalence_margin
                and float(ci_high) <= equivalence_margin
            )
            range_inside = bool(
                fidelity_range is not None
                and fidelity_range <= equivalence_margin
            )
            stable = ci_inside and range_inside
            output.append(
                {
                    "model_name": model_name,
                    "support_size": support_size,
                    "comparison": f"n6-n{support_size}",
                    "assessable": True,
                    "equivalence_margin_rate": equivalence_margin,
                    "mean_paired_count_fidelity_delta": point_delta,
                    "mean_paired_count_fidelity_delta_ci_low": ci_low,
                    "mean_paired_count_fidelity_delta_ci_high": ci_high,
                    "ci_entirely_within_margin": ci_inside,
                    "replicate_count_fidelities": json.dumps(
                        fidelities, separators=(",", ":")
                    ),
                    "count_fidelity_range_across_replicates": (
                        fidelity_range
                    ),
                    "replicate_range_within_margin": range_inside,
                    "practically_stable": stable,
                    "verdict": (
                        "practically_stable"
                        if stable
                        else "stability_not_reached_through_six_shots"
                    ),
                }
            )
    return output


def display_rate(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return "—"
    prefix = "+" if signed else ""
    return f"{float(value) * 100:{prefix}.2f}"


def display_number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def human_readable_report(
    points: Sequence[dict[str, Any]],
    deltas: Sequence[dict[str, Any]],
    panel_c_order: Sequence[dict[str, Any]],
    stability_assessment: Sequence[dict[str, Any]],
    validation: dict[str, Any],
    n_bootstraps: int,
    seed: int,
) -> str:
    lines = [
        "# Few-shot support ablation results",
        "",
        (
            "Status: **complete frozen-matrix analysis**."
            if validation["missing_rows"] == 0
            else "**Interim analysis:** incomplete condition blocks were "
            "excluded as whole blocks."
        ),
        "",
        f"- Models: `{', '.join(validation['models'])}`",
        f"- Missing prediction rows: `{validation['missing_rows']}`",
        f"- Bootstrap: `{n_bootstraps}` task/cell-stratified paired image "
        f"resamples, seed `{seed}`",
        "- Rates are displayed as percentages; `A-H` and paired differences "
        "are percentage points.",
        "",
        "## Panel A: shot size and support replicate",
        "",
        "| Model | Shots | Mean count fidelity | Across-replicate range | "
        "Mean A-H | Mean first-label match | Conflict-only anchoring |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    panel_a = [
        row
        for row in points
        if row["panel"] == "A" and not row["is_alias"]
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in panel_a:
        grouped[(str(row["model_name"]), int(row["support_size"]))].append(
            row
        )
    for (model_name, support_size), rows in sorted(grouped.items()):
        if len(rows) != 3:
            continue
        fidelities = [
            row["count_fidelity_human_valid"] for row in rows
        ]
        finite = [
            float(value) for value in fidelities if value is not None
        ]
        fidelity_range = (
            max(finite) - min(finite) if len(finite) == 3 else None
        )
        lines.append(
            f"| {model_name} | {support_size} | "
            f"{display_rate(strict_mean(fidelities))} | "
            f"{display_rate(fidelity_range)} | "
            f"{display_rate(strict_mean([row['A_minus_H'] for row in rows]), signed=True)} | "
            f"{display_rate(strict_mean([row['first_support_label_match_rate'] for row in rows]))} | "
            f"{display_rate(strict_mean([row['first_support_conflict_anchor_rate'] for row in rows]))} |"
        )

    individual_anchor_maxima: dict[str, float] = {}
    for row in panel_a:
        value = row["first_support_conflict_anchor_rate"]
        if value is None:
            continue
        model_name = str(row["model_name"])
        individual_anchor_maxima[model_name] = max(
            individual_anchor_maxima.get(model_name, 0.0), float(value)
        )
    if individual_anchor_maxima:
        maxima_text = ", ".join(
            f"{model_name} {display_rate(value)}"
            for model_name, value in sorted(individual_anchor_maxima.items())
        )
        lines.extend(
            [
                "",
                "The conflict-only anchoring column averages the three "
                "support-identity replicates at each shot size. Individual "
                f"condition maxima are {maxima_text}. This does not support "
                "a global always-copy account, but it shows localized "
                "first-label matching consistent with "
                "support-identity-sensitive anchoring.",
            ]
        )

    lines.extend(
        [
            "",
            "The practical-stability rule, specified and frozen before "
            "aggregate-result inspection, requires both the paired `n6-n` "
            "count-fidelity CI to lie wholly inside ±3 pp and the "
            "three-replicate count-fidelity range to be at most 3 pp.",
            "",
            "| Model | Shots vs 6 | Paired fidelity delta [95% CI] | "
            "Replicate range | Verdict |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in stability_assessment:
        if not row.get("assessable"):
            lines.append(
                f"| {row['model_name']} | {row['support_size']} | — | — | "
                f"{row['verdict']} |"
            )
            continue
        interval = (
            f"{display_rate(row['mean_paired_count_fidelity_delta'], signed=True)} "
            f"[{display_rate(row['mean_paired_count_fidelity_delta_ci_low'], signed=True)}, "
            f"{display_rate(row['mean_paired_count_fidelity_delta_ci_high'], signed=True)}]"
        )
        lines.append(
            f"| {row['model_name']} | {row['support_size']} | {interval} | "
            f"{display_rate(row['count_fidelity_range_across_replicates'])} | "
            f"{row['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## Panel B: replacing one valid support with a rejected support",
            "",
            "Each row is the mean of three within-replicate paired contrasts, "
            "`2valid+1reject - 3valid`.",
            "",
            "| Model | Δ count fidelity [95% CI] | Δ reject precision | "
            "Δ reject recall | Δ reject F1 | Δ false reject | Δ A-H |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        if (
            row["panel"] != "B"
            or row["comparison_level"] != "mean_over_replicates"
        ):
            continue
        fidelity_interval = (
            f"{display_rate(row['delta_count_fidelity_human_valid'], signed=True)} "
            f"[{display_rate(row['delta_count_fidelity_human_valid_ci_low'], signed=True)}, "
            f"{display_rate(row['delta_count_fidelity_human_valid_ci_high'], signed=True)}]"
        )
        lines.append(
            f"| {row['model_name']} | "
            f"{fidelity_interval} | "
            f"{display_rate(row['delta_reject_precision'], signed=True)} | "
            f"{display_rate(row['delta_reject_recall'], signed=True)} | "
            f"{display_rate(row['delta_reject_f1'], signed=True)} | "
            f"{display_rate(row['delta_false_reject_rate_human_valid'], signed=True)} | "
            f"{display_rate(row['delta_A_minus_H'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "The frozen Panel-B universe contains 52 human-valid and only "
            "3 human-rejected queries. Rejection-rate results are therefore "
            "targeted diagnostics with low effective support, not precise "
            "general estimates of reject recall. This deployed composition "
            "intervention simultaneously introduces a reject exemplar and "
            "removes one numeric-count exemplar; it does not identify a pure "
            "causal effect of the reject cue alone.",
            "",
            "## Panel C: exact support-order reversal",
            "",
            "| Model | Task | Intervention | Prediction changed | "
            "Forward/Reverse fidelity | Δ fidelity | "
            "Forward/Reverse conflict anchoring | "
            "Switched old→new first | Forward first/last match | "
            "Reverse first/last match |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |",
        ]
    )
    for row in panel_c_order:
        lines.append(
            f"| {row['model_name']} | {row['task_name']} | "
            f"{row['intervention_type']} | "
            f"{display_rate(row['prediction_changed_rate'])} | "
            f"{display_rate(row['forward_count_fidelity_human_valid'])}/"
            f"{display_rate(row['reverse_count_fidelity_human_valid'])} | "
            f"{display_rate(row['delta_count_fidelity_human_valid'], signed=True)} | "
            f"{display_rate(row['forward_first_support_conflict_anchor_rate'])}/"
            f"{display_rate(row['reverse_first_support_conflict_anchor_rate'])} | "
            f"{display_rate(row['switched_old_to_new_first_given_changed'])} | "
            f"{display_rate(row['forward_first_support_label_match_rate'])}/"
            f"{display_rate(row['forward_last_support_label_match_rate'])} | "
            f"{display_rate(row['reverse_first_support_label_match_rate'])}/"
            f"{display_rate(row['reverse_last_support_label_match_rate'])} |"
        )
    lines.extend(
        [
            "",
            "For clover, the three-shot count-label sequence `[3,4,3]` is "
            "palindromic. Its reversal changes support identity order but not "
            "the first/last count labels; it must not be interpreted as a "
            "first-versus-last label intervention.",
            "",
            "Machine-readable point estimates, paired intervals, task "
            "breakdowns, and provenance are in the adjacent CSV/JSON files.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    scored_blocks: dict[tuple[str, str], list[dict[str, Any]]],
    points: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    panel_a_stability: list[dict[str, Any]],
    panel_c_order: list[dict[str, Any]],
    stability_assessment: list[dict[str, Any]],
    validation: dict[str, Any],
    provenance: dict[str, Any],
    report: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_rows = [
        row
        for key in sorted(scored_blocks)
        for row in scored_blocks[key]
    ]
    write_csv(output_dir / "scored_rows.csv", scored_rows)
    write_csv(output_dir / "condition_points.csv", points)
    for panel in ("A", "B", "C"):
        write_csv(
            output_dir / f"panel_{panel.lower()}_points.csv",
            [row for row in points if row["panel"] == panel],
        )
    write_csv(output_dir / "paired_deltas.csv", deltas)
    write_csv(
        output_dir / "panel_a_cross_replicate_stability.csv",
        panel_a_stability,
    )
    write_csv(
        output_dir / "panel_c_order_diagnostics.csv", panel_c_order
    )
    write_csv(
        output_dir / "practical_stability_assessment.csv",
        stability_assessment,
    )
    write_json(output_dir / "input_validation.json", validation)
    write_json(output_dir / "scoring_provenance.json", provenance)
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_self_test() -> None:
    base = {
        "human_valid": True,
        "human_reject": False,
        "schema_compliant": True,
        "usable_numeric": True,
        "prediction_reject": False,
        "first_support_count": 2,
        "last_support_count": 4,
    }
    rows = [
        {
            **base,
            "human_count": 2,
            "prediction_count": 2,
            "count_correct": True,
            "human_success": True,
            "automatic_pass": True,
        },
        {
            **base,
            "human_count": 3,
            "prediction_count": 4,
            "count_correct": False,
            "human_success": False,
            "automatic_pass": False,
        },
        {
            **base,
            "human_valid": False,
            "human_reject": True,
            "human_count": None,
            "prediction_count": None,
            "schema_compliant": True,
            "usable_numeric": False,
            "prediction_reject": True,
            "count_correct": False,
            "human_success": False,
            "automatic_pass": False,
        },
    ]
    summary = summarize(rows, "B")
    assert summary["compliance_rate"] == 1.0
    assert summary["count_fidelity_human_valid"] == 0.5
    assert summary["mae_human_valid"] == 0.5
    assert summary["signed_bias_human_valid"] == 0.5
    assert summary["reject_precision"] == 1.0
    assert summary["reject_recall"] == 1.0
    assert summary["false_reject_rate_human_valid"] == 0.0

    queries = [
        {
            "query_id": f"q{cell}_{image}",
            "task_name": "task",
            "cell_id": f"cell{cell}",
        }
        for cell in range(2)
        for image in range(5)
    ]
    draws = make_clustered_draws(queries, 7, random.Random(DEFAULT_SEED))
    assert len(draws) == 7
    assert all(len(draw) == 10 for draw in draws)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.csv"
        write_csv(path, [{"a": 1, "b": 2.5}])
        with path.open(encoding="utf-8") as handle:
            assert list(csv.DictReader(handle)) == [{"a": "1", "b": "2.5"}]
    print("score_results.py self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one or more frozen few-shot ablation model runs with "
            "paired task/cell-stratified image-cluster bootstrap CIs."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Model run directory (repeatable). Each directory may contain "
            "one predictions.jsonl or condition shards recursively."
        ),
    )
    parser.add_argument(
        "--design-dir", type=Path, default=DEFAULT_DESIGN_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Interim mode: skip entire incomplete model-condition blocks. "
            "Never computes metrics on a partial condition."
        ),
    )
    parser.add_argument(
        "--validate-design-only",
        action="store_true",
        help="Validate frozen design hashes/scopes without loading predictions.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small synthetic metrics/bootstrap test and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bootstraps < 1:
        raise SystemExit("--bootstraps must be positive")
    design = load_design(args.design_dir)
    if args.validate_design_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "design_version": design["summary"]["design_version"],
                    "queries": len(design["queries"]),
                    "conditions": len(design["conditions"]),
                    "planned_evaluations_per_model": design["summary"][
                        "planned_evaluations_per_model"
                    ],
                },
                indent=2,
            )
        )
        return
    if not args.run_dir:
        raise SystemExit(
            "At least one --run-dir is required unless "
            "--validate-design-only or --self-test is used"
        )

    by_block, input_provenance = load_predictions(args.run_dir, design)
    try:
        complete, validation = validate_completeness(
            by_block, design, args.allow_incomplete
        )
    except ScoringError as error:
        # Default mode fails closed before point estimates are written.
        raise SystemExit(f"ERROR: {error}") from error
    if not complete:
        raise SystemExit(
            "No complete model-condition block is available for scoring"
        )

    rng = random.Random(args.seed)
    all_queries = list(design["queries"])
    panel_b_queries = [
        row
        for row in all_queries
        if row["query_id"] in design["panel_b_query_ids"]
    ]
    # A single RNG stream creates the two fixed draw matrices. Every
    # model/condition within a scope reuses exactly the same matrix.
    draws_by_scope = {
        "all_queries": make_clustered_draws(
            all_queries, args.bootstraps, rng
        ),
        "panel_b_intersection": make_clustered_draws(
            panel_b_queries, args.bootstraps, rng
        ),
    }
    points = condition_points(
        complete, design, draws_by_scope, args.bootstraps
    )
    deltas = paired_deltas(
        complete, design, draws_by_scope, args.bootstraps
    )
    panel_a_stability = panel_a_stability_diagnostics(
        complete,
        design,
        draws_by_scope["all_queries"],
        args.bootstraps,
    )
    panel_c_order = panel_c_order_diagnostics(
        complete,
        design,
        draws_by_scope["all_queries"],
        args.bootstraps,
    )
    stability_assessment = practical_stability_assessment(
        points, deltas
    )
    provenance = {
        "design_version": design["summary"]["design_version"],
        "design_fingerprints": design["fingerprints"],
        "run_inputs": input_provenance,
        "bootstrap": {
            "resamples": args.bootstraps,
            "seed": args.seed,
            "unit": "query image",
            "scheme": (
                "resample cells with replacement within task, then resample "
                "images with replacement within each selected cell; carry "
                "each selected query's outputs for all conditions/models"
            ),
            "shared_draws": True,
            "query_universes": {
                "all_queries": len(all_queries),
                "panel_b_intersection": len(panel_b_queries),
            },
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "noncompliance_policy": (
            "Non-compliant outputs remain in all denominators and are "
            "automatic/count-fidelity failures; MAE and signed bias require "
            "a usable protocol-valid numeric output and are accompanied by "
            "numeric coverage."
        ),
        "incomplete_policy": (
            "fail closed"
            if not args.allow_incomplete
            else "skip whole incomplete model-condition blocks"
        ),
    }
    report = human_readable_report(
        points,
        deltas,
        panel_c_order,
        stability_assessment,
        validation,
        args.bootstraps,
        args.seed,
    )
    write_outputs(
        args.output_dir,
        complete,
        points,
        deltas,
        panel_a_stability,
        panel_c_order,
        stability_assessment,
        validation,
        provenance,
        report,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": validation["models"],
                "complete_blocks_including_alias": len(complete),
                "condition_point_rows": len(points),
                "paired_delta_rows": len(deltas),
                "panel_a_stability_rows": len(panel_a_stability),
                "panel_c_order_rows": len(panel_c_order),
                "stability_assessment_rows": len(stability_assessment),
                "output_dir": args.output_dir.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ScoringError as error:
        raise SystemExit(f"ERROR: {error}") from error
