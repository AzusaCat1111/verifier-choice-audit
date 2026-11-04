#!/usr/bin/env python3
"""Reproduce the released open-MLLM score tables from raw predictions.

This wrapper bridges two release-layout details without modifying either input:

1. released raw-prediction directories are flat and indexed by
   ``raw_prediction_index.csv``, while the scorer accepts a JSON ``--run-spec``;
2. prediction records store annotation paths below ``dataset/``, so an isolated
   working directory with a read-only ``dataset`` symlink is created.

Only the supplied output directory is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_MAP = {
    "zero_shot_count_only": "zero_count",
    "zero_shot_count_reject": "zero_validity",
    "few_shot_count_only": "few_shot_count",
    "few_shot_count_reject": "few_shot_validity",
}
POINT_FILES = [
    "summary_by_run.csv",
    "summary_by_run_generator.csv",
    "summary_by_run_generator_task.csv",
    "summary_by_cell.csv",
]
CI_FILES = [
    "ci_by_run.csv",
    "ci_by_run_generator.csv",
    "ci_by_run_generator_task.csv",
    "summary_by_run_with_ci.csv",
    "summary_by_run_generator_with_ci.csv",
]
REQUIRED_SCORED_COLUMNS = {
    "run_id",
    "model_name",
    "protocol",
    "item_id",
    "image_rel_path",
    "generator_model",
    "task_name",
    "target_count",
    "human_validity",
    "human_count",
    "prediction_status",
    "prediction_count",
    "prediction_validity",
    "auto_target_pass",
    "human_benchmark_success",
    "agreement_all",
    "row_uid",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "tracked_files_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "tracked_files_dirty": None}


def ensure_output_is_separate(
    output_root: Path, code_root: Path, dataset_root: Path
) -> None:
    for input_root, label in ((code_root, "code root"), (dataset_root, "dataset root")):
        if output_root == input_root or output_root.is_relative_to(input_root):
            raise SystemExit(f"Output directory must not be inside the {label}: {output_root}")


def build_run_spec(dataset_root: Path) -> list[dict[str, Any]]:
    index_path = (
        dataset_root
        / "results/open_mllm_matrix/raw_predictions/raw_prediction_index.csv"
    )
    if not index_path.is_file():
        raise SystemExit(f"Missing raw prediction index: {index_path}")
    rows = read_csv(index_path)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for row in rows:
        model = row.get("model_key", "")
        protocol_key = row.get("protocol_key", "")
        if protocol_key not in PROTOCOL_MAP:
            raise SystemExit(f"Unsupported protocol_key in release index: {protocol_key!r}")
        artifact_relative = row.get("artifact_dir", "")
        run_dir = (dataset_root / artifact_relative).resolve()
        run_dir.relative_to(dataset_root)
        prediction_path = run_dir / "predictions.jsonl"
        if not prediction_path.is_file():
            raise SystemExit(f"Missing indexed prediction file: {prediction_path}")
        groups[(model, PROTOCOL_MAP[protocol_key])].append(run_dir)
    if len(rows) != 26 or len(groups) != 20:
        raise SystemExit(
            f"Expected 26 indexed directories forming 20 runs; "
            f"found {len(rows)} directories and {len(groups)} runs"
        )

    specs: list[dict[str, Any]] = []
    for (model, protocol), run_dirs in sorted(groups.items()):
        # The released CI tables were seeded after concatenating Qwen-Image
        # rows before FLUX rows. Bootstrap point estimates are invariant to
        # this order, but a fixed pseudo-random index stream is not. Preserve
        # the historical concatenation order for byte-level CI reproduction.
        def historical_order(path: Path) -> tuple[int, str]:
            name = path.name
            if "qwenimage2512" in name:
                return (0, name)
            if "flux2dev_full" in name:
                return (1, name)
            return (2, name)

        specs.append(
            {
                "run_id": f"{model}/{protocol}",
                "model_name": model,
                "protocol": protocol,
                "run_dirs": [
                    str(path) for path in sorted(run_dirs, key=historical_order)
                ],
                "note": "constructed from released raw_prediction_index.csv",
            }
        )
    return specs


def compare_csv(generated: Path, released: Path) -> dict[str, Any]:
    if not generated.is_file() or not released.is_file():
        return {
            "match": False,
            "detail": "generated or released file is missing",
        }
    generated_rows = read_csv(generated)
    released_rows = read_csv(released)
    if generated_rows == released_rows:
        return {
            "match": True,
            "detail": f"exact CSV row/value match ({len(generated_rows)} rows)",
            "generated_sha256": sha256(generated),
            "released_sha256": sha256(released),
        }

    first_difference: dict[str, Any] | None = None
    for index, (left, right) in enumerate(
        zip(generated_rows, released_rows, strict=False)
    ):
        if left != right:
            differing = {
                key: {"generated": left.get(key), "released": right.get(key)}
                for key in sorted(set(left) | set(right))
                if left.get(key) != right.get(key)
            }
            first_difference = {
                "row_index": index,
                "columns": differing,
            }
            break
    return {
        "match": False,
        "detail": (
            f"generated rows={len(generated_rows)}, released rows={len(released_rows)}"
        ),
        "first_difference": first_difference,
    }


def validate_outputs(generated_dir: Path) -> dict[str, Any]:
    required = {
        "run_inventory.csv",
        "scored_rows.csv",
        "summary_by_run.csv",
        "summary_by_run_generator.csv",
        "summary_by_run_generator_task.csv",
        "summary_by_cell.csv",
        "ci_by_run.csv",
        "ci_by_run_generator.csv",
        "ci_by_run_generator_task.csv",
        "summary_by_run_with_ci.csv",
        "summary_by_run_generator_with_ci.csv",
        "report.md",
    }
    missing = sorted(name for name in required if not (generated_dir / name).is_file())
    if missing:
        return {"valid": False, "missing": missing}

    scored = read_csv(generated_dir / "scored_rows.csv")
    columns = set(scored[0]) if scored else set()
    row_uids = [row.get("row_uid", "") for row in scored]
    summary = read_csv(generated_dir / "summary_by_run.csv")
    run_ids = {row.get("run_id", "") for row in summary}
    errors: list[str] = []
    if len(scored) != 45600:
        errors.append(f"expected 45,600 scored rows, found {len(scored)}")
    if not REQUIRED_SCORED_COLUMNS.issubset(columns):
        errors.append(
            f"missing scored columns: {sorted(REQUIRED_SCORED_COLUMNS - columns)}"
        )
    if len(row_uids) != len(set(row_uids)) or "" in row_uids:
        errors.append("row_uid is empty or non-unique")
    if len(summary) != 20 or len(run_ids) != 20:
        errors.append(
            f"expected 20 run summaries, rows={len(summary)}, unique IDs={len(run_ids)}"
        )
    return {
        "valid": not errors,
        "scored_rows": len(scored),
        "unique_row_uids": len(set(row_uids)),
        "summary_runs": len(run_ids),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the released scorer.",
    )
    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument(
        "--keep-scored-rows",
        action="store_true",
        help=(
            "Retain the regenerated ~38 MB scored_rows.csv. By default it is "
            "validated and then removed because the dataset already releases it."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_root = args.code_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    scorer = code_root / "scripts/evaluate_mllm_counting.py"
    released_eval = dataset_root / "results/open_mllm_matrix/evaluation"

    if sys.version_info < (3, 10):
        raise SystemExit("Python >=3.10 is required by the released scorer.")
    if not scorer.is_file():
        raise SystemExit(f"Missing released scorer: {scorer}")
    if not dataset_root.is_dir():
        raise SystemExit(f"Missing dataset root: {dataset_root}")
    ensure_output_is_separate(output_root, code_root, dataset_root)

    output_root.mkdir(parents=True, exist_ok=True)
    generated_dir = output_root / "generated"
    work_dir = output_root / "work"
    for path in (generated_dir, work_dir):
        if path.exists() and any(path.iterdir()):
            raise SystemExit(
                f"Refusing to overwrite non-empty directory: {path}. "
                "Choose a new --output-dir."
            )
        path.mkdir(parents=True, exist_ok=True)

    dataset_link = work_dir / "dataset"
    if dataset_link.exists() or dataset_link.is_symlink():
        raise SystemExit(f"Refusing to replace existing path: {dataset_link}")
    os.symlink(dataset_root, dataset_link, target_is_directory=True)

    run_spec = build_run_spec(dataset_root)
    run_spec_path = output_root / "run_spec.json"
    run_spec_path.write_text(
        json.dumps({"runs": run_spec}, indent=2) + "\n", encoding="utf-8"
    )

    command = [
        args.python,
        str(scorer),
        "--run-spec",
        str(run_spec_path),
        "--out-dir",
        str(generated_dir),
        "--n-boot",
        str(args.n_boot),
        "--seed",
        str(args.seed),
        "--include-scored-rows",
    ]
    result = subprocess.run(
        command,
        cwd=work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_root / "scorer.log").write_text(result.stdout, encoding="utf-8")

    output_validation: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if result.returncode == 0:
        output_validation = validate_outputs(generated_dir)
        for name in POINT_FILES:
            comparisons[name] = compare_csv(
                generated_dir / name, released_eval / name
            )
        if args.n_boot == 1000:
            for name in CI_FILES:
                comparisons[name] = compare_csv(
                    generated_dir / name, released_eval / name
                )

    regenerated_scored = generated_dir / "scored_rows.csv"
    if not args.keep_scored_rows and regenerated_scored.is_file():
        regenerated_scored.unlink()

    report = {
        "wrapper": "reproducibility/scripts/reproduce_open_mllm_scores.py",
        "python_version": sys.version.split()[0],
        "code_revision": git_revision(code_root),
        "dataset_revision": git_revision(dataset_root),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "scored_rows_retained": bool(
            args.keep_scored_rows and regenerated_scored.is_file()
        ),
        "run_groups": len(run_spec),
        "indexed_prediction_directories": sum(
            len(spec["run_dirs"]) for spec in run_spec
        ),
        "scorer_returncode": result.returncode,
        "output_validation": output_validation,
        "released_table_comparisons": comparisons,
    }
    report_path = output_root / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Scorer log: {output_root / 'scorer.log'}")
    print(f"Reproduction report: {report_path}")
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        return result.returncode
    if not output_validation.get("valid"):
        print(f"Output validation failed: {output_validation}", file=sys.stderr)
        return 1
    failed_comparisons = [
        name for name, result in comparisons.items() if not result.get("match")
    ]
    if failed_comparisons:
        print(
            f"Released-table comparison failed: {failed_comparisons}",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: regenerated {output_validation['scored_rows']:,} scored rows "
        f"and matched {len(comparisons)} released CSV tables."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
