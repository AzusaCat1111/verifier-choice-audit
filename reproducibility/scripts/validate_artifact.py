#!/usr/bin/env python3
"""Validate the released dataset metadata, schemas, paths, and score inputs.

The validator is intentionally read-only with respect to ``--dataset-root``.
It uses only the Python standard library for core checks. Optional
``huggingface_hub`` and ``mlcroissant`` checks run when requested and installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TASK_ROWS = {
    "cell_phone_object_count": 540,
    "chair_object_count": 540,
    "traffic_light_object_count": 540,
    "clover_leaf_substructure_count": 300,
    "dresser_substructure_count": 450,
    "power_strip_substructure_count": 450,
    "turbine_substructure_count": 300,
}
EXPECTED_GENERATORS = {"qwenimage2512": 3120, "flux2dev_full": 3120}
EXPECTED_VALIDITY = {"valid", "invalid", "clear_fail"}
EXPECTED_PROTOCOL_KEYS = {
    "zero_shot_count_only",
    "zero_shot_count_reject",
    "few_shot_count_only",
    "few_shot_count_reject",
}
REQUIRED_RAI_FIELDS = {
    "rai:dataUseCases",
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:dataPersonalSensitiveInformation",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
    "rai:dataCollection",
    "rai:dataPreprocessing",
    "rai:dataAnnotationProtocol",
    "rai:dataReleaseMaintenancePlan",
}


@dataclass
class Check:
    level: str
    name: str
    detail: str


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def passed(self, name: str, detail: str) -> None:
        self.checks.append(Check("PASS", name, detail))

    def warning(self, name: str, detail: str) -> None:
        self.checks.append(Check("WARN", name, detail))

    def error(self, name: str, detail: str) -> None:
        self.checks.append(Check("ERROR", name, detail))

    def require(self, condition: bool, name: str, success: str, failure: str) -> bool:
        if condition:
            self.passed(name, success)
            return True
        self.error(name, failure)
        return False

    def counts(self) -> Counter[str]:
        return Counter(check.level for check in self.checks)

    def emit(self) -> None:
        for check in self.checks:
            print(f"[{check.level}] {check.name}: {check.detail}")
        counts = self.counts()
        print(
            "Summary: "
            f"{counts['PASS']} passed, {counts['WARN']} warnings, "
            f"{counts['ERROR']} errors"
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_dataset_path(dataset_root: Path, relative: str) -> Path:
    path = (dataset_root / relative).resolve()
    path.relative_to(dataset_root)
    return path


def dataset_prefixed_path(dataset_root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    if normalized.startswith("dataset/"):
        normalized = normalized[len("dataset/") :]
    return safe_dataset_path(dataset_root, normalized)


def parse_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def validate_readme(report: Report, dataset_root: Path, readme: Path) -> None:
    if not report.require(
        readme.is_file(),
        "dataset card",
        f"found {readme}",
        f"missing {readme}",
    ):
        return
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        report.error("dataset card YAML", "README has no YAML front matter")
        return
    end = text.find("\n---\n", 4)
    if end < 0:
        report.error("dataset card YAML", "README front matter is not closed")
        return

    metadata: dict[str, Any] | None = None
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text[4:end])
        if isinstance(payload, dict):
            metadata = payload
            report.passed("dataset card YAML", "PyYAML parsed the front matter")
        else:
            report.error("dataset card YAML", "front matter did not parse to an object")
    except ImportError:
        report.warning(
            "dataset card YAML",
            "PyYAML unavailable; only delimiter-level validation was performed",
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        report.error("dataset card YAML", f"PyYAML parse failed: {exc}")

    if metadata is not None:
        configs = metadata.get("configs")
        if not isinstance(configs, list) or not configs:
            report.error("viewer configs", "metadata has no non-empty configs list")
        else:
            names: list[str] = []
            bad_paths: list[str] = []
            defaults = 0
            for config in configs:
                if not isinstance(config, dict):
                    continue
                names.append(str(config.get("config_name") or ""))
                defaults += int(config.get("default") is True)
                for data_file in config.get("data_files") or []:
                    if not isinstance(data_file, dict):
                        continue
                    path_value = str(data_file.get("path") or "")
                    try:
                        exists = safe_dataset_path(dataset_root, path_value).is_file()
                    except (ValueError, OSError):
                        exists = False
                    if not exists:
                        bad_paths.append(path_value)
            required = {
                "human1_annotations",
                "human2_annotations",
                "prompt_templates",
                "open_mllm_scored_rows",
            }
            report.require(
                required.issubset(set(names)),
                "viewer configs",
                f"{len(names)} explicit configs; core configs present",
                f"missing core configs: {sorted(required - set(names))}",
            )
            report.require(
                defaults == 1,
                "viewer default",
                "exactly one default config",
                f"expected one default config, found {defaults}",
            )
            report.require(
                not bad_paths,
                "viewer data files",
                "all configured data files exist",
                f"missing or unsafe paths: {bad_paths}",
            )

    try:
        from huggingface_hub import DatasetCard  # type: ignore

        card = DatasetCard(text)
        card.data.to_dict()
        report.passed(
            "huggingface_hub card parse",
            "DatasetCard parsed the metadata locally",
        )
    except ImportError:
        report.warning(
            "huggingface_hub card parse",
            "huggingface_hub unavailable; skipped",
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        report.error("huggingface_hub card parse", str(exc))


def validate_croissant(
    report: Report,
    dataset_root: Path,
    run_mlcroissant: bool,
) -> None:
    path = dataset_root / "croissant.json"
    if not report.require(
        path.is_file(),
        "Croissant file",
        "croissant.json exists",
        "croissant.json is missing",
    ):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.error("Croissant JSON", str(exc))
        return

    report.require(
        payload.get("@type") == "Dataset",
        "Croissant type",
        "@type is Dataset",
        f"unexpected @type: {payload.get('@type')!r}",
    )
    conforms = set(payload.get("conformsTo") or [])
    expected_conformance = {
        "http://mlcommons.org/croissant/1.0",
        "http://mlcommons.org/croissant/RAI/1.0",
    }
    report.require(
        expected_conformance.issubset(conforms),
        "Croissant conformance",
        "Croissant 1.0 and RAI 1.0 declared",
        f"missing: {sorted(expected_conformance - conforms)}",
    )
    missing_rai = sorted(field for field in REQUIRED_RAI_FIELDS if field not in payload)
    report.require(
        not missing_rai,
        "Croissant RAI fields",
        f"all {len(REQUIRED_RAI_FIELDS)} required audit fields present",
        f"missing: {missing_rai}",
    )
    bad_url_fields = [
        field
        for field in ("url", "sameAs")
        if "audit-realease" in str(payload.get(field) or "")
        or not str(payload.get(field) or "").endswith("/audit-release")
    ]
    report.require(
        not bad_url_fields,
        "Croissant repository URL",
        "url and sameAs point to audit-release",
        f"incorrect fields: {bad_url_fields}",
    )

    distributions = payload.get("distribution") or []
    missing_distributions: list[str] = []
    hash_mismatches: list[str] = []
    for distribution in distributions:
        if not isinstance(distribution, dict):
            continue
        content_url = str(distribution.get("contentUrl") or "")
        try:
            local_path = safe_dataset_path(dataset_root, content_url)
        except (ValueError, OSError):
            missing_distributions.append(content_url)
            continue
        if not local_path.exists():
            missing_distributions.append(content_url)
        expected_hash = str(distribution.get("sha256") or "")
        if expected_hash and local_path.is_file() and sha256(local_path) != expected_hash:
            hash_mismatches.append(content_url)
    report.require(
        bool(distributions) and not missing_distributions,
        "Croissant distributions",
        f"all {len(distributions)} distribution paths exist",
        f"missing or unsafe: {missing_distributions}",
    )
    report.require(
        not hash_mismatches,
        "Croissant file hashes",
        "declared SHA-256 hashes match",
        f"mismatches: {hash_mismatches}",
    )

    if run_mlcroissant:
        executable = shutil.which("mlcroissant")
        sibling_entrypoint = Path(sys.executable).resolve().parent / "mlcroissant"
        if executable is None and sibling_entrypoint.is_file():
            executable = str(sibling_entrypoint)
        if executable is None:
            report.warning(
                "official Croissant validator",
                "mlcroissant executable unavailable",
            )
        else:
            result = subprocess.run(
                [executable, "validate", f"--jsonld={path}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            detail = result.stdout.strip().splitlines()
            tail = detail[-1] if detail else "no output"
            report.require(
                result.returncode == 0,
                "official Croissant validator",
                f"mlcroissant validate passed ({tail})",
                f"exit {result.returncode}: {tail}",
            )


def validate_human1(report: Report, dataset_root: Path) -> dict[str, dict[str, str]]:
    path = dataset_root / "annotations/human1_annotations.csv"
    if not path.is_file():
        report.error("Human-1 table", f"missing {path}")
        return {}
    rows = read_csv(path)
    required_columns = {
        "image_id",
        "generator_model",
        "task_name",
        "task_type",
        "prompt_id",
        "target_count",
        "image_path",
        "annotation_path",
        "validity_label",
        "observed_count",
        "observed_part_count",
        "human_count",
        "benchmark_success",
    }
    columns = set(rows[0]) if rows else set()
    report.require(
        required_columns.issubset(columns),
        "Human-1 schema",
        f"all {len(required_columns)} core columns present",
        f"missing: {sorted(required_columns - columns)}",
    )
    report.require(
        len(rows) == 6240,
        "Human-1 row count",
        "6,240 rows",
        f"expected 6,240, found {len(rows)}",
    )

    ids = [row.get("image_id", "") for row in rows]
    report.require(
        len(set(ids)) == len(ids) and "" not in ids,
        "Human-1 image IDs",
        "all IDs are non-empty and unique",
        "empty or duplicate image IDs found",
    )
    generator_counts = Counter(row.get("generator_model", "") for row in rows)
    report.require(
        dict(generator_counts) == EXPECTED_GENERATORS,
        "generator distribution",
        str(dict(sorted(generator_counts.items()))),
        f"expected {EXPECTED_GENERATORS}, found {dict(generator_counts)}",
    )

    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        task_counts[row.get("generator_model", "")][row.get("task_name", "")] += 1
    bad_task_counts = {
        generator: dict(counts)
        for generator, counts in task_counts.items()
        if dict(counts) != EXPECTED_TASK_ROWS
    }
    report.require(
        not bad_task_counts,
        "task distribution",
        "both generators match the seven-task 3,120-image design",
        f"unexpected counts: {bad_task_counts}",
    )

    validity_counts = Counter(row.get("validity_label", "") for row in rows)
    report.require(
        set(validity_counts) <= EXPECTED_VALIDITY,
        "validity vocabulary",
        str(dict(sorted(validity_counts.items()))),
        f"unexpected labels: {sorted(set(validity_counts) - EXPECTED_VALIDITY)}",
    )

    semantic_errors: list[str] = []
    path_errors: list[str] = []
    json_errors: list[str] = []
    for row in rows:
        image_id = row.get("image_id", "")
        validity = row.get("validity_label", "")
        task_type = row.get("task_type", "")
        human_count = parse_int(row.get("human_count"))
        observed = parse_int(
            row.get("observed_part_count")
            if task_type == "substructure"
            else row.get("observed_count")
        )
        success = parse_bool(row.get("benchmark_success"))
        target = parse_int(row.get("target_count"))
        if validity == "valid":
            if human_count is None or human_count != observed:
                semantic_errors.append(f"{image_id}: valid count mismatch")
            expected_success = human_count is not None and human_count == target
            if success is not expected_success:
                semantic_errors.append(f"{image_id}: benchmark_success mismatch")
        elif success is not False:
            semantic_errors.append(f"{image_id}: reject marked successful")

        try:
            image_path = safe_dataset_path(dataset_root, row.get("image_path", ""))
            annotation_path = safe_dataset_path(
                dataset_root, row.get("annotation_path", "")
            )
        except (ValueError, OSError):
            path_errors.append(f"{image_id}: unsafe path")
            continue
        if not image_path.is_file() or not annotation_path.is_file():
            path_errors.append(f"{image_id}: image or annotation missing")
            continue
        if image_path.stem != image_id:
            path_errors.append(f"{image_id}: image stem mismatch")
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception as exc:
            json_errors.append(f"{image_id}: {exc}")
            continue
        expected_pairs = {
            "image_id": image_id,
            "task_id": row.get("task_name", ""),
            "task_type": task_type,
            "prompt_id": row.get("prompt_id", ""),
            "validity_label": validity,
        }
        for field, expected in expected_pairs.items():
            if str(annotation.get(field) or "") != str(expected):
                json_errors.append(f"{image_id}: {field} mismatch")
        annotation_count = (
            parse_int(annotation.get("observed_part_count"))
            if task_type == "substructure"
            else parse_int(annotation.get("observed_count"))
        )
        if annotation_count != human_count:
            json_errors.append(f"{image_id}: JSON/CSV human count mismatch")

    report.require(
        not semantic_errors,
        "Human-1 semantics",
        "validity, counts, target, and success are internally consistent",
        f"{len(semantic_errors)} errors; first: {semantic_errors[:3]}",
    )
    report.require(
        not path_errors,
        "Human-1 paths",
        "all 6,240 image and annotation paths resolve inside the dataset",
        f"{len(path_errors)} errors; first: {path_errors[:3]}",
    )
    report.require(
        not json_errors,
        "Human-1 JSON alignment",
        "all per-image JSON labels align with the canonical CSV",
        f"{len(json_errors)} errors; first: {json_errors[:3]}",
    )
    return {row["image_id"]: row for row in rows if row.get("image_id")}


def validate_human2(
    report: Report,
    dataset_root: Path,
    human1: dict[str, dict[str, str]],
) -> None:
    path = dataset_root / "annotations/human2_annotations.csv"
    if not path.is_file():
        report.error("Human-2 table", f"missing {path}")
        return
    rows = read_csv(path)
    required_columns = {
        "image_id",
        "human1_validity_label",
        "human1_count",
        "human2_validity_label",
        "human2_count",
        "human2_status",
        "human2_audit_subset",
    }
    columns = set(rows[0]) if rows else set()
    report.require(
        required_columns.issubset(columns),
        "Human-2 schema",
        f"all {len(required_columns)} core columns present",
        f"missing: {sorted(required_columns - columns)}",
    )
    ids = [row.get("image_id", "") for row in rows]
    report.require(
        len(rows) == 600 and len(set(ids)) == 600,
        "Human-2 row count",
        "600 unique double-audited images",
        f"rows={len(rows)}, unique IDs={len(set(ids))}",
    )
    missing_ids = sorted(set(ids) - set(human1))
    report.require(
        not missing_ids,
        "Human-2 membership",
        "all Human-2 IDs are in Human-1",
        f"missing from Human-1: {missing_ids[:5]}",
    )

    subset_counts = Counter(row.get("human2_audit_subset", "") for row in rows)
    expected_subsets = {"broad": 240, "traffic_test": 320, "broad;traffic_test": 40}
    report.require(
        dict(subset_counts) == expected_subsets,
        "Human-2 subset flags",
        str(dict(sorted(subset_counts.items()))),
        f"expected {expected_subsets}, found {dict(subset_counts)}",
    )

    alignment_errors: list[str] = []
    for row in rows:
        image_id = row.get("image_id", "")
        source = human1.get(image_id)
        if source is None:
            continue
        if row.get("human1_validity_label", "") != source.get("validity_label", ""):
            alignment_errors.append(f"{image_id}: Human-1 validity mismatch")
        if parse_int(row.get("human1_count")) != parse_int(source.get("human_count")):
            alignment_errors.append(f"{image_id}: Human-1 count mismatch")
        if row.get("human2_status", "") != "completed":
            alignment_errors.append(f"{image_id}: incomplete Human-2 status")
        if row.get("human2_validity_label", "") not in EXPECTED_VALIDITY:
            alignment_errors.append(f"{image_id}: unknown Human-2 validity")
        if (
            row.get("human2_validity_label", "") == "valid"
            and parse_int(row.get("human2_count")) is None
        ):
            alignment_errors.append(f"{image_id}: valid Human-2 count missing")
    report.require(
        not alignment_errors,
        "Human-2 alignment",
        "Human-1 copies, Human-2 status, labels, and counts are consistent",
        f"{len(alignment_errors)} errors; first: {alignment_errors[:3]}",
    )


def validate_human2_contexts(
    report: Report,
    dataset_root: Path,
    human1: dict[str, dict[str, str]],
    broad_manifest: Path | None,
    contextual_human2: Path | None,
) -> None:
    if broad_manifest is None:
        return
    broad_manifest = broad_manifest.resolve()
    if not broad_manifest.is_file():
        report.error(
            "Human-2 context audit",
            f"broad-audit manifest is missing: {broad_manifest}",
        )
        return
    broad_rows = read_csv(broad_manifest)
    canonical_rows = read_csv(
        dataset_root / "annotations/human2_annotations.csv"
    )
    canonical_by_id = {row["image_id"]: row for row in canonical_rows}
    report.require(
        len(broad_rows) == 280
        and len({row.get("image_id", "") for row in broad_rows}) == 280,
        "broad Human-2 source",
        "280 unique broad-audit records",
        (
            f"rows={len(broad_rows)}, unique IDs="
            f"{len({row.get('image_id', '') for row in broad_rows})}"
        ),
    )

    broad_both_valid = 0
    broad_agreements = 0
    canonical_both_valid = 0
    canonical_agreements = 0
    context_differences: list[str] = []
    for broad in broad_rows:
        image_id = broad.get("image_id", "")
        h1 = human1.get(image_id)
        canonical = canonical_by_id.get(image_id)
        if h1 is None or canonical is None:
            continue
        h1_valid = h1.get("validity_label", "") == "valid"
        h1_count = parse_int(h1.get("human_count"))
        broad_valid = broad.get("human2_validity", "") == "valid"
        broad_count = parse_int(broad.get("human2_count"))
        canonical_valid = canonical.get("human2_validity_label", "") == "valid"
        canonical_count = parse_int(canonical.get("human2_count"))
        if h1_valid and broad_valid:
            broad_both_valid += 1
            broad_agreements += int(h1_count == broad_count)
        if h1_valid and canonical_valid:
            canonical_both_valid += 1
            canonical_agreements += int(h1_count == canonical_count)
        if (broad_valid, broad_count) != (canonical_valid, canonical_count):
            context_differences.append(image_id)

    report.require(
        (broad_both_valid, broad_agreements) == (248, 244),
        "submitted broad agreement",
        "context-specific records reproduce 244/248 = 98.39%",
        (
            f"expected 244/248; found "
            f"{broad_agreements}/{broad_both_valid}"
        ),
    )
    report.require(
        (canonical_both_valid, canonical_agreements) == (248, 246),
        "canonical broad-filter diagnostic",
        "one-row-per-image CSV yields 246/248 = 99.19%",
        (
            f"expected diagnostic 246/248; found "
            f"{canonical_agreements}/{canonical_both_valid}"
        ),
    )
    report.require(
        len(context_differences) == 3,
        "Human-2 context collisions",
        (
            "three overlap images have context-specific Human-2 counts: "
            + ", ".join(sorted(context_differences))
        ),
        f"expected 3 context differences, found {len(context_differences)}",
    )

    if contextual_human2 is None:
        report.error(
            "context-preserving Human-2 release",
            (
                "current HF CSV collapses broad/traffic observations; provide "
                "the proposed 640-row context-long table or separate 280/360 tables"
            ),
        )
        return
    contextual_human2 = contextual_human2.resolve()
    if not contextual_human2.is_file():
        report.error(
            "context-preserving Human-2 release",
            f"missing proposed table: {contextual_human2}",
        )
        return
    context_rows = read_csv(contextual_human2)
    context_keys = {
        (row.get("audit_context", ""), row.get("image_id", ""))
        for row in context_rows
    }
    context_counts = Counter(row.get("audit_context", "") for row in context_rows)
    unique_images = {row.get("image_id", "") for row in context_rows}
    proposed_broad = [
        row
        for row in context_rows
        if row.get("audit_context") == "broad"
        and row.get("human1_validity_label") == "valid"
        and row.get("human2_validity_label") == "valid"
    ]
    proposed_agree = sum(
        parse_int(row.get("human1_count")) == parse_int(row.get("human2_count"))
        for row in proposed_broad
    )
    report.require(
        len(context_rows) == 640
        and len(context_keys) == 640
        and len(unique_images) == 600
        and dict(context_counts) == {"broad": 280, "traffic_test": 360}
        and (len(proposed_broad), proposed_agree) == (248, 244),
        "context-preserving Human-2 release",
        (
            "640 unique context/image records preserve broad=280, "
            "traffic=360, 600 images, and the submitted 244/248 result"
        ),
        (
            f"rows={len(context_rows)}, keys={len(context_keys)}, "
            f"images={len(unique_images)}, contexts={dict(context_counts)}, "
            f"broad={proposed_agree}/{len(proposed_broad)}"
        ),
    )


def validate_prompts(report: Report, dataset_root: Path) -> None:
    path = dataset_root / "prompts/t2i_prompt_templates.csv"
    if not path.is_file():
        report.error("prompt table", f"missing {path}")
        return
    rows = read_csv(path)
    keys = {
        (
            row.get("task_name", ""),
            row.get("prompt_id", ""),
            row.get("target_count", ""),
        )
        for row in rows
    }
    report.require(
        len(rows) == 57 and len(keys) == 57,
        "prompt templates",
        "57 unique task/prompt/target templates",
        f"rows={len(rows)}, unique keys={len(keys)}",
    )
    covered_generators = {
        token
        for row in rows
        for token in row.get("applies_to_generators", "").split(";")
        if token
    }
    report.require(
        set(EXPECTED_GENERATORS) <= covered_generators,
        "prompt generator coverage",
        "both released generators are declared",
        f"found {sorted(covered_generators)}",
    )


def validate_release_counts(report: Report, dataset_root: Path) -> None:
    release = dataset_root / "release"
    corpora = [release / generator for generator in EXPECTED_GENERATORS]
    pngs = [path for corpus in corpora for path in corpus.glob("**/*.png")]
    annotation_json = [path for corpus in corpora for path in corpus.glob("**/*.json")]
    report.require(
        len(pngs) == 6240,
        "released PNG count",
        "6,240 PNG files",
        f"expected 6,240, found {len(pngs)}",
    )
    report.require(
        len(annotation_json) == 6240,
        "per-image JSON count",
        "6,240 JSON annotation files",
        f"expected 6,240, found {len(annotation_json)}",
    )
    spatial = release / "spatial_relation"
    if spatial.is_dir():
        spatial_pngs = list(spatial.glob("**/*.png"))
        report.require(
            len(spatial_pngs) == 400,
            "spatial-relation PNG count",
            "400 PNG files",
            f"expected 400, found {len(spatial_pngs)}",
        )

    manifest_path = release / "flux2dev_full/results_manifest.jsonl"
    if not manifest_path.is_file():
        report.error("FLUX generation manifest", f"missing {manifest_path}")
        return
    parse_errors: list[str] = []
    rows = 0
    for line_number, row in read_jsonl(manifest_path):
        rows += 1
        relative_path = str(row.get("relative_path") or "")
        try:
            image_path = safe_dataset_path(dataset_root, relative_path)
        except (ValueError, OSError):
            parse_errors.append(f"line {line_number}: unsafe path")
            continue
        if not image_path.is_file():
            parse_errors.append(f"line {line_number}: missing image")
        digest = str(row.get("sha256") or "")
        if len(digest) != 64:
            parse_errors.append(f"line {line_number}: invalid SHA-256 field")
    report.require(
        rows == 3120 and not parse_errors,
        "FLUX generation manifest",
        "3,120 rows with resolvable images and SHA-256 fields",
        f"rows={rows}; errors={parse_errors[:3]}",
    )
    qwen_manifest = release / "qwenimage2512/results_manifest.jsonl"
    if not qwen_manifest.exists():
        report.warning(
            "Qwen generation manifest",
            "no generator-level results_manifest.jsonl; canonical CSV and per-image JSON still cover all 3,120 images",
        )


def validate_raw_predictions(
    report: Report,
    dataset_root: Path,
    deep_results: bool,
) -> None:
    index_path = (
        dataset_root
        / "results/open_mllm_matrix/raw_predictions/raw_prediction_index.csv"
    )
    if not index_path.is_file():
        report.error("raw prediction index", f"missing {index_path}")
        return
    rows = read_csv(index_path)
    report.require(
        len(rows) == 26,
        "raw prediction index",
        "26 released prediction directories",
        f"expected 26 rows, found {len(rows)}",
    )
    protocol_keys = {row.get("protocol_key", "") for row in rows}
    report.require(
        protocol_keys <= EXPECTED_PROTOCOL_KEYS,
        "raw prediction protocols",
        str(sorted(protocol_keys)),
        f"unexpected: {sorted(protocol_keys - EXPECTED_PROTOCOL_KEYS)}",
    )

    grouped = {
        (row.get("model_key", ""), row.get("protocol_key", "")) for row in rows
    }
    report.require(
        len(grouped) == 20,
        "scorer run groups",
        "26 directories combine into 20 model/protocol runs",
        f"found {len(grouped)} groups",
    )

    missing: list[str] = []
    parse_errors: list[str] = []
    total_predictions = 0
    for index_row in rows:
        artifact_dir = str(index_row.get("artifact_dir") or "")
        try:
            prediction_path = safe_dataset_path(
                dataset_root, f"{artifact_dir}/predictions.jsonl"
            )
        except (ValueError, OSError):
            missing.append(artifact_dir)
            continue
        if not prediction_path.is_file():
            missing.append(artifact_dir)
            continue
        if not deep_results:
            try:
                first_line = next(read_jsonl(prediction_path))
                if not isinstance(first_line[1], dict):
                    parse_errors.append(f"{artifact_dir}: first row is not an object")
            except StopIteration:
                parse_errors.append(f"{artifact_dir}: empty predictions")
            except Exception as exc:
                parse_errors.append(f"{artifact_dir}: {exc}")
            continue
        for line_number, prediction in read_jsonl(prediction_path):
            total_predictions += 1
            required = {
                "annotation_path",
                "item_id",
                "generator_model",
                "task_name",
                "task_type",
                "target_count",
                "status",
                "prediction_count",
                "prediction_validity",
            }
            absent = sorted(required - set(prediction))
            if absent:
                parse_errors.append(
                    f"{artifact_dir}:{line_number}: missing {absent}"
                )
                continue
            try:
                annotation = dataset_prefixed_path(
                    dataset_root, str(prediction["annotation_path"])
                )
            except (ValueError, OSError):
                parse_errors.append(
                    f"{artifact_dir}:{line_number}: unsafe annotation path"
                )
                continue
            if not annotation.is_file():
                parse_errors.append(
                    f"{artifact_dir}:{line_number}: annotation missing"
                )
    report.require(
        not missing,
        "raw prediction files",
        "all indexed predictions.jsonl files exist",
        f"missing: {missing}",
    )
    detail = (
        f"{total_predictions:,} JSONL rows parsed and linked to annotations"
        if deep_results
        else "first row of each prediction file parsed"
    )
    report.require(
        not parse_errors,
        "raw prediction schema",
        detail,
        f"{len(parse_errors)} errors; first: {parse_errors[:3]}",
    )
    if deep_results:
        report.require(
            total_predictions == 45600,
            "raw prediction row count",
            "45,600 predictions (20 runs × 2,280 images)",
            f"expected 45,600, found {total_predictions}",
        )

    scored_path = (
        dataset_root / "results/open_mllm_matrix/evaluation/scored_rows.csv"
    )
    if scored_path.is_file():
        with scored_path.open("r", encoding="utf-8", newline="") as handle:
            scored_count = sum(1 for _ in csv.DictReader(handle))
        report.require(
            scored_count == 45600,
            "released scored rows",
            "45,600 scored image-level rows",
            f"expected 45,600, found {scored_count}",
        )
    else:
        report.error("released scored rows", f"missing {scored_path}")


def validate_detector_support(report: Report, dataset_root: Path) -> None:
    expectations = {
        "object": (120, 90, 30),
        "substructure": (160, 120, 40),
    }
    for family, (expected_rows, expected_train, expected_val) in expectations.items():
        manifest = dataset_root / f"detector/yolo/{family}_manifest.jsonl"
        train_list = dataset_root / f"detector/yolo/{family}_train.txt"
        val_list = dataset_root / f"detector/yolo/{family}_val.txt"
        errors: list[str] = []
        manifest_rows = 0
        if manifest.is_file():
            for line_number, row in read_jsonl(manifest):
                manifest_rows += 1
                for key in ("dataset_image", "dataset_label", "source_annotation"):
                    try:
                        path = dataset_prefixed_path(dataset_root, str(row.get(key) or ""))
                    except (ValueError, OSError):
                        errors.append(f"line {line_number}: unsafe {key}")
                        continue
                    if not path.is_file():
                        errors.append(f"line {line_number}: missing {key}")
        else:
            errors.append("manifest missing")
        train_rows = (
            len([line for line in train_list.read_text().splitlines() if line.strip()])
            if train_list.is_file()
            else -1
        )
        val_rows = (
            len([line for line in val_list.read_text().splitlines() if line.strip()])
            if val_list.is_file()
            else -1
        )
        report.require(
            manifest_rows == expected_rows
            and train_rows == expected_train
            and val_rows == expected_val
            and not errors,
            f"{family} detector support",
            (
                f"manifest={manifest_rows}, train={train_rows}, val={val_rows}; "
                "all referenced files exist"
            ),
            (
                f"manifest={manifest_rows}, train={train_rows}, val={val_rows}; "
                f"errors={errors[:3]}"
            ),
        )


def write_json_report(path: Path, report: Report, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "validator": "reproducibility/scripts/validate_artifact.py",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "readme": str(Path(args.readme).resolve()) if args.readme else "",
        "deep_results": bool(args.deep_results),
        "broad_audit_manifest": (
            str(Path(args.broad_audit_manifest).resolve())
            if args.broad_audit_manifest
            else ""
        ),
        "contextual_human2": (
            str(Path(args.contextual_human2).resolve())
            if args.contextual_human2
            else ""
        ),
        "counts": dict(report.counts()),
        "checks": [asdict(check) for check in report.checks],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Dataset card to validate. Defaults to <dataset-root>/README.md.",
    )
    parser.add_argument(
        "--deep-results",
        action="store_true",
        help="Parse and path-check all 45,600 released raw prediction rows.",
    )
    parser.add_argument(
        "--run-mlcroissant",
        action="store_true",
        help="Run `mlcroissant validate` in addition to structural checks.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable validation report.",
    )
    parser.add_argument(
        "--broad-audit-manifest",
        type=Path,
        default=None,
        help=(
            "Optional retained broad-280 source manifest. When supplied, "
            "diagnose whether context-specific Human-2 labels are reproducible."
        ),
    )
    parser.add_argument(
        "--contextual-human2",
        type=Path,
        default=None,
        help=(
            "Optional proposed 640-row context-long Human-2 table used to "
            "resolve the broad/traffic overlap collision."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    readme = (args.readme or (dataset_root / "README.md")).resolve()
    report = Report()

    if not dataset_root.is_dir():
        report.error("dataset root", f"not a directory: {dataset_root}")
    else:
        report.passed("dataset root", str(dataset_root))
        validate_readme(report, dataset_root, readme)
        validate_croissant(report, dataset_root, args.run_mlcroissant)
        human1 = validate_human1(report, dataset_root)
        validate_human2(report, dataset_root, human1)
        validate_human2_contexts(
            report,
            dataset_root,
            human1,
            args.broad_audit_manifest,
            args.contextual_human2,
        )
        validate_prompts(report, dataset_root)
        validate_release_counts(report, dataset_root)
        validate_raw_predictions(report, dataset_root, args.deep_results)
        validate_detector_support(report, dataset_root)

    report.emit()
    if args.json_out:
        write_json_report(args.json_out, report, args)
        print(f"JSON report: {args.json_out}")
    return 1 if report.counts()["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
