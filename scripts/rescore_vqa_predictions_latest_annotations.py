#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_annotation_path(row: dict[str, Any], release_root: Path) -> Path:
    annotation_path = row.get("annotation_path")
    if annotation_path:
        path = Path(str(annotation_path))
        if path.exists():
            return path
        if not path.is_absolute():
            candidate = release_root.parent / path
            if candidate.exists():
                return candidate

    image_rel = row.get("image_rel_path") or row.get("image_path")
    if image_rel:
        rel = Path(str(image_rel))
        if rel.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            rel = rel.with_suffix(".json")
        candidates = [
            release_root / rel,
            release_root.parent / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve annotation for row {row.get('item_id')} {row.get('image_rel_path')}")


def latest_count(annotation: dict[str, Any], task_type: str) -> int | None:
    validity = str(annotation.get("validity_label") or "")
    if validity != "valid":
        return None
    if task_type == "substructure":
        value = annotation.get("observed_part_count")
    else:
        value = annotation.get("observed_count")
    if value is None or value == "":
        return None
    return int(value)


def rescore_row(row: dict[str, Any], release_root: Path) -> dict[str, Any]:
    annotation_path = resolve_annotation_path(row, release_root)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    task_type = str(row.get("task_type") or annotation.get("task_type") or "")
    count = latest_count(annotation, task_type)
    validity = str(annotation.get("validity_label") or "")
    candidate_count = int(row["candidate_count"])
    expected = "yes" if count is not None and candidate_count == count else "no"
    prediction = row.get("prediction_answer")
    new_row = dict(row)
    new_row["old_expected_answer"] = row.get("expected_answer")
    new_row["old_human_count"] = row.get("human_count")
    new_row["old_correct"] = row.get("correct")
    new_row["latest_annotation_path"] = str(annotation_path)
    new_row["human_validity"] = validity
    new_row["human_valid"] = validity == "valid"
    new_row["human_reject"] = validity != "valid"
    new_row["human_count"] = count
    new_row["expected_answer"] = expected
    new_row["correct"] = bool(prediction == expected) if prediction in {"yes", "no"} else False
    new_row["gold_changed"] = (
        str(row.get("expected_answer")) != expected
        or (row.get("human_count") is not None and str(row.get("human_count")) != str(count))
        or (row.get("human_count") is None and count is not None)
    )
    return new_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore VQA yes/no predictions against latest release annotations.")
    parser.add_argument("--prediction", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path("dataset/release"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.prediction:
        rows.extend(read_jsonl(path))
    rescored = [rescore_row(row, args.release_root) for row in rows]
    write_jsonl(args.out, rescored)
    changed = sum(1 for row in rescored if row.get("gold_changed"))
    print(json.dumps({"rows": len(rescored), "gold_changed_rows": changed, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
