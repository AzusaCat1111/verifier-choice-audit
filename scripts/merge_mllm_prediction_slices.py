#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("**/predictions.jsonl"))


def load_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        paths = prediction_paths(run_dir)
        if not paths:
            raise FileNotFoundError(f"No predictions.jsonl under {run_dir}")
        for path in paths:
            rows.extend(read_jsonl(path))
    return rows


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge corrected prediction slices into a full MLLM run.")
    parser.add_argument("--base-run-dir", action="append", type=Path, required=True)
    parser.add_argument("--slice-run-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    base_rows = load_rows(args.base_run_dir)
    slice_rows = load_rows(args.slice_run_dir)
    replacements: dict[str, dict[str, Any]] = {}
    duplicate_slice_keys: list[str] = []
    for row in slice_rows:
        key = str(row.get("image_rel_path") or "")
        if not key:
            raise RuntimeError(f"Slice row missing image_rel_path: {row}")
        if key in replacements:
            duplicate_slice_keys.append(key)
        replacements[key] = row
    if duplicate_slice_keys:
        raise RuntimeError(f"Duplicate replacement image_rel_path values: {duplicate_slice_keys[:10]}")

    replaced = 0
    merged_rows: list[dict[str, Any]] = []
    seen_base_keys: set[str] = set()
    for base_row in base_rows:
        key = str(base_row.get("image_rel_path") or "")
        seen_base_keys.add(key)
        replacement = replacements.get(key)
        if replacement is None:
            merged_rows.append(base_row)
            continue
        out_row = dict(replacement)
        out_row["merged_replaced_old_item_id"] = base_row.get("item_id")
        out_row["merged_replaced_old_status"] = base_row.get("status")
        out_row["merged_from_slice"] = True
        merged_rows.append(out_row)
        replaced += 1

    missing = sorted(set(replacements) - seen_base_keys)
    if missing:
        raise RuntimeError(f"Replacement rows not found in base rows: {missing[:20]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "predictions.jsonl", merged_rows)

    # Preserve human-readable prompt/config artifacts from the first base run when present.
    first_base = args.base_run_dir[0]
    for name in ("config.json", "prompt_template.md", "fewshot_examples.json"):
        copy_if_exists(first_base / name, args.out_dir / name)

    summary = {
        "base_run_dirs": [path.as_posix() for path in args.base_run_dir],
        "slice_run_dirs": [path.as_posix() for path in args.slice_run_dir],
        "out_dir": args.out_dir.as_posix(),
        "base_rows": len(base_rows),
        "slice_rows": len(slice_rows),
        "merged_rows": len(merged_rows),
        "replaced_rows": replaced,
        "note": args.note,
    }
    (args.out_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
