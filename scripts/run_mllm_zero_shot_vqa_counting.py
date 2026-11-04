#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_mllm_zero_shot_counting import (
    SYSTEM_PROMPT,
    TASK_SPECS,
    append_jsonl,
    detect_truncation_signals,
    extract_response_artifacts,
    format_annotation_protocol,
    get_system_prompt,
    get_task_specs,
    image_data_url,
    parse_json_object,
    run_curl_json,
    utc_timestamp,
    write_jsonl,
)


OBJECT_TASKS = [
    "cell_phone_object_count",
    "chair_object_count",
    "traffic_light_object_count",
]

SUBSTRUCTURE_TASKS = [
    "clover_leaf_substructure_count",
    "dresser_substructure_count",
    "power_strip_substructure_count",
    "turbine_substructure_count",
]


USER_PROMPT_TEMPLATE = """You are evaluating a generated image for a visual question answering benchmark.

Question: Are there exactly {candidate_count} {target_units} in the image?

Count only the requested visible target units.
Ignore: {ignore}.

Answer the question directly. Return exactly one valid JSON object and no extra text:
{{
  "answer": "yes" | "no",
  "confidence": "low" | "medium" | "high",
  "evidence": "<one short sentence>"
}}"""


USER_PROMPT_TEMPLATE_TASK_GUIDED = """You are evaluating a generated image for a structural counting VQA benchmark.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

Human annotation protocol for this task:
{annotation_protocol}

Question: Are there exactly {candidate_count} countable target units in the image?

Rules:
1. Use the human annotation protocol to define the counting scope.
2. Answer "yes" only if the image contains exactly {candidate_count} countable target units under that protocol.
3. If the image would be rejected under the annotation protocol, or the count cannot be determined reliably, answer "no".
4. Do not infer the count from common sense, typical object shapes, or likely object defaults.

Return exactly one valid JSON object and no extra text:
{{
  "answer": "yes" | "no",
  "confidence": "low" | "medium" | "high",
  "evidence": "<one short sentence>"
}}"""


@dataclass(frozen=True)
class VQAItem:
    item_id: str
    generator_model: str
    task_name: str
    task_type: str
    prompt_id: str
    count_key: str
    target_count: int | None
    human_count: int
    candidate_count: int
    expected_answer: str
    cell_id: str
    query_cell_id: str
    cell_sample_index: int
    image_path: str
    image_rel_path: str
    annotation_path: str


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_count_from_path(path: Path) -> tuple[str, int] | None:
    for part in path.parts:
        match = re.fullmatch(r"([kc])_(\d+)", part)
        if match:
            return match.group(1), int(match.group(2))
    return None


def read_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def human_count_from_annotation(annotation: dict[str, Any], task_type: str) -> int | None:
    if annotation.get("validity_label") != "valid":
        return None
    if task_type == "object_count":
        return as_int(annotation.get("observed_count"))
    if task_type == "substructure":
        return as_int(annotation.get("observed_part_count"))
    return None


def collect_base_samples(
    release_root: Path,
    generator_model: str,
    task_names: list[str],
    samples_per_cell: int,
    task_specs: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    annotation_root = release_root / generator_model / "annotation_datasets"
    if not annotation_root.exists():
        raise FileNotFoundError(f"Missing annotation root: {annotation_root}")

    cell_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_name in task_names:
        if task_name not in task_specs:
            raise ValueError(f"Unknown task: {task_name}")
        task_spec = task_specs[task_name]
        task_root = annotation_root / task_name / task_name
        if not task_root.exists():
            raise FileNotFoundError(f"Missing task root: {task_root}")
        for image_path in sorted(task_root.rglob("*.png")):
            annotation_path = image_path.with_suffix(".json")
            if not annotation_path.exists():
                continue
            annotation = read_annotation(annotation_path)
            human_count = human_count_from_annotation(annotation, task_spec["task_type"])
            if human_count is None:
                continue
            parsed = parse_count_from_path(image_path)
            if parsed is None:
                continue
            count_key, target_count = parsed
            prompt_id = image_path.parent.name
            cell_id = f"{generator_model}__{task_name}__{prompt_id}__{count_key}{target_count}"
            cell_buckets[cell_id].append(
                {
                    "generator_model": generator_model,
                    "task_name": task_name,
                    "task_type": task_spec["task_type"],
                    "prompt_id": prompt_id,
                    "count_key": count_key,
                    "target_count": target_count,
                    "human_count": human_count,
                    "cell_id": cell_id,
                    "image_path": image_path.resolve(),
                    "image_rel_path": image_path.relative_to(release_root).as_posix(),
                    "annotation_path": annotation_path.resolve(),
                }
            )

    base_samples: list[dict[str, Any]] = []
    for _cell_id, samples in sorted(cell_buckets.items()):
        base_samples.extend(samples[:samples_per_cell])
    return base_samples


def candidate_counts_for_samples(base_samples: list[dict[str, Any]], mode: str) -> dict[str, list[int]]:
    if mode == "per_image_pm1":
        return {}
    if mode != "task_distribution_pm1":
        raise ValueError(f"Unsupported candidate mode: {mode}")
    by_task: dict[str, list[int]] = defaultdict(list)
    for sample in base_samples:
        by_task[str(sample["task_name"])].append(int(sample["human_count"]))
    candidates_by_task: dict[str, list[int]] = {}
    for task_name, counts in by_task.items():
        min_count = min(counts)
        max_count = max(counts)
        candidates_by_task[task_name] = list(range(max(0, min_count - 1), max_count + 2))
    return candidates_by_task


def build_items(
    release_root: Path,
    generator_model: str,
    task_names: list[str],
    samples_per_cell: int,
    candidate_mode: str,
    task_specs: dict[str, dict[str, str]],
) -> list[VQAItem]:
    base_samples = collect_base_samples(
        release_root=release_root,
        generator_model=generator_model,
        task_names=task_names,
        samples_per_cell=samples_per_cell,
        task_specs=task_specs,
    )
    candidates_by_task = candidate_counts_for_samples(base_samples, candidate_mode)
    items: list[VQAItem] = []
    sample_index_by_cell: Counter[str] = Counter()
    for sample in base_samples:
        sample_index_by_cell[str(sample["cell_id"])] += 1
        if candidate_mode == "per_image_pm1":
            candidate_counts = list(range(max(0, int(sample["human_count"]) - 1), int(sample["human_count"]) + 2))
        else:
            candidate_counts = candidates_by_task[str(sample["task_name"])]
        for candidate_count in candidate_counts:
            expected_answer = "yes" if candidate_count == int(sample["human_count"]) else "no"
            item_id = f"{len(items) + 1:06d}"
            query_cell_id = f"{sample['cell_id']}__ask{candidate_count}"
            items.append(
                VQAItem(
                    item_id=item_id,
                    generator_model=str(sample["generator_model"]),
                    task_name=str(sample["task_name"]),
                    task_type=str(sample["task_type"]),
                    prompt_id=str(sample["prompt_id"]),
                    count_key=str(sample["count_key"]),
                    target_count=as_int(sample["target_count"]),
                    human_count=int(sample["human_count"]),
                    candidate_count=candidate_count,
                    expected_answer=expected_answer,
                    cell_id=str(sample["cell_id"]),
                    query_cell_id=query_cell_id,
                    cell_sample_index=sample_index_by_cell[str(sample["cell_id"])],
                    image_path=Path(sample["image_path"]).as_posix(),
                    image_rel_path=str(sample["image_rel_path"]),
                    annotation_path=Path(sample["annotation_path"]).as_posix(),
                )
            )
    return items


def get_vqa_prompt_template(protocol_version: str) -> str:
    if protocol_version == "plain-count-only":
        return USER_PROMPT_TEMPLATE
    return USER_PROMPT_TEMPLATE_TASK_GUIDED


def build_prompt(item: VQAItem, task_specs: dict[str, dict[str, str]], protocol_version: str) -> str:
    task_spec = task_specs[item.task_name]
    template = get_vqa_prompt_template(protocol_version)
    return template.format(
        candidate_count=item.candidate_count,
        target_units=task_spec["target_units"],
        ignore=task_spec["ignore"],
        annotation_protocol=format_annotation_protocol(task_spec),
    )


def build_payload(item: VQAItem, args: argparse.Namespace, task_specs: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "model": args.api_model,
        "messages": [
            {"role": "system", "content": get_system_prompt(args.protocol_version)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(item, task_specs, args.protocol_version)},
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(item.image_path))}},
                ],
            },
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }


def normalize_answer(parsed: dict[str, Any]) -> dict[str, Any]:
    raw = parsed.get("answer")
    answer = ""
    if isinstance(raw, bool):
        answer = "yes" if raw else "no"
    elif isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"yes", "true", "1", "y"}:
            answer = "yes"
        elif lowered in {"no", "false", "0", "n"}:
            answer = "no"

    confidence_raw = parsed.get("confidence")
    confidence = ""
    if isinstance(confidence_raw, str):
        confidence = confidence_raw.strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = ""

    evidence_raw = parsed.get("evidence")
    evidence = evidence_raw.strip() if isinstance(evidence_raw, str) else ""
    return {"answer": answer, "confidence": confidence, "evidence": evidence}


def load_done_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                done.add(str(row.get("item_id")))
    return done


def summarize(output_path: Path, planned_items: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    usable = [row for row in rows if row.get("status") == "ok" and row.get("prediction_answer") in {"yes", "no"}]
    correct = sum(1 for row in usable if row.get("prediction_answer") == row.get("expected_answer"))
    by_task: dict[str, dict[str, Any]] = {}
    for task_name in sorted({row.get("task_name") for row in usable}):
        task_rows = [row for row in usable if row.get("task_name") == task_name]
        task_correct = sum(1 for row in task_rows if row.get("prediction_answer") == row.get("expected_answer"))
        by_task[str(task_name)] = {
            "n": len(task_rows),
            "accuracy": task_correct / len(task_rows) if task_rows else None,
            "answer_distribution": dict(Counter(row.get("prediction_answer") for row in task_rows)),
        }
    return {
        "planned_items": planned_items,
        "completed_rows": len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "usable_rows": len(usable),
        "accuracy": correct / len(usable) if usable else None,
        "expected_answer_distribution": dict(Counter(row.get("expected_answer") for row in usable)),
        "prediction_answer_distribution": dict(Counter(row.get("prediction_answer") for row in usable)),
        "by_task": by_task,
    }


def summarize_manifest(items: list[VQAItem]) -> dict[str, Any]:
    return {
        "planned_items": len(items),
        "base_images": len({(item.cell_id, item.cell_sample_index) for item in items}),
        "query_cells": len({item.query_cell_id for item in items}),
        "by_task": dict(Counter(item.task_name for item in items)),
        "candidate_counts_by_task": {
            task_name: sorted({item.candidate_count for item in items if item.task_name == task_name})
            for task_name in sorted({item.task_name for item in items})
        },
        "expected_answer_distribution": dict(Counter(item.expected_answer for item in items)),
    }


def write_run_config(run_dir: Path, args: argparse.Namespace, items: list[VQAItem]) -> None:
    (run_dir / "prompt_template.md").write_text(
        "\n".join(
            [
                "# MLLM Zero-Shot VQA Counting Prompt",
                "",
                "## System Prompt",
                "",
                "```text",
                get_system_prompt(args.protocol_version),
                "```",
                "",
                "## User Prompt Template",
                "",
                "```text",
                get_vqa_prompt_template(args.protocol_version),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "created_at_utc": utc_timestamp(),
        "api_base": args.api_base,
        "api_model": args.api_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "protocol_version": args.protocol_version,
        "release_root": args.release_root,
        "generator_model": args.generator_model,
        "task_names": args.task_names,
        "samples_per_cell": args.samples_per_cell,
        "candidate_mode": args.candidate_mode,
        "item_count": len(items),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_inference(args: argparse.Namespace) -> None:
    if not args.bearer_token and args.bearer_token_env:
        args.bearer_token = os.environ.get(args.bearer_token_env, "")
    if not args.api_base:
        raise ValueError("Provide --api-base for API calls.")
    if not args.bearer_token:
        raise ValueError("Missing bearer token. Pass --bearer-token or --bearer-token-env.")

    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/mllm_vqa") / f"vqa_zeroshot_{utc_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "manifest.jsonl"
    truncation_path = run_dir / "truncation_report.jsonl"
    task_specs = get_task_specs(args.protocol_version)

    items = build_items(
        release_root=Path(args.release_root),
        generator_model=args.generator_model,
        task_names=args.task_names,
        samples_per_cell=args.samples_per_cell,
        candidate_mode=args.candidate_mode,
        task_specs=task_specs,
    )
    if args.start_index is not None or args.end_index is not None:
        start_index = max(args.start_index or 1, 1)
        end_index = args.end_index or len(items)
        items = items[start_index - 1 : end_index]
    if args.limit_items is not None:
        items = items[: args.limit_items]

    write_jsonl(manifest_path, [asdict(item) for item in items])
    write_run_config(run_dir, args, items)
    if args.dry_run:
        print(json.dumps(summarize_manifest(items), indent=2))
        return

    done = load_done_item_ids(output_path) if args.resume else set()
    url = args.api_base.rstrip("/") + "/v1/chat/completions"
    start_time = time.time()
    for index, item in enumerate(items, start=1):
        if item.item_id in done:
            continue
        request_started = time.time()
        row: dict[str, Any] = {
            **asdict(item),
            "judge_model": args.api_model,
            "api_base": args.api_base,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "status": "error",
            "raw_response": None,
            "assistant_message": None,
            "assistant_content": "",
            "assistant_json_text": "",
            "assistant_think_text": "",
            "assistant_reasoning_fields": {},
            "assistant_reasoning_text": "",
            "finish_reason": "",
            "stop_reason": "",
            "truncation_suspected": False,
            "truncation_signals": [],
            "parsed": None,
            "prediction_answer": "",
            "confidence": "",
            "evidence": "",
            "correct": None,
            "error": "",
            "latency_s": None,
            "usage": {},
        }
        try:
            status_code, body = run_curl_json(
                url=url,
                payload=build_payload(item, args, task_specs),
                timeout=args.request_timeout,
                bearer_token=args.bearer_token,
                retries=args.retries,
            )
            if status_code >= 400:
                raise RuntimeError(f"HTTP {status_code}: {body.decode('utf-8', errors='replace')[:500]}")
            artifacts = extract_response_artifacts(body)
            row.update(
                {
                    "raw_response": artifacts["response_json"],
                    "assistant_message": artifacts["assistant_message"],
                    "assistant_content": artifacts["assistant_content"],
                    "assistant_json_text": artifacts["assistant_json_text"],
                    "assistant_think_text": artifacts["assistant_think_text"],
                    "assistant_reasoning_fields": artifacts["assistant_reasoning_fields"],
                    "assistant_reasoning_text": artifacts["assistant_reasoning_text"],
                    "finish_reason": str(artifacts["choice"].get("finish_reason") or ""),
                    "stop_reason": str(artifacts["choice"].get("stop_reason") or ""),
                    "usage": artifacts["usage"],
                }
            )
            truncation_signals = detect_truncation_signals(artifacts["choice"], artifacts["assistant_json_text"])
            row["truncation_signals"] = truncation_signals
            row["truncation_suspected"] = bool(truncation_signals)
            parsed = parse_json_object(artifacts["assistant_json_text"])
            normalized = normalize_answer(parsed)
            row.update(
                {
                    "status": "ok",
                    "parsed": parsed,
                    "prediction_answer": normalized["answer"],
                    "confidence": normalized["confidence"],
                    "evidence": normalized["evidence"],
                    "correct": normalized["answer"] == item.expected_answer if normalized["answer"] else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        finally:
            row["latency_s"] = round(time.time() - request_started, 3)
            append_jsonl(output_path, row)
            if row["truncation_suspected"]:
                append_jsonl(truncation_path, row)

        if index % args.log_every == 0 or index == len(items):
            elapsed = time.time() - start_time
            print(
                f"[{index}/{len(items)}] item={item.item_id} status={row['status']} "
                f"task={item.task_name} ask={item.candidate_count} expected={item.expected_answer} "
                f"pred={row.get('prediction_answer')} latency={row['latency_s']}s elapsed={elapsed/60:.1f}m",
                flush=True,
            )

    summary = summarize(output_path, len(items))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot yes/no VQA counting on countable-structure tasks.")
    parser.add_argument("--release-root", default="dataset/release")
    parser.add_argument("--generator-model", default="qwenimage2512")
    parser.add_argument("--task-names", nargs="+", default=OBJECT_TASKS)
    parser.add_argument("--samples-per-cell", type=int, default=20)
    parser.add_argument("--candidate-mode", choices=["task_distribution_pm1", "per_image_pm1"], default="task_distribution_pm1")
    parser.add_argument(
        "--protocol-version",
        choices=[
            "count-only",
            "reject-count",
            "task-guided-count-only",
            "task-guided-reject-count",
            "traffic-light-canonical-count-only",
            "traffic-light-canonical-reject-count",
        ],
        default="task-guided-count-only",
    )
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-model", default="qwen3.5-27b")
    parser.add_argument("--bearer-token", default="")
    parser.add_argument("--bearer-token-env", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
