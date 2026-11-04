#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_mllm_zero_shot_counting import (
    SYSTEM_PROMPT,
    TASK_SPECS,
    chat_payload_to_responses_payload,
    InferenceItem,
    append_jsonl,
    build_items,
    detect_truncation_signals,
    extract_response_artifacts,
    extract_responses_artifacts,
    image_data_url,
    format_annotation_protocol,
    get_system_prompt,
    get_task_specs,
    normalize_prediction,
    parse_json_object,
    run_curl_json,
    utc_timestamp,
    write_jsonl,
)


FEW_SHOT_USER_PROMPT_TEMPLATE = """You are evaluating a generated image for a counting benchmark.

Count only the requested visible target units.
Target units to count: {target_units}.
Ignore: {ignore}.

You will first see labeled examples from the same prompt family. Use the examples only to calibrate what should be counted. Then count the final query image.

Reason silently if needed, but do not write step-by-step reasoning in the final answer.
If hidden reasoning is enabled, keep it concise.
Return exactly one valid JSON object for the final query image with these keys:
{{
  "count": <integer>,
  "confidence": "low" | "medium" | "high",
  "evidence": "<one short sentence>"
}}"""


FEW_SHOT_USER_PROMPT_TEMPLATE_COUNT_ONLY_PREFIX = """You are evaluating a generated image for a structural counting benchmark.

This is a count-only protocol. Do not reject the image. Return your best visual count for the final query image.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

You will first see labeled examples from the same prompt family.
The examples are provided only to clarify the counting scope and the JSON format.
Do not copy the first example, the last example, the majority example, or any typical count.
The final query image may have a different count from all examples."""


FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY_PREFIX = """You are evaluating a generated image for a structural counting benchmark.

This is a count-only protocol. Do not reject the image. Return your best visual count for the final query image.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

Human annotation protocol for this task:
{annotation_protocol}

The word "Reject" in the annotation protocol describes human audit behavior. In this count-only protocol, do not output reject; use those cases only as warnings about unreliable counting scope.

You will first see labeled examples from the same prompt family.
The examples are provided only to clarify the counting scope, the human annotation protocol, and the JSON format.
Do not copy the first example, the last example, the majority example, or any typical count.
The final query image may have a different count from all examples."""


FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_REJECT_COUNT_PREFIX = """You are evaluating a generated image for a structural counting benchmark.

This is a reject-or-count protocol. First decide whether the final query image is valid for this counting task. If it is valid, return the visual count. If it is not reliably countable, reject it.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

Human annotation protocol for this task:
{annotation_protocol}

You will first see labeled examples from the same prompt family.
The examples are provided only to clarify the counting scope, the human annotation protocol, the reject boundary, and the JSON format.
Do not copy the first example, the last example, the majority example, or any typical count.
The final query image may have a different validity label and a different count from all examples."""


FEW_SHOT_FINAL_INSTRUCTIONS_COUNT_ONLY = """Rules:
1. Count only visible target units in the final query image.
2. Use the examples only to understand what should be counted.
3. Do not copy any example count.
4. Do not infer the count from common sense, typical object shapes, or likely object defaults.
5. Do not use any desired or requested count; no target count is provided.

Return exactly one valid JSON object for the final query image:
{"count": <integer>}"""


FEW_SHOT_FINAL_INSTRUCTIONS_REJECT_COUNT = """Validity labels:
- "valid": the image has a clear counting scope and the target units can be counted reliably.
- "reject": the image should not be counted because the target is missing, the wrong object is shown, the image violates the human annotation protocol, or the count is not reliable because of blur, occlusion, cropping, merged units, malformed structure, unclear boundaries, or unclear counting scope.

Rules:
1. Count only visible target units in the final query image.
2. Use the examples only to understand what should be counted and when to reject.
3. Do not copy any example count or validity label.
4. Do not infer the count from common sense, typical object shapes, or likely object defaults.
5. Do not use any desired or requested count; no target count is provided.
6. If the final query image is valid, return an integer count.
7. If the final query image should be rejected, set count to null.

Return exactly one valid JSON object for the final query image:
{"validity": "valid" | "reject", "count": <integer or null>}"""


def get_few_shot_prompt_template(protocol_version: str) -> str:
    if protocol_version == "plain-count-only":
        return FEW_SHOT_USER_PROMPT_TEMPLATE
    if protocol_version == "count-only":
        return (
            FEW_SHOT_USER_PROMPT_TEMPLATE_COUNT_ONLY_PREFIX
            + "\n\n[Examples and final query image are interleaved as image messages.]\n\n"
            + FEW_SHOT_FINAL_INSTRUCTIONS_COUNT_ONLY
        )
    if protocol_version in {"task-guided-count-only", "traffic-light-canonical-count-only"}:
        return (
            FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY_PREFIX
            + "\n\n[Examples and final query image are interleaved as image messages.]\n\n"
            + FEW_SHOT_FINAL_INSTRUCTIONS_COUNT_ONLY
        )
    if protocol_version in {"task-guided-reject-count", "traffic-light-canonical-reject-count"}:
        return (
            FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_REJECT_COUNT_PREFIX
            + "\n\n[Examples and final query image are interleaved as image messages.]\n\n"
            + FEW_SHOT_FINAL_INSTRUCTIONS_REJECT_COUNT
        )
    raise ValueError(f"Unsupported few-shot protocol version: {protocol_version}")


@dataclass(frozen=True)
class FewShotExample:
    generator_model: str
    task_name: str
    task_type: str
    prompt_id: str
    count_key: str
    target_count: int
    human_count: int | None
    sample_index: int
    image_id: str
    image_path: str
    image_rel_path: str
    annotation_path: str
    validity_label: str = "valid"


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_sample_index(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.[^.]+$", path.name)
    if not match:
        return None
    return int(match.group(1))


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


def example_validity_from_annotation(annotation: dict[str, Any], task_type: str) -> str:
    return "valid" if human_count_from_annotation(annotation, task_type) is not None else "reject"


def parse_count_dir(path: Path) -> tuple[str, int] | None:
    match = re.fullmatch(r"([kc])_(\d+)", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def support_shuffle_seed(base_seed: int, generator_model: str, task_name: str, prompt_id: str) -> int:
    key = f"{generator_model}__{task_name}__{prompt_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return base_seed + int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_few_shot_example(
    *,
    release_root: Path,
    generator_model: str,
    task_name: str,
    task_type: str,
    prompt_id: str,
    count_key: str,
    target_count: int,
    image_path: Path,
    annotation: dict[str, Any],
    human_count: int | None,
    validity_label: str,
) -> FewShotExample:
    sample_index = parse_sample_index(image_path)
    if sample_index is None:
        raise ValueError(f"Could not parse sample index from {image_path}")
    return FewShotExample(
        generator_model=generator_model,
        task_name=task_name,
        task_type=task_type,
        prompt_id=prompt_id,
        count_key=count_key,
        target_count=target_count,
        human_count=human_count,
        sample_index=sample_index,
        image_id=str(annotation.get("image_id") or image_path.stem),
        image_path=image_path.resolve().as_posix(),
        image_rel_path=image_path.relative_to(release_root).as_posix(),
        annotation_path=image_path.with_suffix(".json").resolve().as_posix(),
        validity_label=validity_label,
    )


def select_few_shot_examples(
    release_root: Path,
    generator_model: str,
    task_name: str,
    prompt_id: str,
    samples_per_cell: int,
    max_examples_per_prompt: int,
    min_examples_per_prompt: int | None,
    include_reject_example: bool = False,
    support_seed: int = 20260503,
) -> list[FewShotExample]:
    task_spec = TASK_SPECS[task_name]
    task_root = release_root / generator_model / "annotation_datasets" / task_name / task_name
    if not task_root.exists():
        raise FileNotFoundError(f"Missing task root: {task_root}")

    valid_by_count: dict[tuple[str, int], list[FewShotExample]] = {}
    reject_candidates: list[FewShotExample] = []
    count_dirs = sorted(path for path in task_root.iterdir() if path.is_dir() and parse_count_dir(path) is not None)

    for count_dir in count_dirs:
        parsed = parse_count_dir(count_dir)
        assert parsed is not None
        count_key, target_count = parsed
        prompt_dir = count_dir / prompt_id
        image_paths = sorted(prompt_dir.glob("*.png"), reverse=True)
        count_examples: list[FewShotExample] = []
        for image_path in image_paths:
            sample_index = parse_sample_index(image_path)
            if sample_index is None or sample_index <= samples_per_cell:
                continue
            annotation_path = image_path.with_suffix(".json")
            annotation = read_annotation(annotation_path)
            human_count = human_count_from_annotation(annotation, task_spec["task_type"])
            if human_count is not None:
                count_examples.append(
                    build_few_shot_example(
                        release_root=release_root,
                        generator_model=generator_model,
                        task_name=task_name,
                        task_type=task_spec["task_type"],
                        prompt_id=prompt_id,
                        count_key=count_key,
                        target_count=target_count,
                        image_path=image_path,
                        annotation=annotation,
                        human_count=human_count,
                        validity_label="valid",
                    )
                )
            elif include_reject_example and example_validity_from_annotation(annotation, task_spec["task_type"]) == "reject":
                reject_candidates.append(
                    build_few_shot_example(
                        release_root=release_root,
                        generator_model=generator_model,
                        task_name=task_name,
                        task_type=task_spec["task_type"],
                        prompt_id=prompt_id,
                        count_key=count_key,
                        target_count=target_count,
                        image_path=image_path,
                        annotation=annotation,
                        human_count=None,
                        validity_label="reject",
                    )
                )
        if count_examples:
            count_examples.sort(key=lambda example: (example.human_count != target_count, -example.sample_index))
            valid_by_count[(count_key, target_count)] = count_examples

    max_valid_examples = max_examples_per_prompt
    if include_reject_example and reject_candidates and max_examples_per_prompt > 1:
        max_valid_examples = max_examples_per_prompt - 1

    examples: list[FewShotExample] = []
    for count_key, target_count in sorted(valid_by_count):
        if len(examples) >= max_valid_examples:
            break
        examples.append(valid_by_count[(count_key, target_count)][0])

    if len(examples) < max_valid_examples:
        used_image_ids = {example.image_id for example in examples}
        for count_key, target_count in sorted(valid_by_count):
            for example in valid_by_count[(count_key, target_count)][1:]:
                if example.image_id in used_image_ids:
                    continue
                examples.append(example)
                used_image_ids.add(example.image_id)
                if len(examples) >= max_valid_examples:
                    break
            if len(examples) >= max_valid_examples:
                break

    if include_reject_example and reject_candidates and len(examples) < max_examples_per_prompt:
        examples.append(reject_candidates[0])

    rng = random.Random(support_shuffle_seed(support_seed, generator_model, task_name, prompt_id))
    rng.shuffle(examples)

    required_examples = min_examples_per_prompt
    if required_examples is None:
        required_examples = min(max_examples_per_prompt, len(count_dirs))
    if len(examples) < required_examples:
        raise RuntimeError(
            f"Only selected {len(examples)} examples for {generator_model}/{task_name}/{prompt_id}; "
            f"need {required_examples}"
        )
    return examples


def build_examples_by_key(
    release_root: Path,
    generator_models: list[str],
    task_names: list[str],
    prompt_ids: list[str],
    samples_per_cell: int,
    max_examples_per_prompt: int,
    min_examples_per_prompt: int | None,
    protocol_version: str = "plain-count-only",
    support_seed: int = 20260503,
) -> dict[str, list[FewShotExample]]:
    examples_by_key: dict[str, list[FewShotExample]] = {}
    for generator_model in generator_models:
        for task_name in task_names:
            for prompt_id in prompt_ids:
                key = f"{generator_model}__{task_name}__{prompt_id}"
                examples_by_key[key] = select_few_shot_examples(
                    release_root=release_root,
                    generator_model=generator_model,
                    task_name=task_name,
                    prompt_id=prompt_id,
                    samples_per_cell=samples_per_cell,
                    max_examples_per_prompt=max_examples_per_prompt,
                    min_examples_per_prompt=min_examples_per_prompt,
                    include_reject_example=protocol_version in {
                        "task-guided-reject-count",
                        "traffic-light-canonical-reject-count",
                    },
                    support_seed=support_seed,
                )
    return examples_by_key


def build_prompt(task_name: str, protocol_version: str = "plain-count-only") -> str:
    task_spec = get_task_specs(protocol_version)[task_name]
    if protocol_version == "count-only":
        return FEW_SHOT_USER_PROMPT_TEMPLATE_COUNT_ONLY_PREFIX.format(
            target_units=task_spec["target_units"],
            ignore=task_spec["ignore"],
        )
    if protocol_version in {"task-guided-count-only", "traffic-light-canonical-count-only"}:
        return FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY_PREFIX.format(
            target_units=task_spec["target_units"],
            ignore=task_spec["ignore"],
            annotation_protocol=format_annotation_protocol(task_spec),
        )
    if protocol_version in {"task-guided-reject-count", "traffic-light-canonical-reject-count"}:
        return FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_REJECT_COUNT_PREFIX.format(
            target_units=task_spec["target_units"],
            ignore=task_spec["ignore"],
            annotation_protocol=format_annotation_protocol(task_spec),
        )
    if protocol_version != "plain-count-only":
        raise ValueError(f"Unsupported few-shot protocol version: {protocol_version}")
    return FEW_SHOT_USER_PROMPT_TEMPLATE.format(
        target_units=task_spec["target_units"],
        ignore=task_spec["ignore"],
    )


def build_payload(item: InferenceItem, examples: list[FewShotExample], args: argparse.Namespace) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_prompt(item.task_name, args.protocol_version)},
    ]
    if args.protocol_version in {
        "count-only",
        "task-guided-count-only",
        "task-guided-reject-count",
        "traffic-light-canonical-count-only",
        "traffic-light-canonical-reject-count",
    }:
        for index, example in enumerate(examples, start=1):
            if args.protocol_version in {
                "task-guided-reject-count",
                "traffic-light-canonical-reject-count",
            }:
                answer = (
                    '{"validity": "reject", "count": null}'
                    if example.validity_label == "reject"
                    else f'{{"validity": "valid", "count": {example.human_count}}}'
                )
            else:
                answer = f'{{"count": {example.human_count}}}'
            content.extend(
                [
                    {"type": "text", "text": f"Example {index} image:"},
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(example.image_path))}},
                    {"type": "text", "text": f"Example {index} answer:\n{answer}"},
                ]
            )
        final_instructions = (
            FEW_SHOT_FINAL_INSTRUCTIONS_REJECT_COUNT
            if args.protocol_version in {
                "task-guided-reject-count",
                "traffic-light-canonical-reject-count",
            }
            else FEW_SHOT_FINAL_INSTRUCTIONS_COUNT_ONLY
        )
        content.extend(
            [
                {"type": "text", "text": "Final query image:"},
                {"type": "image_url", "image_url": {"url": image_data_url(Path(item.image_path))}},
                {"type": "text", "text": final_instructions},
            ]
        )
    else:
        for index, example in enumerate(examples, start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Example {index}: the correct count for this example image is "
                            f"{example.human_count}."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(example.image_path))}},
                ]
            )
        content.extend(
            [
                {
                    "type": "text",
                    "text": "Now count the target units in the final query image. Return only the JSON object.",
                },
                {"type": "image_url", "image_url": {"url": image_data_url(Path(item.image_path))}},
            ]
        )
    payload = {
        "model": args.api_model,
        "messages": [
            {"role": "system", "content": get_system_prompt(args.protocol_version)},
            {"role": "user", "content": content},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
        if args.reasoning_exclude:
            payload["reasoning"]["exclude"] = True
        payload["include_reasoning"] = True
    return payload


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    items: list[InferenceItem],
    examples_by_key: dict[str, list[FewShotExample]],
) -> None:
    prompt_path = run_dir / "prompt_template.md"
    prompt_path.write_text(
        "\n".join(
            [
                "# MLLM Few-Shot Counting Prompt",
                "",
                "Examples are selected per `(generator_model, task_name, prompt_id)` from the back of each cell.",
                f"The first `{args.samples_per_cell}` images per cell are reserved for evaluation and are never used as examples.",
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
                get_few_shot_prompt_template(args.protocol_version),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    examples_payload = {
        key: [asdict(example) for example in examples] for key, examples in sorted(examples_by_key.items())
    }
    (run_dir / "fewshot_examples.json").write_text(
        json.dumps(examples_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = {
        "created_at_utc": utc_timestamp(),
        "api_base": args.api_base,
        "api_format": args.api_format,
        "api_model": args.api_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_exclude": args.reasoning_exclude,
        "protocol_version": args.protocol_version,
        "samples_per_cell": args.samples_per_cell,
        "max_examples_per_prompt": args.max_examples_per_prompt,
        "min_examples_per_prompt": args.min_examples_per_prompt,
        "support_seed": args.support_seed,
        "release_root": args.release_root,
        "generator_models": args.generator_models,
        "task_names": args.task_names,
        "item_count": len(items),
        "prompt_template_path": prompt_path.as_posix(),
        "fewshot_examples_path": (run_dir / "fewshot_examples.json").as_posix(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def summarize(items: list[InferenceItem], output_path: Path) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    status_counts["invalid_jsonl"] += 1
                    continue
                status_counts[str(row.get("status", ""))] += 1
                cell_counts[str(row.get("cell_id", ""))] += 1
                task_counts[str(row.get("task_name", ""))] += 1
                model_counts[str(row.get("generator_model", ""))] += 1
    return {
        "planned_items": len(items),
        "status_counts": dict(status_counts),
        "model_counts": dict(model_counts),
        "task_counts": dict(task_counts),
        "completed_cells": sum(1 for count in cell_counts.values() if count > 0),
        "total_cells": len({item.cell_id for item in items}),
    }


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


def format_prompt_dump(
    item: InferenceItem,
    examples: list[FewShotExample],
    chat_payload: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    image_labels = [
        f"Example {index} image: {example.image_rel_path}"
        for index, example in enumerate(examples, start=1)
    ]
    image_labels.append(f"Final query image: {item.image_rel_path}")
    image_iter = iter(image_labels)

    lines = [
        "# GPT Prompt Dump",
        "",
        "This dump omits base64 image payloads and shows image placeholders instead.",
        "",
        "## Request",
        "",
        f"- model: {args.api_model}",
        f"- api_format: {args.api_format}",
        f"- protocol_version: {args.protocol_version}",
        f"- item_id: {item.item_id}",
        f"- cell_id: {item.cell_id}",
        f"- image: {item.image_rel_path}",
        "",
        "## Chat Messages Before API Conversion",
        "",
    ]
    for message_index, message in enumerate(chat_payload.get("messages") or [], start=1):
        role = message.get("role", "")
        lines.extend([f"### Message {message_index}: {role}", ""])
        content = message.get("content")
        if isinstance(content, str):
            lines.extend(["```text", content, "```", ""])
            continue
        if not isinstance(content, list):
            lines.extend(["```text", str(content), "```", ""])
            continue
        for part in content:
            if not isinstance(part, dict):
                lines.extend(["```text", str(part), "```", ""])
                continue
            part_type = part.get("type")
            if part_type == "text":
                lines.extend(["```text", str(part.get("text") or ""), "```", ""])
            elif part_type == "image_url":
                lines.append(f"[IMAGE: {next(image_iter, 'image payload omitted')}]")
                lines.append("")
            else:
                lines.extend(["```text", json.dumps(part, ensure_ascii=False), "```", ""])
    if args.api_format == "responses":
        responses_payload = chat_payload_to_responses_payload(
            chat_payload,
            max_output_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            reasoning_exclude=args.reasoning_exclude,
            enable_reasoning=args.enable_thinking,
        )
        lines.extend(
            [
                "## Responses API Controls",
                "",
                "```json",
                json.dumps(
                    {
                        "model": responses_payload.get("model"),
                        "temperature": responses_payload.get("temperature"),
                        "top_p": responses_payload.get("top_p"),
                        "max_output_tokens": responses_payload.get("max_output_tokens"),
                        "text": responses_payload.get("text"),
                        "reasoning": responses_payload.get("reasoning"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def run_inference(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if not args.bearer_token and args.bearer_token_env:
        args.bearer_token = os.environ.get(args.bearer_token_env, "")
    if not args.dry_run and not args.dump_first_prompt and not args.api_base:
        raise ValueError("Provide --api-base for API calls.")
    if not args.dry_run and not args.dump_first_prompt and not args.bearer_token:
        raise ValueError("Provide --bearer-token or --bearer-token-env for API calls.")
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/mllm_few_shot") / f"fewshot_{utc_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "manifest.jsonl"
    truncation_path = run_dir / "truncation_report.jsonl"

    items = build_items(
        release_root=Path(args.release_root),
        generator_models=args.generator_models,
        task_names=args.task_names,
        samples_per_cell=args.samples_per_cell,
    )
    if args.item_ids:
        wanted = {str(int(item_id)): f"{int(item_id):06d}" for item_id in args.item_ids}
        wanted_ids = set(wanted.values())
        items = [item for item in items if item.item_id in wanted_ids]
        missing = sorted(wanted_ids - {item.item_id for item in items})
        if missing:
            raise ValueError(f"Requested item_ids not found: {missing}")
    if args.limit_items is not None:
        items = items[: args.limit_items]
    if args.start_index is not None or args.end_index is not None:
        start_index = args.start_index or 1
        end_index = args.end_index or len(items)
        if start_index < 1 or end_index < start_index:
            raise ValueError(f"Invalid item range: start={start_index}, end={end_index}")
        items = items[start_index - 1 : end_index]
    if args.num_shards is not None:
        if args.num_shards < 1:
            raise ValueError("--num-shards must be >= 1")
        if args.shard_index is None:
            raise ValueError("--shard-index is required when --num-shards is set")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")
        items = [item for index, item in enumerate(items) if index % args.num_shards == args.shard_index]
    prompt_ids = sorted({item.prompt_id for item in items})
    examples_by_key = build_examples_by_key(
        release_root=Path(args.release_root),
        generator_models=args.generator_models,
        task_names=args.task_names,
        prompt_ids=prompt_ids,
        samples_per_cell=args.samples_per_cell,
        max_examples_per_prompt=args.max_examples_per_prompt,
        min_examples_per_prompt=args.min_examples_per_prompt,
        protocol_version=args.protocol_version,
        support_seed=args.support_seed,
    )

    write_jsonl(manifest_path, [asdict(item) for item in items])
    write_run_config(run_dir, args, items, examples_by_key)

    if args.dump_first_prompt:
        if not items:
            raise RuntimeError("No items available for prompt dump.")
        item = items[0]
        examples_key = f"{item.generator_model}__{item.task_name}__{item.prompt_id}"
        examples = examples_by_key[examples_key]
        prompt_dump = format_prompt_dump(item, examples, build_payload(item, examples, args), args)
        dump_path = Path(args.dump_prompt_path) if args.dump_prompt_path else run_dir / "first_prompt_dump.md"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(prompt_dump + "\n", encoding="utf-8")
        print(prompt_dump)
        print(f"\nPrompt dump written to: {dump_path.as_posix()}", flush=True)
        return

    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned_items": len(items),
                    "cells": len({item.cell_id for item in items}),
                    "by_model": dict(Counter(item.generator_model for item in items)),
                    "by_task": dict(Counter(item.task_name for item in items)),
                    "fewshot_groups": {
                        key: [example.image_id for example in examples] for key, examples in examples_by_key.items()
                    },
                    "run_dir": run_dir.as_posix(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    done = load_done_item_ids(output_path) if args.resume else set()
    endpoint = "/v1/responses" if args.api_format == "responses" else "/v1/chat/completions"
    url = args.api_base.rstrip("/") + endpoint
    start_time = time.time()

    def run_one(index: int, item: InferenceItem) -> tuple[int, dict[str, Any]]:
        examples_key = f"{item.generator_model}__{item.task_name}__{item.prompt_id}"
        examples = examples_by_key[examples_key]
        request_started = time.time()
        row: dict[str, Any] = {
            **asdict(item),
            "judge_model": args.api_model,
            "api_base": args.api_base,
            "api_format": args.api_format,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "fewshot_examples_key": examples_key,
            "fewshot_examples": [asdict(example) for example in examples],
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
            "prediction_count": None,
            "prediction_validity": "",
            "confidence": "",
            "evidence": "",
            "error": "",
            "latency_s": None,
            "usage": {},
        }
        try:
            chat_payload = build_payload(item, examples, args)
            payload = chat_payload
            if args.api_format == "responses":
                payload = chat_payload_to_responses_payload(
                    chat_payload,
                    max_output_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_exclude=args.reasoning_exclude,
                    enable_reasoning=args.enable_thinking,
                )
            status_code, body = run_curl_json(
                url=url,
                payload=payload,
                timeout=args.request_timeout,
                bearer_token=args.bearer_token,
                retries=args.retries,
            )
            if status_code >= 400:
                raise RuntimeError(f"HTTP {status_code}: {body.decode('utf-8', errors='replace')[:500]}")
            artifacts = (
                extract_responses_artifacts(body)
                if args.api_format == "responses"
                else extract_response_artifacts(body)
            )
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
            normalized = normalize_prediction(parsed)
            row.update(
                {
                    "status": "ok",
                    "parsed": parsed,
                    "prediction_count": normalized["count"],
                    "prediction_validity": normalized["validity"],
                    "confidence": normalized["confidence"],
                    "evidence": normalized["evidence"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        finally:
            row["latency_s"] = round(time.time() - request_started, 3)
        return index, row

    pending_items = [(index, item) for index, item in enumerate(items, start=1) if item.item_id not in done]
    total_pending = len(pending_items)
    completed = 0

    def record_row(index: int, row: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        append_jsonl(output_path, row)
        if row["truncation_suspected"]:
            append_jsonl(truncation_path, row)
        if completed % args.log_every == 0 or completed == total_pending:
            elapsed = time.time() - start_time
            print(
                f"[{completed}/{total_pending}] item_index={index}/{len(items)} "
                f"item={row['item_id']} status={row['status']} "
                f"cell={row['cell_id']} pred={row.get('prediction_count')} "
                f"trunc={int(bool(row.get('truncation_suspected')))} "
                f"latency={row['latency_s']}s elapsed={elapsed/60:.1f}m",
                flush=True,
            )

    if args.concurrency == 1:
        for index, item in pending_items:
            record_row(*run_one(index, item))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_one, index, item) for index, item in pending_items]
            for future in as_completed(futures):
                record_row(*future.result())

    summary = summarize(items, output_path)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run few-shot MLLM counting on release image cells.")
    parser.add_argument("--release-root", default="dataset/release")
    parser.add_argument("--generator-models", nargs="+", default=["qwenimage2512", "flux2dev_full"])
    parser.add_argument("--task-names", nargs="+", default=list(TASK_SPECS))
    parser.add_argument("--samples-per-cell", type=int, default=20)
    parser.add_argument("--max-examples-per-prompt", type=int, default=3)
    parser.add_argument("--min-examples-per-prompt", type=int, default=None)
    parser.add_argument("--support-seed", type=int, default=20260503)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-format", choices=["chat", "responses"], default="chat")
    parser.add_argument("--api-model", default="qwen3.5-9b")
    parser.add_argument("--bearer-token", default="")
    parser.add_argument(
        "--bearer-token-env",
        default="",
        help="Read bearer token from this environment variable instead of passing it on the command line.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--protocol-version",
        choices=[
            "count-only",
            "task-guided-count-only",
            "task-guided-reject-count",
            "traffic-light-canonical-count-only",
            "traffic-light-canonical-reject-count",
        ],
        default="task-guided-count-only",
    )
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="Optional OpenRouter/OpenAI reasoning effort passed as reasoning.effort.",
    )
    parser.add_argument(
        "--reasoning-exclude",
        action="store_true",
        help="Request hidden reasoning without returning reasoning text/details when supported.",
    )
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dump-first-prompt",
        action="store_true",
        help="Print the first request prompt with image placeholders, write it to disk, and exit before API calls.",
    )
    parser.add_argument(
        "--dump-prompt-path",
        default="",
        help="Optional output path for --dump-first-prompt. Defaults to <run-dir>/first_prompt_dump.md.",
    )
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument(
        "--item-ids",
        nargs="+",
        default=None,
        help="Run only these manifest item ids, e.g. 80 312 or 000080 000312.",
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
