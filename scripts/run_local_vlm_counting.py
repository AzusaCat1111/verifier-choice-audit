#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from run_mllm_few_shot_counting import (
    FEW_SHOT_USER_PROMPT_TEMPLATE,
    FEW_SHOT_FINAL_INSTRUCTIONS_COUNT_ONLY,
    FEW_SHOT_FINAL_INSTRUCTIONS_REJECT_COUNT,
    FEW_SHOT_USER_PROMPT_TEMPLATE_COUNT_ONLY_PREFIX,
    FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY_PREFIX,
    FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_REJECT_COUNT_PREFIX,
    build_examples_by_key,
    get_few_shot_prompt_template,
)
from run_mllm_zero_shot_counting import (
    SYSTEM_PROMPT,
    TASK_SPECS,
    USER_PROMPT_TEMPLATE,
    append_jsonl,
    build_items,
    detect_truncation_signals,
    format_annotation_protocol,
    get_system_prompt,
    get_task_specs,
    get_zero_prompt_template,
    normalize_prediction,
    parse_json_object,
    strip_markdown_fences,
    utc_timestamp,
    write_jsonl,
)

COUNT_ONLY_FEWSHOT_PROTOCOLS = {
    "count-only",
    "task-guided-count-only",
    "traffic-light-canonical-count-only",
}

VALIDITY_AWARE_FEWSHOT_PROTOCOLS = {
    "task-guided-reject-count",
    "traffic-light-canonical-reject-count",
}

STRUCTURED_FEWSHOT_PROTOCOLS = COUNT_ONLY_FEWSHOT_PROTOCOLS | VALIDITY_AWARE_FEWSHOT_PROTOCOLS


def load_model_and_processor(model_path: str, dtype: str, device_map: str):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", ""))
    architectures = [str(value) for value in getattr(config, "architectures", []) or []]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if model_type == "qwen3_5" or any("Qwen3_5" in value for value in architectures):
        from transformers import Qwen3_5ForConditionalGeneration

        model_cls = Qwen3_5ForConditionalGeneration
    elif model_type == "gemma3n" or any("Gemma3n" in value for value in architectures):
        from transformers import Gemma3nForConditionalGeneration

        model_cls = Gemma3nForConditionalGeneration
    else:
        model_cls = AutoModelForImageTextToText

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    setattr(model, "_local_vlm_model_type", model_type)
    setattr(processor, "_local_vlm_model_type", model_type)
    return model, processor


def model_input_device(model) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cuda"


def build_zero_prompt(task_name: str, protocol_version: str = "plain-count-only") -> str:
    spec = get_task_specs(protocol_version)[task_name]
    return get_zero_prompt_template(protocol_version).format(
        target_units=spec["target_units"],
        ignore=spec["ignore"],
        annotation_protocol=format_annotation_protocol(spec),
    )


def build_fewshot_prompt(task_name: str, protocol_version: str = "plain-count-only") -> str:
    spec = get_task_specs(protocol_version)[task_name]
    if protocol_version == "count-only":
        return FEW_SHOT_USER_PROMPT_TEMPLATE_COUNT_ONLY_PREFIX.format(
            target_units=spec["target_units"],
            ignore=spec["ignore"],
        )
    if protocol_version in {"task-guided-count-only", "traffic-light-canonical-count-only"}:
        return FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY_PREFIX.format(
            target_units=spec["target_units"],
            ignore=spec["ignore"],
            annotation_protocol=format_annotation_protocol(spec),
        )
    if protocol_version in {"task-guided-reject-count", "traffic-light-canonical-reject-count"}:
        return FEW_SHOT_USER_PROMPT_TEMPLATE_TASK_GUIDED_REJECT_COUNT_PREFIX.format(
            target_units=spec["target_units"],
            ignore=spec["ignore"],
            annotation_protocol=format_annotation_protocol(spec),
        )
    if protocol_version != "plain-count-only":
        raise ValueError(f"Unsupported local few-shot protocol version: {protocol_version}")
    return FEW_SHOT_USER_PROMPT_TEMPLATE.format(target_units=spec["target_units"], ignore=spec["ignore"])


def format_fewshot_answer(example: Any, protocol_version: str) -> str:
    if protocol_version in VALIDITY_AWARE_FEWSHOT_PROTOCOLS:
        if getattr(example, "validity_label", "valid") == "reject":
            return '{"validity": "reject", "count": null}'
        return f'{{"validity": "valid", "count": {example.human_count}}}'
    return f'{{"count": {example.human_count}}}'


def fewshot_final_instructions(protocol_version: str) -> str:
    if protocol_version in VALIDITY_AWARE_FEWSHOT_PROTOCOLS:
        return FEW_SHOT_FINAL_INSTRUCTIONS_REJECT_COUNT
    return FEW_SHOT_FINAL_INSTRUCTIONS_COUNT_ONLY


def open_image(path: str, max_image_edge: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_image_edge and max(image.size) > max_image_edge:
        image.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
    return image


def build_raw_prompt_and_images(
    item: Any,
    examples: list[Any],
    mode: str,
    max_image_edge: int | None,
    protocol_version: str,
) -> tuple[str, list[Image.Image]]:
    images: list[Image.Image] = []
    lines: list[str] = [get_system_prompt(protocol_version), ""]
    if mode == "fewshot":
        lines.append(build_fewshot_prompt(item.task_name, protocol_version))
        if protocol_version in STRUCTURED_FEWSHOT_PROTOCOLS:
            for index, example in enumerate(examples, start=1):
                lines.append("")
                lines.append(f"Example {index} image:")
                lines.append("<image_soft_token>")
                images.append(open_image(example.image_path, max_image_edge))
                lines.append(f"Example {index} answer:")
                lines.append(format_fewshot_answer(example, protocol_version))
            lines.append("")
            lines.append("Final query image:")
            lines.append("<image_soft_token>")
            lines.append(fewshot_final_instructions(protocol_version))
        else:
            for index, example in enumerate(examples, start=1):
                lines.append("")
                lines.append("<image_soft_token>")
                lines.append(f"Example {index}: the correct count for this example image is {example.human_count}.")
                images.append(open_image(example.image_path, max_image_edge))
            lines.append("")
            lines.append("<image_soft_token>")
            lines.append("Now count the target units in the final query image. Return only the JSON object.")
    else:
        lines.append("<image_soft_token>")
        lines.append(build_zero_prompt(item.task_name, protocol_version))
    images.append(open_image(item.image_path, max_image_edge))
    return "\n".join(lines), images


def build_chat_prompt_and_images(
    processor: Any,
    item: Any,
    examples: list[Any],
    mode: str,
    max_image_edge: int | None,
    enable_thinking: bool,
    protocol_version: str,
) -> tuple[str, list[Image.Image]]:
    images: list[Image.Image] = []
    content: list[dict[str, str]] = []
    if mode == "fewshot":
        content.append({"type": "text", "text": build_fewshot_prompt(item.task_name, protocol_version)})
        if protocol_version in STRUCTURED_FEWSHOT_PROTOCOLS:
            for index, example in enumerate(examples, start=1):
                content.append({"type": "text", "text": f"Example {index} image:"})
                content.append({"type": "image"})
                images.append(open_image(example.image_path, max_image_edge))
                content.append(
                    {"type": "text", "text": f"Example {index} answer:\n{format_fewshot_answer(example, protocol_version)}"}
                )
            content.append({"type": "text", "text": "Final query image:"})
            content.append({"type": "image"})
            content.append({"type": "text", "text": fewshot_final_instructions(protocol_version)})
        else:
            for index, example in enumerate(examples, start=1):
                content.append({"type": "image"})
                content.append({"type": "text", "text": f"Example {index}: the correct count for this example image is {example.human_count}."})
                images.append(open_image(example.image_path, max_image_edge))
            content.append({"type": "image"})
            content.append({"type": "text", "text": "Now count the target units in the final query image. Return only the JSON object."})
    else:
        content.append({"type": "image"})
        content.append({"type": "text", "text": build_zero_prompt(item.task_name, protocol_version)})
    images.append(open_image(item.image_path, max_image_edge))
    messages = [
        {"role": "system", "content": get_system_prompt(protocol_version)},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt, images


def extract_count_json_text(decoded: str) -> str:
    text = decoded.strip()
    text = re.sub(
        r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    text = strip_markdown_fences(text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    return text


def generate_one(
    model: Any,
    processor: Any,
    item: Any,
    examples: list[Any],
    mode: str,
    max_new_tokens: int,
    max_image_edge: int | None,
    enable_thinking: bool,
    protocol_version: str,
) -> dict[str, Any]:
    import torch

    model_type = str(getattr(model, "_local_vlm_model_type", ""))
    if model_type in {"qwen3_5", "gemma3n"}:
        prompt, images = build_chat_prompt_and_images(
            processor=processor,
            item=item,
            examples=examples,
            mode=mode,
            max_image_edge=max_image_edge,
            enable_thinking=enable_thinking,
            protocol_version=protocol_version,
        )
    else:
        prompt, images = build_raw_prompt_and_images(item, examples, mode, max_image_edge, protocol_version)
    device = model_input_device(model)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = inputs.to(device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0][input_len:]
    decoded = processor.decode(new_tokens, skip_special_tokens=True)
    json_text = extract_count_json_text(decoded)
    parsed = parse_json_object(json_text)
    normalized = normalize_prediction(parsed)
    truncation_signals = detect_truncation_signals({"finish_reason": "stop"}, json_text)
    return {
        "assistant_content": decoded,
        "assistant_json_text": json_text,
        "parsed": parsed,
        "prediction_count": normalized["count"],
        "prediction_validity": normalized["validity"],
        "confidence": normalized["confidence"],
        "evidence": normalized["evidence"],
        "truncation_signals": truncation_signals,
        "truncation_suspected": bool(truncation_signals),
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


def summarize(items: list[Any], output_path: Path) -> dict[str, Any]:
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


def write_run_config(run_dir: Path, args: argparse.Namespace, items: list[Any], examples_by_key: dict[str, list[Any]]) -> None:
    prompt_path = run_dir / "prompt_template.md"
    prompt_path.write_text(
        "\n".join(
            [
                "# Local VLM Counting Prompt",
                "",
                f"Mode: `{args.mode}`",
                f"Model path: `{args.model_path}`",
                "",
                "## System Prompt",
                "",
                "```text",
                get_system_prompt(args.protocol_version),
                "```",
                "",
                "## Zero-Shot User Prompt Template",
                "",
                "```text",
                get_zero_prompt_template(args.protocol_version),
                "```",
                "",
                "## Few-Shot User Prompt Template",
                "",
                "```text",
                get_few_shot_prompt_template(args.protocol_version)
                if args.protocol_version != "reject-count"
                else "",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if examples_by_key:
        (run_dir / "fewshot_examples.json").write_text(
            json.dumps({key: [asdict(example) for example in value] for key, value in examples_by_key.items()}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    config = {
        "created_at_utc": utc_timestamp(),
        "mode": args.mode,
        "model_path": args.model_path,
        "judge_model": args.judge_model,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "max_image_edge": args.max_image_edge,
        "enable_thinking": args.enable_thinking,
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
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_inference(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/mllm_local") / f"{args.judge_model}_{args.mode}_{utc_timestamp()}"
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
    if args.start_index is not None or args.end_index is not None:
        start_index = max(args.start_index or 1, 1)
        end_index = args.end_index or len(items)
        items = items[start_index - 1 : end_index]
    if args.limit_items is not None:
        items = items[: args.limit_items]

    examples_by_key: dict[str, list[Any]] = {}
    if args.mode == "fewshot":
        examples_by_key = build_examples_by_key(
            release_root=Path(args.release_root),
            generator_models=args.generator_models,
            task_names=args.task_names,
            prompt_ids=sorted({item.prompt_id for item in items}),
            samples_per_cell=args.samples_per_cell,
            max_examples_per_prompt=args.max_examples_per_prompt,
            min_examples_per_prompt=args.min_examples_per_prompt,
            protocol_version=args.protocol_version,
            support_seed=args.support_seed,
        )

    write_jsonl(manifest_path, [asdict(item) for item in items])
    write_run_config(run_dir, args, items, examples_by_key)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned_items": len(items),
                    "cells": len({item.cell_id for item in items}),
                    "by_model": dict(Counter(item.generator_model for item in items)),
                    "by_task": dict(Counter(item.task_name for item in items)),
                    "fewshot_groups": {
                        key: [example.image_id for example in value] for key, value in examples_by_key.items()
                    },
                    "run_dir": run_dir.as_posix(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    model, processor = load_model_and_processor(args.model_path, args.dtype, args.device_map)
    done = load_done_item_ids(output_path) if args.resume else set()
    start_time = time.time()
    for index, item in enumerate(items, start=1):
        if item.item_id in done:
            continue
        examples_key = f"{item.generator_model}__{item.task_name}__{item.prompt_id}"
        examples = examples_by_key.get(examples_key, [])
        request_started = time.time()
        row: dict[str, Any] = {
            **asdict(item),
            "judge_model": args.judge_model,
            "model_path": args.model_path,
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "fewshot_examples_key": examples_key if examples else "",
            "fewshot_examples": [asdict(example) for example in examples],
            "status": "error",
            "assistant_content": "",
            "assistant_json_text": "",
            "finish_reason": "stop",
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
            result = generate_one(
                model=model,
                processor=processor,
                item=item,
                examples=examples,
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                max_image_edge=args.max_image_edge,
                enable_thinking=args.enable_thinking,
                protocol_version=args.protocol_version,
            )
            row.update(result)
            row["status"] = "ok"
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
                f"cell={item.cell_id} pred={row.get('prediction_count')} "
                f"trunc={int(bool(row.get('truncation_suspected')))} "
                f"latency={row['latency_s']}s elapsed={elapsed/60:.1f}m",
                flush=True,
            )

    summary = summarize(items, output_path)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local VLM counting on release image cells.")
    parser.add_argument("--release-root", default="dataset/release")
    parser.add_argument("--generator-models", nargs="+", default=["qwenimage2512", "flux2dev_full"])
    parser.add_argument("--task-names", nargs="+", default=list(TASK_SPECS))
    parser.add_argument("--samples-per-cell", type=int, default=20)
    parser.add_argument("--mode", choices=["zero", "fewshot"], default="zero")
    parser.add_argument("--max-examples-per-prompt", type=int, default=3)
    parser.add_argument("--min-examples-per-prompt", type=int, default=None)
    parser.add_argument("--support-seed", type=int, default=20260503)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--judge-model", default="gemma-3n-E4B-it")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-image-edge", type=int, default=384)
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
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument(
        "--item-ids",
        nargs="+",
        default=None,
        help="Run only these manifest item ids, e.g. 41 101 or 000041 000101.",
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
