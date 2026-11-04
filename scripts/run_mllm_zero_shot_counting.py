#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_SPECS = {
    "cell_phone_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": "visible smartphones / cell phones",
        "ignore": "laptops, tablets, remote controls, reflections, logos, text, hands, people, and background clutter",
    },
    "chair_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": "visible chairs",
        "ignore": "sofas, benches, stools, tables, reflections, partial background furniture, people, and clutter",
    },
    "traffic_light_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": "visible traffic lights",
        "ignore": "pedestrian signals, walk signs, street lamps, road signs, vehicles, reflections, and background clutter",
    },
    "clover_leaf_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible leaflets attached to the single main clover leaf cluster",
        "ignore": "other plants, background leaves, stems, soil, pot edges, shadows, and decorative symbols",
    },
    "dresser_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible drawers in the single dresser cabinet",
        "ignore": "drawer handles, cabinet panels that are not drawers, other furniture, reflections, and background clutter",
    },
    "power_strip_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible electrical outlets / sockets on the single power strip",
        "ignore": "plugs, cables, wall sockets, USB ports unless they are clearly the same outlet type, adapters, logos, and text",
    },
    "turbine_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible blades attached to the main wind turbine rotor",
        "ignore": "the tower, hub, background turbines, partial unrelated rotors, clouds, text, and logos",
    },
}


TASK_SPECS = {
    "cell_phone_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": "visible smartphones or cell phones",
        "ignore": "laptops, tablets, remote controls, reflections, logos, text, hands, people, and background clutter",
        "counting_notes": (
            "Count each visible phone as one unit. A phone counts whether the screen is on or off."
        ),
        "reject_if": (
            "Reject if a visible candidate may be a phone but cannot be confidently identified as one. "
            "Reject if a phone-like object is too cropped, occluded, or fused with another object such that the number of phones is unclear."
        ),
    },
    "chair_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": "visible chairs",
        "ignore": "sofas, benches, stools, tables, reflections, partial background furniture, people, and background clutter",
        "counting_notes": (
            "Count each visible chair as one unit. Do not count stools, benches, or sofas as chairs."
        ),
        "reject_if": (
            "Reject if a visible furniture item may be a chair but cannot be confidently distinguished from non-chair furniture. "
            "Reject if heavy cropping, abnormal overlap, or fusion makes the number of chairs unclear."
        ),
    },
    "traffic_light_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": "visible traffic-light devices",
        "ignore": "individual colored bulbs within the same signal head, pedestrian signals, walk signs, street lamps, road signs, vehicles, reflections, and background clutter",
        "counting_notes": (
            "Count traffic-light devices, not individual colored bulbs. If several lamps clearly belong to one traffic-light device, "
            "count them as one unit."
        ),
        "reject_if": (
            "Reject if the image contains multi-head or clustered traffic-light structures that cannot be reliably mapped to distinct "
            "traffic-light devices. Reject if there is a suspicious ground-level, fallen, detached, or malformed traffic-light-like "
            "object whose status as a countable traffic light is unclear. Reject if a visible traffic-light head/device is too cropped, "
            "fused, or occluded to determine whether it is one separate traffic light."
        ),
    },
    "clover_leaf_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible leaflets attached to the single main clover leaf cluster",
        "ignore": "other plants, background leaves, stems, soil, pot edges, shadows, decorative symbols, and unrelated leaves",
        "counting_notes": (
            "Count only leaflets attached to the single main clover cluster. Count leaflets, not stems."
        ),
        "reject_if": (
            "Reject if there is no single clear main clover cluster. Reject if any other clover leaf or clover-like leaf cluster is visible, "
            "even if it appears in the background. Reject if nearby grass or plant leaves could plausibly be interpreted as belonging to "
            "the main clover cluster. Reject if leaflet boundaries are fused, heavily overlapped, or unclear so that the number of attached "
            "leaflets cannot be determined."
        ),
    },
    "dresser_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible drawer fronts in the single main dresser cabinet",
        "ignore": "drawer handles, knobs, cabinet panels that are not drawers, shelves, other furniture, reflections, and background clutter",
        "counting_notes": (
            "Count each visible drawer front as one unit. If one row contains multiple separate drawers, count each drawer separately."
        ),
        "reject_if": (
            "Reject if there is no single clear main dresser cabinet. Reject if visible panels may or may not be drawers and a reasonable "
            "annotator could disagree. Reject if cropping, fusion, or abnormal structure makes the number of drawers unclear."
        ),
    },
    "power_strip_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible electrical outlets or sockets on the single main power strip",
        "ignore": "plugs, cables, wall sockets, USB ports, switches, screws, adapters, logos, and text",
        "counting_notes": (
            "Count only outlet/socket units on the main power strip. Do not count switches or USB ports as outlets."
        ),
        "reject_if": (
            "Reject if there is no single clear main power strip. Reject if a candidate socket does not resemble a realistic outlet "
            "structure closely enough for a reasonable annotator to decide whether it is an outlet. Reject if outlet boundaries are "
            "malformed, fused, or unclear so that the number of outlets cannot be determined."
        ),
    },
    "turbine_substructure_count": {
        "task_type": "substructure",
        "count_key": "c",
        "target_units": "visible blades attached to the main wind turbine rotor",
        "ignore": "the tower, hub, background turbines, unrelated rotors, clouds, text, logos, and shadows",
        "counting_notes": (
            "Count only blades attached to the single main turbine rotor. Do not count blades from background turbines."
        ),
        "reject_if": (
            "Reject if there is no single clear main turbine rotor. Reject if blades are close to overlapping, fused, or aligned in a "
            "way that makes the number of separate blades unclear. Reject if a background turbine or unrelated rotor could plausibly "
            "be counted as part of the main rotor."
        ),
    },
}


TASK_SPECS_TL_CANONICAL = {
    **TASK_SPECS,
    "traffic_light_object_count": {
        "task_type": "object_count",
        "count_key": "k",
        "target_units": (
            "foreground canonical traffic-light assemblies: each countable unit must be one upright pole "
            "with exactly one clear standard vertical signal head"
        ),
        "ignore": (
            "individual colored bulbs inside one signal head, background signals that are clearly a different type "
            "from the foreground canonical units, horizontal/overhead intersection signal heads in the background, "
            "pedestrian signals, walk signs, countdown signals, arrow signals, very blurry or unidentifiable distant "
            "signal-like shapes, street lamps, road signs, vehicles, reflections, text, logos, and general street clutter"
        ),
        "counting_notes": (
            "This task follows the clean T2I prompt scope. Count only foreground traffic lights that match the canonical form: "
            "one fully visible upright pole with one standard vertical signal head, clearly separated from other traffic lights. "
            "Do not count individual colored bulbs. Background signal lights that are clearly a different type from the foreground "
            "canonical units should be treated as valid background clutter and not counted. For example, if the foreground target "
            "units are upright pole-mounted vertical signal heads, a clear horizontal or overhead vehicle signal in the background "
            "does not make the image invalid and should not be counted. Very blurry, tiny, or unidentifiable distant signal-like "
            "shapes also do not make the image invalid and should not be counted."
        ),
        "reject_if": (
            "Reject if any visible traffic-light-like candidate would make the clean one-pole-one-head foreground scope ambiguous "
            "or violated. Reject if a foreground pole/support has multiple signal heads, if a second foreground head is visible "
            "from the side or behind the main head, or if a traffic light appears floating, fallen, detached, malformed, fused, "
            "heavily cropped, or occluded enough that the clean counting scope is not reliable. For background signals, reject only "
            "when there is a clearly visible extra traffic light that is similar to the foreground canonical units and could reasonably "
            "be counted as another upright pole-mounted vertical signal head. Do not reject merely because the background contains a "
            "different signal type, such as a horizontal/overhead intersection signal or pedestrian signal, or because a distant "
            "signal-like shape is too blurry or small to identify; in those cases, treat it as background clutter and do not count it."
        ),
    },
}


GENERAL_REJECT_RULES = """General reject rules:
1. Reject if there is a visible candidate that may be a target unit, but a reasonable annotator cannot confidently decide whether it should count.
2. Reject if a target candidate is so cropped or truncated that its inclusion/exclusion could change the count and cannot be determined confidently. As a rough heuristic, if only a small fragment, about one-third or less, is visible and this could change the count, reject.
3. Reject if targets or substructures appear fused, abnormally overlapped, or implausibly occluded, so that the number of countable units is unclear.
4. Reject if the image does not provide a single clear counting scope required by the task, such as no clear main parent object or multiple competing parents when the task expects one main object."""


def format_annotation_protocol(task_spec: dict[str, str]) -> str:
    if "annotation_protocol" in task_spec:
        return str(task_spec["annotation_protocol"])
    return (
        f"{GENERAL_REJECT_RULES}\n\n"
        f"Counting notes:\n{task_spec.get('counting_notes', '').strip()}\n\n"
        f"Reject if:\n{task_spec.get('reject_if', '').strip()}"
    ).strip()


SYSTEM_PROMPT = (
    "You are a visual counting assistant for a generated-image benchmark. "
    "If thinking is enabled by the chat template, keep the hidden reasoning concise. "
    "Always return exactly one valid JSON object and no extra text in the final answer."
)


SYSTEM_PROMPT = """You are a visual counting verifier for a generated-image benchmark.

Always return exactly one valid JSON object and no extra text.
Do not write step-by-step reasoning in the final answer.
If hidden reasoning is enabled by the chat template, keep it concise."""


USER_PROMPT_TEMPLATE = """You are evaluating a generated image for a counting benchmark.

Count only the requested visible target units.
Target units to count: {target_units}.
Ignore: {ignore}.

Reason silently if needed, but do not write step-by-step reasoning in the final answer.
If hidden reasoning is enabled, keep it concise.
Return exactly one valid JSON object with these keys:
{{
  "count": <integer>,
  "confidence": "low" | "medium" | "high",
  "evidence": "<one short sentence>"
}}"""


USER_PROMPT_TEMPLATE_COUNT_ONLY = """You are evaluating a generated image for a structural counting benchmark.

This is a count-only protocol. Do not reject the image. Return your best visual count.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

Rules:
1. Count only visible target units in the query image.
2. Do not infer the count from common sense, typical object shapes, or likely object defaults.
3. Do not use any desired or requested count; no target count is provided.
4. If the image is unclear, still return your best visual count as an integer.

Return exactly one valid JSON object:
{{"count": <integer>}}"""


VALIDITY_AWARE_USER_PROMPT_TEMPLATE = """You are evaluating a generated image for a structural counting benchmark.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

First decide whether the image is valid for this counting task.

Validity labels:
- "valid": the image has a clear counting scope and the target units can be counted.
- "reject": the image should not be counted because the target is missing, the wrong object is shown, the image is unclear, or the count is not reliable because of blur, occlusion, cropping, merged units, malformed structure, unclear boundaries, or unclear counting scope.

Rules:
1. If validity is "valid", count only visible target units.
2. If validity is "reject", set count to null.
3. Do not infer the count from common sense, typical object shapes, or likely object defaults.
4. Do not use any desired or requested count; no target count is provided.
5. If you are not confident that the target units can be counted reliably, choose "reject".

Return exactly one valid JSON object:
{{
  "validity": "valid" | "reject",
  "count": <integer or null>
}}"""


USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY = """You are evaluating a generated image for a structural counting benchmark.

This is a count-only protocol. Do not reject the image. Return your best visual count.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

Human annotation protocol for this task:
{annotation_protocol}

The word "Reject" in the annotation protocol describes human audit behavior. In this count-only protocol, do not output reject; use those cases only as warnings about unreliable counting scope.

Rules:
1. Count only visible target units in the query image.
2. Use the human annotation protocol only to define the counting scope.
3. Do not infer the count from common sense, typical object shapes, or likely object defaults.
4. Do not use any desired or requested count; no target count is provided.
5. If the image would be rejected under the annotation protocol, still return your best visual count as an integer.

Return exactly one valid JSON object:
{{"count": <integer>}}"""


VALIDITY_AWARE_USER_PROMPT_TEMPLATE_TASK_GUIDED = """You are evaluating a generated image for a structural counting benchmark.

Count only the requested visible target units.
Target units to count: {target_units}
Do not count / ignore: {ignore}

Human annotation protocol for this task:
{annotation_protocol}

First decide whether the image is valid for this counting task.

Validity labels:
- "valid": the image has a clear counting scope and the target units can be counted according to the human annotation protocol.
- "reject": the image should not be counted because the target is missing, the wrong object is shown, the image violates the human annotation protocol, or the count is not reliable because of blur, occlusion, cropping, merged units, malformed structure, unclear boundaries, or unclear counting scope.

Rules:
1. If validity is "valid", count only visible target units.
2. If validity is "reject", set count to null.
3. Do not infer the count from common sense, typical object shapes, or likely object defaults.
4. Do not use any desired or requested count; no target count is provided.
5. If you are not confident that the target units can be counted reliably, choose "reject".

Return exactly one valid JSON object:
{{
  "validity": "valid" | "reject",
  "count": <integer or null>
}}"""


def get_task_specs(protocol_version: str) -> dict[str, dict[str, str]]:
    if protocol_version == "plain-count-only":
        return TASK_SPECS
    if protocol_version in {
        "traffic-light-canonical-count-only",
        "traffic-light-canonical-reject-count",
    }:
        return TASK_SPECS_TL_CANONICAL
    if protocol_version in {
        "count-only",
        "reject-count",
        "task-guided-count-only",
        "task-guided-reject-count",
    }:
        return TASK_SPECS
    raise ValueError(f"Unknown protocol version: {protocol_version}")


def get_system_prompt(protocol_version: str) -> str:
    if protocol_version == "plain-count-only":
        return SYSTEM_PROMPT
    if protocol_version in {
        "count-only",
        "reject-count",
        "task-guided-count-only",
        "task-guided-reject-count",
        "traffic-light-canonical-count-only",
        "traffic-light-canonical-reject-count",
    }:
        return SYSTEM_PROMPT
    raise ValueError(f"Unknown protocol version: {protocol_version}")


def get_zero_prompt_template(protocol_version: str) -> str:
    if protocol_version == "plain-count-only":
        return USER_PROMPT_TEMPLATE
    if protocol_version == "count-only":
        return USER_PROMPT_TEMPLATE_COUNT_ONLY
    if protocol_version == "reject-count":
        return VALIDITY_AWARE_USER_PROMPT_TEMPLATE
    if protocol_version == "task-guided-count-only":
        return USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY
    if protocol_version == "task-guided-reject-count":
        return VALIDITY_AWARE_USER_PROMPT_TEMPLATE_TASK_GUIDED
    if protocol_version == "traffic-light-canonical-count-only":
        return USER_PROMPT_TEMPLATE_TASK_GUIDED_COUNT_ONLY
    if protocol_version == "traffic-light-canonical-reject-count":
        return VALIDITY_AWARE_USER_PROMPT_TEMPLATE_TASK_GUIDED
    raise ValueError(f"Unknown protocol version: {protocol_version}")


@dataclass(frozen=True)
class InferenceItem:
    item_id: str
    generator_model: str
    task_name: str
    task_type: str
    prompt_id: str
    count_key: str
    target_count: int
    cell_id: str
    cell_sample_index: int
    image_path: str
    image_rel_path: str
    annotation_path: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_count_dir(path: Path) -> tuple[str, int] | None:
    match = re.fullmatch(r"([kc])_(\d+)", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def build_items(
    release_root: Path,
    generator_models: list[str],
    task_names: list[str],
    samples_per_cell: int,
) -> list[InferenceItem]:
    items: list[InferenceItem] = []
    for generator_model in generator_models:
        annotation_root = release_root / generator_model / "annotation_datasets"
        if not annotation_root.exists():
            raise FileNotFoundError(f"Missing annotation root: {annotation_root}")
        for task_name in task_names:
            if task_name not in TASK_SPECS:
                raise ValueError(f"Unknown task: {task_name}")
            task_root = annotation_root / task_name / task_name
            if not task_root.exists():
                raise FileNotFoundError(f"Missing task root: {task_root}")
            task_spec = TASK_SPECS[task_name]
            count_dirs = sorted(
                path
                for path in task_root.iterdir()
                if path.is_dir() and parse_count_dir(path) is not None
            )
            for count_dir in count_dirs:
                parsed = parse_count_dir(count_dir)
                assert parsed is not None
                count_key, target_count = parsed
                if count_key != task_spec["count_key"]:
                    raise ValueError(f"Unexpected count key in {count_dir}; expected {task_spec['count_key']}")
                for prompt_id in ("p1", "p2", "p3"):
                    prompt_dir = count_dir / prompt_id
                    image_paths = sorted(prompt_dir.glob("*.png"))
                    if len(image_paths) < samples_per_cell:
                        raise ValueError(
                            f"Cell has only {len(image_paths)} images, need {samples_per_cell}: {prompt_dir}"
                        )
                    cell_id = f"{generator_model}__{task_name}__{prompt_id}__{count_key}{target_count}"
                    for sample_index, image_path in enumerate(image_paths[:samples_per_cell], start=1):
                        item_id = f"{len(items) + 1:06d}"
                        items.append(
                            InferenceItem(
                                item_id=item_id,
                                generator_model=generator_model,
                                task_name=task_name,
                                task_type=task_spec["task_type"],
                                prompt_id=prompt_id,
                                count_key=count_key,
                                target_count=target_count,
                                cell_id=cell_id,
                                cell_sample_index=sample_index,
                                image_path=image_path.resolve().as_posix(),
                                image_rel_path=image_path.relative_to(release_root).as_posix(),
                                annotation_path=image_path.with_suffix(".json").resolve().as_posix(),
                            )
                        )
    return items


def build_prompt(task_name: str, protocol_version: str = "plain-count-only") -> str:
    task_spec = get_task_specs(protocol_version)[task_name]
    return get_zero_prompt_template(protocol_version).format(
        target_units=task_spec["target_units"],
        ignore=task_spec["ignore"],
        annotation_protocol=format_annotation_protocol(task_spec),
    )


def image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_payload(item: InferenceItem, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.api_model,
        "messages": [
            {"role": "system", "content": get_system_prompt(args.protocol_version)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(item.task_name, args.protocol_version)},
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
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
        if getattr(args, "reasoning_exclude", False):
            payload["reasoning"]["exclude"] = True
        payload["include_reasoning"] = True
    return payload


def chat_content_to_responses_content(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI chat multimodal content into OpenResponses content items."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": flatten_text_content(content)}]

    converted: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            converted.append({"type": "input_text", "text": part})
            continue
        if not isinstance(part, dict):
            converted.append({"type": "input_text", "text": flatten_text_content(part)})
            continue
        part_type = part.get("type")
        if part_type == "text":
            converted.append({"type": "input_text", "text": str(part.get("text") or "")})
        elif part_type == "image_url":
            image_url = part.get("image_url") or {}
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            converted.append({"type": "input_image", "image_url": str(url or "")})
        else:
            converted.append({"type": "input_text", "text": flatten_text_content(part)})
    return converted


def chat_payload_to_responses_payload(
    chat_payload: dict[str, Any],
    *,
    max_output_tokens: int,
    reasoning_effort: str = "",
    reasoning_exclude: bool = False,
    enable_reasoning: bool = False,
) -> dict[str, Any]:
    """Convert our chat-completions payload to OpenRouter Responses API shape."""
    input_messages: list[dict[str, Any]] = []
    instructions: str | None = None
    for message in chat_payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        if role == "system":
            instructions = flatten_text_content(content)
            continue
        input_messages.append({"role": role, "content": chat_content_to_responses_content(content)})

    payload: dict[str, Any] = {
        "model": chat_payload["model"],
        "input": input_messages,
        "temperature": chat_payload.get("temperature", 0.0),
        "top_p": chat_payload.get("top_p", 1.0),
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    if instructions:
        payload["instructions"] = instructions
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
        if reasoning_exclude:
            payload["reasoning"]["exclude"] = True
    elif enable_reasoning:
        payload["reasoning"] = {"enabled": True}
        if reasoning_exclude:
            payload["reasoning"]["exclude"] = True
    return payload


def run_curl_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    bearer_token: str,
    retries: int,
) -> tuple[int, bytes]:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        str(int(timeout)),
        "--output",
        "-",
        "--write-out",
        "\n%{http_code}",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {bearer_token}",
        "--data-binary",
        "@-",
        url,
    ]
    last_error: Exception | None = None
    stdin = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(command, input=stdin, check=True, capture_output=True)
            body, _, status_text = result.stdout.rpartition(b"\n")
            return int(status_text.decode("utf-8").strip()), body
        except (subprocess.CalledProcessError, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"curl request failed after {retries} attempts for {url}") from last_error


def flatten_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "value"):
                    if key in item:
                        parts.append(flatten_text_content(item[key]))
                        break
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return flatten_text_content(value[key])
    return str(value)


def extract_tagged_reasoning(content: str) -> tuple[str, str]:
    think_chunks = re.findall(
        r"<(?:think|thinking|reasoning)>\s*(.*?)\s*</(?:think|thinking|reasoning)>",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    think_text = "\n\n".join(chunk.strip() for chunk in think_chunks if chunk.strip())
    content_wo_think = re.sub(
        r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return think_text, content_wo_think


def strip_markdown_fences(text: str) -> str:
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def extract_response_artifacts(response_body: bytes) -> dict[str, Any]:
    response = json.loads(response_body.decode("utf-8"))
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_raw = message.get("content")
    content_text = flatten_text_content(content_raw)
    think_text, content_without_think = extract_tagged_reasoning(content_text)
    reasoning_fields: dict[str, Any] = {}
    for container_name, container in (("message", message), ("choice", choice)):
        if not isinstance(container, dict):
            continue
        for key in ("reasoning", "reasoning_content", "thinking", "analysis", "thought"):
            if key in container and container[key] not in (None, "", []):
                reasoning_fields[f"{container_name}.{key}"] = container[key]
    reasoning_text = "\n\n".join(
        flatten_text_content(value).strip()
        for value in reasoning_fields.values()
        if flatten_text_content(value).strip()
    )
    json_text = strip_markdown_fences(content_without_think)
    return {
        "response_json": response,
        "choice": choice,
        "assistant_message": message,
        "assistant_content": content_text,
        "assistant_json_text": json_text,
        "assistant_think_text": think_text,
        "assistant_reasoning_fields": reasoning_fields,
        "assistant_reasoning_text": reasoning_text,
        "usage": response.get("usage") or {},
    }


def extract_responses_artifacts(response_body: bytes) -> dict[str, Any]:
    """Extract comparable artifacts from OpenRouter/OpenResponses API output."""
    response = json.loads(response_body.decode("utf-8"))
    output_text = response.get("output_text")
    if not output_text:
        parts: list[str] = []
        for output in response.get("output") or []:
            if not isinstance(output, dict):
                continue
            for content in output.get("content") or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text") or ""))
        output_text = "\n".join(part for part in parts if part).strip()

    reasoning_fields: dict[str, Any] = {}
    if response.get("reasoning") not in (None, "", [], {}):
        reasoning_fields["response.reasoning"] = response["reasoning"]
    reasoning_outputs = [
        output
        for output in response.get("output") or []
        if isinstance(output, dict) and str(output.get("type") or "").startswith("reasoning")
    ]
    if reasoning_outputs:
        reasoning_fields["response.output.reasoning"] = reasoning_outputs

    reasoning_text = "\n\n".join(
        flatten_text_content(value).strip()
        for value in reasoning_fields.values()
        if flatten_text_content(value).strip()
    )
    status = str(response.get("status") or "")
    incomplete_reason = ""
    incomplete_details = response.get("incomplete_details") or {}
    if isinstance(incomplete_details, dict):
        incomplete_reason = str(incomplete_details.get("reason") or "")
    choice = {
        "finish_reason": "stop" if status in {"completed", ""} else status,
        "stop_reason": incomplete_reason,
    }
    return {
        "response_json": response,
        "choice": choice,
        "assistant_message": {"role": "assistant", "content": output_text},
        "assistant_content": output_text,
        "assistant_json_text": strip_markdown_fences(output_text or ""),
        "assistant_think_text": "",
        "assistant_reasoning_fields": reasoning_fields,
        "assistant_reasoning_text": reasoning_text,
        "usage": response.get("usage") or {},
    }


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}
        if not parsed:
            recovered: dict[str, Any] = {}
            count_match = re.search(r'"count"\s*:\s*(-?\d+)', content)
            if count_match:
                recovered["count"] = int(count_match.group(1))
            confidence_match = re.search(r'"confidence"\s*:\s*("([^"]+)"|([0-9]*\.?[0-9]+))', content)
            if confidence_match:
                recovered["confidence"] = confidence_match.group(2) or confidence_match.group(3)
            evidence_match = re.search(r'"evidence"\s*:\s*(\[.*?\]|"[^"]*)', content, flags=re.DOTALL)
            if evidence_match:
                evidence_text = evidence_match.group(1).strip()
                if evidence_text.startswith("["):
                    recovered["evidence"] = evidence_text
                else:
                    recovered["evidence"] = evidence_text.strip('"')
            if not recovered:
                raise
            parsed = recovered
    if not isinstance(parsed, dict):
        raise ValueError("Assistant content was not a JSON object")
    return parsed


def detect_truncation_signals(choice: dict[str, Any], assistant_json_text: str) -> list[str]:
    signals: list[str] = []
    finish_reason = str(choice.get("finish_reason") or "")
    stop_reason = str(choice.get("stop_reason") or "")
    if finish_reason and finish_reason != "stop":
        signals.append(f"finish_reason:{finish_reason}")
    if stop_reason and stop_reason not in {"", "null", "stop"}:
        signals.append(f"stop_reason:{stop_reason}")
    if assistant_json_text:
        if assistant_json_text.count("{") > assistant_json_text.count("}"):
            signals.append("unbalanced_json_braces")
        if assistant_json_text.count("[") > assistant_json_text.count("]"):
            signals.append("unbalanced_json_brackets")
    return signals


def normalize_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    raw_count = parsed.get("count")
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        count = None
    confidence_raw = parsed.get("confidence")
    confidence = ""
    if isinstance(confidence_raw, str):
        confidence = confidence_raw.strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = ""
    elif isinstance(confidence_raw, (int, float)):
        value = float(confidence_raw)
        if value >= 0.8:
            confidence = "high"
        elif value >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

    evidence_raw = parsed.get("evidence")
    if isinstance(evidence_raw, str):
        evidence = evidence_raw
    elif isinstance(evidence_raw, list):
        evidence = "; ".join(str(item).strip() for item in evidence_raw if str(item).strip())
    else:
        evidence = ""
    validity_raw = parsed.get("validity")
    validity = ""
    if isinstance(validity_raw, str):
        validity = validity_raw.strip().lower()
        if validity not in {"valid", "reject"}:
            validity = ""
    return {
        "count": count,
        "validity": validity,
        "confidence": confidence,
        "evidence": evidence.strip(),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def write_run_config(run_dir: Path, args: argparse.Namespace, items: list[InferenceItem]) -> None:
    prompt_path = run_dir / "prompt_template.md"
    prompt_path.write_text(
        "\n".join(
            [
                "# MLLM Zero-Shot Counting Prompt",
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
                get_zero_prompt_template(args.protocol_version),
                "```",
            ]
        )
        + "\n",
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
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_exclude": args.reasoning_exclude,
        "protocol_version": args.protocol_version,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "samples_per_cell": args.samples_per_cell,
        "release_root": args.release_root,
        "generator_models": args.generator_models,
        "task_names": args.task_names,
        "item_count": len(items),
        "prompt_template_path": prompt_path.as_posix(),
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


def run_inference(args: argparse.Namespace) -> None:
    if not args.bearer_token and args.bearer_token_env:
        args.bearer_token = os.environ.get(args.bearer_token_env, "")
    if not args.dry_run and not args.api_base:
        raise ValueError("Provide --api-base for API calls.")
    if not args.dry_run and not args.bearer_token:
        raise ValueError("Provide --bearer-token or --bearer-token-env for API calls.")
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/mllm_zero_shot") / f"zero_shot_{utc_timestamp()}"
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
    if args.start_index is not None or args.end_index is not None:
        start_index = max(args.start_index or 1, 1)
        end_index = args.end_index or len(items)
        if end_index < start_index:
            raise ValueError(f"end_index {end_index} must be >= start_index {start_index}")
        items = items[start_index - 1 : end_index]
    if args.limit_items is not None:
        items = items[: args.limit_items]

    write_jsonl(manifest_path, [asdict(item) for item in items])
    write_run_config(run_dir, args, items)

    if args.dry_run:
        summary = {
            "planned_items": len(items),
            "cells": len({item.cell_id for item in items}),
            "by_model": dict(Counter(item.generator_model for item in items)),
            "by_task": dict(Counter(item.task_name for item in items)),
            "run_dir": run_dir.as_posix(),
        }
        print(json.dumps(summary, indent=2))
        return

    done = load_done_item_ids(output_path) if args.resume else set()
    endpoint = "/v1/responses" if args.api_format == "responses" else "/v1/chat/completions"
    url = args.api_base.rstrip("/") + endpoint
    start_time = time.time()
    for index, item in enumerate(items, start=1):
        if item.item_id in done:
            continue
        request_started = time.time()
        row: dict[str, Any] = {
            **asdict(item),
            "judge_model": args.api_model,
            "api_base": args.api_base,
            "api_format": args.api_format,
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
            "prediction_count": None,
            "prediction_validity": "",
            "confidence": "",
            "evidence": "",
            "error": "",
            "latency_s": None,
            "usage": {},
        }
        try:
            chat_payload = build_payload(item, args)
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
        except Exception as exc:  # noqa: BLE001 - keep batch running and record errors
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
    parser = argparse.ArgumentParser(description="Run zero-shot MLLM counting on release image cells.")
    parser.add_argument("--release-root", default="dataset/release")
    parser.add_argument("--generator-models", nargs="+", default=["qwenimage2512", "flux2dev_full"])
    parser.add_argument("--task-names", nargs="+", default=list(TASK_SPECS))
    parser.add_argument("--samples-per-cell", type=int, default=20)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-format", choices=["chat", "responses"], default="chat")
    parser.add_argument("--api-model", default="qwen3.5-27b")
    parser.add_argument("--bearer-token", default="")
    parser.add_argument(
        "--bearer-token-env",
        default="",
        help="Read bearer token from this environment variable instead of passing it on the command line.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--protocol-version",
        choices=[
            "plain-count-only",
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
    parser.set_defaults(enable_thinking=True)
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
