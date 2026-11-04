#!/usr/bin/env python3
"""Build the support-ablation scoring inputs from the released raw predictions.

Gemma predictions are copied unchanged. For Qwen3.5-4B/9B, the rows whose input
exceeded the 8,192-token context (status "error", six-shot conditions only) are
left out, so `score_results.py --allow-incomplete` drops those condition blocks
whole instead of scoring them as model failures.
"""
import argparse
import json
from pathlib import Path

MODELS = ("gemma3n_e4b", "gemma4_e4b", "qwen35_4b", "qwen35_9b")
OVERFLOW = ("is longer than the model", "context length (8192 tokens)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for model in MODELS:
        source = args.dataset_root / "results/support_ablation/raw_predictions" / model / "predictions.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        dropped = [row for row in rows if row.get("status") != "ok"]
        for row in dropped:
            if not (model.startswith("qwen") and str(row.get("condition_id", "")).startswith("A_count_n6_")
                    and all(part in str(row.get("error", "")) for part in OVERFLOW)):
                raise ValueError(f"Unexpected failed row in {source}: {row.get('run_id')}")
        target = args.out_dir / model / "predictions.jsonl"
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows if row.get("status") == "ok"))
        print(f"{model}: kept {len(rows) - len(dropped)}, left out {len(dropped)} context-overflow rows")


if __name__ == "__main__":
    main()
