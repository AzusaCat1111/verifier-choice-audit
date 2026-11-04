#!/usr/bin/env python3
"""Capture a small, path- and credential-safe reproducibility environment record."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "Pillow",
    "tqdm",
    "requests",
    "ultralytics",
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "mlcroissant",
    "PyYAML",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
        "note": (
            "This whitelist intentionally omits environment variables, command "
            "history, executable paths, local package URLs, and credentials."
        ),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
