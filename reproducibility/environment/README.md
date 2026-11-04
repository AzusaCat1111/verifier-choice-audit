# Environment strategy

The released `requirements.txt` is an unpinned union of scoring, API inference,
local-VLM inference, plotting, and YOLO dependencies. Installing that file is
unnecessarily heavy for the headline score-only reproduction and does not lock
the historical environment.

Use three tiers:

1. **Score-only:** CPython 3.11; no third-party package. This is the minimum
   path for regenerating the 45,600 scored rows and headline open-MLLM tables.
2. **Analysis and metadata:** install
   `requirements-analysis.reference.txt`. These exact versions passed the local
   dataset-card, Croissant, schema, and analysis import checks on 2026-07-23.
3. **Full local inference/detectors:** start from
   `requirements-inference.reference.txt`, then select the PyTorch build that
   matches the target CUDA driver. Model weights are intentionally external.

The reference pins record the versions that passed these checks. The
historical experiments did not record a lock file or an exported environment.
For any new run, capture a safe whitelist of the environment:

```bash
python reproducibility/scripts/capture_environment.py \
  --output outputs/environment.json
```

The capture script excludes environment variables, credentials, shell history,
executable paths, and local package URLs.

To share an exactly reproducible environment, build and test a
platform-specific lock from a fresh environment and record:

- Python version and OS/architecture;
- exact package versions;
- CUDA, driver, GPU model, and PyTorch build for GPU paths;
- external model identifiers and immutable revisions;
- API provider, model name, call dates, protocol controls, and raw outputs;
- a SHA-256 inventory for model-independent release inputs.
