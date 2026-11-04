# Structural T2I Verifier Protocol Audit

Code for the NeurIPS 2026 Evaluations and Datasets Track paper
**Verifier Choice is a Benchmark Design Variable: Auditing Structural Counting
Evaluation in Text-to-Image Models** (Shurun Li, Michael Hua Wang, Haibo Zeng,
Long Wang).

- Dataset: <https://huggingface.co/datasets/neurips2026-ed443/audit-release>
- Code: this repository

The dataset contains generated images, human annotations, prompt metadata,
verifier outputs, detector support labels, and scored result tables. Keep the
code and dataset separate so that generated images and model outputs remain
governed by their source model and provider terms.

## Repository layout

- `scripts/`: verifier inference (API and local), evaluation, target-query
  analysis and detector baselines.
- `reproducibility/scripts/`: score-only reproduction, dataset validation, YOLO
  layout, Human-2 context tables and environment capture.
- `reproducibility/spatial/`: scorers for the spatial-relation experiment.
- `reproducibility/support_set/`: input preparation and scorer for the
  support-set ablation.
- `reproducibility/environment/`: reference environments for scoring, analysis
  and local inference.

## Getting the dataset

Place the dataset at `./dataset`, for example:

```sh
git lfs install
git clone https://huggingface.co/datasets/neurips2026-ed443/audit-release dataset
```

The scripts expect:

```text
./dataset/release
./dataset/annotations
./dataset/results
./dataset/detector/yolo
```

`./dataset/annotations` contains the canonical Human-1 table, the Human-2
tables and the Human-3 consensus labels. Detector support labels and split
lists live under `./dataset/detector/yolo`. Outputs are written under
`./outputs` unless another path is provided.

## Reproducing the reported numbers

### Main counting results

The main results use the original-input matrix (5 verifiers x 4 protocols x
2,280 images). Its image-level statistics bundle ships with the dataset and
needs only Python 3.10+ and NumPy:

```sh
cd dataset/results/original_input_matrix/statistics
python -m pip install -r requirements.txt
python score_imagelevel.py --input image_level_rows.csv --output-dir reproduced --n-boot 10000 --seed 20260727
```

The summary tables it reproduces are in
`dataset/results/original_input_matrix/evaluation/`. The dataset README
documents the Human-3 labels, the consensus reference and the supplementary
analyses.

### Earlier open-verifier matrix (score-only)

`reproducibility/scripts/reproduce_open_mllm_scores.py` regenerates the tables
under `dataset/results/open_mllm_matrix/`, the historical configuration in the
paper's appendix, from the released predictions without calling an API or
loading model weights:

```sh
python reproducibility/scripts/reproduce_open_mllm_scores.py \
  --code-root . --dataset-root dataset --output-dir outputs/score-only --n-boot 1000 --seed 20260503
```

With `--n-boot 0` it checks point estimates only. It validates 45,600 scored
rows and compares every regenerated table with the released one.

### Spatial-relation experiment

```sh
python reproducibility/spatial/score_human1_final_autofail.py \
  --manifest dataset/release/spatial_relation/geneval_position_400.jsonl \
  --human1-responses dataset/annotations/spatial_relation/human1_responses.json \
  --human1-row-map dataset/annotations/spatial_relation/human1_row_map.jsonl \
  --predictions dataset/results/spatial_relation/raw_predictions/{gemma3n_e4b,gemma4_e4b,qwen35_4b,qwen35_9b,qwen35_27b}/predictions.jsonl \
  --output-dir outputs/spatial --bootstrap-iterations 5000 --bootstrap-seed 260725
```

`score_human1_final.py` is the scorer frozen before the original-input runs. It
refuses any schema-noncompliant output, while the experiment's scoring rule
counts such outputs as automatic failures; ten Gemma 4 E4B-it outputs are
noncompliant. `score_human1_final_autofail.py` differs only in that check: it
requires every noncompliant row to carry the automatic-failure encoding
(`auto_pass` false, empty `prediction_validity`) and records the count in
`coverage_audit.json`.

### Support-set ablation

```sh
python reproducibility/support_set/prepare_support_inputs.py --dataset-root dataset --out-dir outputs/support_inputs
python reproducibility/support_set/score_results.py \
  --run-dir outputs/support_inputs/gemma3n_e4b --run-dir outputs/support_inputs/gemma4_e4b \
  --run-dir outputs/support_inputs/qwen35_4b --run-dir outputs/support_inputs/qwen35_9b \
  --design-dir dataset/results/support_ablation/design \
  --output-dir outputs/support --bootstraps 1000 --seed 4432509 --allow-incomplete
```

`prepare_support_inputs.py` leaves out the six-shot Qwen3.5-4B/9B rows whose
input exceeded the 8,192-token context, so `--allow-incomplete` drops those
condition blocks whole and reports them as not assessable.

The spatial and support scorers use only the Python standard library. Their
summary tables reproduce `dataset/results/spatial_relation/evaluation/` and
`dataset/results/support_ablation/evaluation/` byte for byte; the support
`scored_rows.csv` differs only in its `source_file` column, which records the
local input path.

### Dataset validation

```sh
python -m pip install -r reproducibility/environment/requirements-analysis.reference.txt
python reproducibility/scripts/validate_artifact.py --dataset-root dataset \
  --deep-results --run-mlcroissant --json-out outputs/dataset_validation.json
```

### YOLO detector baselines

The dataset stores detector images and labels in separate trees.
`reproducibility/scripts/build_yolo_release_datasets.py` builds the
stem-aligned layout that Ultralytics expects:

```sh
python reproducibility/scripts/build_yolo_release_datasets.py \
  --dataset-root dataset --out-root outputs/yolo-data --image-mode symlink
```

Then train and evaluate with `scripts/train_yolo_detector.py` and
`scripts/evaluate_yolo_*.py`. Starting and trained detector weights are not
released, so exact detector numbers depend on obtaining the identified starting
checkpoint and a matching GPU environment.

## Scripts

- `run_mllm_zero_shot_counting.py`: zero-shot count-only and reject/count MLLM
  inference through OpenAI/OpenRouter-compatible `chat/completions` and
  `responses` endpoints (`--api-base`, `--api-model`, `--api-format`).
- `run_mllm_few_shot_counting.py`: same-task, same-prompt-family few-shot MLLM
  inference, including the targeted GPT-5.5/Claude hard-slice runs.
- `run_local_vlm_counting.py`: local open VLM inference with the same prompt
  schemas.
- `run_mllm_zero_shot_vqa_counting.py`: TIFA-like target-query inference.
- `merge_mllm_prediction_slices.py`: merge sharded prediction files.
- `evaluate_mllm_counting.py`: benchmark score, count fidelity, rejection
  metrics, bias, Cohen's kappa and bootstrap confidence intervals.
- `eval_mllm_vqa_counting.py`, `analyze_vqa_*.py`: target-query aggregation
  and diagnostics.
- `build_ranking_flip_analysis.py`, `check_valid_only_score_shift.py`:
  analyses behind the main-paper claims.
- `train_yolo_detector.py`, `build_yolo_object_task_datasets.py`,
  `evaluate_yolo_*.py`, `calibrate_yolo_*.py`, `sweep_yolo_count_threshold.py`:
  detector baseline training, thresholding and evaluation.

Supply service credentials through environment variables, never on command
lines, in config files or in shared logs. The API scripts call HTTP endpoints
with `curl` and `requests`; they do not require the `openai` SDK.

## Citation

```bibtex
@inproceedings{li2026verifier,
  title     = {Verifier Choice is a Benchmark Design Variable: Auditing Structural Counting Evaluation in Text-to-Image Models},
  author    = {Li, Shurun and Wang, Michael Hua and Zeng, Haibo and Wang, Long},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

## License

Code, scripts, and repository documentation are released under the Apache
License 2.0. See `LICENSE`.

AI-generated images and verifier/model outputs are governed by the licenses and
terms of their respective generator, verifier, model, and service providers.
They are redistributed for research and audit use subject to those terms. Users
are responsible for reviewing and complying with the applicable upstream terms
before reuse or redistribution.
