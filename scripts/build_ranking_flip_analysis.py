#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


DEFAULT_INPUT = "outputs/mllm_evaluation/scored_rows.csv"
DEFAULT_OUT_DIR = "outputs/ranking_flip_analysis"


TASK_ORDER = [
    "cell_phone_object_count",
    "chair_object_count",
    "traffic_light_object_count",
    "clover_leaf_substructure_count",
    "dresser_substructure_count",
    "power_strip_substructure_count",
    "turbine_substructure_count",
]


def normalize_mode(mode: str) -> str:
    mode = str(mode)
    if mode in {"zero", "zero_shot"}:
        return "zero_shot"
    if mode in {"fewshot", "few_shot"}:
        return "few_shot"
    return mode


def winner(delta: float, eps: float) -> str:
    if abs(delta) <= eps:
        return "tie"
    return "generator_a" if delta > 0 else "generator_b"


def is_flip(human_delta: float, metric_delta: float, eps: float) -> bool:
    if abs(human_delta) <= eps or abs(metric_delta) <= eps:
        return False
    return (human_delta > 0) != (metric_delta > 0)


def pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100 * value:.1f}\\%"


def build_scores(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["mode_norm"] = df["mode"].map(normalize_mode)
    df["verifier_id"] = df["judge_model"].astype(str) + "/" + df["mode_norm"].astype(str)
    df["human_success_on_prediction_subset"] = pd.to_numeric(
        df["human_success_on_prediction_subset"], errors="coerce"
    ).fillna(0)
    df["auto_pass_target_count"] = pd.to_numeric(df["auto_pass_target_count"], errors="coerce").fillna(0)
    df["auto_correct_valid_only"] = (
        (df["validity_label"] == "valid")
        & (df["prediction_status"] == "ok")
        & (pd.to_numeric(df["prediction_count"], errors="coerce") == pd.to_numeric(df["human_count"], errors="coerce"))
    ).astype(int)
    df["valid_countable"] = ((df["validity_label"] == "valid") & df["human_count"].notna()).astype(int)

    group_cols = ["verifier_id", "judge_model", "mode_norm", "generator_model", "task_name"]
    records: list[dict[str, object]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        verifier_id, judge_model, mode_norm, generator_model, task_name = key
        valid_group = group[group["valid_countable"] == 1]
        records.append(
            {
                "verifier_id": verifier_id,
                "judge_model": judge_model,
                "mode": mode_norm,
                "generator_model": generator_model,
                "task_name": task_name,
                "n_images": len(group),
                "n_valid_countable": len(valid_group),
                "human_benchmark_score": group["human_success_on_prediction_subset"].mean(),
                "auto_target_pass": group["auto_pass_target_count"].mean(),
                "count_fidelity": valid_group["auto_correct_valid_only"].mean() if len(valid_group) else np.nan,
            }
        )
    return pd.DataFrame(records)


def build_pairwise_flips(scores: pd.DataFrame, eps: float) -> pd.DataFrame:
    metrics = ["human_benchmark_score", "auto_target_pass", "count_fidelity"]
    rows: list[dict[str, object]] = []
    group_cols = ["verifier_id", "judge_model", "mode", "task_name"]
    for key, group in scores.groupby(group_cols, dropna=False):
        verifier_id, judge_model, mode, task_name = key
        generators = sorted(group["generator_model"].unique())
        if len(generators) < 2:
            continue
        by_gen = group.set_index("generator_model")
        for generator_a, generator_b in itertools.combinations(generators, 2):
            human_a = float(by_gen.loc[generator_a, "human_benchmark_score"])
            human_b = float(by_gen.loc[generator_b, "human_benchmark_score"])
            human_delta = human_a - human_b
            for metric in metrics:
                metric_a = float(by_gen.loc[generator_a, metric])
                metric_b = float(by_gen.loc[generator_b, metric])
                metric_delta = metric_a - metric_b
                missing_metric = math.isnan(metric_a) or math.isnan(metric_b)
                rows.append(
                    {
                        "verifier_id": verifier_id,
                        "judge_model": judge_model,
                        "mode": mode,
                        "task_name": task_name,
                        "metric": metric,
                        "generator_a": generator_a,
                        "generator_b": generator_b,
                        "human_score_a": human_a,
                        "human_score_b": human_b,
                        "human_delta_a_minus_b": human_delta,
                        "metric_score_a": metric_a,
                        "metric_score_b": metric_b,
                        "metric_delta_a_minus_b": metric_delta,
                        "human_winner": winner(human_delta, eps),
                        "metric_winner": "missing" if missing_metric else winner(metric_delta, eps),
                        "ranking_flip": False if missing_metric else is_flip(human_delta, metric_delta, eps),
                        "human_tie": abs(human_delta) <= eps,
                        "metric_tie": False if missing_metric else abs(metric_delta) <= eps,
                        "n_a": int(by_gen.loc[generator_a, "n_images"]),
                        "n_b": int(by_gen.loc[generator_b, "n_images"]),
                        "n_valid_countable_a": int(by_gen.loc[generator_a, "n_valid_countable"]),
                        "n_valid_countable_b": int(by_gen.loc[generator_b, "n_valid_countable"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_by_verifier(pairwise: pd.DataFrame) -> pd.DataFrame:
    usable = pairwise[(pairwise["metric"] != "human_benchmark_score") & (pairwise["metric_winner"] != "missing")]
    rows: list[dict[str, object]] = []
    for (verifier_id, judge_model, mode, metric), group in usable.groupby(
        ["verifier_id", "judge_model", "mode", "metric"], dropna=False
    ):
        evaluated = group[~group["human_tie"] & ~group["metric_tie"]]
        rows.append(
            {
                "verifier_id": verifier_id,
                "judge_model": judge_model,
                "mode": mode,
                "metric": metric,
                "n_pairs": len(group),
                "n_decisive_pairs": len(evaluated),
                "n_flips": int(group["ranking_flip"].sum()),
                "flip_rate": float(group["ranking_flip"].mean()) if len(group) else np.nan,
                "decisive_flip_rate": float(evaluated["ranking_flip"].mean()) if len(evaluated) else np.nan,
                "flipped_tasks": ", ".join(sorted(group.loc[group["ranking_flip"], "task_name"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "verifier_id"])


def write_pairwise_tex(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot_table(
        index=["verifier_id", "judge_model", "mode"],
        columns="metric",
        values=["flip_rate", "n_pairs", "n_flips"],
        aggfunc="first",
    ).reset_index()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Pairwise generator-ranking flip rates. Human benchmark score is the reference ranking. Auto target pass is the main benchmark score; count fidelity is a diagnostic count-readout metric.}",
        r"\label{tab:pairwise-flip-rate}",
        r"\begin{tabular}{l l r r r r}",
        r"\toprule",
        r"Verifier & Mode & Target flips & Target rate & Fidelity flips & Fidelity rate \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        verifier = str(row[("judge_model", "")]) if ("judge_model", "") in pivot.columns else str(row["judge_model"])
        mode = str(row[("mode", "")]) if ("mode", "") in pivot.columns else str(row["mode"])
        def get(metric: str, field: str) -> float:
            key = (field, metric)
            if key in row.index:
                return row[key]
            return np.nan
        target_flips = get("auto_target_pass", "n_flips")
        target_pairs = get("auto_target_pass", "n_pairs")
        target_rate = get("auto_target_pass", "flip_rate")
        fidelity_flips = get("count_fidelity", "n_flips")
        fidelity_pairs = get("count_fidelity", "n_pairs")
        fidelity_rate = get("count_fidelity", "flip_rate")
        target_count = "-" if math.isnan(target_flips) else f"{int(target_flips)}/{int(target_pairs)}"
        fidelity_count = "-" if math.isnan(fidelity_flips) else f"{int(fidelity_flips)}/{int(fidelity_pairs)}"
        lines.append(
            f"{verifier} & {mode} & {target_count} & {pct(target_rate)} & {fidelity_count} & {pct(fidelity_rate)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_heatmap(pairwise: pd.DataFrame, out_path: Path, metric: str = "auto_target_pass") -> None:
    data = pairwise[pairwise["metric"] == metric].copy()
    if data.empty:
        return
    data["value"] = np.where(data["ranking_flip"], 1.0, 0.0)
    data.loc[data["human_tie"] | data["metric_tie"], "value"] = 0.5
    tasks = [task for task in TASK_ORDER if task in set(data["task_name"])]
    tasks += [task for task in sorted(set(data["task_name"])) if task not in tasks]
    verifiers = sorted(data["verifier_id"].unique())
    matrix = np.full((len(verifiers), len(tasks)), np.nan)
    for i, verifier in enumerate(verifiers):
        for j, task in enumerate(tasks):
            subset = data[(data["verifier_id"] == verifier) & (data["task_name"] == task)]
            if len(subset):
                matrix[i, j] = float(subset["value"].iloc[0])

    cmap = ListedColormap(["#d8f3dc", "#fff3b0", "#d00000"])
    plot_matrix = matrix.copy()
    fig_w = max(8, 0.8 * len(tasks) + 3)
    fig_h = max(3, 0.55 * len(verifiers) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(plot_matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([task.replace("_", "\n") for task in tasks], fontsize=8)
    ax.set_yticks(range(len(verifiers)))
    ax.set_yticklabels(verifiers, fontsize=8)
    ax.set_title("Ranking flips vs. human benchmark ranking (auto target pass)")
    for i in range(len(verifiers)):
        for j in range(len(tasks)):
            if np.isnan(matrix[i, j]):
                label = "NA"
            elif matrix[i, j] == 1:
                label = "flip"
            elif matrix[i, j] == 0.5:
                label = "tie"
            else:
                label = "ok"
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["same", "tie", "flip"])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generator-ranking flip analysis from auto scored rows.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.input)
    scores = build_scores(rows)
    pairwise = build_pairwise_flips(scores, eps=args.eps)
    summary = summarize_by_verifier(pairwise)

    scores.to_csv(out_dir / "ranking_scores_by_verifier_task_generator.csv", index=False)
    pairwise.to_csv(out_dir / "ranking_flip_by_task_metric.csv", index=False)
    summary.to_csv(out_dir / "ranking_flip_by_verifier.csv", index=False)
    write_pairwise_tex(summary, out_dir / "pairwise_flip_rate.tex")
    write_heatmap(pairwise, out_dir / "ranking_flip_heatmap.pdf", metric="auto_target_pass")

    readme = [
        "# Ranking Flip Analysis",
        "",
        f"- Input: `{args.input}`",
        "- Reference ranking: human benchmark score on the same prediction subset.",
        "- Main automatic benchmark score: `auto_target_pass`, i.e. verifier predicted count equals target prompt count.",
        "- Diagnostic metric: `count_fidelity`, i.e. verifier predicted count equals human observed count on valid/countable images only.",
        "- A flip means the generator ordering under an automatic metric is opposite to the human benchmark ordering for the same task and generator pair.",
        "",
        "## Artifacts",
        "",
        "- `ranking_scores_by_verifier_task_generator.csv`: aggregated scores before pairwise ranking.",
        "- `ranking_flip_by_task_metric.csv`: task/metric/generator-pair flip records.",
        "- `ranking_flip_by_verifier.csv`: flip-rate summary by verifier and metric.",
        "- `pairwise_flip_rate.tex`: LaTeX table for the paper.",
        "- `ranking_flip_heatmap.pdf`: heatmap for auto target pass flips.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
