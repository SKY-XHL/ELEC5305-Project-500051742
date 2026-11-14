"""Create final statistical summaries and report-ready figures for the project.

This script uses only saved experiment outputs.  It does not retrain any model
or re-extract any feature.  The predeclared primary zero-shot prompt is always
prompt_1; prompt_2 and prompt_3 are reported only as sensitivity analyses.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BUDGET_ORDER = ["1", "2", "5", "10", "20", "full"]
BUDGET_TO_LABELS = {"1": 1, "2": 2, "5": 5, "10": 10, "20": 20, "full": 32}
REPRESENTATION_ORDER = ["MFCC", "Log-Mel", "CLAP"]
PRIMARY_PROMPT_ID = "prompt_1"

COLORS = {
    "MFCC": "#4C78A8",
    "Log-Mel": "#F58518",
    "CLAP": "#2CA02C",
    "Zero-shot CLAP": "#D62728",
}

COARSE_NAMES = {
    0: "Animals",
    1: "Natural soundscapes & water",
    2: "Human non-speech sounds",
    3: "Interior & domestic sounds",
    4: "Exterior & urban noises",
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")


def load_inputs() -> dict[str, object]:
    paths = {
        "summary": RESULTS_DIR / "label_efficiency_summary.csv",
        "raw": RESULTS_DIR / "label_efficiency_raw.csv",
        "predictions": RESULTS_DIR / "label_efficiency_predictions.csv.gz",
        "zero_summary": RESULTS_DIR / "clap_zero_shot_summary.csv",
        "zero_predictions": RESULTS_DIR / "clap_zero_shot_predictions.csv",
        "label_metadata": RESULTS_DIR / "label_efficiency_metadata.json",
        "embedding_summary": RESULTS_DIR / "clap_embedding_summary.json",
    }
    for path in paths.values():
        require_file(path)

    data: dict[str, object] = {
        "summary": pd.read_csv(paths["summary"], dtype={"labels_per_class": str}),
        "raw": pd.read_csv(paths["raw"], dtype={"labels_per_class": str}),
        "predictions": pd.read_csv(
            paths["predictions"], dtype={"labels_per_class": str}
        ),
        "zero_summary": pd.read_csv(paths["zero_summary"]),
        "zero_predictions": pd.read_csv(paths["zero_predictions"]),
    }
    with paths["label_metadata"].open("r", encoding="utf-8") as handle:
        data["label_metadata"] = json.load(handle)
    with paths["embedding_summary"].open("r", encoding="utf-8") as handle:
        data["embedding_summary"] = json.load(handle)
    return data


def validate_inputs(data: dict[str, object]) -> None:
    summary = data["summary"]
    raw = data["raw"]
    predictions = data["predictions"]
    zero_summary = data["zero_summary"]
    zero_predictions = data["zero_predictions"]

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)
    assert isinstance(zero_summary, pd.DataFrame)
    assert isinstance(zero_predictions, pd.DataFrame)

    observed_representations = set(summary["representation"])
    if observed_representations != set(REPRESENTATION_ORDER):
        raise ValueError(
            "Unexpected representations: "
            f"{sorted(observed_representations)}"
        )

    for representation in REPRESENTATION_ORDER:
        observed_budgets = set(
            summary.loc[
                summary["representation"] == representation, "labels_per_class"
            ].astype(str)
        )
        if observed_budgets != set(BUDGET_ORDER):
            raise ValueError(
                f"Incomplete budgets for {representation}: {sorted(observed_budgets)}"
            )

    if len(summary) != 18 or len(raw) != 390 or len(predictions) != 156_000:
        raise ValueError(
            "Unexpected supervised result dimensions: "
            f"summary={len(summary)}, raw={len(raw)}, "
            f"predictions={len(predictions)}"
        )

    prompt_ids = set(zero_summary["prompt_id"])
    if prompt_ids != {"prompt_1", "prompt_2", "prompt_3"}:
        raise ValueError(f"Unexpected zero-shot prompts: {sorted(prompt_ids)}")
    if len(zero_predictions) != 6_000:
        raise ValueError(
            f"Expected 6000 zero-shot prediction rows, found {len(zero_predictions)}"
        )

    primary_rows = zero_summary.loc[zero_summary["is_primary"].astype(bool)]
    if len(primary_rows) != 1 or primary_rows.iloc[0]["prompt_id"] != PRIMARY_PROMPT_ID:
        raise ValueError("The predeclared primary zero-shot prompt must be prompt_1")


def summary_row(
    summary: pd.DataFrame, representation: str, budget: str
) -> pd.Series:
    rows = summary.loc[
        (summary["representation"] == representation)
        & (summary["labels_per_class"].astype(str) == budget)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {representation}/{budget}")
    return rows.iloc[0]


def paired_confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean
    standard_error = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    half_width = critical * standard_error
    return mean - half_width, mean + half_width


def create_paired_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparator in ["MFCC", "Log-Mel"]:
        for budget in BUDGET_ORDER:
            clap = raw.loc[
                (raw["representation"] == "CLAP")
                & (raw["labels_per_class"].astype(str) == budget)
            ].copy()
            baseline = raw.loc[
                (raw["representation"] == comparator)
                & (raw["labels_per_class"].astype(str) == budget)
            ].copy()
            paired = clap.merge(
                baseline,
                on=["test_fold", "labels_per_class", "seed"],
                suffixes=("_clap", "_baseline"),
                validate="one_to_one",
            )
            if len(paired) != len(clap) or len(paired) != len(baseline):
                raise ValueError(f"Pairing failed for CLAP vs {comparator}, {budget}")

            result: dict[str, object] = {
                "comparison": f"CLAP - {comparator}",
                "comparator": comparator,
                "labels_per_class": budget,
                "paired_runs": len(paired),
            }
            for metric in ["accuracy", "macro_f1"]:
                differences = (
                    paired[f"{metric}_clap"] - paired[f"{metric}_baseline"]
                ).to_numpy(dtype=float)
                ci_low, ci_high = paired_confidence_interval(differences)
                test = stats.ttest_rel(
                    paired[f"{metric}_clap"], paired[f"{metric}_baseline"]
                )
                result[f"{metric}_difference_mean"] = float(np.mean(differences))
                result[f"{metric}_difference_std"] = float(
                    np.std(differences, ddof=1)
                )
                result[f"{metric}_ci95_low"] = ci_low
                result[f"{metric}_ci95_high"] = ci_high
                result[f"{metric}_paired_t"] = float(test.statistic)
                result[f"{metric}_paired_p"] = float(test.pvalue)
            rows.append(result)
    paired_results = pd.DataFrame(rows)
    paired_results["labels_per_class"] = pd.Categorical(
        paired_results["labels_per_class"], BUDGET_ORDER, ordered=True
    )
    paired_results = paired_results.sort_values(
        ["comparator", "labels_per_class"]
    ).reset_index(drop=True)
    paired_results["labels_per_class"] = paired_results[
        "labels_per_class"
    ].astype(str)
    return paired_results


def create_threshold_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation in REPRESENTATION_ORDER:
        representation_rows = summary.loc[
            summary["representation"] == representation
        ].copy()
        for metric in ["accuracy", "macro_f1"]:
            for threshold in [0.90, 0.95, 0.97]:
                first_budget: str | None = None
                first_value: float | None = None
                for budget in BUDGET_ORDER:
                    row = representation_rows.loc[
                        representation_rows["labels_per_class"].astype(str) == budget
                    ].iloc[0]
                    value = float(row[f"{metric}_mean"])
                    if value >= threshold:
                        first_budget = budget
                        first_value = value
                        break
                rows.append(
                    {
                        "representation": representation,
                        "metric": metric,
                        "threshold": threshold,
                        "first_budget_reaching_threshold": first_budget,
                        "labels_per_class_numeric": (
                            BUDGET_TO_LABELS[first_budget]
                            if first_budget is not None
                            else np.nan
                        ),
                        "metric_value": first_value,
                    }
                )
    return pd.DataFrame(rows)


def create_class_recall_tables(
    predictions: pd.DataFrame, zero_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_frames: list[pd.DataFrame] = []

    zero = zero_predictions.loc[
        zero_predictions["prompt_id"] == PRIMARY_PROMPT_ID,
        ["true_target", "true_category", "correct"],
    ].rename(columns={"true_target": "target", "true_category": "category"})
    zero_grouped = (
        zero.groupby(["target", "category"], as_index=False)
        .agg(recall=("correct", "mean"), prediction_rows=("correct", "size"))
        .assign(condition="Zero-shot CLAP", evaluation_runs=1)
    )
    condition_frames.append(zero_grouped)

    supervised_conditions = [
        ("CLAP", "1", "CLAP 1-shot"),
        ("CLAP", "2", "CLAP 2-shot"),
        ("CLAP", "5", "CLAP 5-shot"),
        ("CLAP", "full", "CLAP full"),
        ("MFCC", "full", "MFCC full"),
        ("Log-Mel", "full", "Log-Mel full"),
    ]
    for representation, budget, condition in supervised_conditions:
        subset = predictions.loc[
            (predictions["representation"] == representation)
            & (predictions["labels_per_class"].astype(str) == budget),
            ["target", "category", "correct", "test_fold", "seed"],
        ]
        run_count = subset[["test_fold", "seed"]].drop_duplicates().shape[0]
        grouped = (
            subset.groupby(["target", "category"], as_index=False)
            .agg(recall=("correct", "mean"), prediction_rows=("correct", "size"))
            .assign(condition=condition, evaluation_runs=run_count)
        )
        condition_frames.append(grouped)

    class_recall = pd.concat(condition_frames, ignore_index=True)
    class_recall = class_recall[
        [
            "target",
            "category",
            "condition",
            "recall",
            "prediction_rows",
            "evaluation_runs",
        ]
    ].sort_values(["target", "condition"])

    recall_wide = class_recall.pivot(
        index=["target", "category"], columns="condition", values="recall"
    ).reset_index()
    recall_wide["clap_5shot_minus_zero_shot"] = (
        recall_wide["CLAP 5-shot"] - recall_wide["Zero-shot CLAP"]
    )
    recall_wide["clap_full_minus_zero_shot"] = (
        recall_wide["CLAP full"] - recall_wide["Zero-shot CLAP"]
    )
    top_changes = recall_wide.sort_values(
        "clap_5shot_minus_zero_shot", ascending=False
    ).reset_index(drop=True)

    coarse = class_recall.copy()
    coarse["coarse_id"] = coarse["target"] // 10
    coarse["coarse_category"] = coarse["coarse_id"].map(COARSE_NAMES)
    # All ESC-50 classes have equal support, so averaging class recall gives
    # macro recall within each official coarse category.
    coarse = (
        coarse.groupby(
            ["coarse_id", "coarse_category", "condition"], as_index=False
        )
        .agg(macro_recall=("recall", "mean"), classes=("target", "nunique"))
        .sort_values(["coarse_id", "condition"])
    )
    return class_recall, top_changes, coarse


def create_confusion_change_table(
    predictions: pd.DataFrame, zero_predictions: pd.DataFrame
) -> pd.DataFrame:
    zero = zero_predictions.loc[
        zero_predictions["prompt_id"] == PRIMARY_PROMPT_ID,
        ["true_target", "predicted_target"],
    ]
    clap_five = predictions.loc[
        (predictions["representation"] == "CLAP")
        & (predictions["labels_per_class"].astype(str) == "5"),
        ["target", "prediction"],
    ]

    labels = list(range(50))
    zero_matrix = pd.crosstab(
        zero["true_target"], zero["predicted_target"]
    ).reindex(index=labels, columns=labels, fill_value=0)
    clap_five_matrix = pd.crosstab(
        clap_five["target"], clap_five["prediction"]
    ).reindex(index=labels, columns=labels, fill_value=0)
    zero_rates = zero_matrix.div(zero_matrix.sum(axis=1), axis=0)
    clap_five_rates = clap_five_matrix.div(clap_five_matrix.sum(axis=1), axis=0)

    category_map = (
        predictions[["target", "category"]]
        .drop_duplicates()
        .set_index("target")["category"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for true_target in labels:
        for predicted_target in labels:
            if true_target == predicted_target:
                continue
            zero_rate = float(zero_rates.loc[true_target, predicted_target])
            five_rate = float(clap_five_rates.loc[true_target, predicted_target])
            rows.append(
                {
                    "true_target": true_target,
                    "true_category": category_map[true_target],
                    "predicted_target": predicted_target,
                    "predicted_category": category_map[predicted_target],
                    "zero_shot_error_rate": zero_rate,
                    "clap_5shot_error_rate": five_rate,
                    "error_rate_reduction": zero_rate - five_rate,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["error_rate_reduction", "zero_shot_error_rate"], ascending=[False, False]
    ).reset_index(drop=True)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_final_learning_curves(
    summary: pd.DataFrame, zero_summary: pd.DataFrame
) -> None:
    x = np.arange(1, len(BUDGET_ORDER) + 1)
    x_ticks = np.arange(0, len(BUDGET_ORDER) + 1)
    x_labels = ["0\nzero-shot", "1", "2", "5", "10", "20", "32\nfull"]
    primary = zero_summary.loc[
        zero_summary["prompt_id"] == PRIMARY_PROMPT_ID
    ].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "macro_f1", "Macro F1"),
    ]:
        for representation in REPRESENTATION_ORDER:
            values = [
                float(summary_row(summary, representation, budget)[f"{metric}_mean"])
                for budget in BUDGET_ORDER
            ]
            axis.plot(
                x,
                values,
                marker="o",
                markersize=5.5 if representation == "CLAP" else 4.5,
                linewidth=2.5 if representation == "CLAP" else 1.8,
                color=COLORS[representation],
                label=representation,
                zorder=3 if representation == "CLAP" else 2,
            )
        zero_value = float(primary[metric])
        clap_one = float(summary_row(summary, "CLAP", "1")[f"{metric}_mean"])
        axis.plot(
            [0, 1],
            [zero_value, clap_one],
            linestyle="--",
            linewidth=1.5,
            color=COLORS["Zero-shot CLAP"],
            alpha=0.75,
            zorder=2,
        )
        axis.scatter(
            [0],
            [zero_value],
            marker="*",
            s=180,
            color=COLORS["Zero-shot CLAP"],
            edgecolor="white",
            linewidth=0.7,
            label="Zero-shot CLAP (primary prompt)",
            zorder=5,
        )
        axis.annotate(
            f"{zero_value:.3f}",
            (0, zero_value),
            xytext=(7, -15),
            textcoords="offset points",
            color=COLORS["Zero-shot CLAP"],
            fontsize=9,
        )
        axis.set_title(title)
        axis.set_xticks(x_ticks, x_labels)
        axis.set_xlabel("Task-specific labelled training clips per class")
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean score")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("ESC-50 label efficiency: frozen CLAP versus classical features")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(FIGURES_DIR / "final_learning_curves_with_zero_shot.png")
    plt.close(fig)


def plot_classical_zoom(summary: pd.DataFrame) -> None:
    x = np.arange(len(BUDGET_ORDER))
    x_labels = ["1", "2", "5", "10", "20", "32\nfull"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "macro_f1", "Macro F1"),
    ]:
        for representation in ["MFCC", "Log-Mel"]:
            rows = [summary_row(summary, representation, b) for b in BUDGET_ORDER]
            means = [float(row[f"{metric}_mean"]) for row in rows]
            stds = [float(row[f"{metric}_std"]) for row in rows]
            axis.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                linewidth=2,
                color=COLORS[representation],
                label=representation,
            )
        axis.set_title(title)
        axis.set_xticks(x, x_labels)
        axis.set_xlabel("Labelled training clips per class")
        axis.set_ylim(0.04, 0.41)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean score (error bars: run SD)")
    axes[1].legend(frameon=False)
    fig.suptitle("Classical-feature baselines (zoomed scale)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / "classical_learning_curves_zoom.png")
    plt.close(fig)


def plot_clap_gain(summary: pd.DataFrame) -> None:
    x = np.arange(len(BUDGET_ORDER))
    width = 0.36
    accuracy_gains: list[float] = []
    f1_gains: list[float] = []
    for budget in BUDGET_ORDER:
        clap = summary_row(summary, "CLAP", budget)
        baselines = [
            summary_row(summary, representation, budget)
            for representation in ["MFCC", "Log-Mel"]
        ]
        accuracy_gains.append(
            float(clap["accuracy_mean"])
            - max(float(row["accuracy_mean"]) for row in baselines)
        )
        f1_gains.append(
            float(clap["macro_f1_mean"])
            - max(float(row["macro_f1_mean"]) for row in baselines)
        )

    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    bars1 = axis.bar(
        x - width / 2,
        accuracy_gains,
        width,
        label="Accuracy gain",
        color="#59A14F",
    )
    bars2 = axis.bar(
        x + width / 2,
        f1_gains,
        width,
        label="Macro-F1 gain",
        color="#B6992D",
    )
    axis.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(x, ["1", "2", "5", "10", "20", "32\nfull"])
    axis.set_xlabel("Labelled training clips per class")
    axis.set_ylabel("Absolute score gain")
    axis.set_ylim(0, max(accuracy_gains + f1_gains) + 0.09)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Frozen CLAP gain over the better classical baseline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "clap_gain_vs_best_classical.png")
    plt.close(fig)


def plot_clap_zero_few_full(
    summary: pd.DataFrame, zero_summary: pd.DataFrame
) -> None:
    primary = zero_summary.loc[
        zero_summary["prompt_id"] == PRIMARY_PROMPT_ID
    ].iloc[0]
    conditions = ["Zero", "1", "2", "5", "10", "20", "Full"]
    accuracy = [float(primary["accuracy"])] + [
        float(summary_row(summary, "CLAP", budget)["accuracy_mean"])
        for budget in BUDGET_ORDER
    ]
    macro_f1 = [float(primary["macro_f1"])] + [
        float(summary_row(summary, "CLAP", budget)["macro_f1_mean"])
        for budget in BUDGET_ORDER
    ]
    x = np.arange(len(conditions))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10, 5))
    bars1 = axis.bar(
        x - width / 2, accuracy, width, color="#4C78A8", label="Accuracy"
    )
    bars2 = axis.bar(
        x + width / 2, macro_f1, width, color="#F58518", label="Macro F1"
    )
    axis.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8, rotation=90)
    axis.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8, rotation=90)
    axis.set_xticks(x, conditions)
    axis.set_xlabel("Task-specific labelled training clips per class")
    axis.set_ylabel("Score")
    axis.set_ylim(0.80, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Frozen CLAP: zero-shot, few-shot, and full-data performance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "clap_zero_few_full_comparison.png")
    plt.close(fig)


def plot_top_corrected_confusions(confusion_changes: pd.DataFrame) -> None:
    top = confusion_changes.loc[
        confusion_changes["error_rate_reduction"] > 0
    ].head(10).copy()
    if top.empty:
        raise ValueError("No zero-shot confusion was reduced by CLAP 5-shot")
    top = top.iloc[::-1].reset_index(drop=True)
    labels = [
        f"{row.true_category} -> {row.predicted_category}"
        for row in top.itertuples(index=False)
    ]
    y = np.arange(len(top))
    height = 0.36

    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    bars1 = axis.barh(
        y + height / 2,
        top["zero_shot_error_rate"],
        height,
        label="Zero-shot CLAP",
        color=COLORS["Zero-shot CLAP"],
        alpha=0.86,
    )
    bars2 = axis.barh(
        y - height / 2,
        top["clap_5shot_error_rate"],
        height,
        label="CLAP 5-shot",
        color=COLORS["CLAP"],
        alpha=0.90,
    )
    axis.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Fraction of the true class assigned to the wrong class")
    axis.set_title("Largest zero-shot confusions reduced by five labelled clips per class")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.set_xlim(0, max(float(top["zero_shot_error_rate"].max()) + 0.08, 0.20))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_corrected_zero_vs_5shot_confusions.png")
    plt.close(fig)


def plot_heatmap(
    table: pd.DataFrame,
    row_column: str,
    column_column: str,
    value_column: str,
    row_order: list[object],
    column_order: list[str],
    row_labels: list[str],
    title: str,
    output_name: str,
    figsize: tuple[float, float],
    annotate: bool,
) -> None:
    pivot = table.pivot(
        index=row_column, columns=column_column, values=value_column
    ).reindex(index=row_order, columns=column_order)
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete data for heatmap {output_name}")

    fig, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axis.set_xticks(np.arange(len(column_order)), column_order, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)
    axis.set_xlabel("Evaluation condition")
    if annotate:
        values = pivot.to_numpy(dtype=float)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                text_color = "white" if value < 0.45 or value > 0.82 else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Recall")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name)
    plt.close(fig)


def create_key_findings(
    summary: pd.DataFrame,
    zero_summary: pd.DataFrame,
    label_metadata: dict[str, object],
    embedding_summary: dict[str, object],
) -> dict[str, object]:
    primary = zero_summary.loc[
        zero_summary["prompt_id"] == PRIMARY_PROMPT_ID
    ].iloc[0]
    best_prompt = zero_summary.sort_values("accuracy", ascending=False).iloc[0]
    clap_rows = {
        budget: summary_row(summary, "CLAP", budget) for budget in BUDGET_ORDER
    }
    full_classical = {
        representation: summary_row(summary, representation, "full")
        for representation in ["MFCC", "Log-Mel"]
    }
    best_full_accuracy_rep = max(
        full_classical,
        key=lambda representation: float(
            full_classical[representation]["accuracy_mean"]
        ),
    )
    best_full_f1_rep = max(
        full_classical,
        key=lambda representation: float(
            full_classical[representation]["macro_f1_mean"]
        ),
    )

    clap_five_accuracy = float(clap_rows["5"]["accuracy_mean"])
    clap_five_f1 = float(clap_rows["5"]["macro_f1_mean"])
    clap_full_accuracy = float(clap_rows["full"]["accuracy_mean"])
    clap_full_f1 = float(clap_rows["full"]["macro_f1_mean"])
    primary_accuracy = float(primary["accuracy"])
    primary_f1 = float(primary["macro_f1"])

    return {
        "analysis_status": "PASS",
        "primary_zero_shot": {
            "prompt_id": PRIMARY_PROMPT_ID,
            "template": str(primary["template"]),
            "accuracy": primary_accuracy,
            "macro_f1": primary_f1,
            "interpretation": (
                "Primary result fixed before reviewing prompt sensitivity results."
            ),
        },
        "prompt_sensitivity": {
            "best_prompt_id_by_accuracy": str(best_prompt["prompt_id"]),
            "best_prompt_accuracy": float(best_prompt["accuracy"]),
            "accuracy_min": float(zero_summary["accuracy"].min()),
            "accuracy_max": float(zero_summary["accuracy"].max()),
            "accuracy_range": float(
                zero_summary["accuracy"].max() - zero_summary["accuracy"].min()
            ),
            "reporting_rule": (
                "prompt_2 and prompt_3 are sensitivity analyses, not replacements "
                "for the primary prompt_1 result"
            ),
        },
        "supervised_clap": {
            budget: {
                "labels_per_class_numeric": BUDGET_TO_LABELS[budget],
                "total_training_clips_mean": float(
                    clap_rows[budget]["train_samples_mean"]
                ),
                "accuracy_mean": float(clap_rows[budget]["accuracy_mean"]),
                "macro_f1_mean": float(clap_rows[budget]["macro_f1_mean"]),
            }
            for budget in BUDGET_ORDER
        },
        "central_comparisons": {
            "zero_shot_minus_clap_1shot_accuracy": primary_accuracy
            - float(clap_rows["1"]["accuracy_mean"]),
            "zero_shot_minus_clap_2shot_accuracy": primary_accuracy
            - float(clap_rows["2"]["accuracy_mean"]),
            "clap_5shot_minus_zero_shot_accuracy": clap_five_accuracy
            - primary_accuracy,
            "clap_5shot_minus_zero_shot_macro_f1": clap_five_f1 - primary_f1,
            "clap_5shot_fraction_of_full_accuracy": clap_five_accuracy
            / clap_full_accuracy,
            "clap_5shot_fraction_of_full_macro_f1": clap_five_f1 / clap_full_f1,
            "clap_5shot_label_reduction_vs_full": 1.0 - (5.0 / 32.0),
            "clap_1shot_minus_best_full_classical_accuracy": float(
                clap_rows["1"]["accuracy_mean"]
            )
            - float(full_classical[best_full_accuracy_rep]["accuracy_mean"]),
            "clap_1shot_minus_best_full_classical_macro_f1": float(
                clap_rows["1"]["macro_f1_mean"]
            )
            - float(full_classical[best_full_f1_rep]["macro_f1_mean"]),
        },
        "best_full_classical": {
            "accuracy_representation": best_full_accuracy_rep,
            "accuracy": float(
                full_classical[best_full_accuracy_rep]["accuracy_mean"]
            ),
            "macro_f1_representation": best_full_f1_rep,
            "macro_f1": float(full_classical[best_full_f1_rep]["macro_f1_mean"]),
        },
        "runtime": {
            "clap_embedding_extraction_seconds": float(
                embedding_summary["embedding_extraction_seconds"]
            ),
            "clap_mean_embedding_seconds_per_clip": float(
                embedding_summary["mean_embedding_seconds_per_clip"]
            ),
            "label_efficiency_experiment_total_seconds": float(
                label_metadata["runtime"]["total_seconds"]
            ),
            "clap_full_linear_probe_training_seconds_mean": float(
                clap_rows["full"]["training_seconds_mean"]
            ),
        },
        "evidence_boundaries": [
            "Zero-shot means no task-specific ESC-50 labels were used for fitting the classifier.",
            "It does not prove that CLAP pretraining contained no related or overlapping audio.",
            "The results establish performance on ESC-50 official folds, not broad out-of-domain generalisation.",
            "Paired t-tests are supportive analyses across the predefined fold/seed runs.",
        ],
    }


def main() -> None:
    print("=" * 72)
    print("FINAL PROJECT RESULT ANALYSIS")
    print("=" * 72)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    validate_inputs(data)
    print("Input validation: PASS")

    summary = data["summary"]
    raw = data["raw"]
    predictions = data["predictions"]
    zero_summary = data["zero_summary"]
    zero_predictions = data["zero_predictions"]
    label_metadata = data["label_metadata"]
    embedding_summary = data["embedding_summary"]

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)
    assert isinstance(zero_summary, pd.DataFrame)
    assert isinstance(zero_predictions, pd.DataFrame)
    assert isinstance(label_metadata, dict)
    assert isinstance(embedding_summary, dict)

    paired = create_paired_comparisons(raw)
    paired_path = RESULTS_DIR / "paired_representation_comparisons.csv"
    paired.to_csv(paired_path, index=False)

    thresholds = create_threshold_table(summary)
    thresholds_path = RESULTS_DIR / "label_efficiency_thresholds.csv"
    thresholds.to_csv(thresholds_path, index=False)

    class_recall, top_changes, coarse = create_class_recall_tables(
        predictions, zero_predictions
    )
    confusion_changes = create_confusion_change_table(predictions, zero_predictions)
    class_recall_path = RESULTS_DIR / "class_recall_comparison.csv"
    top_changes_path = RESULTS_DIR / "top_class_changes.csv"
    coarse_path = RESULTS_DIR / "coarse_category_recall.csv"
    confusion_changes_path = RESULTS_DIR / "zero_vs_5shot_confusion_changes.csv"
    class_recall.to_csv(class_recall_path, index=False)
    top_changes.to_csv(top_changes_path, index=False)
    coarse.to_csv(coarse_path, index=False)
    confusion_changes.to_csv(confusion_changes_path, index=False)

    findings = create_key_findings(
        summary, zero_summary, label_metadata, embedding_summary
    )
    findings_path = RESULTS_DIR / "final_key_findings.json"
    with findings_path.open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2, ensure_ascii=False, allow_nan=False)

    setup_plot_style()
    plot_final_learning_curves(summary, zero_summary)
    plot_classical_zoom(summary)
    plot_clap_gain(summary)
    plot_clap_zero_few_full(summary, zero_summary)
    plot_top_corrected_confusions(confusion_changes)

    class_condition_order = [
        "Zero-shot CLAP",
        "CLAP 1-shot",
        "CLAP 2-shot",
        "CLAP 5-shot",
        "CLAP full",
        "MFCC full",
        "Log-Mel full",
    ]
    class_names = (
        class_recall[["target", "category"]]
        .drop_duplicates()
        .sort_values("target")["category"]
        .tolist()
    )
    plot_heatmap(
        table=class_recall,
        row_column="target",
        column_column="condition",
        value_column="recall",
        row_order=list(range(50)),
        column_order=class_condition_order,
        row_labels=class_names,
        title="Per-class recall across selected evaluation conditions",
        output_name="class_recall_heatmap.png",
        figsize=(11.5, 14.5),
        annotate=False,
    )
    plot_heatmap(
        table=coarse,
        row_column="coarse_id",
        column_column="condition",
        value_column="macro_recall",
        row_order=list(range(5)),
        column_order=class_condition_order,
        row_labels=[COARSE_NAMES[index] for index in range(5)],
        title="Macro recall by ESC-50 coarse category",
        output_name="coarse_category_recall_heatmap.png",
        figsize=(11.5, 4.6),
        annotate=True,
    )

    primary = zero_summary.loc[
        zero_summary["prompt_id"] == PRIMARY_PROMPT_ID
    ].iloc[0]
    clap_five = summary_row(summary, "CLAP", "5")
    clap_full = summary_row(summary, "CLAP", "full")

    print("\nKey verified results")
    print(
        f"Primary zero-shot: accuracy={primary['accuracy']:.4f}, "
        f"macro-F1={primary['macro_f1']:.4f}"
    )
    print(
        f"CLAP 5-shot:       accuracy={clap_five['accuracy_mean']:.4f}, "
        f"macro-F1={clap_five['macro_f1_mean']:.4f}"
    )
    print(
        f"CLAP full:         accuracy={clap_full['accuracy_mean']:.4f}, "
        f"macro-F1={clap_full['macro_f1_mean']:.4f}"
    )
    print("\nSaved tables")
    for path in [
        findings_path,
        thresholds_path,
        paired_path,
        class_recall_path,
        top_changes_path,
        coarse_path,
        confusion_changes_path,
    ]:
        print(f"  {path}")
    print("\nSaved figures")
    for name in [
        "final_learning_curves_with_zero_shot.png",
        "classical_learning_curves_zoom.png",
        "clap_gain_vs_best_classical.png",
        "clap_zero_few_full_comparison.png",
        "class_recall_heatmap.png",
        "coarse_category_recall_heatmap.png",
        "top_corrected_zero_vs_5shot_confusions.png",
    ]:
        print(f"  {FIGURES_DIR / name}")
    print("\nAnalysis status: PASS")


if __name__ == "__main__":
    main()
