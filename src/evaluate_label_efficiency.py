from pathlib import Path
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FEATURE_PATH = PROJECT_ROOT / "cache" / "classical_features.npz"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_configuration():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_features():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature cache not found: {FEATURE_PATH}")

    with np.load(FEATURE_PATH, allow_pickle=False) as feature_data:
        required_keys = {
            "mfcc",
            "logmel",
            "filename",
            "target",
            "category",
            "fold",
        }
        missing_keys = required_keys - set(feature_data.files)
        if missing_keys:
            raise ValueError(
                f"Feature cache is missing keys: {sorted(missing_keys)}"
            )

        feature_matrices = {
            "MFCC": feature_data["mfcc"].astype(np.float32),
            "Log-Mel": feature_data["logmel"].astype(np.float32),
        }
        filenames = feature_data["filename"].astype(str)
        targets = feature_data["target"].astype(np.int64)
        categories = feature_data["category"].astype(str)
        folds = feature_data["fold"].astype(np.int64)

    sample_count = len(targets)
    for representation, matrix in feature_matrices.items():
        if matrix.shape[0] != sample_count:
            raise ValueError(
                f"{representation} row count does not match labels."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"{representation} contains NaN or infinite values."
            )

    if len(filenames) != sample_count or len(folds) != sample_count:
        raise ValueError("Feature-cache metadata lengths do not match.")

    return (
        feature_matrices,
        filenames,
        targets,
        categories,
        folds,
    )


def select_labelled_training_indices(
    candidate_indices,
    targets,
    labels_per_class,
    seed,
):
    if labels_per_class == "full":
        return np.sort(candidate_indices)

    random_generator = np.random.default_rng(seed)
    selected_indices = []

    for class_id in sorted(np.unique(targets[candidate_indices])):
        class_candidates = candidate_indices[
            targets[candidate_indices] == class_id
        ]

        if len(class_candidates) < labels_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_candidates)} "
                f"training samples; requested {labels_per_class}."
            )

        class_selection = random_generator.choice(
            class_candidates,
            size=labels_per_class,
            replace=False,
        )
        selected_indices.extend(class_selection.tolist())

    return np.sort(np.asarray(selected_indices, dtype=np.int64))


def build_linear_probe(seed):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    dual="auto",
                    max_iter=20000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_summary(raw_metrics):
    summary = (
        raw_metrics.groupby(
            ["representation", "labels_per_class"],
            sort=False,
        )
        .agg(
            runs=("accuracy", "size"),
            train_samples_mean=("train_samples", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            training_seconds_mean=("training_seconds", "mean"),
            inference_seconds_mean=("inference_seconds", "mean"),
        )
        .reset_index()
    )

    numeric_columns = [
        "accuracy_std",
        "macro_f1_std",
    ]
    summary[numeric_columns] = summary[numeric_columns].fillna(0.0)
    return summary


def plot_label_efficiency(summary, budget_order):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharex=True,
        sharey=True,
    )

    x_positions = np.arange(len(budget_order))
    colours = {
        "MFCC": "#4C72B0",
        "Log-Mel": "#DD8452",
    }

    for representation in ["MFCC", "Log-Mel"]:
        representation_summary = (
            summary[summary["representation"] == representation]
            .set_index("labels_per_class")
            .reindex(budget_order)
        )

        axes[0].errorbar(
            x_positions,
            representation_summary["accuracy_mean"],
            yerr=representation_summary["accuracy_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=representation,
            color=colours[representation],
        )
        axes[1].errorbar(
            x_positions,
            representation_summary["macro_f1_mean"],
            yerr=representation_summary["macro_f1_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=representation,
            color=colours[representation],
        )

    axes[0].set_title("Accuracy")
    axes[1].set_title("Macro-F1")

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(budget_order)
        axis.set_xlabel("Labelled training examples per class")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    figure.suptitle(
        "ESC-50 Label Efficiency with a Common Linear SVM Probe",
        fontsize=16,
    )
    figure.tight_layout()

    output_path = FIGURES_DIR / "classical_label_efficiency.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    configuration = load_configuration()
    label_configuration = configuration["label_efficiency"]
    label_budgets = label_configuration["label_budgets"]
    seeds = label_configuration["seeds"]

    (
        feature_matrices,
        filenames,
        targets,
        categories,
        folds,
    ) = load_features()

    official_folds = sorted(np.unique(folds).tolist())
    class_count = len(np.unique(targets))
    metric_records = []
    sampling_records = []

    print("=" * 72)
    print("ESC-50 LABEL-EFFICIENCY EXPERIMENT")
    print("=" * 72)
    print(f"Feature cache: {FEATURE_PATH}")
    print(f"Samples: {len(targets)}")
    print(f"Classes: {class_count}")
    print(f"Official folds: {official_folds}")
    print(f"Label budgets: {label_budgets}")
    print(f"Seeds: {seeds}")

    experiment_start = time.perf_counter()

    for test_fold in official_folds:
        candidate_indices = np.flatnonzero(folds != test_fold)
        testing_indices = np.flatnonzero(folds == test_fold)

        print("\n" + "=" * 72)
        print(f"OUTER TEST FOLD {test_fold}")
        print("=" * 72)

        for budget in label_budgets:
            budget_label = str(budget)
            active_seeds = [0] if budget == "full" else seeds

            for seed in active_seeds:
                selected_indices = select_labelled_training_indices(
                    candidate_indices=candidate_indices,
                    targets=targets,
                    labels_per_class=budget,
                    seed=seed,
                )

                for selected_index in selected_indices:
                    sampling_records.append(
                        {
                            "test_fold": test_fold,
                            "labels_per_class": budget_label,
                            "seed": seed,
                            "sample_index": int(selected_index),
                            "filename": filenames[selected_index],
                            "target": int(targets[selected_index]),
                            "category": categories[selected_index],
                            "source_fold": int(folds[selected_index]),
                        }
                    )

                print(
                    f"\nBudget {budget_label:>4} | seed {seed} | "
                    f"training samples {len(selected_indices)}"
                )

                for representation, feature_matrix in feature_matrices.items():
                    model = build_linear_probe(seed)

                    training_start = time.perf_counter()
                    model.fit(
                        feature_matrix[selected_indices],
                        targets[selected_indices],
                    )
                    training_seconds = time.perf_counter() - training_start

                    inference_start = time.perf_counter()
                    predictions = model.predict(
                        feature_matrix[testing_indices]
                    )
                    inference_seconds = (
                        time.perf_counter() - inference_start
                    )

                    accuracy = accuracy_score(
                        targets[testing_indices],
                        predictions,
                    )
                    macro_f1 = f1_score(
                        targets[testing_indices],
                        predictions,
                        average="macro",
                        zero_division=0,
                    )

                    metric_records.append(
                        {
                            "representation": representation,
                            "test_fold": test_fold,
                            "labels_per_class": budget_label,
                            "seed": seed,
                            "train_samples": len(selected_indices),
                            "test_samples": len(testing_indices),
                            "accuracy": accuracy,
                            "macro_f1": macro_f1,
                            "training_seconds": training_seconds,
                            "inference_seconds": inference_seconds,
                        }
                    )

                    print(
                        f"  {representation:<7} | "
                        f"accuracy {accuracy:.4f} | "
                        f"macro-F1 {macro_f1:.4f}"
                    )

    total_seconds = time.perf_counter() - experiment_start

    raw_metrics = pd.DataFrame(metric_records)
    sampling_manifest = pd.DataFrame(sampling_records)

    budget_order = [str(budget) for budget in label_budgets]
    raw_metrics["labels_per_class"] = pd.Categorical(
        raw_metrics["labels_per_class"],
        categories=budget_order,
        ordered=True,
    )

    summary = make_summary(raw_metrics)
    summary["labels_per_class"] = pd.Categorical(
        summary["labels_per_class"],
        categories=budget_order,
        ordered=True,
    )
    summary = summary.sort_values(
        ["representation", "labels_per_class"]
    ).reset_index(drop=True)

    raw_metrics_path = RESULTS_DIR / "label_efficiency_raw.csv"
    summary_path = RESULTS_DIR / "label_efficiency_summary.csv"
    sampling_path = RESULTS_DIR / "label_efficiency_samples.csv"
    metadata_path = RESULTS_DIR / "label_efficiency_metadata.json"

    raw_metrics.to_csv(raw_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    sampling_manifest.to_csv(sampling_path, index=False)
    figure_path = plot_label_efficiency(summary, budget_order)

    metadata = {
        "experiment": "ESC-50 classical feature label efficiency",
        "feature_cache": str(FEATURE_PATH),
        "representations": list(feature_matrices.keys()),
        "classifier": {
            "name": "LinearSVC",
            "C": 1.0,
            "preprocessing": "StandardScaler fitted on labelled training data",
        },
        "evaluation": {
            "outer_test_folds": official_folds,
            "label_budgets_per_class": budget_order,
            "partial_budget_seeds": seeds,
            "full_budget_seed": 0,
            "metrics": ["accuracy", "macro_f1"],
            "sampling_without_replacement": True,
            "shared_samples_between_representations": True,
        },
        "runtime": {
            "total_seconds": total_seconds,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            "raw_metrics": str(raw_metrics_path),
            "summary": str(summary_path),
            "sampling_manifest": str(sampling_path),
            "figure": str(figure_path),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 72)
    print("FINAL LABEL-EFFICIENCY SUMMARY")
    print("=" * 72)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nTotal experiment time: {total_seconds:.2f} seconds")
    print(f"Saved: {raw_metrics_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {sampling_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
