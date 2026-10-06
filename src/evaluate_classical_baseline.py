from pathlib import Path
import json
import platform
import time

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FEATURE_PATH = PROJECT_ROOT / "cache" / "classical_features.npz"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_feature_cache(feature_data):
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
            f"Feature cache is missing required keys: {sorted(missing_keys)}"
        )

    mfcc = feature_data["mfcc"]
    logmel = feature_data["logmel"]
    target = feature_data["target"]
    fold = feature_data["fold"]

    expected_samples = 2000

    if mfcc.shape != (expected_samples, 80):
        raise ValueError(
            f"Unexpected MFCC shape: {mfcc.shape}; expected (2000, 80)."
        )

    if logmel.shape != (expected_samples, 256):
        raise ValueError(
            f"Unexpected Log-Mel shape: {logmel.shape}; "
            "expected (2000, 256)."
        )

    if target.shape != (expected_samples,):
        raise ValueError(f"Unexpected target shape: {target.shape}")

    if fold.shape != (expected_samples,):
        raise ValueError(f"Unexpected fold shape: {fold.shape}")

    if not np.isfinite(mfcc).all():
        raise ValueError("MFCC feature matrix contains NaN or Inf.")

    if not np.isfinite(logmel).all():
        raise ValueError("Log-Mel feature matrix contains NaN or Inf.")

    unique_targets, target_counts = np.unique(
        target,
        return_counts=True,
    )

    if len(unique_targets) != 50:
        raise ValueError(
            f"Expected 50 classes, found {len(unique_targets)}."
        )

    if not np.all(target_counts == 40):
        raise ValueError(
            "Expected exactly 40 samples per class."
        )

    unique_folds, fold_counts = np.unique(
        fold,
        return_counts=True,
    )

    if not np.array_equal(
        unique_folds,
        np.asarray([1, 2, 3, 4, 5]),
    ):
        raise ValueError(
            f"Unexpected fold labels: {unique_folds.tolist()}"
        )

    if not np.all(fold_counts == 400):
        raise ValueError(
            f"Unexpected fold sizes: {fold_counts.tolist()}"
        )


def build_category_names(target, category):
    category_table = (
        pd.DataFrame(
            {
                "target": target,
                "category": category,
            }
        )
        .drop_duplicates()
        .sort_values("target")
    )

    expected_targets = np.arange(50)

    if not np.array_equal(
        category_table["target"].to_numpy(),
        expected_targets,
    ):
        raise ValueError(
            "Could not construct a complete ordered category list."
        )

    return category_table["category"].astype(str).tolist()


def build_inner_cv(training_folds):
    unique_training_folds = sorted(
        int(value)
        for value in np.unique(training_folds)
    )

    if len(unique_training_folds) != 4:
        raise ValueError(
            "Each outer training set should contain four official folds."
        )

    fold_mapping = {
        official_fold: split_index
        for split_index, official_fold
        in enumerate(unique_training_folds)
    }

    predefined_labels = np.asarray(
        [
            fold_mapping[int(official_fold)]
            for official_fold in training_folds
        ],
        dtype=np.int64,
    )

    return PredefinedSplit(
        test_fold=predefined_labels
    )


def plot_metric_comparison(fold_metrics):
    metric_names = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    ]

    representations = ["MFCC", "Log-Mel"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharey=True,
    )

    colors = {
        "MFCC": "#4472C4",
        "Log-Mel": "#ED7D31",
    }

    for axis, (metric_column, metric_label) in zip(
        axes,
        metric_names,
    ):
        positions = np.arange(len(representations))

        means = []
        standard_deviations = []

        for representation in representations:
            values = fold_metrics.loc[
                fold_metrics["representation"] == representation,
                metric_column,
            ].to_numpy()

            means.append(float(np.mean(values)))
            standard_deviations.append(
                float(np.std(values, ddof=1))
            )

        axis.bar(
            positions,
            means,
            yerr=standard_deviations,
            capsize=6,
            color=[
                colors[representation]
                for representation in representations
            ],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

        for representation_index, representation in enumerate(
            representations
        ):
            values = fold_metrics.loc[
                fold_metrics["representation"] == representation,
                metric_column,
            ].to_numpy()

            jitter = np.linspace(
                -0.08,
                0.08,
                len(values),
            )

            axis.scatter(
                np.full(len(values), representation_index) + jitter,
                values,
                color="black",
                s=28,
                zorder=3,
                label=(
                    "Official ESC-50 folds"
                    if representation_index == 0
                    and metric_column == "accuracy"
                    else None
                ),
            )

        axis.set_xticks(
            positions,
            representations,
        )
        axis.set_ylabel(metric_label)
        axis.set_ylim(0.0, 1.0)
        axis.grid(
            axis="y",
            alpha=0.25,
        )
        axis.set_title(
            f"{metric_label} across five official folds"
        )

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
        )

    figure.suptitle(
        "ESC-50 Classical Feature Baseline",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0.06, 1, 0.94)
    )

    output_path = (
        FIGURES_DIR
        / "classical_metric_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_confusion_matrix(
    true_targets,
    predicted_targets,
    category_names,
    representation,
):
    labels = np.arange(50)

    matrix = confusion_matrix(
        true_targets,
        predicted_targets,
        labels=labels,
        normalize="true",
    )

    figure, axis = plt.subplots(
        figsize=(16, 14),
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    axis.set_title(
        f"{representation} Normalised Confusion Matrix",
        fontsize=15,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    axis.set_xticks(
        labels,
        category_names,
        rotation=90,
        fontsize=6,
    )
    axis.set_yticks(
        labels,
        category_names,
        fontsize=6,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(
        "Proportion within true class"
    )

    figure.tight_layout()

    safe_name = (
        representation
        .lower()
        .replace("-", "")
        .replace(" ", "_")
    )

    output_path = (
        FIGURES_DIR
        / f"{safe_name}_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {FEATURE_PATH}"
        )

    config = load_config()

    with np.load(
        FEATURE_PATH,
        allow_pickle=False,
    ) as feature_data:
        validate_feature_cache(feature_data)

        feature_matrices = {
            "MFCC": feature_data["mfcc"].astype(
                np.float64,
                copy=False,
            ),
            "Log-Mel": feature_data["logmel"].astype(
                np.float64,
                copy=False,
            ),
        }

        filenames = feature_data["filename"].astype(str)
        targets = feature_data["target"].astype(np.int64)
        categories = feature_data["category"].astype(str)
        folds = feature_data["fold"].astype(np.int64)

    category_names = build_category_names(
        targets,
        categories,
    )

    svm_config = config["classical_baseline"]

    parameter_grid = {
        "classifier__C": [
            float(value)
            for value in svm_config["C_values"]
        ],
        "classifier__gamma": [
            (
                value
                if isinstance(value, str)
                else float(value)
            )
            for value in svm_config["gamma_values"]
        ],
    }

    fold_rows = []
    prediction_rows = []

    print("=" * 72)
    print("ESC-50 CLASSICAL FEATURE BASELINE")
    print("=" * 72)
    print(f"Feature cache: {FEATURE_PATH}")
    print(f"Samples: {len(targets)}")
    print(f"Classes: {len(np.unique(targets))}")
    print(f"Official folds: {sorted(np.unique(folds).tolist())}")
    print()

    for representation, feature_matrix in feature_matrices.items():
        print("=" * 72)
        print(f"REPRESENTATION: {representation}")
        print("=" * 72)

        for outer_fold in sorted(
            int(value)
            for value in np.unique(folds)
        ):
            training_mask = folds != outer_fold
            testing_mask = folds == outer_fold

            training_indices = np.flatnonzero(
                training_mask
            )
            testing_indices = np.flatnonzero(
                testing_mask
            )

            X_train = feature_matrix[training_indices]
            X_test = feature_matrix[testing_indices]

            y_train = targets[training_indices]
            y_test = targets[testing_indices]

            training_folds = folds[training_indices]

            inner_cv = build_inner_cv(
                training_folds
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            cache_size=2048,
                        ),
                    ),
                ]
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring={
                    "accuracy": "accuracy",
                    "macro_f1": "f1_macro",
                },
                refit="macro_f1",
                cv=inner_cv,
                n_jobs=-1,
                verbose=1,
                return_train_score=False,
                error_score="raise",
            )

            print(
                f"\n{representation} | "
                f"outer test fold {outer_fold}"
            )
            print(
                f"Training samples: {len(training_indices)} | "
                f"Testing samples: {len(testing_indices)}"
            )

            training_start = time.perf_counter()

            search.fit(
                X_train,
                y_train,
            )

            training_seconds = (
                time.perf_counter()
                - training_start
            )

            inference_start = time.perf_counter()

            predictions = search.predict(
                X_test
            )

            inference_seconds = (
                time.perf_counter()
                - inference_start
            )

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )

            best_parameters = search.best_params_

            fold_row = {
                "representation": representation,
                "test_fold": int(outer_fold),
                "training_samples": int(
                    len(training_indices)
                ),
                "testing_samples": int(
                    len(testing_indices)
                ),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "best_C": float(
                    best_parameters["classifier__C"]
                ),
                "best_gamma": (
                    best_parameters[
                        "classifier__gamma"
                    ]
                ),
                "best_inner_macro_f1": float(
                    search.best_score_
                ),
                "grid_search_seconds": float(
                    training_seconds
                ),
                "inference_seconds": float(
                    inference_seconds
                ),
            }

            fold_rows.append(fold_row)

            for local_index, prediction in enumerate(
                predictions
            ):
                original_index = int(
                    testing_indices[local_index]
                )

                prediction_rows.append(
                    {
                        "representation": representation,
                        "sample_index": original_index,
                        "filename": filenames[
                            original_index
                        ],
                        "test_fold": int(outer_fold),
                        "true_target": int(
                            y_test[local_index]
                        ),
                        "true_category": categories[
                            original_index
                        ],
                        "predicted_target": int(
                            prediction
                        ),
                        "predicted_category": (
                            category_names[
                                int(prediction)
                            ]
                        ),
                        "correct": bool(
                            prediction
                            == y_test[local_index]
                        ),
                    }
                )

            print(
                f"Accuracy: {accuracy:.4f} | "
                f"Macro-F1: {macro_f1:.4f}"
            )
            print(
                f"Best C: "
                f"{best_parameters['classifier__C']} | "
                f"Best gamma: "
                f"{best_parameters['classifier__gamma']}"
            )
            print(
                f"Grid search: "
                f"{training_seconds:.2f} seconds | "
                f"Inference: "
                f"{inference_seconds:.4f} seconds"
            )

    fold_metrics = pd.DataFrame(
        fold_rows
    ).sort_values(
        [
            "representation",
            "test_fold",
        ]
    )

    predictions_frame = pd.DataFrame(
        prediction_rows
    ).sort_values(
        [
            "representation",
            "sample_index",
        ]
    )

    expected_prediction_rows = (
        len(targets)
        * len(feature_matrices)
    )

    if len(predictions_frame) != expected_prediction_rows:
        raise RuntimeError(
            "Unexpected number of out-of-fold predictions: "
            f"{len(predictions_frame)}; "
            f"expected {expected_prediction_rows}."
        )

    duplicate_prediction_counts = (
        predictions_frame
        .groupby(
            [
                "representation",
                "sample_index",
            ]
        )
        .size()
    )

    if not (
        duplicate_prediction_counts == 1
    ).all():
        raise RuntimeError(
            "One or more samples received an unexpected "
            "number of out-of-fold predictions."
        )

    summary_rows = []

    for representation in feature_matrices:
        representation_rows = fold_metrics.loc[
            fold_metrics["representation"]
            == representation
        ]

        summary_rows.append(
            {
                "representation": representation,
                "folds": int(
                    len(representation_rows)
                ),
                "accuracy_mean": float(
                    representation_rows[
                        "accuracy"
                    ].mean()
                ),
                "accuracy_std": float(
                    representation_rows[
                        "accuracy"
                    ].std(ddof=1)
                ),
                "macro_f1_mean": float(
                    representation_rows[
                        "macro_f1"
                    ].mean()
                ),
                "macro_f1_std": float(
                    representation_rows[
                        "macro_f1"
                    ].std(ddof=1)
                ),
                "grid_search_seconds_total": float(
                    representation_rows[
                        "grid_search_seconds"
                    ].sum()
                ),
                "grid_search_seconds_mean": float(
                    representation_rows[
                        "grid_search_seconds"
                    ].mean()
                ),
                "inference_seconds_total": float(
                    representation_rows[
                        "inference_seconds"
                    ].sum()
                ),
            }
        )

    summary_frame = pd.DataFrame(
        summary_rows
    )

    fold_metrics_path = (
        RESULTS_DIR
        / "classical_fold_metrics.csv"
    )
    summary_path = (
        RESULTS_DIR
        / "classical_summary.csv"
    )
    predictions_path = (
        RESULTS_DIR
        / "oof_predictions.csv"
    )
    summary_json_path = (
        RESULTS_DIR
        / "classical_baseline_summary.json"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    predictions_frame.to_csv(
        predictions_path,
        index=False,
    )

    comparison_figure_path = (
        plot_metric_comparison(
            fold_metrics
        )
    )

    confusion_figure_paths = {}

    for representation in feature_matrices:
        representation_predictions = (
            predictions_frame.loc[
                predictions_frame["representation"]
                == representation
            ]
        )

        confusion_figure_paths[
            representation
        ] = str(
            plot_confusion_matrix(
                true_targets=(
                    representation_predictions[
                        "true_target"
                    ].to_numpy()
                ),
                predicted_targets=(
                    representation_predictions[
                        "predicted_target"
                    ].to_numpy()
                ),
                category_names=category_names,
                representation=representation,
            )
        )

    machine_readable_summary = {
        "experiment": (
            "Nested official-fold RBF-SVM "
            "classical baseline"
        ),
        "feature_cache": str(
            FEATURE_PATH
        ),
        "samples": int(
            len(targets)
        ),
        "classes": int(
            len(np.unique(targets))
        ),
        "outer_folds": [
            int(value)
            for value in sorted(
                np.unique(folds)
            )
        ],
        "inner_selection_metric": (
            "macro_f1"
        ),
        "parameter_grid": {
            "C": parameter_grid[
                "classifier__C"
            ],
            "gamma": parameter_grid[
                "classifier__gamma"
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "librosa": librosa.__version__,
        },
        "summary": summary_rows,
        "outputs": {
            "fold_metrics": str(
                fold_metrics_path
            ),
            "summary_csv": str(
                summary_path
            ),
            "oof_predictions": str(
                predictions_path
            ),
            "comparison_figure": str(
                comparison_figure_path
            ),
            "confusion_figures": (
                confusion_figure_paths
            ),
        },
    }

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            machine_readable_summary,
            file,
            indent=4,
        )

    print()
    print("=" * 72)
    print("FINAL FIVE-FOLD SUMMARY")
    print("=" * 72)
    print(
        summary_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print(f"Saved: {fold_metrics_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {summary_json_path}")
    print(f"Saved: {comparison_figure_path}")

    for path in confusion_figure_paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()