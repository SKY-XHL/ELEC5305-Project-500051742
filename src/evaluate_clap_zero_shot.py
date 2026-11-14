import argparse
import importlib.metadata as package_metadata
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_CACHE = PROJECT_ROOT / "models" / "huggingface"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CLAP_CACHE_PATH = CACHE_DIR / "clap_embeddings.npz"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_CACHE)
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import yaml
from msclap import CLAP
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot Microsoft CLAP on ESC-50."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached model and tokenizer files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing zero-shot result files.",
    )
    return parser.parse_args()


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_configuration():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as input_file:
        configuration = yaml.safe_load(input_file)

    clap_configuration = configuration.get("clap", {})
    prompt_templates = clap_configuration.get("prompt_templates", [])
    if not prompt_templates:
        raise ValueError("No CLAP prompt templates are defined in config.yaml.")

    for template in prompt_templates:
        try:
            rendered = str(template).format("test class")
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid prompt template {template!r}: {error}"
            ) from error
        if "test class" not in rendered:
            raise ValueError(
                f"Prompt template does not insert the class name: {template!r}"
            )

    return configuration


def load_clap_cache(expected_dimension):
    if not CLAP_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"CLAP embedding cache not found: {CLAP_CACHE_PATH}"
        )

    with np.load(CLAP_CACHE_PATH, allow_pickle=False) as cache:
        required_keys = {"clap", "filename", "target", "category", "fold"}
        missing_keys = required_keys - set(cache.files)
        if missing_keys:
            raise ValueError(
                f"CLAP cache is missing keys: {sorted(missing_keys)}"
            )

        audio_embeddings = cache["clap"].astype(np.float32)
        filenames = cache["filename"].astype(str)
        targets = cache["target"].astype(np.int64)
        categories = cache["category"].astype(str)
        folds = cache["fold"].astype(np.int64)

    expected_shape = (2000, expected_dimension)
    if audio_embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected CLAP shape {expected_shape}, found "
            f"{audio_embeddings.shape}."
        )

    if not np.isfinite(audio_embeddings).all():
        raise ValueError("CLAP audio embeddings contain NaN or infinite values.")

    if not (
        len(filenames)
        == len(targets)
        == len(categories)
        == len(folds)
        == 2000
    ):
        raise ValueError("CLAP cache metadata lengths are inconsistent.")

    return audio_embeddings, filenames, targets, categories, folds


def build_class_mapping(targets, categories):
    mapping_table = (
        pd.DataFrame({"target": targets, "category": categories})
        .drop_duplicates()
        .sort_values("target")
        .reset_index(drop=True)
    )

    if len(mapping_table) != 50:
        raise ValueError(
            f"Expected 50 target/category pairs, found {len(mapping_table)}."
        )

    expected_targets = np.arange(50, dtype=np.int64)
    actual_targets = mapping_table["target"].astype(np.int64).to_numpy()
    if not np.array_equal(actual_targets, expected_targets):
        raise ValueError(
            "ESC-50 target identifiers are not the expected contiguous 0-49."
        )

    category_counts = (
        pd.DataFrame({"target": targets, "category": categories})
        .groupby("target")["category"]
        .nunique()
    )
    if not (category_counts == 1).all():
        raise ValueError("At least one target maps to multiple category names.")

    return mapping_table


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Cannot normalize an embedding with an invalid norm.")
    return matrix / norms


def make_unicode_array(values):
    return np.asarray([str(value) for value in values], dtype=np.str_)


def top_confusion_records(
    raw_confusion,
    class_targets,
    class_categories,
    prompt_id,
    template,
):
    records = []
    for true_index, true_target in enumerate(class_targets):
        for predicted_index, predicted_target in enumerate(class_targets):
            if true_index == predicted_index:
                continue
            count = int(raw_confusion[true_index, predicted_index])
            if count == 0:
                continue
            records.append(
                {
                    "prompt_id": prompt_id,
                    "template": template,
                    "true_target": int(true_target),
                    "true_category": class_categories[true_index],
                    "predicted_target": int(predicted_target),
                    "predicted_category": class_categories[predicted_index],
                    "count": count,
                }
            )

    return sorted(records, key=lambda record: record["count"], reverse=True)


def main():
    arguments = parse_arguments()
    configuration = load_configuration()
    clap_configuration = configuration["clap"]

    expected_dimension = int(clap_configuration["embedding_dimension"])
    model_version = str(clap_configuration["version"])
    prompt_templates = [
        str(template) for template in clap_configuration["prompt_templates"]
    ]

    output_files = {
        "summary_csv": RESULTS_DIR / "clap_zero_shot_summary.csv",
        "fold_metrics": RESULTS_DIR / "clap_zero_shot_fold_metrics.csv",
        "predictions": RESULTS_DIR / "clap_zero_shot_predictions.csv",
        "class_metrics": RESULTS_DIR / "clap_zero_shot_class_metrics.csv",
        "top_confusions": RESULTS_DIR / "clap_zero_shot_top_confusions.csv",
        "prompt_definitions": RESULTS_DIR
        / "clap_zero_shot_prompt_definitions.csv",
        "summary_json": RESULTS_DIR / "clap_zero_shot_summary.json",
        "text_cache": CACHE_DIR / "clap_zero_shot_text_embeddings.npz",
    }

    if output_files["summary_json"].exists() and not arguments.force:
        print(
            "Existing zero-shot results found. Use --force only if they "
            "must be regenerated."
        )
        print(output_files["summary_json"])
        return

    (
        audio_embeddings,
        filenames,
        targets,
        categories,
        folds,
    ) = load_clap_cache(expected_dimension)

    class_mapping = build_class_mapping(targets, categories)
    class_targets = class_mapping["target"].astype(np.int64).to_numpy()
    class_categories = class_mapping["category"].astype(str).tolist()
    class_text = [category.replace("_", " ") for category in class_categories]
    target_to_category = dict(
        zip(class_targets.tolist(), class_categories, strict=True)
    )

    audio_normalized = normalize_rows(audio_embeddings)

    print("=" * 68)
    print("MICROSOFT CLAP ZERO-SHOT ESC-50 EVALUATION")
    print("=" * 68)
    print(f"Audio embeddings:      {audio_embeddings.shape}")
    print(f"Classes:               {len(class_targets)}")
    print(f"Prompt templates:      {len(prompt_templates)}")
    print(f"Model version:         {model_version}")
    print("Primary prompt:        prompt_1")
    print("Loading frozen CLAP model for text embeddings...")

    model_load_start = time.perf_counter()
    clap_model = CLAP(version=model_version, use_cuda=False)
    model_load_seconds = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_seconds:.2f} seconds.")

    prediction_records = []
    summary_records = []
    fold_metric_records = []
    class_metric_records = []
    confusion_records = []
    prompt_definition_records = []
    text_embedding_arrays = []
    prompt_text_arrays = []
    evaluation_start = time.perf_counter()

    for template_index, template in enumerate(prompt_templates, start=1):
        prompt_id = f"prompt_{template_index}"
        prompts = [template.format(label) for label in class_text]

        text_start = time.perf_counter()
        text_tensor = clap_model.get_text_embeddings(prompts)
        text_seconds = time.perf_counter() - text_start
        text_embeddings = tensor_to_numpy(text_tensor).astype(np.float32)

        expected_text_shape = (50, expected_dimension)
        if text_embeddings.shape != expected_text_shape:
            raise ValueError(
                f"Expected text shape {expected_text_shape}, found "
                f"{text_embeddings.shape}."
            )

        if not np.isfinite(text_embeddings).all():
            raise ValueError(
                f"Text embeddings for {prompt_id} contain NaN or Inf."
            )

        text_normalized = normalize_rows(text_embeddings)
        cosine_similarity = audio_normalized @ text_normalized.T
        predicted_positions = np.argmax(cosine_similarity, axis=1)
        predicted_targets = class_targets[predicted_positions]
        predicted_categories = np.asarray(
            [target_to_category[int(value)] for value in predicted_targets],
            dtype=np.str_,
        )
        maximum_cosine = cosine_similarity[
            np.arange(len(targets)), predicted_positions
        ]

        accuracy = float(accuracy_score(targets, predicted_targets))
        macro_f1 = float(
            f1_score(
                targets,
                predicted_targets,
                labels=class_targets,
                average="macro",
                zero_division=0,
            )
        )

        fold_accuracies = []
        fold_macro_f1_values = []
        for fold in sorted(np.unique(folds).tolist()):
            fold_mask = folds == fold
            fold_accuracy = float(
                accuracy_score(targets[fold_mask], predicted_targets[fold_mask])
            )
            fold_macro_f1 = float(
                f1_score(
                    targets[fold_mask],
                    predicted_targets[fold_mask],
                    labels=class_targets,
                    average="macro",
                    zero_division=0,
                )
            )
            fold_accuracies.append(fold_accuracy)
            fold_macro_f1_values.append(fold_macro_f1)
            fold_metric_records.append(
                {
                    "prompt_id": prompt_id,
                    "template": template,
                    "fold": int(fold),
                    "test_samples": int(fold_mask.sum()),
                    "accuracy": fold_accuracy,
                    "macro_f1": fold_macro_f1,
                }
            )

        summary_records.append(
            {
                "prompt_id": prompt_id,
                "template": template,
                "is_primary": template_index == 1,
                "samples": len(targets),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "fold_accuracy_mean": float(np.mean(fold_accuracies)),
                "fold_accuracy_std": float(np.std(fold_accuracies, ddof=1)),
                "fold_macro_f1_mean": float(
                    np.mean(fold_macro_f1_values)
                ),
                "fold_macro_f1_std": float(
                    np.std(fold_macro_f1_values, ddof=1)
                ),
                "mean_max_cosine_similarity": float(
                    np.mean(maximum_cosine)
                ),
                "text_embedding_seconds": text_seconds,
            }
        )

        precision, recall, class_f1, support = (
            precision_recall_fscore_support(
                targets,
                predicted_targets,
                labels=class_targets,
                zero_division=0,
            )
        )
        for class_index, target in enumerate(class_targets):
            class_metric_records.append(
                {
                    "prompt_id": prompt_id,
                    "template": template,
                    "target": int(target),
                    "category": class_categories[class_index],
                    "precision": float(precision[class_index]),
                    "recall": float(recall[class_index]),
                    "f1": float(class_f1[class_index]),
                    "support": int(support[class_index]),
                }
            )

        raw_confusion = confusion_matrix(
            targets,
            predicted_targets,
            labels=class_targets,
        )
        row_totals = raw_confusion.sum(axis=1, keepdims=True)
        normalized_confusion = np.divide(
            raw_confusion,
            row_totals,
            out=np.zeros_like(raw_confusion, dtype=float),
            where=row_totals != 0,
        )

        raw_confusion_path = (
            RESULTS_DIR / f"clap_zero_shot_confusion_{prompt_id}_counts.csv"
        )
        normalized_confusion_path = (
            RESULTS_DIR
            / f"clap_zero_shot_confusion_{prompt_id}_normalized.csv"
        )
        pd.DataFrame(
            raw_confusion,
            index=class_categories,
            columns=class_categories,
        ).rename_axis("true_category").to_csv(raw_confusion_path)
        pd.DataFrame(
            normalized_confusion,
            index=class_categories,
            columns=class_categories,
        ).rename_axis("true_category").to_csv(normalized_confusion_path)

        confusion_records.extend(
            top_confusion_records(
                raw_confusion,
                class_targets,
                class_categories,
                prompt_id,
                template,
            )
        )

        for sample_index in range(len(targets)):
            prediction_records.append(
                {
                    "prompt_id": prompt_id,
                    "template": template,
                    "sample_index": sample_index,
                    "filename": filenames[sample_index],
                    "fold": int(folds[sample_index]),
                    "true_target": int(targets[sample_index]),
                    "true_category": categories[sample_index],
                    "predicted_target": int(predicted_targets[sample_index]),
                    "predicted_category": predicted_categories[sample_index],
                    "correct": bool(
                        predicted_targets[sample_index] == targets[sample_index]
                    ),
                    "maximum_cosine_similarity": float(
                        maximum_cosine[sample_index]
                    ),
                }
            )

        for class_index, target in enumerate(class_targets):
            prompt_definition_records.append(
                {
                    "prompt_id": prompt_id,
                    "template": template,
                    "target": int(target),
                    "category": class_categories[class_index],
                    "class_text": class_text[class_index],
                    "prompt_text": prompts[class_index],
                }
            )

        text_embedding_arrays.append(text_embeddings)
        prompt_text_arrays.append(make_unicode_array(prompts))

        print(
            f"{prompt_id}: accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}, text time={text_seconds:.2f}s"
        )

    evaluation_seconds = time.perf_counter() - evaluation_start

    summary_table = pd.DataFrame(summary_records)
    fold_metric_table = pd.DataFrame(fold_metric_records)
    prediction_table = pd.DataFrame(prediction_records)
    class_metric_table = pd.DataFrame(class_metric_records)
    confusion_table = pd.DataFrame(confusion_records)
    prompt_definition_table = pd.DataFrame(prompt_definition_records)

    summary_table.to_csv(output_files["summary_csv"], index=False)
    fold_metric_table.to_csv(output_files["fold_metrics"], index=False)
    prediction_table.to_csv(output_files["predictions"], index=False)
    class_metric_table.to_csv(output_files["class_metrics"], index=False)
    confusion_table.to_csv(output_files["top_confusions"], index=False)
    prompt_definition_table.to_csv(
        output_files["prompt_definitions"], index=False
    )

    np.savez_compressed(
        output_files["text_cache"],
        text_embeddings=np.stack(text_embedding_arrays).astype(np.float32),
        prompts=np.stack(prompt_text_arrays),
        prompt_id=make_unicode_array(
            [f"prompt_{index}" for index in range(1, len(prompt_templates) + 1)]
        ),
        template=make_unicode_array(prompt_templates),
        target=class_targets.astype(np.int64),
        category=make_unicode_array(class_categories),
    )

    primary_record = summary_records[0]
    best_record = max(summary_records, key=lambda record: record["macro_f1"])
    summary_document = {
        "status": "PASS",
        "experiment": "Microsoft CLAP zero-shot ESC-50 classification",
        "model_version": model_version,
        "frozen_encoder": True,
        "msclap_version": package_metadata.version("msclap"),
        "torch_version": torch.__version__,
        "device": "CPU",
        "samples": len(targets),
        "classes": len(class_targets),
        "folds": sorted(np.unique(folds).astype(int).tolist()),
        "prompt_count": len(prompt_templates),
        "primary_prompt_id": "prompt_1",
        "primary_prompt_template": primary_record["template"],
        "primary_accuracy": primary_record["accuracy"],
        "primary_macro_f1": primary_record["macro_f1"],
        "best_observed_prompt_id": best_record["prompt_id"],
        "best_observed_prompt_template": best_record["template"],
        "best_observed_accuracy": best_record["accuracy"],
        "best_observed_macro_f1": best_record["macro_f1"],
        "selection_note": (
            "prompt_1 is the predeclared primary result; the other prompts "
            "are sensitivity analyses and are not used to replace it."
        ),
        "model_load_seconds": model_load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "outputs": {
            key: str(value) for key, value in output_files.items()
        },
    }

    with output_files["summary_json"].open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(summary_document, output_file, indent=2)

    print()
    print("=" * 68)
    print("ZERO-SHOT SUMMARY")
    print("=" * 68)
    for record in summary_records:
        primary_marker = " [PRIMARY]" if record["is_primary"] else ""
        print(
            f"{record['prompt_id']}{primary_marker}: "
            f"accuracy={record['accuracy']:.4f}, "
            f"macro-F1={record['macro_f1']:.4f}"
        )
    print(f"Predictions saved:     {output_files['predictions']}")
    print(f"Summary saved:         {output_files['summary_json']}")
    print("Zero-shot status:      PASS")


if __name__ == "__main__":
    main()
