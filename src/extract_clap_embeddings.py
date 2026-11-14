import argparse
import importlib.metadata as package_metadata
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATASET_ROOT = PROJECT_ROOT / "data" / "ESC-50"
METADATA_PATH = DATASET_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = DATASET_ROOT / "audio"
MODEL_CACHE = PROJECT_ROOT / "models" / "huggingface"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CLASSICAL_CACHE_PATH = CACHE_DIR / "classical_features.npz"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep all Hugging Face files inside the final-project workspace. These
# variables must be set before importing msclap or transformers.
os.environ["HF_HOME"] = str(MODEL_CACHE)
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
import yaml
from msclap import CLAP


REQUIRED_METADATA_COLUMNS = {
    "filename",
    "fold",
    "target",
    "category",
}


class SoundFileCLAP(CLAP):
    """CLAP wrapper that reads ESC-50 WAV files without TorchCodec."""

    def read_audio(self, audio_path, resample=True):
        """Read WAV audio and return a channels-first float tensor."""
        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )

        # SoundFile returns [samples, channels]. ESC-50 is mono, but the
        # averaging keeps the loader safe if a different file is stereo.
        if audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1, keepdims=True)

        waveform = torch.from_numpy(audio_data.T.copy())
        output_sample_rate = int(sample_rate)
        target_sample_rate = int(self.args.sampling_rate)

        if resample and output_sample_rate != target_sample_rate:
            waveform = audio_functional.resample(
                waveform,
                output_sample_rate,
                target_sample_rate,
            )
            output_sample_rate = target_sample_rate

        return waveform, output_sample_rate


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract frozen Microsoft CLAP embeddings for ESC-50."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke50", "full"),
        default="smoke50",
        help=(
            "smoke50 extracts one clip per class; full extracts all "
            "2,000 clips."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size stored in config.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing output/checkpoint files and start again.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only model and tokenizer files already in the local cache.",
    )
    return parser.parse_args()


def load_configuration():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as input_file:
        configuration = yaml.safe_load(input_file)

    if "clap" not in configuration:
        raise KeyError("config.yaml does not contain a clap section.")

    return configuration


def load_and_validate_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {METADATA_PATH}")

    metadata_table = pd.read_csv(METADATA_PATH)
    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata_table.columns)

    if missing_columns:
        raise ValueError(
            "ESC-50 metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(metadata_table) != 2000:
        raise ValueError(
            f"Expected 2,000 ESC-50 records, found {len(metadata_table)}."
        )

    if metadata_table["filename"].duplicated().any():
        raise ValueError("ESC-50 metadata contains duplicate filenames.")

    if metadata_table["target"].nunique() != 50:
        raise ValueError("ESC-50 metadata does not contain exactly 50 classes.")

    missing_audio = [
        filename
        for filename in metadata_table["filename"].astype(str)
        if not (AUDIO_DIR / filename).exists()
    ]
    if missing_audio:
        raise FileNotFoundError(
            f"Missing {len(missing_audio)} audio files; first: {missing_audio[0]}"
        )

    return metadata_table


def validate_metadata_against_classical_cache(metadata_table):
    if not CLASSICAL_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Classical feature cache not found: {CLASSICAL_CACHE_PATH}"
        )

    with np.load(CLASSICAL_CACHE_PATH, allow_pickle=False) as classical_data:
        required_keys = {"filename", "target", "category", "fold"}
        missing_keys = required_keys - set(classical_data.files)
        if missing_keys:
            raise ValueError(
                "Classical cache is missing metadata keys: "
                f"{sorted(missing_keys)}"
            )

        comparisons = {
            "filename": np.array_equal(
                metadata_table["filename"].astype(str).to_numpy(),
                classical_data["filename"].astype(str),
            ),
            "target": np.array_equal(
                metadata_table["target"].astype(np.int64).to_numpy(),
                classical_data["target"].astype(np.int64),
            ),
            "category": np.array_equal(
                metadata_table["category"].astype(str).to_numpy(),
                classical_data["category"].astype(str),
            ),
            "fold": np.array_equal(
                metadata_table["fold"].astype(np.int64).to_numpy(),
                classical_data["fold"].astype(np.int64),
            ),
        }

    failed_comparisons = [
        field for field, matches in comparisons.items() if not matches
    ]
    if failed_comparisons:
        raise ValueError(
            "ESC-50 metadata order does not match the classical cache for: "
            f"{failed_comparisons}"
        )


def select_records(metadata_table, mode):
    if mode == "full":
        return metadata_table.reset_index(drop=True).copy()

    selected = (
        metadata_table.sort_values(["target", "filename"])
        .groupby("target", sort=True, as_index=False)
        .head(1)
        .sort_values("target")
        .reset_index(drop=True)
    )

    if len(selected) != 50 or selected["target"].nunique() != 50:
        raise ValueError(
            "The smoke50 selection did not produce one clip for each class."
        )

    return selected


def output_paths(mode):
    if mode == "full":
        return (
            CACHE_DIR / "clap_embeddings.npz",
            CACHE_DIR / "clap_embeddings_checkpoint.npz",
            RESULTS_DIR / "clap_embedding_summary.json",
        )

    return (
        CACHE_DIR / "clap_embeddings_smoke_50.npz",
        CACHE_DIR / "clap_embeddings_smoke_50_checkpoint.npz",
        RESULTS_DIR / "clap_embedding_smoke_50_summary.json",
    )


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_embedding_cache(output_path, embeddings, records):
    filenames = np.asarray(
        records["filename"].astype(str).tolist(),
        dtype=np.str_,
    )
    categories = np.asarray(
        records["category"].astype(str).tolist(),
        dtype=np.str_,
    )

    np.savez_compressed(
        output_path,
        clap=embeddings.astype(np.float32),
        filename=filenames,
        target=records["target"].astype(np.int64).to_numpy(),
        category=categories,
        fold=records["fold"].astype(np.int64).to_numpy(),
    )


def load_checkpoint(checkpoint_path, selected_records, embedding_dimension):
    if not checkpoint_path.exists():
        return np.empty((0, embedding_dimension), dtype=np.float32)

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        required_keys = {"clap", "filename", "target", "category", "fold"}
        missing_keys = required_keys - set(checkpoint.files)
        if missing_keys:
            raise ValueError(
                f"Checkpoint is missing keys: {sorted(missing_keys)}"
            )

        embeddings = checkpoint["clap"].astype(np.float32)
        completed_count = embeddings.shape[0]

        if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dimension:
            raise ValueError(
                f"Checkpoint has unexpected shape: {embeddings.shape}"
            )

        if completed_count > len(selected_records):
            raise ValueError("Checkpoint contains more rows than requested.")

        completed_records = selected_records.iloc[:completed_count]
        checks = [
            np.array_equal(
                checkpoint["filename"].astype(str),
                completed_records["filename"].astype(str).to_numpy(),
            ),
            np.array_equal(
                checkpoint["target"].astype(np.int64),
                completed_records["target"].astype(np.int64).to_numpy(),
            ),
            np.array_equal(
                checkpoint["category"].astype(str),
                completed_records["category"].astype(str).to_numpy(),
            ),
            np.array_equal(
                checkpoint["fold"].astype(np.int64),
                completed_records["fold"].astype(np.int64).to_numpy(),
            ),
        ]

        if not all(checks):
            raise ValueError(
                "Checkpoint metadata does not match the requested records."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Checkpoint contains NaN or infinite values.")

    return embeddings


def validate_final_cache(output_path, expected_records, embedding_dimension):
    with np.load(output_path, allow_pickle=False) as saved_data:
        required_keys = {"clap", "filename", "target", "category", "fold"}
        missing_keys = required_keys - set(saved_data.files)
        if missing_keys:
            raise ValueError(
                f"Final cache is missing keys: {sorted(missing_keys)}"
            )

        embeddings = saved_data["clap"]
        expected_shape = (len(expected_records), embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected CLAP shape {expected_shape}, found {embeddings.shape}."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Final CLAP cache contains NaN or infinite values.")

        metadata_checks = {
            "filename": np.array_equal(
                saved_data["filename"].astype(str),
                expected_records["filename"].astype(str).to_numpy(),
            ),
            "target": np.array_equal(
                saved_data["target"].astype(np.int64),
                expected_records["target"].astype(np.int64).to_numpy(),
            ),
            "category": np.array_equal(
                saved_data["category"].astype(str),
                expected_records["category"].astype(str).to_numpy(),
            ),
            "fold": np.array_equal(
                saved_data["fold"].astype(np.int64),
                expected_records["fold"].astype(np.int64).to_numpy(),
            ),
        }

        failed_checks = [
            field for field, matches in metadata_checks.items() if not matches
        ]
        if failed_checks:
            raise ValueError(
                "Final cache metadata mismatch for: "
                f"{failed_checks}"
            )

    return expected_shape


def main():
    arguments = parse_arguments()
    configuration = load_configuration()
    clap_configuration = configuration["clap"]

    batch_size = (
        arguments.batch_size
        if arguments.batch_size is not None
        else int(clap_configuration["batch_size"])
    )
    if batch_size < 1:
        raise ValueError("Batch size must be at least one.")

    model_version = str(clap_configuration["version"])
    use_cuda_requested = bool(clap_configuration["use_cuda"])
    embedding_dimension = int(clap_configuration["embedding_dimension"])
    expected_model_sample_rate = int(
        clap_configuration["model_sample_rate"]
    )
    expected_model_duration = int(
        clap_configuration["model_duration_seconds"]
    )
    checkpoint_every_batches = int(
        clap_configuration["checkpoint_every_batches"]
    )

    metadata_table = load_and_validate_metadata()
    validate_metadata_against_classical_cache(metadata_table)
    selected_records = select_records(metadata_table, arguments.mode)

    output_path, checkpoint_path, summary_path = output_paths(arguments.mode)

    if output_path.exists() and not arguments.force:
        shape = validate_final_cache(
            output_path,
            selected_records,
            embedding_dimension,
        )
        print(f"Existing validated output: {output_path}")
        print(f"Embedding shape: {shape}")
        print("Use --force only if the cache must be regenerated.")
        return

    if arguments.force:
        existing_embeddings = np.empty(
            (0, embedding_dimension),
            dtype=np.float32,
        )
    else:
        existing_embeddings = load_checkpoint(
            checkpoint_path,
            selected_records,
            embedding_dimension,
        )

    completed_count = existing_embeddings.shape[0]
    embedding_chunks = []
    if completed_count:
        embedding_chunks.append(existing_embeddings)

    actual_use_cuda = use_cuda_requested and torch.cuda.is_available()
    device_name = (
        torch.cuda.get_device_name(0) if actual_use_cuda else "CPU"
    )

    print("=" * 68)
    print("FROZEN MICROSOFT CLAP EMBEDDING EXTRACTION")
    print("=" * 68)
    print(f"Mode:                  {arguments.mode}")
    print(f"Records requested:     {len(selected_records)}")
    print(f"Already completed:     {completed_count}")
    print(f"Batch size:            {batch_size}")
    print(f"Model version:         {model_version}")
    print(f"Device:                {device_name}")
    print(f"Output:                {output_path}")
    print(f"Checkpoint:            {checkpoint_path}")
    print("Loading frozen CLAP model...")

    model_load_start = time.perf_counter()
    clap_model = SoundFileCLAP(
        version=model_version,
        use_cuda=actual_use_cuda,
    )
    model_load_seconds = time.perf_counter() - model_load_start

    actual_model_sample_rate = int(clap_model.args.sampling_rate)
    if actual_model_sample_rate != expected_model_sample_rate:
        raise ValueError(
            "CLAP sample-rate mismatch: configuration expects "
            f"{expected_model_sample_rate}, model uses "
            f"{actual_model_sample_rate}."
        )

    actual_model_duration = int(clap_model.args.duration)
    if actual_model_duration != expected_model_duration:
        raise ValueError(
            "CLAP duration mismatch: configuration expects "
            f"{expected_model_duration}, model uses "
            f"{actual_model_duration}."
        )

    print(f"Model loaded in {model_load_seconds:.2f} seconds.")
    print(f"Model sample rate:     {actual_model_sample_rate} Hz")
    print(f"Model input duration:  {actual_model_duration} seconds")
    print()

    extraction_start = time.perf_counter()
    batch_number = 0

    for batch_start in range(completed_count, len(selected_records), batch_size):
        batch_number += 1
        batch_end = min(batch_start + batch_size, len(selected_records))
        batch_records = selected_records.iloc[batch_start:batch_end]
        audio_paths = [
            str(AUDIO_DIR / filename)
            for filename in batch_records["filename"].astype(str)
        ]

        batch_start_time = time.perf_counter()
        embedding_tensor = clap_model.get_audio_embeddings(audio_paths)
        batch_seconds = time.perf_counter() - batch_start_time

        batch_embeddings = tensor_to_numpy(embedding_tensor).astype(
            np.float32
        )
        expected_batch_shape = (
            len(batch_records),
            embedding_dimension,
        )

        if batch_embeddings.shape != expected_batch_shape:
            raise ValueError(
                "Unexpected batch embedding shape: expected "
                f"{expected_batch_shape}, found {batch_embeddings.shape}."
            )

        if not np.isfinite(batch_embeddings).all():
            raise ValueError(
                f"Batch {batch_start}:{batch_end} contains NaN or Inf."
            )

        embedding_chunks.append(batch_embeddings)
        processed_embeddings = np.concatenate(embedding_chunks, axis=0)
        processed_count = processed_embeddings.shape[0]

        print(
            f"Processed {processed_count:4d}/{len(selected_records)} "
            f"clips | batch {batch_seconds:.2f} s"
        )

        should_checkpoint = (
            batch_number % checkpoint_every_batches == 0
            or processed_count == len(selected_records)
        )
        if should_checkpoint:
            save_embedding_cache(
                checkpoint_path,
                processed_embeddings,
                selected_records.iloc[:processed_count],
            )
            print(f"Checkpoint saved at {processed_count} clips.")

    if embedding_chunks:
        all_embeddings = np.concatenate(embedding_chunks, axis=0)
    else:
        all_embeddings = existing_embeddings

    extraction_seconds = time.perf_counter() - extraction_start

    if all_embeddings.shape != (
        len(selected_records),
        embedding_dimension,
    ):
        raise ValueError(
            f"Final in-memory shape is incorrect: {all_embeddings.shape}"
        )

    save_embedding_cache(output_path, all_embeddings, selected_records)
    validated_shape = validate_final_cache(
        output_path,
        selected_records,
        embedding_dimension,
    )

    summary = {
        "status": "PASS",
        "mode": arguments.mode,
        "model": str(clap_configuration["implementation"]),
        "model_version": model_version,
        "frozen_encoder": True,
        "msclap_version": package_metadata.version("msclap"),
        "torch_version": torch.__version__,
        "device": device_name,
        "use_cuda": actual_use_cuda,
        "audio_loader": "soundfile",
        "source_sample_rate_hz": int(configuration["audio"]["sample_rate"]),
        "model_sample_rate_hz": actual_model_sample_rate,
        "model_input_duration_seconds": actual_model_duration,
        "records_requested": len(selected_records),
        "records_successful": len(selected_records),
        "records_failed": 0,
        "classes": int(selected_records["target"].nunique()),
        "folds": sorted(
            selected_records["fold"].astype(int).unique().tolist()
        ),
        "embedding_shape": list(validated_shape),
        "embedding_dimension": embedding_dimension,
        "batch_size": batch_size,
        "model_load_seconds": model_load_seconds,
        "embedding_extraction_seconds": extraction_seconds,
        "mean_embedding_seconds_per_clip": (
            extraction_seconds / len(selected_records)
        ),
        "output_file": str(output_path),
        "checkpoint_file": str(checkpoint_path),
        "metadata_order_matches_classical_cache": True,
        "finite_values_only": True,
    }

    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print()
    print("=" * 68)
    print("CLAP EMBEDDING SUMMARY")
    print("=" * 68)
    print(f"Embedding shape:       {validated_shape}")
    print(f"Classes represented:   {summary['classes']}")
    print(f"Records failed:        {summary['records_failed']}")
    print(f"Extraction time:       {extraction_seconds:.2f} seconds")
    print(
        "Mean time per clip:    "
        f"{summary['mean_embedding_seconds_per_clip']:.4f} seconds"
    )
    print(f"Saved cache:           {output_path}")
    print(f"Saved summary:         {summary_path}")
    print("Extraction status:     PASS")


if __name__ == "__main__":
    main()
