from pathlib import Path
import json

import pandas as pd
import soundfile as sf


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "ESC-50"
METADATA_PATH = DATASET_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = DATASET_ROOT / "audio"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_JSON = RESULTS_DIR / "dataset_validation.json"
CLASS_DISTRIBUTION_CSV = RESULTS_DIR / "class_distribution.csv"


# ---------------------------------------------------------
# Expected ESC-50 properties
# ---------------------------------------------------------
EXPECTED_RECORDS = 2000
EXPECTED_CLASSES = 50
EXPECTED_CLIPS_PER_CLASS = 40
EXPECTED_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_CLIPS_PER_FOLD = 400
EXPECTED_SAMPLE_RATE = 44100
EXPECTED_CHANNELS = 1
EXPECTED_DURATION = 5.0
DURATION_TOLERANCE = 0.01


def main():
    print("=" * 60)
    print("ESC-50 DATASET VALIDATION")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Basic path checks
    # -----------------------------------------------------
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_DIR}"
        )

    # -----------------------------------------------------
    # 2. Load metadata
    # -----------------------------------------------------
    df = pd.read_csv(METADATA_PATH)

    required_columns = {
        "filename",
        "fold",
        "target",
        "category",
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Required metadata columns missing: {missing_columns}"
        )

    # -----------------------------------------------------
    # 3. Metadata statistics
    # -----------------------------------------------------
    num_records = len(df)
    num_classes = df["category"].nunique()

    class_distribution = (
        df.groupby(["target", "category"])
        .size()
        .reset_index(name="count")
        .sort_values("target")
    )

    fold_distribution = (
        df["fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    folds_present = set(int(x) for x in df["fold"].unique())

    duplicate_filenames = int(df["filename"].duplicated().sum())

    # -----------------------------------------------------
    # 4. Validate audio files
    # -----------------------------------------------------
    missing_files = []
    unreadable_files = []

    sample_rate_counts = {}
    channel_counts = {}
    durations = []

    for index, row in df.iterrows():
        filename = row["filename"]
        audio_path = AUDIO_DIR / filename

        if not audio_path.exists():
            missing_files.append(filename)
            continue

        try:
            info = sf.info(audio_path)

            sample_rate_counts[info.samplerate] = (
                sample_rate_counts.get(info.samplerate, 0) + 1
            )

            channel_counts[info.channels] = (
                channel_counts.get(info.channels, 0) + 1
            )

            durations.append(info.duration)

        except Exception as exc:
            unreadable_files.append(
                {
                    "filename": filename,
                    "error": str(exc),
                }
            )

        if (index + 1) % 250 == 0:
            print(f"Checked {index + 1}/{num_records} audio files...")

    # -----------------------------------------------------
    # 5. Duration checks
    # -----------------------------------------------------
    duration_min = min(durations) if durations else None
    duration_max = max(durations) if durations else None
    duration_mean = (
        sum(durations) / len(durations)
        if durations
        else None
    )

    unexpected_durations = sum(
        abs(duration - EXPECTED_DURATION) > DURATION_TOLERANCE
        for duration in durations
    )

    # -----------------------------------------------------
    # 6. Validation conditions
    # -----------------------------------------------------
    class_counts_correct = bool(
        len(class_distribution) == EXPECTED_CLASSES
        and (class_distribution["count"] == EXPECTED_CLIPS_PER_CLASS).all()
    )

    fold_counts_correct = bool(
        folds_present == EXPECTED_FOLDS
        and all(
            fold_distribution.get(fold, 0) == EXPECTED_CLIPS_PER_FOLD
            for fold in EXPECTED_FOLDS
        )
    )

    sample_rate_correct = (
        sample_rate_counts == {EXPECTED_SAMPLE_RATE: EXPECTED_RECORDS}
    )

    channels_correct = (
        channel_counts == {EXPECTED_CHANNELS: EXPECTED_RECORDS}
    )

    validation_checks = {
        "records_correct": num_records == EXPECTED_RECORDS,
        "classes_correct": num_classes == EXPECTED_CLASSES,
        "clips_per_class_correct": class_counts_correct,
        "folds_correct": fold_counts_correct,
        "no_duplicate_filenames": duplicate_filenames == 0,
        "no_missing_files": len(missing_files) == 0,
        "no_unreadable_files": len(unreadable_files) == 0,
        "sample_rate_correct": sample_rate_correct,
        "channels_correct": channels_correct,
        "durations_correct": unexpected_durations == 0,
    }

    overall_pass = all(validation_checks.values())

    # -----------------------------------------------------
    # 7. Save results
    # -----------------------------------------------------
    results = {
        "dataset": "ESC-50",
        "dataset_root": str(DATASET_ROOT),
        "records": num_records,
        "classes": num_classes,
        "clips_per_class_expected": EXPECTED_CLIPS_PER_CLASS,
        "folds_present": sorted(folds_present),
        "fold_distribution": {
            str(k): int(v)
            for k, v in fold_distribution.items()
        },
        "duplicate_filenames": duplicate_filenames,
        "missing_files": len(missing_files),
        "missing_file_names": missing_files,
        "unreadable_files": len(unreadable_files),
        "unreadable_file_details": unreadable_files,
        "sample_rate_distribution": {
            str(k): int(v)
            for k, v in sample_rate_counts.items()
        },
        "channel_distribution": {
            str(k): int(v)
            for k, v in channel_counts.items()
        },
        "duration_seconds": {
            "minimum": duration_min,
            "maximum": duration_max,
            "mean": duration_mean,
            "unexpected_count": unexpected_durations,
        },
        "checks": validation_checks,
        "validation_status": "PASS" if overall_pass else "FAIL",
    }

    with open(VALIDATION_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_CSV,
        index=False,
    )

    # -----------------------------------------------------
    # 8. Console summary
    # -----------------------------------------------------
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Dataset records:       {num_records}")
    print(f"Classes:               {num_classes}")
    print(f"Folds:                 {sorted(folds_present)}")
    print(f"Fold distribution:     {fold_distribution}")
    print(f"Duplicate filenames:   {duplicate_filenames}")
    print(f"Files missing:         {len(missing_files)}")
    print(f"Unreadable files:      {len(unreadable_files)}")
    print(f"Sample rates:          {sample_rate_counts}")
    print(f"Channels:              {channel_counts}")

    if durations:
        print(
            "Duration range:        "
            f"{duration_min:.6f} - {duration_max:.6f} seconds"
        )
        print(f"Mean duration:         {duration_mean:.6f} seconds")

    print(f"Unexpected durations:  {unexpected_durations}")

    print("-" * 60)

    for name, passed in validation_checks.items():
        print(f"{name:<30} {'PASS' if passed else 'FAIL'}")

    print("-" * 60)
    print(
        f"Validation status:     "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    print()
    print(f"Saved: {VALIDATION_JSON}")
    print(f"Saved: {CLASS_DISTRIBUTION_CSV}")


if __name__ == "__main__":
    main()