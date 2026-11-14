from pathlib import Path
import argparse
import json
import time

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_features(audio_path, config):
    audio_cfg = config["audio"]
    feat_cfg = config["features"]

    target_sr = int(audio_cfg["sample_rate"])

    y, sr = librosa.load(
        audio_path,
        sr=target_sr,
        mono=bool(audio_cfg["mono"]),
    )

    # -----------------------------------------------------
    # Shared Log-Mel representation
    # -----------------------------------------------------
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(feat_cfg["n_fft"]),
        hop_length=int(feat_cfg["hop_length"]),
        win_length=int(feat_cfg["win_length"]),
        window=feat_cfg["window"],
        n_mels=int(feat_cfg["n_mels"]),
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel_power,
        ref=np.max,
        top_db=float(feat_cfg["top_db"]),
    )

    # 128 means + 128 standard deviations = 256 dimensions
    logmel_vector = np.concatenate(
        [
            np.mean(log_mel, axis=1),
            np.std(log_mel, axis=1),
        ]
    ).astype(np.float32)

    # -----------------------------------------------------
    # MFCC derived from the same Log-Mel representation
    # -----------------------------------------------------
    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=int(feat_cfg["n_mfcc"]),
    )

    # 40 means + 40 standard deviations = 80 dimensions
    mfcc_vector = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
        ]
    ).astype(np.float32)

    return mfcc_vector, logmel_vector


def main():
    parser = argparse.ArgumentParser(
        description="Extract MFCC and Log-Mel features from ESC-50."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of clips for a smoke test.",
    )

    args = parser.parse_args()

    config = load_config()

    dataset_root = PROJECT_ROOT / config["dataset"]["root"]
    metadata_path = PROJECT_ROOT / config["dataset"]["metadata"]
    audio_dir = PROJECT_ROOT / config["dataset"]["audio_dir"]

    cache_dir = PROJECT_ROOT / "cache"
    results_dir = PROJECT_ROOT / "results"

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)

    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
        output_name = f"classical_features_smoke_{args.limit}.npz"
        summary_name = f"feature_extraction_smoke_{args.limit}.json"
    else:
        output_name = "classical_features.npz"
        summary_name = "feature_extraction_summary.json"

    print("=" * 60)
    print("CLASSICAL FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Audio clips: {len(df)}")
    print(f"Sample rate: {config['audio']['sample_rate']} Hz")
    print(
        f"STFT: n_fft={config['features']['n_fft']}, "
        f"win_length={config['features']['win_length']}, "
        f"hop_length={config['features']['hop_length']}"
    )
    print(f"Mel bands: {config['features']['n_mels']}")
    print(f"MFCC coefficients: {config['features']['n_mfcc']}")
    print()

    mfcc_features = []
    logmel_features = []

    filenames = []
    targets = []
    categories = []
    folds = []

    failed_files = []

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------
    # Librosa/Numba may perform one-time initialisation or JIT
    # compilation on the first feature extraction. Run one clip
    # before timing so that reported extraction time better
    # reflects steady-state processing cost.
    if len(df) > 0:
        warmup_path = audio_dir / df.iloc[0]["filename"]

        print(f"Warm-up clip: {df.iloc[0]['filename']}")

        warmup_start = time.perf_counter()

        extract_features(
            warmup_path,
            config,
        )

        warmup_seconds = time.perf_counter() - warmup_start

        print(f"Warm-up time: {warmup_seconds:.2f} seconds")
        print()

    else:
        warmup_seconds = 0.0

    start_time = time.perf_counter()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting features",
    ):
        audio_path = audio_dir / row["filename"]

        try:
            mfcc_vector, logmel_vector = extract_features(
                audio_path,
                config,
            )

            mfcc_features.append(mfcc_vector)
            logmel_features.append(logmel_vector)

            filenames.append(row["filename"])
            targets.append(int(row["target"]))
            categories.append(row["category"])
            folds.append(int(row["fold"]))

        except Exception as exc:
            failed_files.append(
                {
                    "filename": row["filename"],
                    "error": str(exc),
                }
            )

    elapsed = time.perf_counter() - start_time

    if len(mfcc_features) == 0:
        raise RuntimeError("No features were successfully extracted.")

    X_mfcc = np.stack(mfcc_features)
    X_logmel = np.stack(logmel_features)

    output_path = cache_dir / output_name

    np.savez_compressed(
        output_path,
        mfcc=X_mfcc,
        logmel=X_logmel,
        filename=np.asarray(filenames),
        target=np.asarray(targets, dtype=np.int64),
        category=np.asarray(categories),
        fold=np.asarray(folds, dtype=np.int64),
    )

    summary = {
        "clips_requested": int(len(df)),
        "clips_successful": int(len(filenames)),
        "clips_failed": int(len(failed_files)),
        "failed_files": failed_files,
        "mfcc_shape": list(X_mfcc.shape),
        "logmel_shape": list(X_logmel.shape),
        "mfcc_dimension": int(X_mfcc.shape[1]),
        "logmel_dimension": int(X_logmel.shape[1]),
        "warmup_seconds": warmup_seconds,
        "total_extraction_seconds": elapsed,
        "mean_seconds_per_clip": elapsed / len(filenames),
        "output_file": str(output_path),
    }

    summary_path = results_dir / summary_name

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Successful clips:      {len(filenames)}")
    print(f"Failed clips:          {len(failed_files)}")
    print(f"MFCC matrix shape:     {X_mfcc.shape}")
    print(f"Log-Mel matrix shape:  {X_logmel.shape}")
    print(f"Total time:            {elapsed:.2f} seconds")
    print(
        f"Mean time per clip:    "
        f"{elapsed / len(filenames):.4f} seconds"
    )
    print(f"Saved feature cache:   {output_path}")
    print(f"Saved summary:         {summary_path}")


if __name__ == "__main__":
    main()