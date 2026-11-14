import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "ESC-50"
METADATA_PATH = DATASET_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = DATASET_ROOT / "audio"
MODEL_CACHE = PROJECT_ROOT / "models" / "huggingface"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# This must be set before importing msclap or transformers.
os.environ["HF_HOME"] = str(MODEL_CACHE)

import importlib.metadata as metadata
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from msclap import CLAP


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class SoundFileCLAP(CLAP):
    """CLAP wrapper that reads ESC-50 WAV files without TorchCodec."""

    def read_audio(self, audio_path, resample=True):
        """Read WAV audio and return a channels-first float tensor."""
        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )

        # SoundFile returns [samples, channels]. ESC-50 is mono, but
        # averaging here keeps the loader safe if another file is stereo.
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


def main():
    print("=" * 60)
    print("CLAP SINGLE-FILE SMOKE TEST")
    print("=" * 60)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    metadata_table = pd.read_csv(METADATA_PATH)

    if len(metadata_table) != 2000:
        raise ValueError(
            f"Expected 2000 ESC-50 records, found {len(metadata_table)}"
        )

    sample = metadata_table.iloc[0]
    filename = str(sample["filename"])
    category = str(sample["category"])
    fold = int(sample["fold"])
    target = int(sample["target"])
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    class_text = category.replace("_", " ")
    prompt = f"this is the sound of {class_text}"

    print(f"Audio file: {filename}")
    print(f"Category:   {category}")
    print(f"Target:     {target}")
    print(f"Fold:       {fold}")
    print(f"Prompt:     {prompt}")
    print(f"Device:     CPU")
    print()
    print("Loading Microsoft CLAP 2023...")
    print("The first run may download model and tokenizer files.")

    model_start = time.perf_counter()
    clap_model = SoundFileCLAP(version="2023", use_cuda=False)
    model_load_seconds = time.perf_counter() - model_start

    print(f"Model loaded in {model_load_seconds:.2f} seconds.")
    print("Extracting audio embedding...")

    audio_start = time.perf_counter()
    audio_embedding = clap_model.get_audio_embeddings([str(audio_path)])
    audio_seconds = time.perf_counter() - audio_start

    print("Extracting text embedding...")

    text_start = time.perf_counter()
    text_embedding = clap_model.get_text_embeddings([prompt])
    text_seconds = time.perf_counter() - text_start

    similarity = clap_model.compute_similarity(
        audio_embedding,
        text_embedding,
    )

    audio_array = tensor_to_numpy(audio_embedding).astype(np.float32)
    text_array = tensor_to_numpy(text_embedding).astype(np.float32)
    similarity_array = tensor_to_numpy(similarity).astype(np.float32)

    if audio_array.shape[0] != 1:
        raise ValueError(
            f"Expected one audio embedding, found shape {audio_array.shape}"
        )

    if text_array.shape[0] != 1:
        raise ValueError(
            f"Expected one text embedding, found shape {text_array.shape}"
        )

    if audio_array.shape[1] != text_array.shape[1]:
        raise ValueError(
            "Audio and text embedding dimensions do not match: "
            f"{audio_array.shape} versus {text_array.shape}"
        )

    if not np.isfinite(audio_array).all():
        raise ValueError("Audio embedding contains NaN or infinite values.")

    if not np.isfinite(text_array).all():
        raise ValueError("Text embedding contains NaN or infinite values.")

    if not np.isfinite(similarity_array).all():
        raise ValueError("Similarity contains NaN or infinite values.")

    cache_path = CACHE_DIR / "clap_smoke_1.npz"

    np.savez_compressed(
        cache_path,
        audio_embedding=audio_array,
        text_embedding=text_array,
        similarity=similarity_array,
        filename=np.asarray([filename]),
        category=np.asarray([category]),
        target=np.asarray([target], dtype=np.int64),
        fold=np.asarray([fold], dtype=np.int64),
        prompt=np.asarray([prompt]),
    )

    summary = {
        "status": "PASS",
        "model": "Microsoft CLAP",
        "model_version": "2023",
        "msclap_version": metadata.version("msclap"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "audio_loader": "soundfile",
        "source_sample_rate_hz": 44100,
        "clap_sample_rate_hz": int(clap_model.args.sampling_rate),
        "filename": filename,
        "category": category,
        "target": target,
        "fold": fold,
        "prompt": prompt,
        "audio_embedding_shape": list(audio_array.shape),
        "text_embedding_shape": list(text_array.shape),
        "similarity_shape": list(similarity_array.shape),
        "similarity_value": float(similarity_array[0, 0]),
        "model_load_seconds": model_load_seconds,
        "audio_embedding_seconds": audio_seconds,
        "text_embedding_seconds": text_seconds,
        "cache_file": str(cache_path),
    }

    summary_path = RESULTS_DIR / "clap_smoke_test.json"

    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print()
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Audio embedding shape: {audio_array.shape}")
    print(f"Text embedding shape:  {text_array.shape}")
    print(f"Similarity shape:      {similarity_array.shape}")
    print(f"Similarity value:      {similarity_array[0, 0]:.6f}")
    print(f"Audio time:            {audio_seconds:.2f} seconds")
    print(f"Text time:             {text_seconds:.2f} seconds")
    print(f"Saved cache:           {cache_path}")
    print(f"Saved summary:         {summary_path}")
    print("Smoke test status:     PASS")


if __name__ == "__main__":
    main()
