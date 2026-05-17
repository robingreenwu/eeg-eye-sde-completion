#!/usr/bin/env python3
"""Print compact summaries for EEG DE and eye-tracking feature zip files."""

from __future__ import annotations

import argparse
import io
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _stats(array: np.ndarray) -> dict:
    arr = np.asarray(array)
    result = {"shape": tuple(arr.shape), "dtype": str(arr.dtype)}
    if np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            result.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "nan": int(np.isnan(arr).sum())
                    if np.issubdtype(arr.dtype, np.floating)
                    else 0,
                }
            )
    return result


def _label_distribution(labels: np.ndarray) -> dict:
    values, counts = np.unique(labels, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def inspect_eeg(zip_path: Path) -> None:
    print("\n=== EEG DE feature dataset ===")
    with zipfile.ZipFile(zip_path) as archive:
        npz_files = sorted(name for name in archive.namelist() if name.endswith(".npz"))
        print(f"zip: {zip_path}")
        print(f"npz files: {len(npz_files)}")
        for group in ("eeg_used_1s", "eeg_used_4s"):
            files = [name for name in npz_files if f"/{group}/" in name]
            print(f"\n[{group}] files: {len(files)}")
            if not files:
                continue

            sample_name = next(
                (name for name in files if name.endswith("/1_1.npz")),
                files[0],
            )
            npz_data = np.load(io.BytesIO(archive.read(sample_name)), allow_pickle=True)
            train_data = pickle.loads(npz_data["train_data"])
            test_data = pickle.loads(npz_data["test_data"])

            print(f"sample file: {sample_name}")
            print(f"npz keys: {list(npz_data.keys())}")
            print(f"feature bands: {list(train_data.keys())}")
            for band, values in train_data.items():
                print(f"  train {band}: {_stats(values)}")
            for band, values in test_data.items():
                print(f"  test  {band}: {_stats(values)}")
            for label_key in ("train_label", "test_label"):
                labels = np.asarray(npz_data[label_key])
                print(
                    f"  {label_key}: shape={labels.shape}, "
                    f"dist={_label_distribution(labels)}, first10={labels[:10].tolist()}"
                )

            shape_counts: Counter = Counter()
            label_counts: dict[str, defaultdict] = {
                "train_label": defaultdict(int),
                "test_label": defaultdict(int),
            }
            for name in files:
                data = np.load(io.BytesIO(archive.read(name)), allow_pickle=True)
                train = pickle.loads(data["train_data"])
                test = pickle.loads(data["test_data"])
                first_train = next(iter(train.values()))
                first_test = next(iter(test.values()))
                shape_counts[
                    (
                        tuple(first_train.shape),
                        tuple(first_test.shape),
                        tuple(data["train_label"].shape),
                        tuple(data["test_label"].shape),
                    )
                ] += 1
                for label_key in ("train_label", "test_label"):
                    for value, count in _label_distribution(data[label_key]).items():
                        label_counts[label_key][value] += count

            print(f"shape signatures: {dict(shape_counts)}")
            print(
                "label totals: "
                f"train={dict(label_counts['train_label'])}, "
                f"test={dict(label_counts['test_label'])}"
            )


def inspect_eye(zip_path: Path) -> None:
    print("\n=== Eye-tracking feature dataset ===")
    with zipfile.ZipFile(zip_path) as archive:
        files = sorted(
            name
            for name in archive.namelist()
            if name.startswith("04-Eye-tracking-feature/eye_tracking_feature/")
            and not name.endswith("/")
        )
        print(f"zip: {zip_path}")
        print(f"pickle files: {len(files)}")
        if not files:
            return

        sample_name = next((name for name in files if name.endswith("/1_1")), files[0])
        sample = pickle.loads(archive.read(sample_name))
        print(f"sample file: {sample_name}")
        print(f"keys: {list(sample.keys())}")
        for key, values in sample.items():
            arr = np.asarray(values)
            print(f"  {key}: {_stats(arr)}")
            if arr.ndim >= 2 and arr.shape[0] > 0:
                print(f"  {key} first row first 12: {np.round(arr[0, :12], 4).tolist()}")

        shape_counts: Counter = Counter()
        for name in files:
            data = pickle.loads(archive.read(name))
            shape_counts[
                (
                    tuple(np.asarray(data["train_data_eye"]).shape),
                    tuple(np.asarray(data["test_data_eye"]).shape),
                )
            ] += 1
        print(f"shape signatures: {dict(shape_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-zip", type=Path, required=True)
    parser.add_argument("--eye-zip", type=Path, required=True)
    args = parser.parse_args()

    inspect_eeg(args.eeg_zip)
    inspect_eye(args.eye_zip)


if __name__ == "__main__":
    main()
