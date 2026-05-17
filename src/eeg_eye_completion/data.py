from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


WINDOW_DIR_NAME = "processed_eeg_eye_4s"
SEED_IV_DIR_NAME = "seed-iv"


@dataclass
class EmotionArrays:
    eeg: np.ndarray
    eye: np.ndarray
    label: np.ndarray
    subject: np.ndarray | None = None
    session: np.ndarray | None = None
    sample_index: np.ndarray | None = None
    source_id: np.ndarray | None = None

    @property
    def num_classes(self) -> int:
        return int(np.max(self.label)) + 1

    def take(self, indices: np.ndarray) -> "EmotionArrays":
        return EmotionArrays(
            eeg=self.eeg[indices],
            eye=self.eye[indices],
            label=self.label[indices],
            subject=None if self.subject is None else self.subject[indices],
            session=None if self.session is None else self.session[indices],
            sample_index=None if self.sample_index is None else self.sample_index[indices],
            source_id=None if self.source_id is None else self.source_id[indices],
        )


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "Dataset"


def _as_label(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if labels.size and labels.min() == 1:
        labels = labels - 1
    return labels


def _limit_arrays(arrays: EmotionArrays, limit: int | None) -> EmotionArrays:
    if limit is None or limit <= 0 or limit >= arrays.label.shape[0]:
        return arrays
    return arrays.take(np.arange(limit))


def load_window_arrays(dataset_root: Path, split: str, limit: int | None = None) -> EmotionArrays:
    path = dataset_root / WINDOW_DIR_NAME / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Window-level data not found: {path}. Run Dataset/process_eeg_eye_4s.py first."
        )
    data = np.load(path)
    arrays = EmotionArrays(
        eeg=np.asarray(data["eeg"], dtype=np.float32),
        eye=np.asarray(data["eye"], dtype=np.float32),
        label=_as_label(data["label"]),
        subject=np.asarray(data["subject"], dtype=np.int64) if "subject" in data else None,
        session=np.asarray(data["session"], dtype=np.int64) if "session" in data else None,
        sample_index=np.asarray(data["sample_index"], dtype=np.int64) if "sample_index" in data else None,
        source_id=np.asarray(data["source_id"]) if "source_id" in data else None,
    )
    return _limit_arrays(arrays, limit)


def _stratified_split(labels: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        n_test = max(1, int(round(indices.shape[0] * test_ratio)))
        test_indices.append(indices[:n_test])
        train_indices.append(indices[n_test:])
    train = np.concatenate(train_indices)
    test = np.concatenate(test_indices)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _load_seed_iv_full(dataset_root: Path) -> EmotionArrays:
    seed_dir = dataset_root / SEED_IV_DIR_NAME
    de_path = seed_dir / "DE_allbands.mat"
    eye_path = seed_dir / "EYE_data.mat"
    if not de_path.exists() or not eye_path.exists():
        raise FileNotFoundError(f"SEED-IV files not found under {seed_dir}")
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading MATLAB v7.3 seed-iv files requires h5py. Install it with `python3 -m pip install h5py`."
        ) from exc

    bands = ("delta", "theta", "alpha", "beta", "gamma")
    with h5py.File(de_path, "r") as f:
        eeg_bands = []
        for band in bands:
            # MATLAB v7.3 stores this as [time_segments, channels, samples].
            band_data = np.asarray(f[f"DE_features/{band}"], dtype=np.float32).transpose(2, 0, 1)
            eeg_bands.append(band_data)
        eeg = np.stack(eeg_bands, axis=2)
        labels = _as_label(np.asarray(f["labels"]))

    with h5py.File(eye_path, "r") as f:
        # Stored as [time_segments, eye_features, samples].
        eye = np.asarray(f["EYE_data"], dtype=np.float32).transpose(2, 0, 1)

    if eeg.shape[0] != eye.shape[0] or eeg.shape[0] != labels.shape[0]:
        raise ValueError(f"SEED-IV shapes are not aligned: eeg={eeg.shape}, eye={eye.shape}, labels={labels.shape}")
    return EmotionArrays(
        eeg=eeg,
        eye=eye,
        label=labels,
        sample_index=np.arange(labels.shape[0], dtype=np.int64),
        source_id=np.asarray([f"seediv_{i:04d}" for i in range(labels.shape[0])]),
    )


def load_seed_iv_splits(
    dataset_root: Path,
    test_ratio: float = 0.2,
    seed: int = 0,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[EmotionArrays, EmotionArrays]:
    full = _load_seed_iv_full(dataset_root)
    train_idx, test_idx = _stratified_split(full.label, test_ratio=test_ratio, seed=seed)
    return _limit_arrays(full.take(train_idx), train_limit), _limit_arrays(full.take(test_idx), test_limit)


def standardize_splits(train: EmotionArrays, test: EmotionArrays) -> dict[str, np.ndarray]:
    eeg_mean = train.eeg.mean(axis=0, keepdims=True)
    eeg_std = train.eeg.std(axis=0, keepdims=True)
    eye_mean = train.eye.mean(axis=0, keepdims=True)
    eye_std = train.eye.std(axis=0, keepdims=True)
    eeg_std[eeg_std < 1e-6] = 1.0
    eye_std[eye_std < 1e-6] = 1.0
    train.eeg = ((train.eeg - eeg_mean) / eeg_std).astype(np.float32)
    test.eeg = ((test.eeg - eeg_mean) / eeg_std).astype(np.float32)
    train.eye = ((train.eye - eye_mean) / eye_std).astype(np.float32)
    test.eye = ((test.eye - eye_mean) / eye_std).astype(np.float32)
    return {
        "eeg_mean": eeg_mean.astype(np.float32),
        "eeg_std": eeg_std.astype(np.float32),
        "eye_mean": eye_mean.astype(np.float32),
        "eye_std": eye_std.astype(np.float32),
    }


def make_missing_mask(
    n_samples: int,
    mode: str,
    missing_rate: float,
    seed: int,
    n_modalities: int = 2,
) -> np.ndarray:
    mode = mode.lower()
    mask = np.ones((n_samples, n_modalities), dtype=np.float32)
    if mode in {"none", "complete"} or missing_rate <= 0:
        return mask
    if mode in {"missing_eeg", "eeg"}:
        mask[:, 0] = 0.0
        return mask
    if mode in {"missing_eye", "eye"}:
        mask[:, 1] = 0.0
        return mask
    if mode not in {"random", "random_single"}:
        raise ValueError(f"Unknown missing mode: {mode}")

    rng = np.random.default_rng(seed)
    dropped = rng.random(n_samples) < missing_rate
    missing_modality = rng.integers(0, n_modalities, size=n_samples)
    mask[np.arange(n_samples)[dropped], missing_modality[dropped]] = 0.0
    return mask


class EmotionMissingDataset(Dataset):
    def __init__(self, arrays: EmotionArrays, missing_mode: str, missing_rate: float, seed: int):
        self.arrays = arrays
        self.mask = make_missing_mask(
            n_samples=arrays.label.shape[0],
            mode=missing_mode,
            missing_rate=missing_rate,
            seed=seed,
        )

    def __len__(self) -> int:
        return int(self.arrays.label.shape[0])

    def __getitem__(self, index: int) -> dict:
        item = {
            "eeg": torch.from_numpy(self.arrays.eeg[index]).float(),
            "eye": torch.from_numpy(self.arrays.eye[index]).float(),
            "label": torch.tensor(int(self.arrays.label[index]), dtype=torch.long),
            "mask": torch.from_numpy(self.mask[index]).float(),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.arrays.subject is not None:
            item["subject"] = torch.tensor(int(self.arrays.subject[index]), dtype=torch.long)
        if self.arrays.session is not None:
            item["session"] = torch.tensor(int(self.arrays.session[index]), dtype=torch.long)
        if self.arrays.sample_index is not None:
            item["sample_index"] = torch.tensor(int(self.arrays.sample_index[index]), dtype=torch.long)
        if self.arrays.source_id is not None:
            item["source_id"] = str(self.arrays.source_id[index])
        return item


def build_emotion_dataloaders(
    data_mode: str,
    dataset_root: Path | None = None,
    batch_size: int = 64,
    missing_mode: str = "random",
    missing_rate: float = 0.3,
    seed: int = 0,
    num_workers: int = 0,
    normalize: bool | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    root = (dataset_root or default_dataset_root()).resolve()
    data_mode = data_mode.lower()
    if data_mode == "window":
        train = load_window_arrays(root, "train", train_limit)
        test = load_window_arrays(root, "test", test_limit)
        should_normalize = False if normalize is None else normalize
    elif data_mode == "trial":
        train, test = load_seed_iv_splits(root, seed=seed, train_limit=train_limit, test_limit=test_limit)
        should_normalize = True if normalize is None else normalize
    else:
        raise ValueError(f"Unknown data mode: {data_mode}")

    stats = {}
    if should_normalize:
        stats = standardize_splits(train, test)

    train_dataset = EmotionMissingDataset(train, missing_mode=missing_mode, missing_rate=missing_rate, seed=seed)
    test_dataset = EmotionMissingDataset(test, missing_mode=missing_mode, missing_rate=missing_rate, seed=seed + 1009)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "data_mode": data_mode,
        "dataset_root": str(root),
        "num_classes": max(train.num_classes, test.num_classes),
        "eeg_shape": tuple(train.eeg.shape[1:]),
        "eye_shape": tuple(train.eye.shape[1:]),
        "train_samples": int(train.label.shape[0]),
        "test_samples": int(test.label.shape[0]),
        "normalized": should_normalize,
        "stats": stats,
    }
    return train_loader, test_loader, meta


def iter_missing_modes() -> Iterable[str]:
    return ("random", "missing_eeg", "missing_eye")
