"""Trajectory conditions aligned with FSQ token windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .fsq_token_dataset import FSQTokenDataset, FSQTokenStore, FSQTokenWindow


@dataclass(frozen=True)
class TrajectoryNormalization:
    """Training-split normalization for root-local trajectory controls."""

    mean: np.ndarray
    std: np.ndarray
    valid_frames: int

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1 or self.mean.shape != self.std.shape:
            raise ValueError("Trajectory mean and std must be same-length vectors")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Trajectory normalization contains non-finite values")
        if np.any(self.std <= 0.0):
            raise ValueError("Trajectory normalization std values must be positive")

    @property
    def trajectory_dim(self) -> int:
        return int(self.mean.shape[0])

    def as_checkpoint(self) -> dict[str, object]:
        return {
            "mean": self.mean.astype(np.float32),
            "std": self.std.astype(np.float32),
            "valid_frames": int(self.valid_frames),
        }

    @classmethod
    def from_checkpoint(cls, values: dict[str, object]) -> "TrajectoryNormalization":
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
            valid_frames=int(values.get("valid_frames", 0)),
        )


class FSQTrajectoryStore:
    """Mmap-backed trajectory shards that exactly mirror an FSQ token store."""

    def __init__(
        self,
        database: Path,
        trajectory_files: list[Path],
        valid_files: list[Path],
        num_frames: np.ndarray,
        trajectory_dim: int,
        future_frames: np.ndarray,
        feature_order: str,
    ) -> None:
        self.database = Path(database)
        self.trajectory_files = trajectory_files
        self.valid_files = valid_files
        self.num_frames = np.asarray(num_frames, dtype=np.int32)
        self.trajectory_dim = int(trajectory_dim)
        self.future_frames = np.asarray(future_frames, dtype=np.int32)
        self.feature_order = str(feature_order)
        self._values: list[np.ndarray] | None = None
        self._valid: list[np.ndarray] | None = None

    def _ensure_open(self) -> None:
        if self._values is not None and self._valid is not None:
            return
        self._values = [np.load(path, mmap_mode="r") for path in self.trajectory_files]
        self._valid = [np.load(path, mmap_mode="r") for path in self.valid_files]
        for shard_idx, (values, valid, num_frames) in enumerate(
            zip(self._values, self._valid, self.num_frames.tolist())
        ):
            expected_values = (int(num_frames), self.trajectory_dim)
            if values.shape != expected_values or valid.shape != (int(num_frames),):
                raise ValueError(
                    f"Trajectory shard {shard_idx} has values={values.shape}, valid={valid.shape}; "
                    f"expected {expected_values} and {(int(num_frames),)}"
                )

    def close(self) -> None:
        """Release mmap references; workers reopen shards lazily when needed."""
        self._values = None
        self._valid = None

    def __getstate__(self) -> dict:
        # Do not pickle full memmap state into spawned DataLoader workers.
        state = dict(self.__dict__)
        state["_values"] = None
        state["_valid"] = None
        return state

    def raw_window(self, shard_idx: int, start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_open()
        if shard_idx < 0 or shard_idx >= len(self.num_frames):
            raise IndexError(f"Invalid trajectory shard index {shard_idx}")
        if start_idx < 0 or end_idx < start_idx or end_idx > int(self.num_frames[shard_idx]):
            raise IndexError(
                f"Trajectory slice [{start_idx}, {end_idx}) exceeds shard {shard_idx} length "
                f"{self.num_frames[shard_idx]}"
            )
        assert self._values is not None and self._valid is not None
        return (
            np.asarray(self._values[shard_idx][start_idx:end_idx], dtype=np.float32).copy(),
            np.asarray(self._valid[shard_idx][start_idx:end_idx], dtype=bool).copy(),
        )

    def normalized_window(
        self,
        shard_idx: int,
        start_idx: int,
        end_idx: int,
        normalization: TrajectoryNormalization,
    ) -> tuple[np.ndarray, np.ndarray]:
        values, valid = self.raw_window(shard_idx, start_idx, end_idx)
        if normalization.trajectory_dim != self.trajectory_dim:
            raise ValueError(
                f"Normalization dim={normalization.trajectory_dim} does not match trajectory dim={self.trajectory_dim}"
            )
        values = (values - normalization.mean) / normalization.std
        values[~valid] = 0.0
        return values.astype(np.float32, copy=False), valid

    def controls_for_input_tokens(
        self,
        shard_idx: int,
        input_start_idx: int,
        input_length: int,
        normalization: TrajectoryNormalization,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return controls for predictions emitted by consecutive input tokens.

        Input token ``x_t`` predicts ``x_{t+1}``, so its trajectory condition
        is taken from local frame ``t+1``.  This makes a newly supplied command
        affect the very next sampled FSQ token rather than waiting a frame.
        """
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        return self.normalized_window(
            shard_idx,
            input_start_idx + 1,
            input_start_idx + 1 + input_length,
            normalization,
        )


def build_fsq_trajectory_store(database: str | Path, token_store: FSQTokenStore) -> FSQTrajectoryStore:
    database = Path(database)
    metadata_path = database / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing aligned trajectory metadata: {metadata_path}")
    data = np.load(metadata_path, allow_pickle=True)
    required = (
        "trajectory_files",
        "valid_files",
        "num_frames",
        "trajectory_dim",
        "future_frames",
        "range_names",
        "range_mirror",
        "tokenizer_checkpoint_sha256",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Trajectory database metadata is missing: {missing}")
    if str(np.asarray(data["tokenizer_checkpoint_sha256"]).item()) != token_store.checkpoint_sha256:
        raise ValueError("Trajectory database and token database use different FSQ tokenizer checkpoints")
    num_frames = np.asarray(data["num_frames"], dtype=np.int32)
    range_names = np.asarray(data["range_names"], dtype=object)
    range_mirror = np.asarray(data["range_mirror"], dtype=bool)
    if not np.array_equal(num_frames, token_store.num_frames):
        raise ValueError("Trajectory database and token database have different shard lengths")
    if [str(name) for name in range_names.tolist()] != [str(name) for name in token_store.range_names.tolist()]:
        raise ValueError("Trajectory database and token database have different range ordering")
    if not np.array_equal(range_mirror, token_store.range_mirror):
        raise ValueError("Trajectory database and token database have different mirror ordering")
    trajectory_rel = [str(path) for path in np.asarray(data["trajectory_files"], dtype=object).tolist()]
    valid_rel = [str(path) for path in np.asarray(data["valid_files"], dtype=object).tolist()]
    if len(trajectory_rel) != len(token_store.token_files) or len(valid_rel) != len(token_store.token_files):
        raise ValueError("Trajectory database has a different number of shards from the token database")
    return FSQTrajectoryStore(
        database=database,
        trajectory_files=[database / path for path in trajectory_rel],
        valid_files=[database / path for path in valid_rel],
        num_frames=num_frames,
        trajectory_dim=int(np.asarray(data["trajectory_dim"]).item()),
        future_frames=np.asarray(data["future_frames"], dtype=np.int32),
        feature_order=(
            str(np.asarray(data["trajectory_feature_order"]).item())
            if "trajectory_feature_order" in data.files
            else "legacy-unspecified"
        ),
    )


def fit_trajectory_normalization(
    trajectory_store: FSQTrajectoryStore,
    train_windows: Iterable[FSQTokenWindow],
    min_std: float = 1e-6,
) -> TrajectoryNormalization:
    """Fit statistics from the training targets, never val/test windows."""
    if min_std <= 0.0:
        raise ValueError("min_std must be positive")
    coverage = [np.zeros(int(length), dtype=bool) for length in trajectory_store.num_frames.tolist()]
    for window in train_windows:
        # Targets are [start + 1, end), matching the shifted control alignment.
        coverage[window.shard_idx][window.start_idx + 1 : window.end_idx] = True
    trajectory_store._ensure_open()
    assert trajectory_store._values is not None and trajectory_store._valid is not None
    total = np.zeros(trajectory_store.trajectory_dim, dtype=np.float64)
    total_squared = np.zeros(trajectory_store.trajectory_dim, dtype=np.float64)
    count = 0
    for values, valid, selected in zip(trajectory_store._values, trajectory_store._valid, coverage):
        mask = np.asarray(valid, dtype=bool) & selected
        if not np.any(mask):
            continue
        selected_values = np.asarray(values[mask], dtype=np.float64)
        total += selected_values.sum(axis=0)
        total_squared += np.square(selected_values).sum(axis=0)
        count += int(selected_values.shape[0])
    if count == 0:
        raise ValueError("No valid trajectory frames overlap the training token windows")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), min_std**2)
    normalization = TrajectoryNormalization(
        mean=mean.astype(np.float32),
        std=np.sqrt(variance).astype(np.float32),
        valid_frames=count,
    )
    trajectory_store.close()
    return normalization


class FSQConditionalDataset(Dataset):
    """Token windows plus controls for their next-token targets."""

    def __init__(
        self,
        split: str,
        token_store: FSQTokenStore,
        trajectory_store: FSQTrajectoryStore,
        normalization: TrajectoryNormalization,
    ) -> None:
        if normalization.trajectory_dim != trajectory_store.trajectory_dim:
            raise ValueError("Trajectory normalization and store dimensions differ")
        self.tokens = FSQTokenDataset(split, token_store)
        self.trajectory_store = trajectory_store
        self.normalization = normalization

    @property
    def windows(self) -> list[FSQTokenWindow]:
        return self.tokens.windows

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        item = self.tokens[index]
        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        shard_idx = int(item["shard_idx"])
        if end_idx - start_idx < 2:
            raise ValueError("Conditional generator requires token windows with at least two frames")
        values, valid = self.trajectory_store.normalized_window(
            shard_idx,
            start_idx + 1,
            end_idx,
            self.normalization,
        )
        item["trajectory"] = torch.from_numpy(values)
        item["trajectory_valid"] = torch.from_numpy(valid)
        return item
