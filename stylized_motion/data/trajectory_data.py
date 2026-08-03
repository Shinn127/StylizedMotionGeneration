"""Schema-v3 trajectory store aligned with a TokenStore."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_data import MMapShardCache, _check_common_index, _load_index, _require_manifest, _require_relative_path
from .sampling import SampleRequest
from .token_data import TokenDataset, TokenStore


@dataclass(frozen=True)
class TrajectoryNormalization:
    mean: np.ndarray
    std: np.ndarray
    valid_frames: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("Trajectory normalization mean/std must be one-dimensional and same-shaped")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
            raise ValueError("Trajectory normalization must be finite with positive std")
        if int(self.valid_frames) <= 0:
            raise ValueError("Trajectory normalization must contain at least one valid frame")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "valid_frames", int(self.valid_frames))

    @property
    def trajectory_dim(self) -> int:
        return int(np.asarray(self.mean).shape[0])

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float32),
            "std": np.asarray(self.std, dtype=np.float32),
            "valid_frames": int(self.valid_frames),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrajectoryNormalization":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
            valid_frames=int(value["valid_frames"]),
        )


@dataclass
class TrajectoryStore:
    database: Path
    manifest: dict[str, object]
    trajectory_files: list[Path]
    valid_files: list[Path]
    shard_num_frames: np.ndarray
    clip_ids: np.ndarray
    source_clip_ids: np.ndarray
    range_shard_indices: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    range_mirror: np.ndarray
    split_ids: np.ndarray
    range_names: tuple[str, ...]
    source_clip_names: tuple[str, ...]
    style_names: tuple[str, ...]
    action_names: tuple[str, ...]
    style_ids: np.ndarray
    action_ids: np.ndarray
    trajectory_dim: int
    future_frames: np.ndarray
    feature_order: str
    checkpoint_sha256: str
    feature_schema_hash: str
    normalization: TrajectoryNormalization
    max_open_shards: int = 32
    _value_cache: MMapShardCache | None = None
    _valid_cache: MMapShardCache | None = None

    @property
    def split_manifest_hash(self) -> str:
        return str(self.manifest["split_manifest_hash"])

    def _read_raw(self, request: SampleRequest, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(request, SampleRequest):
            raise TypeError("TrajectoryStore.read_aligned requires a validated SampleRequest")
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        variant_idx = int(request.variant_idx)
        if variant_idx < 0 or variant_idx >= len(self.range_names):
            raise IndexError(f"Invalid trajectory range index {variant_idx}")
        shard_idx = int(request.shard_idx)
        if int(self.range_shard_indices[variant_idx]) != shard_idx:
            raise ValueError("SampleRequest shard_idx does not match variant_idx")
        start = int(request.target_start) + 1
        stop = start + int(target_frames)
        if int(request.target_start) < int(self.range_starts[variant_idx]) or stop > int(self.range_stops[variant_idx]):
            raise IndexError("Trajectory target exceeds its source range")
        if self._value_cache is None:
            self._value_cache = MMapShardCache(self.trajectory_files, self.max_open_shards)
        if self._valid_cache is None:
            self._valid_cache = MMapShardCache(self.valid_files, self.max_open_shards)
        values = self._value_cache.get(shard_idx)
        valid = self._valid_cache.get(shard_idx)
        expected = (int(self.shard_num_frames[shard_idx]), self.trajectory_dim)
        if values.dtype != np.float32 or values.shape != expected:
            raise ValueError(f"Trajectory shard has dtype/shape {values.dtype}/{values.shape}, expected float32/{expected}")
        if valid.dtype != np.bool_ or valid.shape != (int(self.shard_num_frames[shard_idx]),):
            raise ValueError("Trajectory valid shard has invalid dtype/shape")
        return (
            np.ascontiguousarray(values[start:stop], dtype=np.float32),
            np.ascontiguousarray(valid[start:stop], dtype=bool),
        )

    def read_aligned(
        self,
        request: SampleRequest,
        target_frames: int,
        normalization: TrajectoryNormalization | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        values, valid = self._read_raw(request, int(target_frames))
        stats = normalization or self.normalization
        if stats.trajectory_dim != self.trajectory_dim:
            raise ValueError("Trajectory normalization and store dimensions differ")
        values = (values - stats.mean) / stats.std
        values[~valid] = 0.0
        return np.ascontiguousarray(values, dtype=np.float32), valid

    def split_intervals(self, split: str) -> np.ndarray:
        split_id = {"train": 0, "val": 1, "test": 2}.get(split)
        if split_id is None:
            raise ValueError(f"Unsupported split {split!r}")
        rows = np.flatnonzero(self.split_ids == split_id)
        return np.column_stack((
            self.range_shard_indices[rows], self.range_starts[rows], self.range_stops[rows], rows,
        )).astype(np.int64, copy=False)

    def close(self) -> None:
        if self._value_cache is not None:
            self._value_cache.close()
        if self._valid_cache is not None:
            self._valid_cache.close()
        self._value_cache = None
        self._valid_cache = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_value_cache"] = None
        state["_valid_cache"] = None
        return state


def open_trajectory_store(
    database: str | Path,
    *,
    token_store: TokenStore | None = None,
    max_open_shards: int = 32,
) -> TrajectoryStore:
    database = Path(database)
    manifest_path = database / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing trajectory manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Trajectory manifest must be a JSON object")
    _require_manifest(manifest, "trajectory")
    valid_values = manifest.get("valid_shard_files")
    if not isinstance(valid_values, list) or len(valid_values) != len(manifest["shard_files"]):
        raise ValueError("Trajectory manifest must contain valid_shard_files for every shard")
    valid_hashes = manifest.get("valid_shard_sha256")
    if not isinstance(valid_hashes, list) or len(valid_hashes) != len(valid_values):
        raise ValueError("Trajectory manifest must contain valid_shard_sha256 for every valid shard")
    for value in valid_values:
        _require_relative_path(value, "valid shard")
    trajectory_files = [database / str(value) for value in manifest["shard_files"]]
    valid_files = [database / str(value) for value in valid_values]
    if any(not path.exists() for path in [*trajectory_files, *valid_files]):
        raise FileNotFoundError("Trajectory manifest references a missing shard")
    index = _load_index(database)
    common = _check_common_index(index, manifest, len(trajectory_files), "Trajectory")
    _, clip_ids, source_clip_ids, range_shards, range_starts, range_stops, range_mirror, split_ids = common
    range_names = tuple(str(value) for value in manifest.get("range_names", []))
    source_clip_names = tuple(str(value) for value in manifest.get("source_clip_names", []))
    if len(range_names) != len(range_starts) or not source_clip_names:
        raise ValueError("Trajectory range/source clip metadata is invalid")
    trajectory_dim = int(manifest.get("trajectory_dim", 0))
    if trajectory_dim <= 0:
        raise ValueError("Trajectory manifest trajectory_dim must be positive")
    future_frames = np.asarray(manifest.get("future_frames", []), dtype=np.int32)
    if future_frames.ndim != 1 or len(future_frames) == 0 or np.any(future_frames <= 0):
        raise ValueError("Trajectory manifest future_frames must be positive and non-empty")
    if not str(manifest.get("feature_order", "")):
        raise ValueError("Trajectory manifest feature_order must be non-empty")
    for values_path, valid_path, frames in zip(trajectory_files, valid_files, index["shard_num_frames"].tolist()):
        values = np.load(values_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (int(frames), trajectory_dim):
            raise ValueError(f"Trajectory shard {values_path} has invalid dtype/shape")
        if valid.dtype != np.bool_ or valid.shape != (int(frames),):
            raise ValueError(f"Trajectory valid shard {valid_path} has invalid dtype/shape")
        del values, valid
    required = {"normalization_mean", "normalization_std"}
    if not required.issubset(index):
        raise ValueError("Trajectory index is missing normalization arrays")
    normalization = TrajectoryNormalization(
        mean=np.asarray(index["normalization_mean"], dtype=np.float32),
        std=np.asarray(index["normalization_std"], dtype=np.float32),
        valid_frames=int(manifest.get("normalization_valid_frames", 0)),
    )
    if normalization.trajectory_dim != trajectory_dim:
        raise ValueError("Trajectory normalization dimension does not match trajectory_dim")
    checkpoint_sha256 = str(manifest.get("checkpoint_sha256", ""))
    if not checkpoint_sha256:
        raise ValueError("Trajectory manifest is missing checkpoint_sha256")
    style_names = tuple(str(value) for value in manifest.get("style_names", []))
    action_names = tuple(str(value) for value in manifest.get("action_names", []))
    range_count = len(range_names)
    style_ids = np.asarray(index.get("style_ids", np.zeros(range_count, dtype=np.int32)), dtype=np.int32)
    action_ids = np.asarray(index.get("action_ids", np.zeros(range_count, dtype=np.int32)), dtype=np.int32)
    if len(style_ids) != range_count or len(action_ids) != range_count:
        raise ValueError("Trajectory style/action index arrays have invalid lengths")
    store = TrajectoryStore(
        database=database,
        manifest=manifest,
        trajectory_files=trajectory_files,
        valid_files=valid_files,
        shard_num_frames=np.asarray(index["shard_num_frames"], dtype=np.int64),
        clip_ids=np.asarray(clip_ids, dtype=np.int32),
        source_clip_ids=np.asarray(source_clip_ids, dtype=np.int32),
        range_shard_indices=np.asarray(range_shards, dtype=np.int32),
        range_starts=np.asarray(range_starts, dtype=np.int64),
        range_stops=np.asarray(range_stops, dtype=np.int64),
        range_mirror=np.asarray(range_mirror, dtype=bool),
        split_ids=np.asarray(split_ids, dtype=np.uint8),
        range_names=range_names,
        source_clip_names=source_clip_names,
        style_names=style_names,
        action_names=action_names,
        style_ids=style_ids,
        action_ids=action_ids,
        trajectory_dim=trajectory_dim,
        future_frames=future_frames,
        feature_order=str(manifest.get("feature_order", "")),
        checkpoint_sha256=checkpoint_sha256,
        feature_schema_hash=str(manifest["feature_schema_hash"]),
        normalization=normalization,
        max_open_shards=int(max_open_shards),
    )
    if token_store is not None:
        for key in ("split_manifest_hash", "feature_schema_hash"):
            if str(getattr(store, key)) != str(getattr(token_store, key)):
                raise ValueError(f"TrajectoryStore does not inherit TokenStore {key}")
        for key in ("shard_num_frames", "clip_ids", "source_clip_ids", "range_shard_indices", "range_starts", "range_stops", "range_mirror", "split_ids"):
            if not np.array_equal(getattr(store, key), getattr(token_store, key)):
                raise ValueError(f"TrajectoryStore index does not align with TokenStore at {key}")
        if store.checkpoint_sha256 != token_store.checkpoint_sha256:
            raise ValueError("TrajectoryStore checkpoint_sha256 does not match TokenStore")
    return store


def fit_trajectory_normalization(trajectory_store: TrajectoryStore, min_std: float = 1e-6) -> TrajectoryNormalization:
    """Fit only from valid frames whose inherited split id is train."""
    if min_std <= 0.0:
        raise ValueError("min_std must be positive")
    if trajectory_store._value_cache is None:
        trajectory_store._value_cache = MMapShardCache(trajectory_store.trajectory_files, trajectory_store.max_open_shards)
    if trajectory_store._valid_cache is None:
        trajectory_store._valid_cache = MMapShardCache(trajectory_store.valid_files, trajectory_store.max_open_shards)
    total = np.zeros(trajectory_store.trajectory_dim, dtype=np.float64)
    total_squared = np.zeros_like(total)
    count = 0
    for range_idx, split_id in enumerate(trajectory_store.split_ids.tolist()):
        if int(split_id) != 0:
            continue
        shard_idx = int(trajectory_store.range_shard_indices[range_idx])
        start = int(trajectory_store.range_starts[range_idx])
        stop = int(trajectory_store.range_stops[range_idx])
        values = trajectory_store._value_cache.get(shard_idx)[start:stop]
        valid = trajectory_store._valid_cache.get(shard_idx)[start:stop]
        values = np.asarray(values[valid], dtype=np.float64)
        if len(values):
            total += values.sum(axis=0)
            total_squared += np.square(values).sum(axis=0)
            count += len(values)
    if count <= 0:
        raise ValueError("No valid trajectory frames overlap the training split")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), float(min_std) ** 2)
    result = TrajectoryNormalization(mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count)
    trajectory_store.close()
    return result


class ConditionalTokenDataset(Dataset):
    """Return token 65-frame sequences plus 64 aligned normalized controls."""

    def __init__(
        self,
        split: str,
        token_store: TokenStore,
        trajectory_store: TrajectoryStore,
        *,
        requests: Iterable[SampleRequest] | None = None,
        normalization: TrajectoryNormalization | None = None,
        max_open_shards: int = 32,
        return_metadata: bool = False,
    ) -> None:
        if trajectory_store.trajectory_dim <= 0:
            raise ValueError("TrajectoryStore has invalid trajectory_dim")
        self.tokens = TokenDataset(
            split,
            token_store,
            requests=requests,
            sequence_frames=65,
            max_open_shards=max_open_shards,
            return_metadata=return_metadata,
        )
        self.trajectory_store = trajectory_store
        self.normalization = normalization or trajectory_store.normalization
        trajectory_store.max_open_shards = int(max_open_shards)
        if self.normalization.trajectory_dim != trajectory_store.trajectory_dim:
            raise ValueError("Trajectory normalization and store dimensions differ")
        for key in ("split_manifest_hash", "feature_schema_hash"):
            if str(getattr(token_store, key)) != str(getattr(trajectory_store, key)):
                raise ValueError(f"Token and trajectory stores differ at {key}")
        for key in (
            "shard_num_frames", "clip_ids", "source_clip_ids", "range_shard_indices",
            "range_starts", "range_stops", "range_mirror", "split_ids",
        ):
            if not np.array_equal(getattr(token_store, key), getattr(trajectory_store, key)):
                raise ValueError(f"Token and trajectory stores are not aligned at {key}")

    def __len__(self) -> int:
        return len(self.tokens)

    def _batch(self, values: list[int | SampleRequest]) -> dict[str, Any]:
        requests = [self.tokens._request(value) for value in values]
        batch = self.tokens.__getitems__(requests)
        target_frames = int(requests[0].target_frames)
        trajectory = np.empty((len(requests), target_frames, self.trajectory_store.trajectory_dim), dtype=np.float32)
        valid = np.empty((len(requests), target_frames), dtype=bool)
        grouped: dict[int, list[tuple[int, SampleRequest]]] = {}
        for batch_idx, request in enumerate(requests):
            grouped.setdefault(int(request.shard_idx), []).append((batch_idx, request))
        for shard_idx in sorted(grouped):
            for batch_idx, request in grouped[shard_idx]:
                trajectory[batch_idx], valid[batch_idx] = self.trajectory_store.read_aligned(
                    request, target_frames, normalization=self.normalization
                )
        batch["trajectory"] = torch.from_numpy(trajectory)
        batch["trajectory_valid"] = torch.from_numpy(valid)
        return batch

    def __getitem__(self, index: int | SampleRequest) -> dict[str, Any]:
        request = self.tokens._request(index)
        batch = self._batch([request])
        return {key: value[0] if isinstance(value, torch.Tensor) else value[0] for key, value in batch.items()}

    def __getitems__(self, values: list[int | SampleRequest]) -> dict[str, Any]:
        return self._batch(values)

    def close(self) -> None:
        self.tokens.close()
        self.trajectory_store.close()


__all__ = ["ConditionalTokenDataset", "TrajectoryStore", "open_trajectory_store"]
