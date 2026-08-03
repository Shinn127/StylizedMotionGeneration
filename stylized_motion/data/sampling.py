"""Deterministic split manifests and shard-free sample request samplers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
from torch.utils.data import Sampler


SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little", signed=False)


@dataclass(frozen=True)
class SampleRequest:
    shard_idx: int
    target_start: int
    target_frames: int
    variant_idx: int

    def __post_init__(self) -> None:
        if int(self.shard_idx) < 0 or int(self.target_start) < 0 or int(self.variant_idx) < 0:
            raise ValueError("SampleRequest shard, target, and variant indices must be non-negative")
        if int(self.target_frames) <= 0:
            raise ValueError("SampleRequest target_frames must be positive")


@dataclass(frozen=True)
class SplitManifest:
    policy: str
    algorithm_version: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    stratify_keys: tuple[str, ...]
    source_clip_names: tuple[str, ...]
    split_by_source_clip: dict[str, str]
    split_manifest_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "algorithm_version": self.algorithm_version,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "stratify_keys": list(self.stratify_keys),
            "source_clip_names": list(self.source_clip_names),
            "split_by_source_clip": dict(sorted(self.split_by_source_clip.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SplitManifest":
        required = {
            "policy", "algorithm_version", "train_ratio", "val_ratio", "test_ratio", "seed",
            "stratify_keys", "source_clip_names", "split_by_source_clip", "split_manifest_hash",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Split manifest is missing required fields: {missing}")
        payload = {key: value[key] for key in required if key != "split_manifest_hash"}
        expected_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        if str(value["split_manifest_hash"]) != expected_hash:
            raise ValueError("split_manifest_hash does not match canonical manifest content")
        assignments = value["split_by_source_clip"]
        names = value["source_clip_names"]
        if not isinstance(assignments, Mapping) or not isinstance(names, (list, tuple)):
            raise ValueError("Split source clip fields have invalid types")
        assignments = {str(key): str(item) for key, item in assignments.items()}
        names = tuple(str(item) for item in names)
        if set(assignments) != set(names) or any(item not in SPLIT_IDS for item in assignments.values()):
            raise ValueError("Split source clip assignments are incomplete or invalid")
        return cls(
            policy=str(value["policy"]),
            algorithm_version=int(value["algorithm_version"]),
            train_ratio=float(value["train_ratio"]),
            val_ratio=float(value["val_ratio"]),
            test_ratio=float(value["test_ratio"]),
            seed=int(value["seed"]),
            stratify_keys=tuple(str(item) for item in value["stratify_keys"]),
            source_clip_names=names,
            split_by_source_clip=assignments,
            split_manifest_hash=expected_hash,
        )


def build_split_manifest(
    source_clip_names: Iterable[str],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
    stratify_keys: Iterable[str] = (),
    labels: Mapping[str, Mapping[str, str]] | None = None,
    algorithm_version: int = 1,
) -> SplitManifest:
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) < 0.0:
        raise ValueError("split ratios must be non-negative and sum to one")
    names = tuple(sorted(set(str(name) for name in source_clip_names)))
    if not names:
        raise ValueError("source_clip_names must not be empty")
    keys = tuple(str(key) for key in stratify_keys)
    labels = labels or {}
    groups: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        label = labels.get(name, {})
        group = tuple(str(label.get(key, "")) for key in keys)
        groups.setdefault(group, []).append(name)
    ordered: list[str] = []
    for group in sorted(groups):
        ordered.extend(sorted(groups[group], key=lambda name: (_stable_u64(f"{seed}:{group}:{name}"), name)))
    counts = [int(np.floor(len(names) * ratio)) for ratio in ratios]
    counts[2] = len(names) - counts[0] - counts[1]
    if len(names) >= 3:
        for split_idx in (1, 2):
            if counts[split_idx] == 0:
                donor = max((idx for idx, count in enumerate(counts) if count > 1), key=lambda idx: counts[idx])
                counts[donor] -= 1
                counts[split_idx] += 1
    if sum(counts) != len(names) or min(counts) < 0:
        raise ValueError(f"Invalid split counts: {counts}")
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in zip(("train", "val", "test"), counts):
        for name in ordered[offset : offset + count]:
            assignments[name] = split
        offset += count
    payload = {
        "policy": "source_clip",
        "algorithm_version": int(algorithm_version),
        "train_ratio": ratios[0],
        "val_ratio": ratios[1],
        "test_ratio": ratios[2],
        "seed": int(seed),
        "stratify_keys": list(keys),
        "source_clip_names": list(names),
        "split_by_source_clip": dict(sorted(assignments.items())),
    }
    split_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return SplitManifest(
        policy="source_clip",
        algorithm_version=int(algorithm_version),
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        seed=int(seed),
        stratify_keys=keys,
        source_clip_names=names,
        split_by_source_clip=assignments,
        split_manifest_hash=split_hash,
    )


def _store_intervals(store: Any, split: str) -> np.ndarray:
    if split not in SPLIT_IDS:
        raise ValueError(f"Unsupported split {split!r}")
    rows = np.flatnonzero(np.asarray(store.split_ids, dtype=np.uint8) == SPLIT_IDS[split])
    return np.column_stack((
        np.asarray(store.range_shard_indices, dtype=np.int64)[rows],
        np.asarray(store.range_starts, dtype=np.int64)[rows],
        np.asarray(store.range_stops, dtype=np.int64)[rows],
        rows.astype(np.int64),
    ))


def _epoch_seed(seed: int, epoch: int) -> int:
    return _stable_u64(f"{int(seed)}:{int(epoch)}")


class TrainWindowSampler(Sampler[SampleRequest]):
    """Dynamic frame-uniform requests with deterministic DDP ordinal partitioning."""

    def __init__(
        self,
        store: Any,
        *,
        target_frames: int = 64,
        samples_per_epoch: int = 100000,
        seed: int = 3407,
        mirror_probability: float = 0.5,
        balance_key: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        required_frames: int | None = None,
    ) -> None:
        if target_frames <= 0 or samples_per_epoch <= 0:
            raise ValueError("target_frames/samples_per_epoch have invalid values")
        if not 0.0 <= mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0,1]")
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("invalid rank/world_size")
        self.store = store
        self.target_frames = int(target_frames)
        self.required_frames = int(required_frames if required_frames is not None else target_frames)
        if self.required_frames < self.target_frames:
            raise ValueError("required_frames cannot be smaller than target_frames")
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.mirror_probability = float(mirror_probability)
        self.balance_key = balance_key
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        intervals = _store_intervals(store, "train")
        valid = intervals[:, 2] - intervals[:, 1] - self.required_frames + 1
        intervals = intervals[valid > 0]
        if len(intervals) == 0:
            raise ValueError("Train split has no valid target windows")
        self._intervals = intervals
        self._valid_lengths = (intervals[:, 2] - intervals[:, 1] - self.required_frames + 1).astype(np.int64)
        source_ids = np.asarray(store.source_clip_ids, dtype=np.int32)
        mirror = np.asarray(store.range_mirror, dtype=bool)
        self._groups: list[np.ndarray] = []
        for source_id in sorted(set(source_ids[intervals[:, 3]].tolist())):
            rows = np.flatnonzero(source_ids[intervals[:, 3]] == source_id)
            self._groups.append(rows)
        if not self._groups:
            raise ValueError("Train split has no source clip groups")
        group_weights = np.asarray([self._valid_lengths[group].sum() for group in self._groups], dtype=np.float64)
        if balance_key in {"style", "action"}:
            values = np.asarray(getattr(store, f"{balance_key}_ids"), dtype=np.int32)[intervals[:, 3]]
            labels = [int(values[group[0]]) for group in self._groups]
            unique = sorted(set(labels))
            group_weights = np.asarray([1.0 / len([label for label in labels if label == value]) for value in labels], dtype=np.float64)
            self._balance_values = unique
        else:
            self._balance_values = None
        if np.any(group_weights <= 0.0) or not np.isfinite(group_weights).all():
            raise ValueError("Train source clip groups must have positive finite sampling weights")
        self._group_weights = group_weights / group_weights.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (self.samples_per_epoch + self.world_size - 1 - self.rank) // self.world_size

    def __iter__(self):
        rng = np.random.default_rng(_epoch_seed(self.seed, self.epoch))
        for ordinal in range(self.samples_per_epoch):
            group_index = int(rng.choice(len(self._groups), p=self._group_weights))
            group = self._intervals[self._groups[group_index]]
            range_rows = group[:, 3]
            mirror_rows = np.asarray(self.store.range_mirror, dtype=bool)[range_rows]
            desired_mirror = bool(rng.random() < self.mirror_probability)
            candidates = np.flatnonzero(mirror_rows == desired_mirror)
            if len(candidates) == 0:
                candidates = np.arange(len(group))
            candidate_lengths = self._valid_lengths[self._groups[group_index]][candidates]
            selected = int(rng.choice(candidates, p=candidate_lengths / candidate_lengths.sum()))
            shard_idx, start, stop, variant_idx = (int(value) for value in group[selected].tolist())
            max_start = stop - self.required_frames
            target_start = int(rng.integers(start, max_start + 1))
            if ordinal % self.world_size == self.rank:
                yield SampleRequest(shard_idx, target_start, self.target_frames, variant_idx)


class FixedWindowSampler(Sampler[SampleRequest]):
    """Deterministic stride windows for validation and test."""

    def __init__(
        self,
        store: Any,
        split: str,
        *,
        target_frames: int = 64,
        stride: int = 64,
        include_tail: bool = False,
        rank: int = 0,
        world_size: int = 1,
        required_frames: int | None = None,
    ) -> None:
        if target_frames <= 0 or stride <= 0:
            raise ValueError("target_frames/stride have invalid values")
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("invalid rank/world_size")
        self.split = split
        self.target_frames = int(target_frames)
        self.required_frames = int(required_frames if required_frames is not None else target_frames)
        if self.required_frames < self.target_frames:
            raise ValueError("required_frames cannot be smaller than target_frames")
        self.rank = int(rank)
        self.world_size = int(world_size)
        rows = _store_intervals(store, split)
        requests: list[tuple[int, int, int, int]] = []
        for shard_idx, start, stop, variant_idx in rows.tolist():
            start, stop = int(start), int(stop)
            starts = list(range(start, stop - self.required_frames + 1, int(stride)))
            if include_tail and stop - self.required_frames >= start:
                tail = stop - self.required_frames
                if not starts or starts[-1] != tail:
                    starts.append(tail)
            requests.extend((int(shard_idx), value, value + self.required_frames, int(variant_idx)) for value in starts)
        self.index = np.asarray(requests, dtype=np.int64).reshape((-1, 4)) if requests else np.empty((0, 4), dtype=np.int64)

    def __len__(self) -> int:
        return (len(self.index) + self.world_size - 1 - self.rank) // self.world_size

    def __iter__(self):
        for row in self.index[self.rank :: self.world_size]:
            shard_idx, target_start, _target_stop, variant_idx = (int(value) for value in row.tolist())
            yield SampleRequest(shard_idx, target_start, self.target_frames, variant_idx)


__all__ = ["FixedWindowSampler", "SampleRequest", "SplitManifest", "TrainWindowSampler", "build_split_manifest"]
