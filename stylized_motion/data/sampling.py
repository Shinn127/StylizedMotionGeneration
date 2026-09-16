"""Deterministic split manifests and compact sample-request samplers.

Plan §3.3/§4.4. The v3 samplers scanned every interval of every source at
construction time, called ``rng.choice`` with a full probability vector per
sample, and materialised one Python tuple per validation window. At BONES-SEED
scale (tens of thousands of sources, close to a million windows) that is the
dominant cost *before* any data is read.

This module builds a compact CSR index once — source group to variant to frame
interval — and samples from it with integer uniforms or a precomputed CDF. It
also separates the sampling strategies the plan asks for:

``clip_uniform``
    Every source group is equally likely; long clips do not dominate. Within a
    group a variant is chosen first (official mirror or original) and only then
    a frame window, so adding a mirrored variant never doubles a group's weight.
``frame_uniform``
    Weighted by the number of valid start frames, i.e. long clips are sampled
    more often. This is a real mode, not an alias of ``clip_uniform``.
``group_balanced``
    Optional package/style mixing for ablations: inverse-frequency weights are
    capped and blended with the natural distribution so a rare class with a
    dozen clips is not repeated endlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from torch.utils.data import Sampler


SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
SAMPLING_STRATEGIES = ("clip_uniform", "frame_uniform", "group_balanced")
TAIL_POLICIES = ("drop", "pad")


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


@dataclass(frozen=True)
class ClipInterval:
    """One logical clip (or v3 range) with its physical frame interval."""

    clip_id: int
    shard_idx: int
    offset: int
    length: int
    source_group: int
    variant: int
    split: int
    mirror: bool

    @property
    def stop(self) -> int:
        return int(self.offset) + int(self.length)


@dataclass
class IntervalTable:
    """Vectorised view of one split's intervals."""

    clip_id: np.ndarray
    shard_idx: np.ndarray
    offset: np.ndarray
    length: np.ndarray
    group_id: np.ndarray
    variant: np.ndarray
    mirror: np.ndarray
    style_id: np.ndarray
    action_id: np.ndarray
    package_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    def __len__(self) -> int:
        return int(len(self.clip_id))

    @property
    def stop(self) -> np.ndarray:
        return self.offset + self.length

    def valid_starts(self, required_frames: int) -> np.ndarray:
        return (self.length - int(required_frames) + 1).astype(np.int64, copy=False)


def _empty_table() -> IntervalTable:
    empty_i = np.zeros(0, dtype=np.int64)
    return IntervalTable(
        clip_id=empty_i,
        shard_idx=np.zeros(0, dtype=np.int32),
        offset=empty_i,
        length=empty_i,
        group_id=np.zeros(0, dtype=np.int32),
        variant=np.zeros(0, dtype=np.int32),
        mirror=np.zeros(0, dtype=bool),
        style_id=np.zeros(0, dtype=np.int32),
        action_id=np.zeros(0, dtype=np.int32),
        package_id=np.zeros(0, dtype=np.int32),
    )


def store_intervals(store: Any, split: str) -> IntervalTable:
    """Extract one split's intervals from a v4 packed store or a v3 store.

    The v4 packed store exposes a clip table; the v3 store exposes range rows.
    Both are presented as the same vectorised interval table so the samplers
    have exactly one code path.
    """
    if split not in SPLIT_IDS:
        raise ValueError(f"Unsupported split {split!r}")
    split_id = SPLIT_IDS[split]
    if hasattr(store, "clip_shard"):
        rows = np.flatnonzero(np.asarray(store.clip_split, dtype=np.uint8) == split_id)
        if len(rows) == 0:
            return _empty_table()
        return IntervalTable(
            clip_id=rows.astype(np.int64),
            shard_idx=np.asarray(store.clip_shard, dtype=np.int32)[rows],
            offset=np.asarray(store.clip_offset, dtype=np.int64)[rows],
            length=np.asarray(store.clip_length, dtype=np.int64)[rows],
            group_id=np.asarray(store.clip_source_group, dtype=np.int32)[rows],
            variant=np.asarray(store.clip_variant, dtype=np.int32)[rows],
            mirror=np.asarray(store.clip_mirror, dtype=bool)[rows],
            style_id=np.asarray(getattr(store, "clip_style_id", np.zeros(len(store.clip_shard))), dtype=np.int32)[rows],
            action_id=np.asarray(getattr(store, "clip_action_id", np.zeros(len(store.clip_shard))), dtype=np.int32)[rows],
            package_id=np.asarray(
                getattr(store, "clip_package_id", np.zeros(len(store.clip_shard))), dtype=np.int32
            )[rows],
        )
    rows = np.flatnonzero(np.asarray(store.split_ids, dtype=np.uint8) == split_id)
    if len(rows) == 0:
        return _empty_table()
    starts = np.asarray(store.range_starts, dtype=np.int64)[rows]
    stops = np.asarray(store.range_stops, dtype=np.int64)[rows]

    def optional(key: str, dtype: Any = np.int32) -> np.ndarray:
        values = getattr(store, key, None)
        if values is None:
            return np.zeros(len(rows), dtype=dtype)
        return np.asarray(values, dtype=dtype)[rows]

    # The v3 store carries no variant table; its ``range_mirror`` flag is the
    # only variant dimension, and ``clip_id`` stays the range row so the
    # resulting SampleRequest keeps pointing at the same v3 dataset row. A
    # minimal store that omits source grouping falls back to one group per
    # interval, which reproduces the historic "every interval is its own unit"
    # behaviour instead of failing.
    mirror = optional("range_mirror", bool)
    groups = getattr(store, "source_clip_ids", None)
    group_ids = rows.astype(np.int32) if groups is None else np.asarray(groups, dtype=np.int32)[rows]
    return IntervalTable(
        clip_id=rows.astype(np.int64),
        shard_idx=np.asarray(store.range_shard_indices, dtype=np.int32)[rows],
        offset=starts,
        length=stops - starts,
        group_id=group_ids,
        variant=mirror.astype(np.int32),
        mirror=mirror,
        style_id=optional("style_ids"),
        action_id=optional("action_ids"),
        package_id=optional("package_ids"),
    )


class GroupIndex:
    """Compact source-group -> variant-slot -> interval index built in one pass.

    Construction is a single ``lexsort`` over the split's intervals, so the
    cost is O(n log n) in the number of intervals of that split rather than the
    v3 O(sources x intervals) scan. Everything the sampler needs afterwards —
    group bounds, variant slots, valid-start counts, sampling CDFs — is a
    numpy array lookup, so per-sample work does not grow with the dataset.
    """

    def __init__(self, table: IntervalTable, *, required_frames: int) -> None:
        valid = table.valid_starts(int(required_frames))
        keep = valid > 0
        if not np.any(keep):
            raise ValueError("No interval in this split can hold the requested window length")
        clip_id = table.clip_id[keep]
        group_id = table.group_id[keep]
        variant = table.variant[keep]
        shard_idx = table.shard_idx[keep]
        offset = table.offset[keep]
        length = table.length[keep]
        mirror = table.mirror[keep]
        style_id = table.style_id[keep]
        action_id = table.action_id[keep]
        package_id = table.package_id[keep] if len(table.package_id) else np.zeros(len(clip_id), dtype=np.int32)
        valid = valid[keep]

        order = np.lexsort((clip_id, variant, group_id))
        self.clip_id = clip_id[order].astype(np.int64)
        self.group_id = group_id[order].astype(np.int32)
        self.variant = variant[order].astype(np.int32)
        self.shard_idx = shard_idx[order].astype(np.int32)
        self.offset = offset[order].astype(np.int64)
        self.length = length[order].astype(np.int64)
        self.mirror = mirror[order].astype(bool)
        self.style_id = style_id[order].astype(np.int32)
        self.action_id = action_id[order].astype(np.int32)
        self.package_id = np.asarray(package_id, dtype=np.int32)[order]
        self.valid_starts = valid[order].astype(np.int64)

        self.groups, self.group_start = np.unique(self.group_id, return_index=True)
        self.group_start = self.group_start.astype(np.int64)
        self.group_stop = np.concatenate(
            [self.group_start[1:], np.asarray([len(self.group_id)], dtype=np.int64)]
        )
        self.group_valid_starts = np.add.reduceat(self.valid_starts, self.group_start)
        self.num_groups = int(len(self.groups))

        # Variant slots: within a group the rows are variant-sorted, so a slot
        # is one contiguous run of one variant.
        new_slot = np.ones(len(self.group_id), dtype=bool)
        if len(self.group_id) > 1:
            new_slot[1:] = (self.group_id[1:] != self.group_id[:-1]) | (self.variant[1:] != self.variant[:-1])
        slot_rows = np.flatnonzero(new_slot)
        self.slot_start = slot_rows.astype(np.int64)
        self.slot_stop = np.concatenate(
            [self.slot_start[1:], np.asarray([len(self.group_id)], dtype=np.int64)]
        )
        self.slot_group = self.group_id[slot_rows]
        self.slot_variant = self.variant[slot_rows]
        self.slot_mirror = self.mirror[slot_rows]
        self.slot_valid_starts = np.add.reduceat(self.valid_starts, self.slot_start)
        self.group_slot_start = np.searchsorted(self.slot_group, self.groups, side="left").astype(np.int64)
        self.group_slot_stop = np.searchsorted(self.slot_group, self.groups, side="right").astype(np.int64)

    @property
    def num_intervals(self) -> int:
        return int(len(self.clip_id))

    def group_weights_from_labels(self, key: str, *, mix: float, max_ratio: float) -> np.ndarray:
        """Inverse-frequency weights blended with the natural distribution.

        ``mix=0`` reproduces the natural (clip-uniform) distribution and
        ``mix=1`` is pure inverse frequency. ``max_ratio`` caps every group's
        weight relative to the natural weight, which is what keeps a rare class
        with a dozen clips from being repeated orders of magnitude more often
        than the plan's "record sampling coverage" check would tolerate.
        """
        if not 0.0 <= float(mix) <= 1.0:
            raise ValueError("balance mix must be in [0, 1]")
        if float(max_ratio) < 1.0:
            raise ValueError("balance max_ratio must be at least 1")
        values = np.asarray(getattr(self, key), dtype=np.int32)
        labels = values[self.group_start]
        counts: dict[int, int] = {}
        for label in labels.tolist():
            counts[int(label)] = counts.get(int(label), 0) + 1
        natural = np.full(self.num_groups, 1.0 / self.num_groups, dtype=np.float64)
        inverse = np.asarray(
            [1.0 / counts[int(label)] for label in labels.tolist()], dtype=np.float64
        )
        inverse = inverse / inverse.sum()
        weights = (1.0 - float(mix)) * natural + float(mix) * inverse
        cap = natural * float(max_ratio)
        # Water-filling: clamp every group to the cap and hand the freed mass to
        # groups that still have room. A plain clamp-then-normalize would push
        # the capped groups back over the cap, which is exactly what the plan
        # warns about for rare classes.
        for _ in range(64):
            weights = np.minimum(weights, cap)
            gap = 1.0 - float(weights.sum())
            if gap <= 1e-12:
                break
            room = np.maximum(cap - weights, 0.0)
            room_total = float(room.sum())
            if room_total <= 1e-12:
                break
            allocation = np.minimum(room * (gap / room_total), room)
            weights = weights + allocation
            if gap - float(allocation.sum()) <= 1e-12:
                break
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("Balanced sampling weights are degenerate")
        weights = weights / total
        if float(weights.max()) > float(cap.max()) * (1.0 + 1e-9):
            raise ValueError("Balanced sampling weights exceed the configured weight cap")
        return weights

    def interval_cdf(self, mask: np.ndarray) -> np.ndarray:
        """Cumulative valid-start distribution over a subset of intervals."""
        mask = np.asarray(mask, dtype=bool)
        counts = np.where(mask, self.valid_starts, 0).astype(np.float64)
        total = counts.sum()
        if total <= 0.0:
            return np.zeros(0, dtype=np.float64)
        return np.cumsum(counts) / total


@dataclass
class SamplingCoverage:
    """How much of the catalogue a sampling run actually visited."""

    group_counts: np.ndarray
    visited_intervals: np.ndarray

    @classmethod
    def empty(cls, num_groups: int, num_intervals: int) -> "SamplingCoverage":
        return cls(
            group_counts=np.zeros(int(num_groups), dtype=np.int64),
            visited_intervals=np.zeros(int(num_intervals), dtype=np.int64),
        )

    def observe(self, group_in_index: int, interval_in_index: int) -> None:
        self.group_counts[int(group_in_index)] += 1
        self.visited_intervals[int(interval_in_index)] += 1

    def summary(self) -> dict[str, float]:
        total = int(self.group_counts.sum())
        visited = int((self.group_counts > 0).sum())
        groups = int(len(self.group_counts))
        intervals = int(len(self.visited_intervals))
        visited_intervals = int((self.visited_intervals > 0).sum())
        if total <= 0:
            return {
                "samples": 0,
                "groups": groups,
                "groups_visited": 0,
                "group_coverage": 0.0,
                "intervals": intervals,
                "intervals_visited": 0,
                "interval_coverage": 0.0,
                "max_group_repeat": 0,
                "normalized_entropy": 0.0,
            }
        share = self.group_counts / total
        nonzero = share[share > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        return {
            "samples": total,
            "groups": groups,
            "groups_visited": visited,
            "group_coverage": visited / groups if groups else 0.0,
            "intervals": intervals,
            "intervals_visited": visited_intervals,
            "interval_coverage": visited_intervals / intervals if intervals else 0.0,
            "max_group_repeat": int(self.group_counts.max()),
            "normalized_entropy": entropy / math.log(groups) if groups > 1 else 0.0,
        }


class TrainWindowSampler(Sampler[SampleRequest]):
    """Sample source groups, then a variant, then a frame window inside a clip."""

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
        strategy: str = "clip_uniform",
        balance_mix: float = 0.0,
        balance_max_ratio: float = 4.0,
        tail: str = "drop",
        start_ordinal: int = 0,
        track_coverage: bool = True,
    ) -> None:
        if target_frames <= 0 or samples_per_epoch <= 0:
            raise ValueError("target_frames/samples_per_epoch have invalid values")
        if not 0.0 <= float(mirror_probability) <= 1.0:
            raise ValueError("mirror_probability must be in [0,1]")
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("invalid rank/world_size")
        if str(strategy) not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unsupported sampling strategy {strategy!r}")
        if str(tail) not in TAIL_POLICIES:
            raise ValueError(f"Unsupported tail policy {tail!r}; expected one of {TAIL_POLICIES}")
        if int(start_ordinal) < 0:
            raise ValueError("start_ordinal must be non-negative")
        self.store = store
        self.target_frames = int(target_frames)
        self.required_frames = int(required_frames if required_frames is not None else target_frames)
        if self.required_frames < self.target_frames:
            raise ValueError("required_frames cannot be smaller than target_frames")
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.mirror_probability = float(mirror_probability)
        self.balance_key = balance_key
        self.strategy = str(strategy)
        self.balance_mix = float(balance_mix)
        self.balance_max_ratio = float(balance_max_ratio)
        self.tail = str(tail)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.start_ordinal = int(start_ordinal)
        self.track_coverage = bool(track_coverage)
        self._stream_key = _stable_u64(f"{self.seed}:sampler-stream") & 0xFFFFFFFF

        table = store_intervals(store, "train")
        if len(table) == 0:
            raise ValueError("Train split has no intervals")
        self.table = table
        self.index = GroupIndex(table, required_frames=self.required_frames)
        self._group_weights = self._build_group_weights()
        self._group_cdf = np.cumsum(self._group_weights) if self.strategy != "clip_uniform" else np.zeros(0)
        if len(self._group_cdf):
            self._group_cdf[-1] = 1.0
        self._mirror_cdf = self.index.interval_cdf(self.index.mirror) if self.strategy == "frame_uniform" else np.zeros(0)
        self._plain_cdf = (
            self.index.interval_cdf(~self.index.mirror) if self.strategy == "frame_uniform" else np.zeros(0)
        )
        self.coverage = (
            SamplingCoverage.empty(self.index.num_groups, self.index.num_intervals)
            if self.track_coverage
            else None
        )
        self._consumed = 0

    # -- construction helpers ----------------------------------------------
    def _build_group_weights(self) -> np.ndarray:
        if self.strategy == "group_balanced":
            if self.balance_key not in {"style", "action", "package"}:
                raise ValueError("group_balanced sampling requires balance_key style/action/package")
            key = f"{self.balance_key}_id" if self.balance_key != "package" else "package_id"
            return self.index.group_weights_from_labels(
                key, mix=self.balance_mix, max_ratio=self.balance_max_ratio
            )
        if self.balance_key is not None and self.strategy == "clip_uniform":
            # A balance key without group_balanced keeps clip-uniform weights;
            # silently switching strategies would make the sampler's behaviour
            # depend on an unrelated config field.
            pass
        return np.full(self.index.num_groups, 1.0 / self.index.num_groups, dtype=np.float64)

    @property
    def group_weights(self) -> np.ndarray:
        return self._group_weights

    # -- epoch / DDP bookkeeping -------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_ordinal = 0

    @property
    def epoch_samples(self) -> int:
        """Samples this rank yields per epoch; identical for every rank."""
        total = int(self.samples_per_epoch)
        if self.tail == "pad":
            return int(math.ceil(total / self.world_size))
        return int(total // self.world_size)

    def __len__(self) -> int:
        return max(0, self.epoch_samples - min(self.start_ordinal, self.epoch_samples))

    def _ordinals(self) -> Iterator[int]:
        total = int(self.samples_per_epoch)
        for local in range(self.start_ordinal, self.epoch_samples):
            ordinal = local * self.world_size + self.rank
            if ordinal >= total:
                if self.tail == "pad":
                    ordinal %= total
                else:  # pragma: no cover - drop already excludes these ordinals
                    continue
            yield ordinal

    def state_dict(self) -> dict[str, int]:
        return {"epoch": int(self.epoch), "next_ordinal": int(self.start_ordinal + self._consumed)}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.epoch = int(state.get("epoch", self.epoch))
        self.start_ordinal = int(state.get("next_ordinal", 0))

    def mark_epoch_complete(self, epoch: int) -> None:
        """Record that an epoch finished, even if fewer samples were drawn.

        A trainer may cap an epoch below the sampler's own budget
        (``steps_per_epoch``). Without this marker a checkpoint written at such
        an epoch boundary would show ``next_ordinal < epoch_samples`` and a
        resume would wrongly continue inside a finished epoch.
        """
        self.epoch = int(epoch)
        self.start_ordinal = self.epoch_samples
        self._consumed = 0

    # -- sampling -----------------------------------------------------------
    def _pick_group(self, rng: np.random.Generator) -> int:
        if self.strategy == "clip_uniform":
            return int(rng.integers(0, self.index.num_groups))
        draw = float(rng.random())
        return int(np.searchsorted(self._group_cdf, draw, side="right"))

    def _pick_slot(self, rng: np.random.Generator, group_index: int) -> int:
        start = int(self.index.group_slot_start[group_index])
        stop = int(self.index.group_slot_stop[group_index])
        if stop - start == 1:
            return start
        slots = np.arange(start, stop)
        mirrors = self.index.slot_mirror[start:stop]
        want_mirror = bool(rng.random() < self.mirror_probability)
        candidates = slots[mirrors == want_mirror]
        if len(candidates) == 0:
            # Fall back among variants that actually have valid windows, so a
            # missing official mirror never changes a group's sampling weight.
            candidates = slots
        return int(candidates[int(rng.integers(0, len(candidates)))])

    def _pick_interval_in_slot(self, rng: np.random.Generator, slot: int) -> int:
        """Uniformly pick an interval inside one variant slot.

        Uniform *within* the slot keeps every clip of a variant equally likely
        so a long clip gains no extra weight from its length; the variant itself
        was already chosen by the mirror policy.
        """
        start = int(self.index.slot_start[slot])
        stop = int(self.index.slot_stop[slot])
        return start + int(rng.integers(0, stop - start))

    def _pick_frame_uniform_interval(self, rng: np.random.Generator) -> int:
        want_mirror = bool(rng.random() < self.mirror_probability)
        cdf = self._mirror_cdf if want_mirror else self._plain_cdf
        if len(cdf) == 0:
            cdf = self._mirror_cdf if len(self._mirror_cdf) else self._plain_cdf
        if len(cdf) == 0:
            return int(rng.integers(0, self.index.num_intervals))
        return int(np.searchsorted(cdf, float(rng.random()), side="right"))

    def _request_for(self, rng: np.random.Generator) -> tuple[SampleRequest, int, int]:
        if self.strategy == "frame_uniform":
            interval_index = self._pick_frame_uniform_interval(rng)
            group_index = int(np.searchsorted(self.index.group_start, interval_index, side="right")) - 1
        else:
            group_index = self._pick_group(rng)
            # Both grouped strategies pick the variant first, so
            # mirror_probability means the same thing under clip_uniform and
            # group_balanced; only the group weights differ between them.
            slot = self._pick_slot(rng, group_index)
            interval_index = self._pick_interval_in_slot(rng, slot)
        offset = int(self.index.offset[interval_index])
        valid = int(self.index.valid_starts[interval_index])
        target_start = offset + int(rng.integers(0, valid))
        request = SampleRequest(
            shard_idx=int(self.index.shard_idx[interval_index]),
            target_start=target_start,
            target_frames=self.target_frames,
            variant_idx=int(self.index.clip_id[interval_index]),
        )
        return request, group_index, interval_index

    def _rng_for_ordinal(self, ordinal: int) -> np.random.Generator:
        """One generator per ordinal, keyed by (seed, epoch, rank, ordinal).

        Deriving every sample from its own ordinal — rather than from a single
        stream consumed in order — is what makes a resumed run replay exactly
        the requests that follow its checkpoint. A generator costs ~1.5 us,
        against a data-loading step measured in milliseconds.
        """
        return np.random.default_rng(
            [self._stream_key, int(self.epoch) & 0xFFFFFFFF, int(self.rank), int(ordinal)]
        )

    def __iter__(self) -> Iterator[SampleRequest]:
        self._consumed = 0
        for ordinal in self._ordinals():
            rng = self._rng_for_ordinal(ordinal)
            request, group_index, interval_index = self._request_for(rng)
            self._consumed += 1
            if self.coverage is not None:
                self.coverage.observe(group_index, interval_index)
            yield request

    def coverage_summary(self) -> dict[str, float]:
        if self.coverage is None:
            return {}
        return self.coverage.summary()

    def reset_coverage(self) -> None:
        """Restart coverage accounting, e.g. at an epoch boundary."""
        if self.coverage is not None:
            self.coverage.group_counts.fill(0)
            self.coverage.visited_intervals.fill(0)


class FixedWindowSampler(Sampler[SampleRequest]):
    """Deterministic stride windows for validation and test.

    The v3 implementation materialised one Python tuple per window. Here the
    window starts are computed as a flat numpy array, so a 100k-window
    validation split costs a few hundred kilobytes instead of a few hundred
    thousand objects.
    """

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
        limit: int | None = None,
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
        self.table = store_intervals(store, split)
        shard, start, clip = [], [], []
        for row in range(len(self.table)):
            offset = int(self.table.offset[row])
            stop = int(self.table.stop[row])
            if stop - self.required_frames < offset:
                continue
            starts = np.arange(offset, stop - self.required_frames + 1, int(stride), dtype=np.int64)
            if include_tail:
                tail = stop - self.required_frames
                if len(starts) == 0 or int(starts[-1]) != tail:
                    starts = np.concatenate([starts, np.asarray([tail], dtype=np.int64)])
            if len(starts) == 0:
                continue
            shard.append(np.full(len(starts), int(self.table.shard_idx[row]), dtype=np.int32))
            start.append(starts)
            clip.append(np.full(len(starts), int(self.table.clip_id[row]), dtype=np.int64))
        if shard:
            self._shard = np.concatenate(shard)
            self._start = np.concatenate(start)
            self._clip = np.concatenate(clip)
        else:
            self._shard = np.zeros(0, dtype=np.int32)
            self._start = np.zeros(0, dtype=np.int64)
            self._clip = np.zeros(0, dtype=np.int64)
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise ValueError("limit must be positive")
            self._shard = self._shard[:limit]
            self._start = self._start[:limit]
            self._clip = self._clip[:limit]

    @property
    def index(self) -> np.ndarray:
        """Compatibility view of the window table as [N, 4] int64 rows."""
        if len(self._start) == 0:
            return np.empty((0, 4), dtype=np.int64)
        return np.column_stack((self._shard, self._start, self._start + self.required_frames, self._clip)).astype(
            np.int64
        )

    def __len__(self) -> int:
        count = len(self._start)
        return max(0, (count + self.world_size - 1 - self.rank) // self.world_size)

    def __iter__(self) -> Iterator[SampleRequest]:
        for index in range(self.rank, len(self._start), self.world_size):
            yield SampleRequest(
                shard_idx=int(self._shard[index]),
                target_start=int(self._start[index]),
                target_frames=self.target_frames,
                variant_idx=int(self._clip[index]),
            )


def sampling_contract(sampling: Mapping[str, object], kind: str) -> dict[str, Any]:
    """Validate a sampling config block and resolve it into sampler kwargs.

    Shared by the loader builder and the benchmark harness so both read the
    same options and reject the same mistakes.
    """
    if not isinstance(sampling, Mapping):
        raise TypeError("sampling config must be a mapping")
    strategy = str(sampling.get("strategy", "clip_uniform"))
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported sampling strategy {strategy!r}")
    target_frames = int(sampling.get("target_frames", 64))
    if target_frames != 64:
        raise ValueError("Canonical data loaders require sampling.target_frames=64")
    balance_key = sampling.get("balance_key")
    if balance_key is not None and balance_key not in {"style", "action", "package"}:
        raise ValueError("sampling.balance_key must be null, style, action, or package")
    if kind == "representation" and balance_key is not None and strategy != "group_balanced":
        raise ValueError("sampling.balance_key requires strategy=group_balanced")
    if strategy == "group_balanced" and balance_key is None:
        raise ValueError("group_balanced sampling requires sampling.balance_key")
    mirror_probability = float(sampling.get("mirror_probability", 0.5))
    if not 0.0 <= mirror_probability <= 1.0:
        raise ValueError("sampling.mirror_probability must be in [0,1]")
    return {
        "strategy": strategy,
        "target_frames": target_frames,
        "samples_per_epoch": int(sampling.get("samples_per_epoch", 100000)),
        "seed": int(sampling.get("seed", 3407)),
        "mirror_probability": mirror_probability,
        "balance_key": balance_key,
        "balance_mix": float(sampling.get("balance_mix", 0.0)),
        "balance_max_ratio": float(sampling.get("balance_max_ratio", 4.0)),
        "tail": str(sampling.get("tail", "drop")),
        "required_frames": target_frames + 1 if kind != "representation" else target_frames,
        "stride": int(sampling.get("stride", 64)),
        "include_tail": bool(sampling.get("include_tail", False)),
        "eval_limit": sampling.get("eval_limit"),
    }


__all__ = [
    "ClipInterval",
    "FixedWindowSampler",
    "GroupIndex",
    "IntervalTable",
    "SAMPLING_STRATEGIES",
    "SPLIT_IDS",
    "SampleRequest",
    "SamplingCoverage",
    "SplitManifest",
    "TAIL_POLICIES",
    "TrainWindowSampler",
    "build_split_manifest",
    "sampling_contract",
    "store_intervals",
]
