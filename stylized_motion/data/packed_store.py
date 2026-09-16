"""Schema-v4 packed feature store: physical shards decoupled from logical clips.

Plan §4.2. The v3 layout writes one file per *(source clip, mirror)* pair and
bakes the training normalization into those bytes, which for BONES-SEED means
~284k small files and a full rewrite every time the split or the statistics
change. The v4 layout instead:

* packs un-normalized float32 ``[N, motion_dim]`` frames into ~256 MiB shards
  (``packed_shard_bytes`` is configurable so 128/256/512 MiB can be benchmarked),
* keeps the logical clip as a *table entry* — ``shard_id``, ``offset``,
  ``length``, ``source_group``, ``variant``, ``split`` — stored as mmap-able
  ``.npy`` arrays rather than repeated strings,
* keeps normalization in a separate versioned artifact, so changing the split
  or the statistics never rewrites feature bytes,
* carries separate ``skeleton_hash``, ``feature_schema_hash``,
  ``split_manifest_hash`` and ``normalization_hash`` so a checkpoint can bind
  all four instead of trusting one opaque digest.

The v3 reader stays available: ``open_any_feature_store`` dispatches on
``data_schema_version`` and never rewrites an existing store in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from stylized_motion.anim.features import MotionFeatureStats, joint_feature_dim
from stylized_motion.data.feature_data import (
    MMapShardCache,
    canonical_json_bytes,
    open_feature_store,
    sha256_file,
)
from stylized_motion.data.normalization import FeatureNormalization, names_sha256
from stylized_motion.data.sampling import SampleRequest


PACKED_SCHEMA_VERSION = 4
PACKED_STORE_TYPE = "feature_packed"
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES = ("train", "val", "test")

#: Columns every store must carry.
CLIP_TABLE_ARRAYS = {
    "clip_shard": np.int32,
    "clip_offset": np.int64,
    "clip_length": np.int64,
    "clip_source_group": np.int32,
    "clip_variant": np.uint8,
    "clip_split": np.uint8,
    "clip_mirror": bool,
    "clip_source_id": np.int32,
    "clip_style_id": np.int32,
    "clip_action_id": np.int32,
    "clip_package_id": np.int32,
}

#: Columns written by newer builds.  A store without them stays loadable and
#: reports the value as unknown (-1) instead of failing: label tables are
#: metadata, not feature bytes.
CLIP_TABLE_OPTIONAL_ARRAYS = {
    "clip_performer_id": np.int32,
}
CLIP_UNKNOWN_LABEL_ID = -1


def skeleton_hash(names: Sequence[str], parents: Sequence[int], joint_subset: str) -> str:
    payload = {
        "names": [str(name) for name in names],
        "parents": [int(value) for value in parents],
        "joint_subset": str(joint_subset),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def feature_schema_hash(names: Sequence[str], parents: Sequence[int], joint_subset: str) -> str:
    payload = {
        "name": "motion_feature_v2",
        "motion_dim": joint_feature_dim(len(names)),
        "joint_subset": str(joint_subset),
        "skeleton_hash": skeleton_hash(names, parents, joint_subset),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass
class ClipTableEntry:
    """One logical clip: where it lives physically and how it is labelled."""

    clip_id: int
    shard_idx: int
    offset: int
    length: int
    source_group: int
    variant: int
    split: int
    mirror: bool
    source_id: int
    style_id: int = 0
    action_id: int = 0
    package_id: int = 0
    performer_id: int = CLIP_UNKNOWN_LABEL_ID
    move_name: str = ""
    relative_path: str = ""
    position_sum: np.ndarray | None = None

    @property
    def stop(self) -> int:
        return int(self.offset) + int(self.length)


class PackedFeatureStoreWriter:
    """Write un-normalized clip features into ~``shard_bytes`` packed shards.

    Clips may be appended in any order; the writer buffers one shard at a time
    and flushes it to staging as soon as the next clip would overflow the
    target size. Nothing is durable until :meth:`finalize` publishes the store,
    so an interrupted build leaves only staging behind.
    """

    def __init__(
        self,
        staging: Path,
        *,
        motion_dim: int,
        num_joints: int,
        shard_bytes: int = DEFAULT_SHARD_BYTES,
        names: Sequence[str] | None = None,
        subdirectory: str = "motion",
        frames_per_shard: int | None = None,
    ) -> None:
        if int(motion_dim) <= 0 or int(num_joints) <= 0:
            raise ValueError("motion_dim and num_joints must be positive")
        if int(shard_bytes) <= 0:
            raise ValueError("shard_bytes must be positive")
        self.staging = Path(staging)
        self.motion_dim = int(motion_dim)
        self.num_joints = int(num_joints)
        self.shard_bytes = int(shard_bytes)
        self._explicit_frames_per_shard = None if frames_per_shard is None else int(frames_per_shard)
        if self._explicit_frames_per_shard is not None and self._explicit_frames_per_shard <= 0:
            raise ValueError("frames_per_shard must be positive")
        self.names = [str(name) for name in (names or [])]
        self.subdirectory = str(subdirectory)
        self.shard_dir = self.staging / self.subdirectory
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[ClipTableEntry] = []
        self.shard_files: list[str] = []
        self.shard_sha256: list[str] = []
        self.shard_num_frames: list[int] = []
        self._buffer: list[np.ndarray] = []
        self._buffer_frames = 0
        self._bytes_written = 0

    @property
    def frames_per_shard(self) -> int:
        """How many frames fit in one shard under the current target size.

        Parallel arrays (such as root channels) pass an explicit value so their
        shard boundaries land exactly where the primary array's do, whatever
        their own width is.
        """
        if self._explicit_frames_per_shard is not None:
            return self._explicit_frames_per_shard
        return max(1, self.shard_bytes // (self.motion_dim * 4))

    def append_clip(
        self,
        features: np.ndarray,
        *,
        source_group: int,
        variant: int,
        split: int,
        mirror: bool,
        source_id: int,
        style_id: int = 0,
        action_id: int = 0,
        package_id: int = 0,
        performer_id: int = CLIP_UNKNOWN_LABEL_ID,
        move_name: str = "",
        relative_path: str = "",
        position_sum: np.ndarray | None = None,
    ) -> ClipTableEntry:
        values = np.ascontiguousarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.motion_dim:
            raise ValueError(f"Clip features must be [N, {self.motion_dim}], got {values.shape}")
        if values.shape[0] <= 0:
            raise ValueError(f"Clip {move_name!r} has no frames")
        if int(split) not in (0, 1, 2):
            raise ValueError(f"Invalid split id {split!r}")
        position_value = None
        if position_sum is not None:
            position_value = np.asarray(position_sum, dtype=np.float64)
            if position_value.shape != (self.num_joints, 3):
                raise ValueError(
                    f"Clip position sum must be [{self.num_joints}, 3], got {position_value.shape}"
                )
        # A clip larger than one shard is written on its own: the shard-size
        # target is a packing hint, not a hard frame ceiling.
        if self._buffer and self._buffer_frames + values.shape[0] > self.frames_per_shard:
            self._flush_shard()
        shard_idx = len(self.shard_files)
        offset = self._buffer_frames
        self._buffer.append(values)
        self._buffer_frames += int(values.shape[0])
        entry = ClipTableEntry(
            clip_id=len(self.entries),
            shard_idx=int(shard_idx),
            offset=int(offset),
            length=int(values.shape[0]),
            source_group=int(source_group),
            variant=int(variant),
            split=int(split),
            mirror=bool(mirror),
            source_id=int(source_id),
            style_id=int(style_id),
            action_id=int(action_id),
            package_id=int(package_id),
            performer_id=int(performer_id),
            move_name=str(move_name),
            relative_path=str(relative_path),
            position_sum=position_value,
        )
        self.entries.append(entry)
        if self._buffer_frames >= self.frames_per_shard:
            self._flush_shard()
        return entry

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        values = np.concatenate(self._buffer, axis=0) if len(self._buffer) > 1 else self._buffer[0]
        shard_idx = len(self.shard_files)
        relative = Path(self.subdirectory) / f"shard_{shard_idx:05d}.npy"
        target = self.staging / relative
        np.save(target, values)
        self.shard_files.append(relative.as_posix())
        self.shard_sha256.append(sha256_file(target))
        self.shard_num_frames.append(int(values.shape[0]))
        self._bytes_written += int(values.nbytes)
        self._buffer = []
        self._buffer_frames = 0

    def close_shards(self) -> None:
        self._flush_shard()

    @property
    def total_frames(self) -> int:
        return int(sum(entry.length for entry in self.entries))

    @property
    def bytes_on_disk(self) -> int:
        return int(self._bytes_written)


def _validate_clip_table(arrays: Mapping[str, np.ndarray], shard_num_frames: np.ndarray, label: str) -> None:
    required = set(CLIP_TABLE_ARRAYS) | {"clip_position_sum"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{label} clip table is missing fields: {missing}")
    lengths = {key: len(arrays[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{label} clip table arrays have inconsistent lengths: {lengths}")
    clip_shard = arrays["clip_shard"]
    if len(clip_shard) and (int(clip_shard.min()) < 0 or int(clip_shard.max()) >= len(shard_num_frames)):
        raise ValueError(f"{label} clip table references an invalid shard")
    offsets = arrays["clip_offset"]
    clip_lengths = arrays["clip_length"]
    if np.any(clip_lengths <= 0):
        raise ValueError(f"{label} clip table contains empty clips")
    if np.any(offsets < 0):
        raise ValueError(f"{label} clip table contains negative offsets")
    stops = offsets + clip_lengths
    limits = shard_num_frames[clip_shard] if len(clip_shard) else np.empty(0, dtype=np.int64)
    if len(clip_shard) and np.any(stops > limits):
        raise ValueError(f"{label} clip table range exceeds its shard frame count")
    if np.any(arrays["clip_split"] > 2):
        raise ValueError(f"{label} clip table splits must be train=0, val=1, test=2")
    if np.any(arrays["clip_variant"] > 2):
        raise ValueError(f"{label} clip table variants must be original=0, official_mirror=1, generated_mirror=2")
    position_sums = arrays["clip_position_sum"]
    if position_sums.ndim == 3 and len(position_sums) != len(clip_shard):
        raise ValueError(f"{label} clip position sums do not align with the clip table")


@dataclass
class PackedFeatureStore:
    """mmap reader over a schema-v4 packed store."""

    database: Path
    manifest: dict[str, Any]
    shard_files: list[Path]
    shard_num_frames: np.ndarray
    clip_shard: np.ndarray
    clip_offset: np.ndarray
    clip_length: np.ndarray
    clip_source_group: np.ndarray
    clip_variant: np.ndarray
    clip_split: np.ndarray
    clip_mirror: np.ndarray
    clip_source_id: np.ndarray
    clip_style_id: np.ndarray
    clip_action_id: np.ndarray
    clip_package_id: np.ndarray
    clip_position_sum: np.ndarray
    range_names: tuple[str, ...]
    names: list[str]
    parents: np.ndarray
    joint_subset: str
    motion_dim: int
    feature_schema_hash: str
    skeleton_hash: str
    #: Optional build metadata: -1 marks an unknown performer (see
    #: ``CLIP_TABLE_OPTIONAL_ARRAYS``), so stores built before this column
    #: existed keep loading and report an empty performer.
    clip_performer_id: np.ndarray | None = None
    source_style_names: tuple[str, ...] = ()
    source_action_names: tuple[str, ...] = ()
    source_package_names: tuple[str, ...] = ()
    source_performer_names: tuple[str, ...] = ()
    root_files: list[Path] = field(default_factory=list)
    max_open_shards: int = 16
    normalization: FeatureNormalization | None = None
    name: str = "packed_feature_store"
    _cache: dict[int, np.ndarray] = field(default_factory=dict)
    _cache_order: list[int] = field(default_factory=list)
    _root_cache: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def root_channels(self) -> int:
        """Width of the parallel root position/heading array (0 when absent)."""
        return int(self.manifest.get("root_channels", 0))

    @property
    def has_root_channels(self) -> bool:
        return bool(self.root_files) and self.root_channels > 0

    def read_root_window(self, clip_idx: int, start: int, frames: int) -> np.ndarray:
        """Read per-frame root position/heading for a window inside one clip.

        Root channels share the feature shards' clipping layout, so the same
        clip/offset/length triple indexes both arrays.
        """
        if not self.has_root_channels:
            raise ValueError("Packed store was built without root channels")
        clip_idx = int(clip_idx)
        if clip_idx < 0 or clip_idx >= self.num_clips:
            raise IndexError(f"Invalid clip index {clip_idx}")
        offset = int(self.clip_offset[clip_idx])
        length = int(self.clip_length[clip_idx])
        start, frames = int(start), int(frames)
        if start < offset or start + frames > offset + length:
            raise IndexError(
                f"Root window [{start}, {start + frames}) leaves clip {clip_idx} interval "
                f"[{offset}, {offset + length})"
            )
        shard_idx = int(self.clip_shard[clip_idx])
        values = self._root_cache.get(shard_idx)
        if values is None:
            values = np.load(self.root_files[shard_idx], mmap_mode="r", allow_pickle=False)
            self._root_cache[shard_idx] = values
        return np.ascontiguousarray(values[start : start + frames], dtype=np.float32)

    def read_root_clip(self, clip_idx: int) -> np.ndarray:
        clip_idx = int(clip_idx)
        return self.read_root_window(clip_idx, int(self.clip_offset[clip_idx]), int(self.clip_length[clip_idx]))

    # -- identity -----------------------------------------------------------
    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_clips(self) -> int:
        return len(self.clip_shard)

    @property
    def total_frames(self) -> int:
        return int(self.shard_num_frames.sum())

    @property
    def split_manifest_hash(self) -> str:
        return str(self.manifest["split_manifest_hash"])

    @property
    def normalization_hash(self) -> str:
        return str(self.manifest.get("normalization_hash", ""))

    @property
    def stats(self) -> MotionFeatureStats:
        """Normalization statistics, loaded from the store's versioned artifact.

        Features are persisted un-normalized, so callers that need the
        statistics (loss contexts, checkpoints, reconstruction) read them from
        here rather than from the feature bytes.
        """
        if self.normalization is None:
            raise ValueError(
                "Packed store has no normalization artifact loaded; build one with "
                "'preprocess train-stats' or open it with load_normalization=True"
            )
        return self.normalization.stats

    def model_feature_weights(self) -> np.ndarray:
        return self.stats.weights.astype(np.float32, copy=True)

    def feature_schema(self) -> dict[str, Any]:
        return {
            "name": "motion_feature_v2",
            "motion_dim": self.motion_dim,
            "joint_subset": self.joint_subset,
            "names_sha256": names_sha256(self.names),
            "skeleton_hash": self.skeleton_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "normalization_hash": self.normalization_hash,
        }

    # -- shard access -------------------------------------------------------
    def _get_shard(self, shard_idx: int) -> np.ndarray:
        shard_idx = int(shard_idx)
        if shard_idx < 0 or shard_idx >= len(self.shard_files):
            raise IndexError(f"Invalid packed shard index {shard_idx}")
        cached = self._cache.get(shard_idx)
        if cached is None:
            cached = np.load(self.shard_files[shard_idx], mmap_mode="r", allow_pickle=False)
            self._cache[shard_idx] = cached
            self._cache_order.append(shard_idx)
            while len(self._cache_order) > max(1, int(self.max_open_shards)):
                victim = self._cache_order.pop(0)
                self._cache.pop(victim, None)
        else:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
        return cached

    def read_frames(self, shard_idx: int, start: int, frames: int) -> np.ndarray:
        values = self._get_shard(shard_idx)
        start, frames = int(start), int(frames)
        if frames <= 0 or start < 0 or start + frames > len(values):
            raise IndexError(
                f"Packed shard {shard_idx} cannot serve [{start}, {start + frames}) of {len(values)} frames"
            )
        return np.ascontiguousarray(values[start : start + frames], dtype=np.float32)

    def read_clip(self, clip_idx: int) -> np.ndarray:
        clip_idx = int(clip_idx)
        if clip_idx < 0 or clip_idx >= self.num_clips:
            raise IndexError(f"Invalid clip index {clip_idx}")
        return self.read_frames(
            int(self.clip_shard[clip_idx]), int(self.clip_offset[clip_idx]), int(self.clip_length[clip_idx])
        )

    def read_window(self, clip_idx: int, start: int, frames: int) -> np.ndarray:
        """Read a window strictly inside one logical clip's valid interval."""
        clip_idx = int(clip_idx)
        if clip_idx < 0 or clip_idx >= self.num_clips:
            raise IndexError(f"Invalid clip index {clip_idx}")
        offset = int(self.clip_offset[clip_idx])
        length = int(self.clip_length[clip_idx])
        start, frames = int(start), int(frames)
        if start < offset or start + frames > offset + length:
            raise IndexError(
                f"Window [{start}, {start + frames}) leaves clip {clip_idx} interval "
                f"[{offset}, {offset + length})"
            )
        return self.read_frames(int(self.clip_shard[clip_idx]), start, frames)

    def clip_label(self, clip_idx: int) -> dict[str, Any]:
        clip_idx = int(clip_idx)
        performer = ""
        if self.clip_performer_id is not None and self.source_performer_names:
            performer_id = int(self.clip_performer_id[clip_idx])
            if 0 <= performer_id < len(self.source_performer_names):
                performer = str(self.source_performer_names[performer_id])
        return {
            "clip_id": clip_idx,
            "source_group": int(self.clip_source_group[clip_idx]),
            "variant": int(self.clip_variant[clip_idx]),
            "split": int(self.clip_split[clip_idx]),
            "mirror": bool(self.clip_mirror[clip_idx]),
            "source_id": int(self.clip_source_id[clip_idx]),
            "style": self.source_style_names[int(self.clip_style_id[clip_idx])]
            if self.source_style_names
            else "",
            "action": self.source_action_names[int(self.clip_action_id[clip_idx])]
            if self.source_action_names
            else "",
            "package": self.source_package_names[int(self.clip_package_id[clip_idx])]
            if self.source_package_names
            else "",
            "performer": performer,
        }

    # -- split-scoped views -------------------------------------------------
    def split_clip_indices(self, split: str) -> np.ndarray:
        if split not in SPLIT_IDS:
            raise ValueError(f"Unsupported split {split!r}")
        return np.flatnonzero(self.clip_split == SPLIT_IDS[split])

    def iter_split_clips(self, split: str) -> Iterator[Any]:
        from stylized_motion.data.sampling import ClipInterval

        for clip_idx in self.split_clip_indices(split).tolist():
            yield ClipInterval(
                clip_id=int(clip_idx),
                shard_idx=int(self.clip_shard[clip_idx]),
                offset=int(self.clip_offset[clip_idx]),
                length=int(self.clip_length[clip_idx]),
                source_group=int(self.clip_source_group[clip_idx]),
                variant=int(self.clip_variant[clip_idx]),
                split=int(self.clip_split[clip_idx]),
                mirror=bool(self.clip_mirror[clip_idx]),
            )

    def split_position_sum(self, split: str) -> tuple[np.ndarray, int]:
        rows = self.split_clip_indices(split)
        if len(rows) == 0:
            raise ValueError(f"Split {split!r} has no clips")
        sums = self.clip_position_sum[rows].sum(axis=0, dtype=np.float64)
        frames = int(self.clip_length[rows].sum())
        return sums, frames

    def iter_split_feature_blocks(self, split: str, chunk_frames: int = 1 << 16) -> Iterator[np.ndarray]:
        if int(chunk_frames) <= 0:
            raise ValueError("chunk_frames must be positive")
        rows = self.split_clip_indices(split)
        order = rows[np.lexsort((self.clip_offset[rows], self.clip_shard[rows]))]
        for clip_idx in order.tolist():
            offset = int(self.clip_offset[clip_idx])
            length = int(self.clip_length[clip_idx])
            for start in range(offset, offset + length, int(chunk_frames)):
                stop = min(offset + length, start + int(chunk_frames))
                yield self.read_frames(int(self.clip_shard[clip_idx]), start, stop - start)

    def close(self) -> None:
        self._cache.clear()
        self._cache_order.clear()
        self._root_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = {}
        state["_cache_order"] = []
        state["_root_cache"] = {}
        return state


def _load_clip_table(database: Path) -> dict[str, np.ndarray]:
    table_dir = database / "clips"
    arrays: dict[str, np.ndarray] = {}
    for key, dtype in CLIP_TABLE_ARRAYS.items():
        path = table_dir / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Packed store clip table is missing {path.name}")
        arrays[key] = np.load(path, mmap_mode="r")
    position_path = table_dir / "clip_position_sum.npy"
    if position_path.exists():
        arrays["clip_position_sum"] = np.load(position_path, mmap_mode="r")
    else:
        arrays["clip_position_sum"] = np.zeros((len(arrays["clip_shard"]), 0, 3), dtype=np.float64)
    for key, dtype in CLIP_TABLE_OPTIONAL_ARRAYS.items():
        optional_path = table_dir / f"{key}.npy"
        if optional_path.exists():
            arrays[key] = np.load(optional_path, mmap_mode="r")
        else:
            arrays[key] = np.full(len(arrays["clip_shard"]), CLIP_UNKNOWN_LABEL_ID, dtype=dtype)
    return arrays


def write_clip_table(staging: Path, entries: Sequence[ClipTableEntry], *, num_joints: int) -> None:
    table_dir = Path(staging) / "clips"
    table_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "clip_shard": np.asarray([entry.shard_idx for entry in entries], dtype=np.int32),
        "clip_offset": np.asarray([entry.offset for entry in entries], dtype=np.int64),
        "clip_length": np.asarray([entry.length for entry in entries], dtype=np.int64),
        "clip_source_group": np.asarray([entry.source_group for entry in entries], dtype=np.int32),
        "clip_variant": np.asarray([entry.variant for entry in entries], dtype=np.uint8),
        "clip_split": np.asarray([entry.split for entry in entries], dtype=np.uint8),
        "clip_mirror": np.asarray([entry.mirror for entry in entries], dtype=bool),
        "clip_source_id": np.asarray([entry.source_id for entry in entries], dtype=np.int32),
        "clip_style_id": np.asarray([entry.style_id for entry in entries], dtype=np.int32),
        "clip_action_id": np.asarray([entry.action_id for entry in entries], dtype=np.int32),
        "clip_package_id": np.asarray([entry.package_id for entry in entries], dtype=np.int32),
    }
    performer_ids = np.asarray(
        [entry.performer_id for entry in entries], dtype=CLIP_TABLE_OPTIONAL_ARRAYS["clip_performer_id"]
    )
    if bool((performer_ids >= 0).any()):
        # Only write the column when a build actually resolved performers, so an
        # older store layout is not silently relabelled as "all unknown".
        arrays["clip_performer_id"] = performer_ids
    for key, values in arrays.items():
        np.save(table_dir / f"{key}.npy", values)
    position = np.zeros((len(entries), int(num_joints), 3), dtype=np.float64)
    for index, entry in enumerate(entries):
        if entry.position_sum is not None:
            position[index] = np.asarray(entry.position_sum, dtype=np.float64)
    np.save(table_dir / "clip_position_sum.npy", position)


def _nested_store_hint(database: Path) -> str:
    """Point at a store one level down when the path names its parent.

    The most common configuration mistake is aiming ``data.fsq_window_index`` at
    the output *parent* rather than the store directory itself; naming the
    candidate stores turns an opaque "missing manifest" into a one-line fix.
    """
    try:
        candidates = sorted(
            child.name
            for child in database.iterdir()
            if child.is_dir() and (child / "manifest.json").exists()
        )
    except OSError:  # pragma: no cover - unreadable path
        return ""
    if not candidates:
        return ""
    listed = ", ".join(str(database / name) for name in candidates[:3])
    return f"; did you mean one of these stores? {listed}"


def open_packed_feature_store(
    database: str | Path,
    *,
    max_open_shards: int = 16,
    load_normalization: bool = True,
) -> PackedFeatureStore:
    database = Path(database)
    manifest_path = database / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing packed store manifest: {manifest_path}{_nested_store_hint(database)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("data_schema_version", 0)) != PACKED_SCHEMA_VERSION:
        raise ValueError(
            f"Expected a schema v{PACKED_SCHEMA_VERSION} packed store, got v{manifest.get('data_schema_version')}"
        )
    if str(manifest.get("store_type")) != PACKED_STORE_TYPE:
        raise ValueError(f"Expected store_type={PACKED_STORE_TYPE!r}")
    if int(manifest.get("frame_rate", 0)) != 60:
        raise ValueError("Canonical data frame_rate must be 60")
    build = manifest.get("build", {})
    if str(build.get("status", "complete")) != "complete":
        raise ValueError(f"Packed store was not published cleanly: {build.get('status')!r}")
    shard_files = []
    for relative in manifest["shard_files"]:
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Packed store shard paths must be relative to the store root")
        shard_files.append(database / path)
    missing = [str(path) for path in shard_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Packed store references missing shards: {missing[:3]}")
    if len(shard_files) != int(manifest["num_shards"]):
        raise ValueError("Packed store manifest shard count is invalid")
    schema = manifest.get("feature_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("Packed store manifest must contain feature_schema")
    names = [str(value) for value in schema.get("names", [])]
    parents = np.asarray(schema.get("parents", []), dtype=np.int32)
    if not names or parents.shape != (len(names),):
        raise ValueError("Packed store skeleton schema is invalid")
    motion_dim = int(manifest["motion_dim"])
    expected_dim = joint_feature_dim(len(names))
    if motion_dim != expected_dim:
        raise ValueError(
            f"Packed store motion_dim {motion_dim} does not match the skeleton-derived width {expected_dim}"
        )
    table = _load_clip_table(database)
    shard_num_frames = np.zeros(len(shard_files), dtype=np.int64)
    for index, path in enumerate(shard_files):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != motion_dim:
            raise ValueError(f"Packed shard {path} has an unexpected dtype/shape {values.dtype}/{values.shape}")
        shard_num_frames[index] = int(values.shape[0])
    recorded = manifest.get("shard_num_frames")
    if recorded is not None and not np.array_equal(np.asarray(recorded, dtype=np.int64), shard_num_frames):
        raise ValueError("Packed store shard frame counts do not match the manifest")
    _validate_clip_table(table, shard_num_frames, "Packed feature store")
    normalization = None
    if load_normalization and manifest.get("normalization_hash"):
        if (database / "normalization.npz").exists():
            normalization = FeatureNormalization.load(database)
    root_files: list[Path] = []
    if manifest.get("root_shard_files"):
        root_channels = int(manifest.get("root_channels", 0))
        if root_channels <= 0:
            raise ValueError("Packed store declares root shards without a root_channel width")
        for relative in manifest["root_shard_files"]:
            path = Path(str(relative))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Packed store root shard paths must be relative to the store root")
            root_files.append(database / path)
        missing = [str(path) for path in root_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Packed store references missing root shards: {missing[:3]}")
        for path, frames in zip(root_files, shard_num_frames.tolist()):
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.dtype != np.float32 or values.shape != (int(frames), root_channels):
                raise ValueError(
                    f"Packed root shard {path} has {values.dtype}/{values.shape}, "
                    f"expected float32/{(int(frames), root_channels)}"
                )
    return PackedFeatureStore(
        database=database,
        manifest=manifest,
        shard_files=shard_files,
        shard_num_frames=shard_num_frames,
        clip_shard=np.asarray(table["clip_shard"], dtype=np.int32),
        clip_offset=np.asarray(table["clip_offset"], dtype=np.int64),
        clip_length=np.asarray(table["clip_length"], dtype=np.int64),
        clip_source_group=np.asarray(table["clip_source_group"], dtype=np.int32),
        clip_variant=np.asarray(table["clip_variant"], dtype=np.uint8),
        clip_split=np.asarray(table["clip_split"], dtype=np.uint8),
        clip_mirror=np.asarray(table["clip_mirror"], dtype=bool),
        clip_source_id=np.asarray(table["clip_source_id"], dtype=np.int32),
        clip_style_id=np.asarray(table["clip_style_id"], dtype=np.int32),
        clip_action_id=np.asarray(table["clip_action_id"], dtype=np.int32),
        clip_package_id=np.asarray(table["clip_package_id"], dtype=np.int32),
        clip_position_sum=np.asarray(table["clip_position_sum"], dtype=np.float64)
        if table["clip_position_sum"].ndim == 3
        else np.zeros((len(table["clip_shard"]), 0, 3), dtype=np.float64),
        range_names=tuple(str(value) for value in manifest.get("clip_names", [])),
        names=names,
        parents=parents,
        joint_subset=str(schema.get("joint_subset", "unknown")),
        motion_dim=motion_dim,
        feature_schema_hash=str(manifest["feature_schema_hash"]),
        skeleton_hash=str(manifest.get("skeleton_hash", "")),
        source_style_names=tuple(str(value) for value in manifest.get("style_names", [])),
        source_action_names=tuple(str(value) for value in manifest.get("action_names", [])),
        source_package_names=tuple(str(value) for value in manifest.get("package_names", [])),
        clip_performer_id=table.get("clip_performer_id"),
        source_performer_names=tuple(str(value) for value in manifest.get("performer_names", [])),
        root_files=root_files,
        max_open_shards=int(max_open_shards),
        normalization=normalization,
    )


class _LegacyFeatureStoreAdapter:
    """Present a v3 ``FeatureStore`` through the v4 reader protocol.

    The v3 store has no packed clip table and no un-normalized bytes, so this
    adapter only supports what the streaming statistics scan needs: iterate the
    train ranges and report the train reference skeleton.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self.motion_dim = int(store.motion_dim)
        self.num_joints = int(store.num_joints)
        self.names = list(store.names)
        self.feature_schema_hash = str(store.feature_schema_hash)
        self.split_manifest_hash = str(store.split_manifest_hash)
        self.name = str(store.database)

    def split_clip_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.store.split_ids == SPLIT_IDS[split])

    def iter_split_feature_blocks(self, split: str, chunk_frames: int = 1 << 16) -> Iterator[np.ndarray]:
        rows = self.split_clip_indices(split)
        shards = self.store.range_shard_indices[rows]
        starts = self.store.range_starts[rows]
        stops = self.store.range_stops[rows]
        order = np.lexsort((starts, shards))
        if self.store._cache is None:
            self.store._cache = MMapShardCache(self.store.motion_files, self.store.max_open_shards)
        for row in order.tolist():
            start = int(starts[row])
            stop = int(stops[row])
            shard_idx = int(shards[row])
            for block_start in range(start, stop, int(chunk_frames)):
                block_stop = min(stop, block_start + int(chunk_frames))
                values = self.store._cache.get(shard_idx)  # noqa: SLF001
                yield np.ascontiguousarray(values[block_start:block_stop], dtype=np.float32)

    def split_position_sum(self, split: str) -> tuple[np.ndarray, int]:
        rows = self.split_clip_indices(split)
        frames = int((self.store.range_stops[rows] - self.store.range_starts[rows]).sum())
        ref_pos = np.asarray(self.store.stats.ref_pos, dtype=np.float64)
        return ref_pos * frames, frames

    def close(self) -> None:
        self.store.close()


def open_any_feature_store(
    database: str | Path,
    *,
    max_open_shards: int = 16,
    load_normalization: bool = True,
) -> Any:
    """Open a v4 packed store or fall back to the v3 reader without rewriting it."""
    path = Path(database)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing store manifest: {manifest_path}{_nested_store_hint(path)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(manifest.get("data_schema_version", 0))
    if version == PACKED_SCHEMA_VERSION:
        return open_packed_feature_store(path, max_open_shards=max_open_shards, load_normalization=load_normalization)
    if version == 3:
        return open_feature_store(path, max_open_shards=max_open_shards)
    raise ValueError(f"Unsupported data schema version {version!r} at {path}")


class PackedFeatureDataset(Dataset):
    """Assemble windows from a packed store, normalizing on the CPU batch.

    ``normalize_on="cpu"`` (the default) applies ``(x - offset) / scale`` to the
    assembled batch array; ``"none"`` returns raw frames plus the statistics so
    a caller can normalize the batch on the GPU instead. Both paths are
    measured by the benchmark harness before one is chosen for a run.
    """

    def __init__(
        self,
        split: str,
        store: PackedFeatureStore,
        *,
        normalize_on: str = "cpu",
        max_open_shards: int = 16,
        return_metadata: bool = False,
    ) -> None:
        if split not in SPLIT_IDS:
            raise ValueError(f"Unsupported split {split!r}")
        if normalize_on not in {"cpu", "none"}:
            raise ValueError("normalize_on must be 'cpu' or 'none'")
        self.split = split
        self.store = store
        self.normalize_on = normalize_on
        self.max_open_shards = int(max_open_shards)
        self.return_metadata = bool(return_metadata)
        self.store.max_open_shards = int(max_open_shards)
        self._split_id = SPLIT_IDS[split]

    def __len__(self) -> int:
        return int((self.store.clip_split == self._split_id).sum())

    def _check(self, request: SampleRequest) -> tuple[int, int, int]:
        clip_idx = int(request.variant_idx)
        if clip_idx < 0 or clip_idx >= self.store.num_clips:
            raise IndexError(f"Invalid packed clip index {clip_idx}")
        if int(self.store.clip_split[clip_idx]) != self._split_id:
            raise ValueError(f"SampleRequest clip {clip_idx} does not belong to split {self.split!r}")
        return clip_idx, int(request.target_start), int(request.target_frames)

    def _read_numpy(self, request: SampleRequest) -> np.ndarray:
        clip_idx, start, frames = self._check(request)
        return self.store.read_window(clip_idx, start, frames)

    def _normalize(self, motion: np.ndarray) -> None:
        if self.normalize_on != "cpu":
            return
        normalization = self.store.normalization
        if normalization is None:
            return
        motion -= normalization.stats.offset
        motion /= normalization.stats.scale

    def _batch(self, values: Sequence[int | SampleRequest]) -> dict[str, Any]:
        if not values:
            raise ValueError("Cannot collate an empty packed feature batch")
        requests = [value if isinstance(value, SampleRequest) else None for value in values]
        if any(request is None for request in requests):
            raise TypeError("PackedFeatureDataset batches require SampleRequest entries")
        typed = [request for request in requests if request is not None]
        frames = int(typed[0].target_frames)
        if any(int(request.target_frames) != frames for request in typed):
            raise ValueError("A packed feature batch must use one window length")
        motion = np.empty((len(typed), frames, self.store.motion_dim), dtype=np.float32)
        grouped: dict[int, list[tuple[int, SampleRequest]]] = {}
        for batch_idx, request in enumerate(typed):
            grouped.setdefault(int(self.store.clip_shard[int(request.variant_idx)]), []).append((batch_idx, request))
        for shard_idx in sorted(grouped):
            for batch_idx, request in grouped[shard_idx]:
                motion[batch_idx] = self._read_numpy(request)
        self._normalize(motion)
        batch: dict[str, Any] = {
            "motion": torch.from_numpy(motion),
            "loss_mask": torch.ones((len(typed), frames), dtype=torch.bool),
        }
        if self.normalize_on == "none" and self.store.normalization is not None:
            batch["normalization"] = {
                "offset": torch.from_numpy(np.asarray(self.store.normalization.stats.offset, dtype=np.float32)),
                "scale": torch.from_numpy(np.asarray(self.store.normalization.stats.scale, dtype=np.float32)),
                "normalization_hash": self.store.normalization_hash,
            }
        if self.return_metadata:
            batch["metadata"] = [
                {
                    "clip_id": int(request.variant_idx),
                    "shard_idx": int(self.store.clip_shard[int(request.variant_idx)]),
                    "target_start": int(request.target_start),
                    "target_frames": int(request.target_frames),
                    **self.store.clip_label(int(request.variant_idx)),
                }
                for request in typed
            ]
        return batch

    def __getitem__(self, index: int | SampleRequest) -> dict[str, Any]:
        if not isinstance(index, SampleRequest):
            raise TypeError("PackedFeatureDataset requires a SampleRequest")
        motion = np.array(self._read_numpy(index), dtype=np.float32, copy=True, order="C")
        self._normalize(motion)
        item: dict[str, Any] = {
            "motion": torch.from_numpy(motion),
            "loss_mask": torch.ones((motion.shape[0],), dtype=torch.bool),
        }
        if self.return_metadata:
            item["metadata"] = {
                "clip_id": int(index.variant_idx),
                "shard_idx": int(self.store.clip_shard[int(index.variant_idx)]),
                "target_start": int(index.target_start),
                "target_frames": int(index.target_frames),
                **self.store.clip_label(int(index.variant_idx)),
            }
        return item

    def __getitems__(self, values: Sequence[int | SampleRequest]) -> dict[str, Any]:
        return self._batch(list(values))

    def close(self) -> None:
        self.store.close()

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


def normalize_batch_on_device(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Apply CPU-computed statistics to a batch tensor on ``device``.

    Used by the benchmark comparison and by callers that read with
    ``normalize_on="none"``; it keeps the arithmetic identical to the CPU path
    so only throughput, not numerics, differs between the two placements.
    """
    stats = batch.get("normalization")
    if not stats:
        return dict(batch)
    motion = batch["motion"]
    offset = stats["offset"].to(device=motion.device, dtype=motion.dtype)
    scale = stats["scale"].to(device=motion.device, dtype=motion.dtype)
    updated = dict(batch)
    updated["motion"] = (motion - offset) / scale
    return updated


def publish_packed_store(staging: Path, output: Path, *, overwrite: bool = False) -> Path:
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Packed store already exists: {output}; pass overwrite=True to replace it")
        import shutil

        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    os.replace(Path(staging), output)
    return output


__all__ = [
    "CLIP_TABLE_ARRAYS",
    "DEFAULT_SHARD_BYTES",
    "PACKED_SCHEMA_VERSION",
    "PACKED_STORE_TYPE",
    "ClipTableEntry",
    "PackedFeatureDataset",
    "PackedFeatureStore",
    "PackedFeatureStoreWriter",
    "feature_schema_hash",
    "normalize_batch_on_device",
    "open_any_feature_store",
    "open_packed_feature_store",
    "publish_packed_store",
    "skeleton_hash",
    "write_clip_table",
]
