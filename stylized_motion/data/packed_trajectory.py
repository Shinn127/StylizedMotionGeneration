"""Schema-v4 trajectory store: clip-local future controls with a validity mask.

Plan §4.6. The v3 path materialised a whole-database NPZ of concatenated future
controls and then kept every shard's value/valid list in memory while building
the trajectory database. Both are unbounded at SEED scale (~142k clips).

This builder instead streams the packed feature store's parallel root channels
one clip at a time, derives the root-relative future offsets *per clip*, and
writes a packed trajectory array that shares the feature store's clipping
layout. The trailing frames of every clip — where the requested future horizon
falls off the end — carry an explicit validity mask so no window can silently
read a future frame belonging to another clip or split.

Only the train split contributes to the normalization statistics, which are
streamed with the same mergeable accumulator the feature normalization uses.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from stylized_motion.anim import quat
from stylized_motion.data.feature_data import sha256_file
from stylized_motion.data.packed_store import (
    PACKED_SCHEMA_VERSION,
    _load_clip_table,
    open_packed_feature_store,
    publish_packed_store,
)
from stylized_motion.data.sampling import SampleRequest

TRAJECTORY_STORE_TYPE = "trajectory_packed"
DEFAULT_FUTURE_FRAMES = (20, 40, 60)
TRAJECTORY_ORDER = "pos(future_frames),dir(future_frames)"


@dataclass
class PackedTrajectoryStore:
    """mmap reader over a schema-v4 packed trajectory store."""

    database: Path
    manifest: dict[str, Any]
    shard_files: list[Path]
    valid_files: list[Path]
    shard_num_frames: np.ndarray
    clip_shard: np.ndarray
    clip_offset: np.ndarray
    clip_length: np.ndarray
    clip_source_group: np.ndarray
    clip_variant: np.ndarray
    clip_split: np.ndarray
    clip_mirror: np.ndarray
    clip_source_id: np.ndarray
    trajectory_dim: int
    future_frames: tuple[int, ...]
    feature_schema_hash: str
    normalization_hash: str
    split_manifest_hash: str
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    normalization_valid_frames: int
    checkpoint_sha256: str = ""
    max_open_shards: int = 16
    name: str = "packed_trajectory_store"

    @property
    def num_clips(self) -> int:
        return int(len(self.clip_shard))

    def read_clip(self, clip_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(values[N, C], valid[N])`` for one clip."""
        clip_idx = int(clip_idx)
        if clip_idx < 0 or clip_idx >= self.num_clips:
            raise IndexError(f"Invalid clip index {clip_idx}")
        shard_idx = int(self.clip_shard[clip_idx])
        start = int(self.clip_offset[clip_idx])
        length = int(self.clip_length[clip_idx])
        values = np.load(self.shard_files[shard_idx], mmap_mode="r", allow_pickle=False)
        valid = np.load(self.valid_files[shard_idx], mmap_mode="r", allow_pickle=False)
        return (
            np.ascontiguousarray(values[start : start + length], dtype=np.float32),
            np.ascontiguousarray(valid[start : start + length], dtype=bool),
        )

    def window_local_start(self, clip_idx: int, target_start: int) -> int:
        """Convert an absolute frame offset into a clip-local one (row alignment)."""
        clip_idx = int(clip_idx)
        local = int(target_start) - int(self.clip_offset[clip_idx])
        if local < 0:
            raise IndexError(f"Target start {target_start} precedes clip {clip_idx}")
        return local

    def read_window(self, clip_idx: int, start: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Read a trajectory window strictly inside one clip, validity included."""
        clip_idx = int(clip_idx)
        offset = int(self.clip_offset[clip_idx])
        length = int(self.clip_length[clip_idx])
        start, frames = int(start), int(frames)
        if start < offset or start + frames > offset + length:
            raise IndexError(
                f"Trajectory window [{start}, {start + frames}) leaves clip {clip_idx} "
                f"interval [{offset}, {offset + length})"
            )
        values, valid = self.read_clip(clip_idx)
        local = start - offset
        return values[local : local + frames], valid[local : local + frames]

    def split_clip_indices(self, split: str) -> np.ndarray:
        split_id = {"train": 0, "val": 1, "test": 2}.get(str(split))
        if split_id is None:
            raise ValueError(f"Unsupported split {split!r}")
        return np.flatnonzero(self.clip_split == split_id)

    def close(self) -> None:
        return None

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


def open_packed_trajectory_store(database: str | Path, *, max_open_shards: int = 16) -> PackedTrajectoryStore:
    database = Path(database)
    manifest_path = database / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing packed trajectory manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("data_schema_version", 0)) != PACKED_SCHEMA_VERSION:
        raise ValueError("Packed trajectory store must use data schema v4")
    if str(manifest.get("store_type")) != TRAJECTORY_STORE_TYPE:
        raise ValueError(f"Expected store_type={TRAJECTORY_STORE_TYPE!r}")
    build = manifest.get("build", {})
    if str(build.get("status", "complete")) != "complete":
        raise ValueError(f"Packed trajectory store was not published cleanly: {build.get('status')!r}")
    shard_files = [database / str(value) for value in manifest["shard_files"]]
    valid_files = [database / str(value) for value in manifest["valid_shard_files"]]
    shard_files = list(shard_files)
    valid_files = list(valid_files)
    for path in [*shard_files, *valid_files]:
        if not path.exists():
            raise FileNotFoundError(f"Packed trajectory store references a missing file: {path}")
    table = _load_clip_table(database)
    trajectory_dim = int(manifest["trajectory_dim"])
    shard_num_frames = np.zeros(len(shard_files), dtype=np.int64)
    for index, (value_path, valid_path) in enumerate(zip(shard_files, valid_files)):
        values = np.load(value_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != trajectory_dim:
            raise ValueError(f"Trajectory shard {value_path} has an unexpected shape {values.shape}")
        if valid.dtype != np.bool_ or valid.shape != (len(values),):
            raise ValueError(f"Trajectory validity shard {valid_path} has an unexpected shape {valid.shape}")
        shard_num_frames[index] = int(values.shape[0])
    store = PackedTrajectoryStore(
        database=database,
        manifest=manifest,
        shard_files=shard_files,
        valid_files=valid_files,
        shard_num_frames=shard_num_frames,
        clip_shard=np.asarray(table["clip_shard"], dtype=np.int32),
        clip_offset=np.asarray(table["clip_offset"], dtype=np.int64),
        clip_length=np.asarray(table["clip_length"], dtype=np.int64),
        clip_source_group=np.asarray(table["clip_source_group"], dtype=np.int32),
        clip_variant=np.asarray(table["clip_variant"], dtype=np.uint8),
        clip_split=np.asarray(table["clip_split"], dtype=np.uint8),
        clip_mirror=np.asarray(table["clip_mirror"], dtype=bool),
        clip_source_id=np.asarray(table["clip_source_id"], dtype=np.int32),
        trajectory_dim=trajectory_dim,
        future_frames=tuple(int(value) for value in manifest.get("future_frames", [])),
        feature_schema_hash=str(manifest["feature_schema_hash"]),
        normalization_hash=str(manifest.get("feature_normalization_hash", "")),
        split_manifest_hash=str(manifest["split_manifest_hash"]),
        normalization_mean=np.asarray(manifest["normalization_mean"], dtype=np.float32),
        normalization_std=np.asarray(manifest["normalization_std"], dtype=np.float32),
        normalization_valid_frames=int(manifest["normalization_valid_frames"]),
        checkpoint_sha256=str(manifest.get("checkpoint_sha256", "")),
        max_open_shards=int(max_open_shards),
    )
    return store


def root_relative_future(
    root: np.ndarray,
    future_frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Root-relative future positions and headings for every frame of one clip.

    Frame ``t`` gets ``pos[t+dt] - pos[t]`` and the future heading, both rotated
    into the root frame at ``t``. A frame is valid only when *every* requested
    horizon stays inside the same clip, so no sample can reach across a clip or
    split boundary.
    """
    root = np.asarray(root, dtype=np.float32)
    if root.ndim != 2 or root.shape[1] != 7:
        raise ValueError(f"Root channels must be [N, 7] (pos xyz + quat wxyz), got {root.shape}")
    frames = np.asarray([int(value) for value in future_frames], dtype=np.int32)
    if frames.ndim != 1 or len(frames) == 0 or np.any(frames <= 0):
        raise ValueError("future_frames must be a non-empty sequence of positive offsets")
    num_frames = int(root.shape[0])
    max_future = int(frames.max())
    positions = root[:, :3]
    rotations = quat.normalize(np.asarray(root[:, 3:7], dtype=np.float32))
    directions = quat.mul_vec(rotations, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    values = np.zeros((num_frames, len(frames) * 6), dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=bool)
    if num_frames <= max_future:
        return values, valid
    indices = np.arange(0, num_frames - max_future, dtype=np.int64)
    future = indices[:, None] + frames[None, :]
    local_pos = quat.inv_mul_vec(rotations[indices][:, None], positions[future] - positions[indices][:, None])
    local_dir = quat.inv_mul_vec(rotations[indices][:, None], directions[future])
    values[indices] = np.concatenate(
        (local_pos.reshape(len(indices), -1), local_dir.reshape(len(indices), -1)), axis=-1
    ).astype(np.float32)
    valid[indices] = True
    return values, valid


def build_packed_trajectory_store(
    feature_store_path: str | Path,
    output: str | Path,
    *,
    future_frames: Sequence[int] = DEFAULT_FUTURE_FRAMES,
    overwrite: bool = False,
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    """Stream root channels into a packed trajectory store, train-only stats."""
    feature_store = open_packed_feature_store(feature_store_path)
    output = Path(output)
    if output.exists() and not overwrite:
        feature_store.close()
        raise FileExistsError(f"Packed trajectory store already exists: {output}")
    if not feature_store.has_root_channels:
        feature_store.close()
        raise ValueError(
            "Packed feature store has no root channels; rebuild it with root capture enabled"
        )
    frames = tuple(int(value) for value in future_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    (staging / "trajectory").mkdir(parents=True, exist_ok=True)
    (staging / "valid").mkdir(parents=True, exist_ok=True)
    trajectory_dim = len(frames) * 6
    # Values share the feature store's physical shard layout: one trajectory
    # shard per feature shard, with identical frame counts.
    values_shards: list[str] = []
    valid_shards: list[str] = []
    value_hashes: list[str] = []
    valid_hashes: list[str] = []
    total = np.zeros(trajectory_dim, dtype=np.float64)
    total_squared = np.zeros(trajectory_dim, dtype=np.float64)
    count = 0
    try:
        for shard_idx in tqdm(range(len(feature_store.shard_files)), desc="Trajectory shards"):
            frames_in_shard = int(feature_store.shard_num_frames[shard_idx])
            values = np.zeros((frames_in_shard, trajectory_dim), dtype=np.float32)
            valid = np.zeros((frames_in_shard,), dtype=bool)
            rows = np.flatnonzero(np.asarray(feature_store.clip_shard) == shard_idx)
            for clip_idx in rows.tolist():
                start = int(feature_store.clip_offset[clip_idx])
                length = int(feature_store.clip_length[clip_idx])
                root = feature_store.read_root_window(clip_idx, start, length)
                clip_values, clip_valid = root_relative_future(root, frames)
                values[start : start + length] = clip_values
                valid[start : start + length] = clip_valid
                if int(feature_store.clip_split[clip_idx]) == 0 and clip_valid.any():
                    selected = clip_values[clip_valid].astype(np.float64)
                    total += selected.sum(axis=0)
                    total_squared += np.square(selected).sum(axis=0)
                    count += int(selected.shape[0])
                del root
            value_relative = Path("trajectory") / f"shard_{shard_idx:05d}.npy"
            valid_relative = Path("valid") / f"shard_{shard_idx:05d}.npy"
            np.save(staging / value_relative, values)
            np.save(staging / valid_relative, valid)
            values_shards.append(value_relative.as_posix())
            valid_shards.append(valid_relative.as_posix())
            value_hashes.append(sha256_file(staging / value_relative))
            valid_hashes.append(sha256_file(staging / valid_relative))
            del values, valid
        if count <= 0:
            raise ValueError("No valid trajectory frames were produced from the train split")
        mean = total / count
        std = np.sqrt(np.maximum(total_squared / count - np.square(mean), 1e-12)).astype(np.float32)
        from stylized_motion.data.packed_store import ClipTableEntry, write_clip_table

        entries = [
            ClipTableEntry(
                clip_id=clip_idx,
                shard_idx=int(feature_store.clip_shard[clip_idx]),
                offset=int(feature_store.clip_offset[clip_idx]),
                length=int(feature_store.clip_length[clip_idx]),
                source_group=int(feature_store.clip_source_group[clip_idx]),
                variant=int(feature_store.clip_variant[clip_idx]),
                split=int(feature_store.clip_split[clip_idx]),
                mirror=bool(feature_store.clip_mirror[clip_idx]),
                source_id=int(feature_store.clip_source_id[clip_idx]),
                style_id=int(feature_store.clip_style_id[clip_idx]),
                action_id=int(feature_store.clip_action_id[clip_idx]),
                package_id=int(feature_store.clip_package_id[clip_idx]),
                position_sum=None,
            )
            for clip_idx in range(feature_store.num_clips)
        ]
        write_clip_table(staging, entries, num_joints=len(feature_store.names))
        manifest: dict[str, Any] = {
            "data_schema_version": PACKED_SCHEMA_VERSION,
            "store_type": TRAJECTORY_STORE_TYPE,
            "layout": "packed",
            "frame_rate": 60,
            "created_by": "stylized_motion.data.packed_trajectory",
            "num_shards": len(values_shards),
            "shard_files": values_shards,
            "shard_sha256": value_hashes,
            "valid_shard_files": valid_shards,
            "valid_shard_sha256": valid_hashes,
            "shard_num_frames": [int(value) for value in feature_store.shard_num_frames.tolist()],
            "num_clips": int(feature_store.num_clips),
            "total_frames": int(feature_store.total_frames),
            "trajectory_dim": int(trajectory_dim),
            "future_frames": list(frames),
            "feature_order": TRAJECTORY_ORDER,
            "feature_schema_hash": feature_store.feature_schema_hash,
            "feature_normalization_hash": feature_store.normalization_hash,
            "split_manifest_hash": feature_store.split_manifest_hash,
            "normalization_mean": mean.astype(np.float32).tolist(),
            "normalization_std": std.tolist(),
            "normalization_valid_frames": int(count),
            "checkpoint_sha256": str(checkpoint_sha256),
            "clip_names": list(feature_store.manifest.get("clip_names", [])),
            "style_names": list(feature_store.source_style_names),
            "action_names": list(feature_store.source_action_names),
            "package_names": list(feature_store.source_package_names),
            "build": {"status": "complete"},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        # The trajectory store must line up with the feature store clip for clip.
        check = open_packed_trajectory_store(staging)
        try:
            for key in ("clip_source_group", "clip_variant", "clip_split", "clip_length", "clip_mirror"):
                if not np.array_equal(np.asarray(getattr(check, key)), np.asarray(getattr(feature_store, key))):
                    raise ValueError(f"Packed trajectory clip table disagrees with the feature store at {key}")
            horizon = int(max(frames))
            for index in range(check.num_clips):
                _, valid = check.read_clip(index)
                length = int(check.clip_length[index])
                usable = max(0, length - horizon)
                if not valid[:usable].all() or valid[usable:].any():
                    raise ValueError(
                        f"Trajectory validity mask of clip {index} is not the expected "
                        f"[0, {usable}) of {length} frames"
                    )
        finally:
            check.close()
        publish_packed_store(staging, output, overwrite=overwrite)
        return {
            "output": str(output),
            "clips": int(feature_store.num_clips),
            "shards": len(values_shards),
            "trajectory_dim": int(trajectory_dim),
            "future_frames": list(frames),
            "normalization_valid_frames": int(count),
            "feature_schema_hash": feature_store.feature_schema_hash,
            "normalization_hash": feature_store.normalization_hash,
            "split_manifest_hash": feature_store.split_manifest_hash,
        }
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    finally:
        feature_store.close()


class PackedConditionalTokenDataset:
    """Tokens plus clip-local trajectory conditions, validity-masked per window."""

    def __init__(
        self,
        split: str,
        token_store: Any,
        trajectory_store: PackedTrajectoryStore,
        *,
        sequence_frames: int = 65,
        normalize_trajectory: bool = True,
        max_open_shards: int = 16,
        return_metadata: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {split!r}")
        if len(trajectory_store.clip_shard) != len(token_store.clip_shard):
            raise ValueError("Trajectory and token stores have different clip tables")
        for key in ("clip_source_group", "clip_variant", "clip_split", "clip_length"):
            if not np.array_equal(
                np.asarray(getattr(trajectory_store, key)), np.asarray(getattr(token_store, key))
            ):
                raise ValueError(f"Trajectory store disagrees with the token store at {key}")
        self.split = split
        self.token_store = token_store
        self.trajectory_store = trajectory_store
        self.sequence_frames = int(sequence_frames)
        self.normalize_trajectory = bool(normalize_trajectory)
        self.return_metadata = bool(return_metadata)
        self._split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.token_store.max_open_shards = int(max_open_shards)

    def __len__(self) -> int:
        return int((self.token_store.clip_split == self._split_id).sum())

    def __getitems__(self, values: Sequence[int | SampleRequest]) -> dict[str, Any]:
        import torch

        if not values:
            raise ValueError("Cannot collate an empty conditional batch")
        typed = [value for value in values if isinstance(value, SampleRequest)]
        if len(typed) != len(values):
            raise TypeError("PackedConditionalTokenDataset requires SampleRequest entries")
        frames = int(typed[0].target_frames)
        tokens = np.empty((len(typed), frames, int(self.token_store.num_coordinates)), dtype=np.uint8)
        trajectory = np.empty((len(typed), frames, int(self.trajectory_store.trajectory_dim)), dtype=np.float32)
        valid = np.empty((len(typed), frames), dtype=bool)
        for row, request in enumerate(typed):
            clip_idx = int(request.variant_idx)
            if int(self.token_store.clip_split[clip_idx]) != self._split_id:
                raise ValueError(f"SampleRequest clip {clip_idx} does not belong to split {self.split!r}")
            # Sample requests carry the sampler store's absolute frame offset.
            # Token and trajectory stores share clip *rows* but not physical
            # shard offsets, so the window is re-based through the clip row.
            local_start = int(request.target_start) - int(self.token_store.clip_offset[clip_idx])
            frames_requested = int(request.target_frames)
            if local_start < 0 or local_start + frames_requested > int(self.token_store.clip_length[clip_idx]):
                raise IndexError(
                    f"SampleRequest [{request.target_start}, "
                    f"{int(request.target_start) + frames_requested}) leaves clip row {clip_idx}"
                )
            tokens[row] = self.token_store.read_window(
                clip_idx, int(request.target_start), frames_requested
            )
            values_window, valid_window = self.trajectory_store.read_window(
                clip_idx,
                int(self.trajectory_store.clip_offset[clip_idx]) + local_start,
                frames_requested,
            )
            if valid_window.shape[0] != frames:
                raise ValueError("Trajectory window length does not match the token window")
            trajectory[row] = values_window
            valid[row] = valid_window
        if self.normalize_trajectory:
            trajectory -= self.trajectory_store.normalization_mean
            trajectory /= np.maximum(self.trajectory_store.normalization_std, 1e-6)
        batch: dict[str, Any] = {
            "tokens": torch.from_numpy(tokens),
            "trajectory": torch.from_numpy(trajectory),
            "trajectory_valid": torch.from_numpy(valid),
            "loss_mask": torch.ones((len(typed), frames), dtype=torch.bool),
        }
        if self.return_metadata:
            batch["metadata"] = [
                {
                    "clip_id": int(request.variant_idx),
                    "target_start": int(request.target_start),
                    "target_frames": int(request.target_frames),
                    **self.token_store.clip_label(int(request.variant_idx)),
                }
                for request in typed
            ]
        return batch

    def __getitem__(self, index: int | SampleRequest) -> dict[str, Any]:
        batch = self.__getitems__([index])
        return {key: value[0] if key != "metadata" else value[0] for key, value in batch.items()}

    def close(self) -> None:
        self.token_store.close()

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


__all__ = [
    "DEFAULT_FUTURE_FRAMES",
    "TRAJECTORY_ORDER",
    "TRAJECTORY_STORE_TYPE",
    "PackedConditionalTokenDataset",
    "PackedTrajectoryStore",
    "build_packed_trajectory_store",
    "open_packed_trajectory_store",
    "root_relative_future",
]
