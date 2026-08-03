from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stylized_motion.anim.features import MotionFeatureStats, deserialize_motion_feature_stats


DEFAULT_WINDOW_SIZE = 64


@dataclass
class FeatureWindow:
    shard_idx: int
    start_idx: int
    end_idx: int
    range_idx: int


@dataclass
class FeatureStore:
    feature_database: Path
    motion_files: list[Path]
    range_names: np.ndarray
    range_mirror: np.ndarray
    window_size: int
    motion_dim: int
    stats: MotionFeatureStats
    names: list[str]
    parents: np.ndarray
    joint_subset: str
    num_joints: int
    clip_names: list[str]
    split_windows: dict[str, list[FeatureWindow]]
    frame_rate: int
    names_sha256: str
    stats_sha256: str
    feature_schema_hash: str

    def feature_schema(self) -> dict[str, object]:
        return {
            "name": "motion_feature_v2",
            "motion_dim": self.motion_dim,
            "joint_subset": self.joint_subset,
            "names_sha256": self.names_sha256,
            "stats_sha256": self.stats_sha256,
            "feature_schema_hash": self.feature_schema_hash,
        }


def _load_windows(data: dict[str, np.ndarray], key: str) -> list[FeatureWindow]:
    window_array = np.asarray(data[key], dtype=np.int32)
    return [
        FeatureWindow(
            shard_idx=int(row[0]),
            start_idx=int(row[1]),
            end_idx=int(row[2]),
            range_idx=int(row[3]),
        )
        for row in window_array
    ]


def _fixed_windows_from_intervals(intervals: np.ndarray, window_size: int) -> list[FeatureWindow]:
    """Build old-style fixed windows from V2 interval metadata.

    The tokenizer in this checkout still trains on fully supervised 64-frame
    windows.  V2 feature databases store split-safe intervals instead of
    materialized window requests, so derive the same non-overlapping plus tail
    window policy without changing the model's 64-frame contract.
    """
    rows = np.asarray(intervals, dtype=np.int32)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"Expected intervals with shape [N, 4], got {rows.shape}")
    windows = []
    for shard_idx, start_idx, stop_idx, range_idx in rows.tolist():
        if stop_idx - start_idx < window_size:
            continue
        last_start = stop_idx - window_size
        starts = list(range(start_idx, last_start + 1, window_size))
        if starts[-1] != last_start:
            starts.append(last_start)
        windows.extend(
            FeatureWindow(
                shard_idx=int(shard_idx),
                start_idx=int(window_start),
                end_idx=int(window_start + window_size),
                range_idx=int(range_idx),
            )
            for window_start in starts
        )
    if not windows:
        raise ValueError(f"No {window_size}-frame windows can be built from the supplied intervals")
    return windows


def build_feature_store(feature_database: str | Path) -> FeatureStore:
    feature_database = Path(feature_database)
    metadata_path = feature_database / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature database metadata: {metadata_path}")
    npz = np.load(metadata_path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}

    motion_files = [(feature_database / str(name)) for name in np.asarray(data["motion_files"], dtype=object).tolist()]
    range_names = np.asarray(data["range_names"], dtype=object)
    range_mirror = np.asarray(data["range_mirror"], dtype=bool)
    motion_dim = int(np.asarray(data["motion_dim"], dtype=np.int32).item())

    stats, stats_meta = deserialize_motion_feature_stats(data)
    if stats.offset.shape[0] != motion_dim:
        raise ValueError(f"motion_dim={motion_dim} does not match stats offset dim {stats.offset.shape[0]}")
    names = list(stats_meta["names"])
    parents = np.asarray(stats_meta["parents"], dtype=np.int32)
    joint_subset = str(stats_meta["joint_subset"])
    num_joints = int(len(names))
    if {"train_windows", "val_windows", "test_windows", "window_size"}.issubset(data):
        window_size = int(np.asarray(data["window_size"], dtype=np.int32).item())
        split_windows = {
            "train": _load_windows(data, "train_windows"),
            "val": _load_windows(data, "val_windows"),
            "test": _load_windows(data, "test_windows"),
        }
        clip_names = sorted(set(str(name) for name in range_names.tolist()))
    elif {"train_intervals", "val_intervals", "test_intervals"}.issubset(data):
        window_size = DEFAULT_WINDOW_SIZE
        split_windows = {
            split: _fixed_windows_from_intervals(data[f"{split}_intervals"], window_size)
            for split in ("train", "val", "test")
        }
        clip_source = data.get("source_clip_ids", range_names)
        clip_names = sorted(set(str(name) for name in np.asarray(clip_source, dtype=object).tolist()))
    else:
        raise ValueError("Feature metadata must contain either split windows or V2 split intervals")

    names_payload = json.dumps(names, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    names_sha256 = hashlib.sha256(names_payload).hexdigest()
    stats_digest = hashlib.sha256()
    for name, array in (
        ("offset", stats.offset),
        ("scale", stats.scale),
        ("weights", stats.weights),
        ("ref_pos", stats.ref_pos),
    ):
        stats_digest.update(name.encode("ascii"))
        stats_digest.update(np.asarray(array, dtype=np.float32).tobytes())
    stats_sha256 = stats_digest.hexdigest()
    schema_payload = json.dumps(
        {
            "name": "motion_feature_v2",
            "motion_dim": motion_dim,
            "joint_subset": joint_subset,
            "names_sha256": names_sha256,
            "stats_sha256": stats_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    feature_schema_hash = hashlib.sha256(schema_payload).hexdigest()
    frame_rate = int(np.asarray(data.get("frame_rate", 60), dtype=np.int32).item())

    return FeatureStore(
        feature_database=feature_database,
        motion_files=motion_files,
        range_names=range_names,
        range_mirror=range_mirror,
        window_size=window_size,
        motion_dim=motion_dim,
        stats=stats,
        names=names,
        parents=parents,
        joint_subset=joint_subset,
        num_joints=num_joints,
        clip_names=clip_names,
        split_windows=split_windows,
        frame_rate=frame_rate,
        names_sha256=names_sha256,
        stats_sha256=stats_sha256,
        feature_schema_hash=feature_schema_hash,
    )


class FeatureDataset(Dataset):
    def __init__(self, split: str, store: FeatureStore) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.feature_database = store.feature_database
        self.store = store
        self.split = split

        self.motion_files = self.store.motion_files
        self.range_names = self.store.range_names
        self.range_mirror = self.store.range_mirror
        self.window_size = self.store.window_size
        self.motion_dim = self.store.motion_dim
        self.stats = self.store.stats
        self.names = self.store.names
        self.parents = self.store.parents
        self.joint_subset = self.store.joint_subset
        self.num_joints = self.store.num_joints
        self.clip_names = self.store.clip_names
        self.split_windows = self.store.split_windows
        self.windows = self.split_windows[self.split]
        self._motion_arrays: list[np.ndarray] | None = None

    def _ensure_open(self) -> None:
        if self._motion_arrays is None:
            self._motion_arrays = [np.load(path, mmap_mode="r") for path in self.motion_files]
            for path, array in zip(self.motion_files, self._motion_arrays):
                if array.ndim != 2 or array.shape[1] != self.motion_dim:
                    raise ValueError(
                        f"Motion shard {path} must have shape [N, {self.motion_dim}], got {tuple(array.shape)}"
                    )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        self._ensure_open()
        window = self.windows[index]
        motion = np.asarray(
            self._motion_arrays[window.shard_idx][window.start_idx : window.end_idx],
            dtype=np.float32,
        ).copy()
        range_name = str(self.range_names[window.range_idx])
        mirror = bool(self.range_mirror[window.range_idx])
        return {
            "motion": torch.from_numpy(motion),
            "start_idx": window.start_idx,
            "end_idx": window.end_idx,
            "range_idx": window.range_idx,
            "range_name": range_name,
            "mirror": mirror,
        }

    def feature_stats(self) -> MotionFeatureStats:
        return self.stats

    def model_feature_weights(self) -> np.ndarray:
        return self.stats.weights.astype(np.float32)

    def split_summary(self) -> dict[str, int]:
        return {
            "num_windows": len(self.windows),
            "num_ranges": int(len(self.range_names)),
            "num_clip_groups": int(len(self.clip_names)),
            "window_size": self.window_size,
            "motion_dim": self.motion_dim,
            "num_joints": self.num_joints,
        }
