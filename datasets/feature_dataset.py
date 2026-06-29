from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from motion_features import MotionFeatureStats, deserialize_motion_feature_stats


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
    root_cond_files: list[Path]
    range_names: np.ndarray
    range_mirror: np.ndarray
    window_size: int
    root_cond_dim: int
    motion_dim: int
    full_motion_dim: int
    stats: MotionFeatureStats
    names: list[str]
    parents: np.ndarray
    joint_subset: str
    num_joints: int
    clip_names: list[str]
    split_windows: dict[str, list[FeatureWindow]]


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


def build_feature_store(feature_database: str | Path) -> FeatureStore:
    feature_database = Path(feature_database)
    metadata_path = feature_database / "metadata.npz"
    npz = np.load(metadata_path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}

    motion_files = [(feature_database / str(name)) for name in np.asarray(data["motion_files"], dtype=object).tolist()]
    root_cond_files = [(feature_database / str(name)) for name in np.asarray(data["root_cond_files"], dtype=object).tolist()]
    range_names = np.asarray(data["range_names"], dtype=object)
    range_mirror = np.asarray(data["range_mirror"], dtype=bool)
    window_size = int(np.asarray(data["window_size"], dtype=np.int32).item())
    root_cond_dim = int(np.asarray(data["root_cond_dim"], dtype=np.int32).item())
    motion_dim = int(np.asarray(data["motion_dim"], dtype=np.int32).item())
    full_motion_dim = int(np.asarray(data["full_motion_dim"], dtype=np.int32).item())

    stats, stats_meta = deserialize_motion_feature_stats(data)
    names = list(stats_meta["names"])
    parents = np.asarray(stats_meta["parents"], dtype=np.int32)
    joint_subset = str(stats_meta["joint_subset"])
    num_joints = int(len(names))
    clip_names = sorted(set(str(name) for name in range_names.tolist()))
    split_windows = {
        "train": _load_windows(data, "train_windows"),
        "val": _load_windows(data, "val_windows"),
        "test": _load_windows(data, "test_windows"),
    }

    return FeatureStore(
        feature_database=feature_database,
        motion_files=motion_files,
        root_cond_files=root_cond_files,
        range_names=range_names,
        range_mirror=range_mirror,
        window_size=window_size,
        root_cond_dim=root_cond_dim,
        motion_dim=motion_dim,
        full_motion_dim=full_motion_dim,
        stats=stats,
        names=names,
        parents=parents,
        joint_subset=joint_subset,
        num_joints=num_joints,
        clip_names=clip_names,
        split_windows=split_windows,
    )


class FeatureDataset(Dataset):
    def __init__(self, split: str, store: FeatureStore) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.feature_database = store.feature_database
        self.store = store
        self.split = split

        self.motion_files = self.store.motion_files
        self.root_cond_files = self.store.root_cond_files
        self.range_names = self.store.range_names
        self.range_mirror = self.store.range_mirror
        self.window_size = self.store.window_size
        self.root_cond_dim = self.store.root_cond_dim
        self.motion_dim = self.store.motion_dim
        self.full_motion_dim = self.store.full_motion_dim
        self.stats = self.store.stats
        self.names = self.store.names
        self.parents = self.store.parents
        self.joint_subset = self.store.joint_subset
        self.num_joints = self.store.num_joints
        self.clip_names = self.store.clip_names
        self.split_windows = self.store.split_windows
        self.windows = self.split_windows[self.split]
        self._motion_arrays: list[np.ndarray] | None = None
        self._root_cond_arrays: list[np.ndarray] | None = None

    def _ensure_open(self) -> None:
        if self._motion_arrays is None:
            self._motion_arrays = [np.load(path, mmap_mode="r") for path in self.motion_files]
            self._root_cond_arrays = [np.load(path, mmap_mode="r") for path in self.root_cond_files]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        self._ensure_open()
        window = self.windows[index]
        motion = np.asarray(
            self._motion_arrays[window.shard_idx][window.start_idx : window.end_idx],
            dtype=np.float32,
        ).copy()
        root_cond = np.asarray(
            self._root_cond_arrays[window.shard_idx][window.start_idx : window.end_idx],
            dtype=np.float32,
        ).copy()
        range_name = str(self.range_names[window.range_idx])
        mirror = bool(self.range_mirror[window.range_idx])
        return {
            "motion": torch.from_numpy(motion),
            "root_cond": torch.from_numpy(root_cond),
            "start_idx": window.start_idx,
            "end_idx": window.end_idx,
            "range_idx": window.range_idx,
            "range_name": range_name,
            "mirror": mirror,
        }

    def feature_stats(self) -> MotionFeatureStats:
        return self.stats

    def model_feature_weights(self) -> np.ndarray:
        return self.stats.weights[self.root_cond_dim :].astype(np.float32)

    def pack_full_motion(self, motion: np.ndarray, root_cond: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(root_cond, dtype=np.float32),
                np.asarray(motion, dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

    def split_summary(self) -> dict[str, int | bool]:
        return {
            "num_windows": len(self.windows),
            "num_ranges": int(len(self.range_names)),
            "num_clip_groups": int(len(self.clip_names)),
            "window_size": self.window_size,
            "motion_dim": self.motion_dim,
            "full_motion_dim": self.full_motion_dim,
            "use_root_cond": True,
            "root_cond_dim": self.root_cond_dim,
            "num_joints": self.num_joints,
        }
