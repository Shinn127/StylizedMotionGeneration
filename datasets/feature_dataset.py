from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from motion_features import MotionFeatureStats, deserialize_motion_feature_stats


@dataclass
class FeatureWindow:
    start_idx: int
    end_idx: int
    range_idx: int


@dataclass
class FeatureStore:
    feature_database: Path
    motion: np.ndarray
    root_cond: np.ndarray
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
            start_idx=int(row[0]),
            end_idx=int(row[1]),
            range_idx=int(row[2]),
        )
        for row in window_array
    ]


def build_feature_store(feature_database: str | Path) -> FeatureStore:
    feature_database = Path(feature_database)
    npz = np.load(feature_database, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}

    motion = np.asarray(data["motion"], dtype=np.float32)
    root_cond = np.asarray(data["root_cond"], dtype=np.float32)
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
        motion=motion,
        root_cond=root_cond,
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
    def __init__(
        self,
        split: str,
        store: FeatureStore,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.feature_database = store.feature_database
        self.store = store
        self.split = split

        self.motion = self.store.motion
        self.root_cond = self.store.root_cond
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

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        window = self.windows[index]
        motion = self.motion[window.start_idx : window.end_idx]
        root_cond = self.root_cond[window.start_idx : window.end_idx]
        range_name = str(self.range_names[window.range_idx])
        mirror = bool(self.range_mirror[window.range_idx])
        return {
            "motion": torch.from_numpy(motion.astype(np.float32)),
            "root_cond": torch.from_numpy(root_cond.astype(np.float32)),
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
        motion = np.asarray(motion, dtype=np.float32)
        root_cond = np.asarray(root_cond, dtype=np.float32)
        return np.concatenate([root_cond, motion], axis=-1).astype(np.float32)

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
