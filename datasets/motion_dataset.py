from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

from motion_features import (
    MotionFeatureStats,
    build_motion_features,
    load_database,
    normalize_motion_features,
    resolve_database_path,
)


@dataclass
class MotionWindow:
    start_idx: int
    end_idx: int
    range_idx: int


def _build_windows_for_ranges(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    valid_range_mask: np.ndarray,
    window_size: int,
) -> list[MotionWindow]:
    windows: list[MotionWindow] = []
    for range_idx, (range_start, range_stop, keep) in enumerate(zip(range_starts, range_stops, valid_range_mask)):
        if not keep:
            continue

        start = int(range_start)
        stop = int(range_stop)
        length = stop - start
        if length < window_size:
            continue

        last_start = stop - window_size
        for window_start in range(start, last_start + 1, window_size):
            windows.append(
                MotionWindow(
                    start_idx=window_start,
                    end_idx=window_start + window_size,
                    range_idx=range_idx,
                )
            )

        if windows and windows[-1].range_idx == range_idx and windows[-1].start_idx != last_start:
            windows.append(
                MotionWindow(
                    start_idx=last_start,
                    end_idx=last_start + window_size,
                    range_idx=range_idx,
                )
            )
    return windows


def _seed_for_group(base_seed: int, group_name: str) -> int:
    return (int(base_seed) + zlib.crc32(group_name.encode("utf-8"))) % (2**32)


def _split_counts(num_items: int) -> dict[str, int]:
    if num_items <= 0:
        return {"train": 0, "val": 0, "test": 0}

    n_train = int(np.floor(num_items * 0.8))
    n_val = int(np.floor(num_items * 0.1))
    n_test = num_items - n_train - n_val

    if num_items >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(n_train - 1, 1)
        if n_test == 0:
            n_test = 1
            n_train = max(n_train - 1, 1)

    while n_train + n_val + n_test > num_items:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    return {"train": n_train, "val": n_val, "test": n_test}


def _relative_window_starts(range_start: int, range_stop: int, window_size: int) -> list[int]:
    length = int(range_stop) - int(range_start)
    if length < window_size:
        return []

    last_start = int(range_stop) - window_size
    return list(range(int(range_start), last_start + 1, window_size))


def _build_windows_within_clip_split(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    range_names: np.ndarray,
    window_size: int,
    seed: int,
) -> dict[str, list[MotionWindow]]:
    split_windows: dict[str, list[MotionWindow]] = {"train": [], "val": [], "test": []}
    unique_clip_names = sorted(set(str(name) for name in range_names.tolist()))

    for clip_name in unique_clip_names:
        group_range_indices = np.nonzero(range_names == clip_name)[0].tolist()
        if not group_range_indices:
            continue

        anchor_range_idx = group_range_indices[0]
        anchor_starts = _relative_window_starts(
            range_start=int(range_starts[anchor_range_idx]),
            range_stop=int(range_stops[anchor_range_idx]),
            window_size=window_size,
        )
        if not anchor_starts:
            continue

        counts = _split_counts(len(anchor_starts))
        rng = np.random.default_rng(_seed_for_group(seed, clip_name))
        perm = rng.permutation(len(anchor_starts))

        split_to_indices = {
            "train": perm[: counts["train"]],
            "val": perm[counts["train"] : counts["train"] + counts["val"]],
            "test": perm[counts["train"] + counts["val"] :],
        }

        for range_idx in group_range_indices:
            starts = _relative_window_starts(
                range_start=int(range_starts[range_idx]),
                range_stop=int(range_stops[range_idx]),
                window_size=window_size,
            )
            if len(starts) != len(anchor_starts):
                raise ValueError(
                    f"Clip {clip_name} has inconsistent mirrored window counts: "
                    f"{len(anchor_starts)} vs {len(starts)}"
                )

            for split_name, local_indices in split_to_indices.items():
                for local_idx in local_indices.tolist():
                    start_idx = int(starts[local_idx])
                    split_windows[split_name].append(
                        MotionWindow(
                            start_idx=start_idx,
                            end_idx=start_idx + window_size,
                            range_idx=range_idx,
                        )
                    )

    return split_windows


def _unique_frame_indices(windows: list[MotionWindow]) -> np.ndarray:
    if not windows:
        raise ValueError("Expected at least one window when building normalization statistics")
    frame_indices = np.concatenate(
        [np.arange(window.start_idx, window.end_idx, dtype=np.int32) for window in windows],
        axis=0,
    )
    return np.unique(frame_indices)


class MotionDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        window_size: int = 64,
        database_path: str | Path | None = None,
        use_full_skeleton: bool = False,
        use_root_cond: bool = True,
        root_cond_dim: int = 6,
        seed: int = 3407,
        normalize: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.normalize = normalize
        self.use_root_cond = bool(use_root_cond)
        self.root_cond_dim = int(root_cond_dim)
        self.database_path = resolve_database_path(use_full_skeleton=use_full_skeleton, database_path=database_path)

        self.database = load_database(self.database_path)
        self.names = self.database["names"].tolist()
        self.parents = self.database["parents"].astype(np.int32)
        self.range_starts = self.database["range_starts"].astype(np.int32)
        self.range_stops = self.database["range_stops"].astype(np.int32)
        self.range_names = self.database["range_names"]
        self.range_mirror = self.database["range_mirror"].astype(bool)
        self.joint_subset = str(self.database["joint_subset"].item())

        self.clip_names = sorted(set(str(name) for name in self.range_names.tolist()))
        self.split_windows = _build_windows_within_clip_split(
            range_starts=self.range_starts,
            range_stops=self.range_stops,
            range_names=self.range_names,
            window_size=self.window_size,
            seed=self.seed,
        )
        self.windows = self.split_windows[self.split]
        train_frame_indices = _unique_frame_indices(self.split_windows["train"])

        self.motion_features, self.stats = build_motion_features(
            self.database,
            stat_frame_indices=train_frame_indices,
        )
        if self.normalize:
            self.motion_features = normalize_motion_features(self.motion_features, self.stats)

        self.full_motion_dim = int(self.motion_features.shape[1])
        self.motion_dim = self.full_motion_dim - self.root_cond_dim if self.use_root_cond else self.full_motion_dim
        self.num_joints = int(len(self.names))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        window = self.windows[index]
        full_motion = self.motion_features[window.start_idx : window.end_idx]
        if self.use_root_cond:
            root_cond = full_motion[:, : self.root_cond_dim]
            motion = full_motion[:, self.root_cond_dim :]
        else:
            root_cond = np.zeros((len(full_motion), 0), dtype=np.float32)
            motion = full_motion
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
        if self.use_root_cond:
            return self.stats.weights[self.root_cond_dim :].astype(np.float32)
        return self.stats.weights.astype(np.float32)

    def pack_full_motion(self, motion: np.ndarray, root_cond: np.ndarray) -> np.ndarray:
        motion = np.asarray(motion, dtype=np.float32)
        if not self.use_root_cond:
            return motion
        root_cond = np.asarray(root_cond, dtype=np.float32)
        return np.concatenate([root_cond, motion], axis=-1).astype(np.float32)

    def split_summary(self) -> dict[str, int]:
        return {
            "num_windows": len(self.windows),
            "num_ranges": int(len(self.range_names)),
            "num_clip_groups": int(len(self.clip_names)),
            "window_size": self.window_size,
            "motion_dim": self.motion_dim,
            "full_motion_dim": self.full_motion_dim,
            "use_root_cond": self.use_root_cond,
            "root_cond_dim": self.root_cond_dim,
            "num_joints": self.num_joints,
        }
