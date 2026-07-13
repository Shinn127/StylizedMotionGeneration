from pathlib import Path
import zlib

import numpy as np
from tqdm import tqdm

from motion_features import (
    MotionFeatureStats,
    build_motion_feature_components,
    default_joint_weights,
    serialize_motion_feature_stats,
)
from preprocess import bvh
from preprocess.build_database import (
    MotionDatabaseWriter,
    build_100style_tags,
    build_lafan_tags,
    iter_motion_pairs,
    source_path_for,
)


def _parse_style_filter(styles_arg):
    if not styles_arg:
        return None
    return {style.strip() for style in styles_arg.split(",") if style.strip()}


def _seed_for_group(base_seed: int, group_name: str) -> int:
    return (int(base_seed) + zlib.crc32(group_name.encode("utf-8"))) % (2**32)


def _relative_window_starts(nframes: int, window_size: int) -> list[int]:
    if nframes < window_size:
        return []
    last_start = nframes - window_size
    starts = list(range(0, last_start + 1, window_size))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


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


class FeatureStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.x_sum = None
        self.x_sumsq = None
        self.ref_pos_sum = None

    @staticmethod
    def _update_moments(sum_arr, sumsq_arr, values):
        if sum_arr is None:
            sum_arr = values.sum(axis=0, dtype=np.float64)
            sumsq_arr = np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        else:
            sum_arr += values.sum(axis=0, dtype=np.float64)
            sumsq_arr += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        return sum_arr, sumsq_arr

    def update(self, components, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != components.x.shape[0]:
            raise ValueError(f"Invalid train frame mask shape {mask.shape} for component length {components.x.shape[0]}")
        if not np.any(mask):
            return

        x = components.x[mask]
        self.x_sum, self.x_sumsq = self._update_moments(self.x_sum, self.x_sumsq, x)
        if self.ref_pos_sum is None:
            self.ref_pos_sum = components.positions[mask].sum(axis=0, dtype=np.float64)
        else:
            self.ref_pos_sum += components.positions[mask].sum(axis=0, dtype=np.float64)
        self.count += int(mask.sum())

    def finalize(self, names):
        if self.count <= 0:
            raise ValueError("No training frames available for statistics")

        def mean_and_std(sum_arr, sumsq_arr):
            mean = (sum_arr / self.count).astype(np.float32)
            var = np.maximum(sumsq_arr / self.count - np.square(mean, dtype=np.float32), 1e-8)
            std = np.sqrt(var).astype(np.float32)
            return mean, std

        x_mean, x_std = mean_and_std(self.x_sum, self.x_sumsq)
        nbones = len(names)
        rot_stop = 9 + (nbones - 1) * 6
        hip_vel_stop = rot_stop + 3
        angular_stop = hip_vel_stop + (nbones - 1) * 3

        scale = np.concatenate(
            [
                np.full(3, x_std[0:3].mean(), dtype=np.float32),
                np.full(3, x_std[3:6].mean(), dtype=np.float32),
                np.full(3, x_std[6:9].mean(), dtype=np.float32),
                np.full(rot_stop - 9, x_std[9:rot_stop].mean(), dtype=np.float32),
                np.full(3, x_std[rot_stop:hip_vel_stop].mean(), dtype=np.float32),
                np.full(angular_stop - hip_vel_stop, x_std[hip_vel_stop:angular_stop].mean(), dtype=np.float32),
                np.full(2, x_std[angular_stop:].mean(), dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        scale = np.maximum(scale, 1e-8)

        joint_weights = default_joint_weights(names)
        weights = np.concatenate(
            [
                np.ones(3, dtype=np.float32),
                np.ones(3, dtype=np.float32),
                np.ones(3, dtype=np.float32),
                joint_weights[1:].repeat(6).astype(np.float32) * (nbones - 1),
                np.ones(3, dtype=np.float32),
                joint_weights[1:].repeat(3).astype(np.float32) * (nbones - 1),
                np.ones(2, dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        dist = (x_std / scale).astype(np.float32)

        return MotionFeatureStats(
            offset=x_mean,
            scale=scale,
            dist=dist.astype(np.float32),
            weights=weights,
            ref_pos=(self.ref_pos_sum / self.count).astype(np.float32),
        )


def _build_shard_specs(dataset_name, styles_arg, max_styles):
    if dataset_name == "lafan":
        tags_data = build_lafan_tags()
    else:
        tags_data = build_100style_tags(
            style_filter=_parse_style_filter(styles_arg),
            max_styles=max_styles,
        )

    bvh_paths = []
    for range_name, tag, _range_start, _range_end in tags_data:
        if tag != "all":
            continue
        bvh_paths.append(source_path_for(dataset_name, range_name))
    bvh_paths = list(dict.fromkeys(bvh_paths))
    if not bvh_paths:
        raise FileNotFoundError(f"No BVH files found for dataset {dataset_name}")

    shard_specs = [
        {
            "range_name": path.stem,
            "mirror": bool(mirror),
            "nframes": int(bvh.read_frame_count(path)),
        }
        for path in bvh_paths
        for mirror in [False, True]
    ]
    return shard_specs, tags_data, bvh_paths


def _build_split_windows(shard_specs, window_size, seed):
    split_windows = {"train": [], "val": [], "test": []}
    range_names = np.asarray([spec["range_name"] for spec in shard_specs], dtype=object)
    unique_clip_names = sorted(set(str(name) for name in range_names.tolist()))

    for clip_name in unique_clip_names:
        group_indices = np.nonzero(range_names == clip_name)[0].tolist()
        if not group_indices:
            continue
        anchor_idx = group_indices[0]
        anchor_starts = _relative_window_starts(shard_specs[anchor_idx]["nframes"], window_size)
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

        for shard_idx in group_indices:
            starts = _relative_window_starts(shard_specs[shard_idx]["nframes"], window_size)
            if len(starts) != len(anchor_starts):
                raise ValueError(
                    f"Clip {clip_name} has inconsistent mirrored window counts: {len(anchor_starts)} vs {len(starts)}"
                )
            for split_name, local_indices in split_to_indices.items():
                for local_idx in local_indices.tolist():
                    start_idx = int(starts[local_idx])
                    split_windows[split_name].append([shard_idx, start_idx, start_idx + window_size, shard_idx])

    return {
        "train_windows": np.asarray(split_windows["train"], dtype=np.int32),
        "val_windows": np.asarray(split_windows["val"], dtype=np.int32),
        "test_windows": np.asarray(split_windows["test"], dtype=np.int32),
    }


def _save_motion_shard(output_dir: Path, shard_idx: int, motion: np.ndarray) -> str:
    motion_dir = output_dir / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    motion_rel = Path("motion") / f"motion_{shard_idx:05d}.npy"
    np.save(output_dir / motion_rel, motion.astype(np.float32))
    return motion_rel.as_posix()


def _normalize_motion_shard(path: Path, stats: MotionFeatureStats, chunk_size: int = 16384) -> None:
    motion = np.load(path, mmap_mode="r+")
    if motion.ndim != 2 or motion.shape[1] != stats.offset.shape[0]:
        raise ValueError(f"Unexpected raw feature shard shape at {path}: {motion.shape}")
    for start in range(0, len(motion), chunk_size):
        stop = min(start + chunk_size, len(motion))
        motion[start:stop] = (motion[start:stop] - stats.offset) / stats.scale
    motion.flush()


def build_processed_data(
    dataset_name,
    output_dir,
    styles_arg=None,
    max_styles=None,
    prune_ends_and_fingers=False,
    window_size=64,
    seed=3407,
    workers=1,
):
    output_dir = Path(output_dir)
    feature_dir = output_dir / "feature_database"
    feature_dir.mkdir(parents=True, exist_ok=True)
    shard_specs, tags_data, bvh_paths = _build_shard_specs(
        dataset_name=dataset_name,
        styles_arg=styles_arg,
        max_styles=max_styles,
    )
    split_windows = _build_split_windows(shard_specs, window_size=window_size, seed=seed)

    train_frame_masks = [np.zeros(spec["nframes"], dtype=bool) for spec in shard_specs]
    for shard_idx, start_idx, end_idx, _range_idx in split_windows["train_windows"].tolist():
        train_frame_masks[int(shard_idx)][int(start_idx):int(end_idx)] = True

    stats_acc = FeatureStatsAccumulator()
    names = None
    parents = None
    joint_subset = "prune_ends_and_fingers" if prune_ends_and_fingers else "full"
    motion_files = []
    database_path = output_dir / "database.npz"
    database = MotionDatabaseWriter(
        database_path,
        total_frames=sum(spec["nframes"] for spec in shard_specs),
        tags_data=tags_data,
        prune_ends_and_fingers=prune_ends_and_fingers,
    )

    shard_idx = 0
    for range_name, motions in iter_motion_pairs(
        bvh_paths,
        prune_ends_and_fingers=prune_ends_and_fingers,
        workers=workers,
        desc="Building motion + features",
    ):
        for mirror, motion in motions:
            spec = shard_specs[shard_idx]
            if (range_name, mirror) != (spec["range_name"], spec["mirror"]):
                raise ValueError(f"Unexpected motion order at shard {shard_idx}: {range_name}, mirror={mirror}")
            expected_frames = int(spec["nframes"])
            if len(motion["positions"]) != expected_frames:
                raise ValueError(
                    f"Frame count changed for {range_name} (mirror={mirror}): "
                    f"metadata={expected_frames}, processed={len(motion['positions'])}"
                )
            if names is None:
                names = motion["names"]
                parents = motion["parents"]
            elif names != motion["names"] or not np.array_equal(parents, motion["parents"]):
                raise ValueError(f"Skeleton mismatch while processing {range_name} (mirror={mirror})")

            components = build_motion_feature_components(motion)
            stats_acc.update(components, train_frame_masks[shard_idx])
            motion_files.append(_save_motion_shard(feature_dir, shard_idx, components.x))
            database.add(range_name, mirror, motion)
            shard_idx += 1

    stats = stats_acc.finalize(names)
    if shard_idx != len(shard_specs):
        raise ValueError(f"Expected {len(shard_specs)} motion shards, processed {shard_idx}")
    for motion_rel in tqdm(motion_files, desc="Normalizing features"):
        _normalize_motion_shard(feature_dir / motion_rel, stats)

    database.save()

    stats_payload = serialize_motion_feature_stats(stats, names=names, parents=parents, joint_subset=joint_subset)
    metadata = {
        "motion_files": np.asarray(motion_files, dtype=object),
        "train_windows": split_windows["train_windows"],
        "val_windows": split_windows["val_windows"],
        "test_windows": split_windows["test_windows"],
        "range_names": np.asarray([spec["range_name"] for spec in shard_specs], dtype=object),
        "range_mirror": np.asarray([spec["mirror"] for spec in shard_specs], dtype=bool),
        "window_size": np.asarray(window_size, dtype=np.int32),
        "motion_dim": np.asarray(stats.offset.shape[0], dtype=np.int32),
        "source_database": np.asarray(str(database_path), dtype=object),
    }
    metadata.update(stats_payload)
    np.savez(feature_dir / "metadata.npz", **metadata)

    print(f"saved={output_dir}")
    print(f"database={database_path}")
    print(f"motion_files={len(motion_files)}")
    print(f"train_windows={len(split_windows['train_windows'])}")
    print(f"val_windows={len(split_windows['val_windows'])}")
    print(f"test_windows={len(split_windows['test_windows'])}")
