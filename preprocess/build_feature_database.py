import argparse
import os
import multiprocessing as mp
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import zlib

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())
PREPROCESS_DIR = PROJECT_ROOT / "preprocess"
if PREPROCESS_DIR.as_posix() not in sys.path:
    sys.path.insert(0, PREPROCESS_DIR.as_posix())

from motion_features import MotionFeatureStats, build_motion_feature_components, default_joint_weights, serialize_motion_feature_stats
from preprocess.build_database import _process_motion, build_100style_tags, build_lafan_tags, source_path_for


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lafan", "100style"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--styles", type=str, default=None)
    parser.add_argument("--max-styles", type=int, default=None)
    parser.add_argument("--prune-ends-and-fingers", action="store_true")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--root-cond-dim", type=int, default=6)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    return parser.parse_args()


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
        self.x_rvel_sum = None
        self.x_rvel_sumsq = None
        self.x_rang_sum = None
        self.x_rang_sumsq = None
        self.x_hip_pos_sum = None
        self.x_hip_pos_sumsq = None
        self.x_rot_6d_sum = None
        self.x_rot_6d_sumsq = None
        self.x_hip_vel_sum = None
        self.x_hip_vel_sumsq = None
        self.x_ang_sum = None
        self.x_ang_sumsq = None
        self.x_contacts_sum = None
        self.x_contacts_sumsq = None
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
        self.x_rvel_sum, self.x_rvel_sumsq = self._update_moments(self.x_rvel_sum, self.x_rvel_sumsq, components.x_rvel[mask])
        self.x_rang_sum, self.x_rang_sumsq = self._update_moments(self.x_rang_sum, self.x_rang_sumsq, components.x_rang[mask])
        self.x_hip_pos_sum, self.x_hip_pos_sumsq = self._update_moments(self.x_hip_pos_sum, self.x_hip_pos_sumsq, components.x_hip_pos[mask])
        self.x_rot_6d_sum, self.x_rot_6d_sumsq = self._update_moments(self.x_rot_6d_sum, self.x_rot_6d_sumsq, components.x_rot_6d[mask])
        self.x_hip_vel_sum, self.x_hip_vel_sumsq = self._update_moments(self.x_hip_vel_sum, self.x_hip_vel_sumsq, components.x_hip_vel[mask])
        self.x_ang_sum, self.x_ang_sumsq = self._update_moments(self.x_ang_sum, self.x_ang_sumsq, components.x_ang_local[mask])
        self.x_contacts_sum, self.x_contacts_sumsq = self._update_moments(self.x_contacts_sum, self.x_contacts_sumsq, components.x_contacts[mask])
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
        x_rvel_mean, x_rvel_std = mean_and_std(self.x_rvel_sum, self.x_rvel_sumsq)
        x_rang_mean, x_rang_std = mean_and_std(self.x_rang_sum, self.x_rang_sumsq)
        x_hip_pos_mean, x_hip_pos_std = mean_and_std(self.x_hip_pos_sum, self.x_hip_pos_sumsq)
        x_rot_6d_mean, x_rot_6d_std = mean_and_std(self.x_rot_6d_sum, self.x_rot_6d_sumsq)
        x_hip_vel_mean, x_hip_vel_std = mean_and_std(self.x_hip_vel_sum, self.x_hip_vel_sumsq)
        x_ang_mean, x_ang_std = mean_and_std(self.x_ang_sum, self.x_ang_sumsq)
        x_contacts_mean, x_contacts_std = mean_and_std(self.x_contacts_sum, self.x_contacts_sumsq)

        scale = np.concatenate(
            [
                np.full(3, x_rvel_std.mean(), dtype=np.float32),
                np.full(3, x_rang_std.mean(), dtype=np.float32),
                np.full(3, x_hip_pos_std.mean(), dtype=np.float32),
                np.full(x_rot_6d_mean.shape[0], x_rot_6d_std.mean(), dtype=np.float32),
                np.full(3, x_hip_vel_std.mean(), dtype=np.float32),
                np.full(x_ang_mean.shape[0], x_ang_std.mean(), dtype=np.float32),
                np.full(2, x_contacts_std.mean(), dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        scale = np.maximum(scale, 1e-8)

        joint_weights = default_joint_weights(names)
        nbones = len(names)
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


def _save_raw_shard(output_dir: Path, shard_idx: int, components) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"motion_{shard_idx:05d}.npy"
    np.save(raw_path, components.x.astype(np.float32))
    return raw_path


def _build_feature_shard(task):
    shard_idx, spec, train_mask, prune_ends_and_fingers, raw_dir = task
    motion = _process_motion(
        spec["path"],
        spec["mirror"],
        prune_ends_and_fingers=prune_ends_and_fingers,
    )
    components = build_motion_feature_components(motion)
    raw_path = _save_raw_shard(raw_dir, shard_idx, components)
    return {
        "shard_idx": shard_idx,
        "range_name": spec["range_name"],
        "mirror": spec["mirror"],
        "nframes": spec["nframes"],
        "names": motion["names"],
        "parents": motion["parents"],
        "components": components,
        "train_mask": np.asarray(train_mask, dtype=bool),
        "raw_path": raw_path,
    }


def _build_shard_specs(dataset_name, styles_arg, max_styles, prune_ends_and_fingers):
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

    shard_specs = []
    for path in bvh_paths:
        for mirror in [False, True]:
            motion = _process_motion(path, mirror, prune_ends_and_fingers=prune_ends_and_fingers)
            shard_specs.append(
                {
                    "range_name": path.stem,
                    "mirror": bool(mirror),
                    "nframes": int(len(motion["positions"])),
                    "path": path,
                }
            )
    return shard_specs


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


def _save_shard_arrays(output_dir: Path, shard_idx: int, motion: np.ndarray, root_cond: np.ndarray) -> tuple[str, str]:
    motion_dir = output_dir / "motion"
    root_dir = output_dir / "root_cond"
    motion_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)

    motion_rel = Path("motion") / f"motion_{shard_idx:05d}.npy"
    root_rel = Path("root_cond") / f"root_cond_{shard_idx:05d}.npy"
    np.save(output_dir / motion_rel, motion.astype(np.float32))
    np.save(output_dir / root_rel, root_cond.astype(np.float32))
    return motion_rel.as_posix(), root_rel.as_posix()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    shard_specs = _build_shard_specs(
        dataset_name=args.dataset,
        styles_arg=args.styles,
        max_styles=args.max_styles,
        prune_ends_and_fingers=args.prune_ends_and_fingers,
    )
    split_windows = _build_split_windows(shard_specs, window_size=args.window_size, seed=args.seed)

    train_frame_masks = [np.zeros(spec["nframes"], dtype=bool) for spec in shard_specs]
    for shard_idx, start_idx, end_idx, _range_idx in split_windows["train_windows"].tolist():
        train_frame_masks[int(shard_idx)][int(start_idx):int(end_idx)] = True

    raw_dir = args.output / "_raw"
    tasks = [
        (shard_idx, spec, train_frame_masks[shard_idx], args.prune_ends_and_fingers, raw_dir)
        for shard_idx, spec in enumerate(shard_specs)
    ]

    stats_acc = FeatureStatsAccumulator()
    names = None
    parents = None
    joint_subset = "prune_ends_and_fingers" if args.prune_ends_and_fingers else "full"
    raw_paths = [None] * len(tasks)

    workers = max(1, int(args.workers))
    if workers == 1:
        results = [_build_feature_shard(task) for task in tqdm(tasks, desc="Building shards")]
    else:
        context = mp.get_context("fork")
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            results = list(tqdm(executor.map(_build_feature_shard, tasks, chunksize=chunksize), total=len(tasks), desc="Building shards"))

    for result in results:
        shard_idx = result["shard_idx"]
        if names is None:
            names = result["names"]
            parents = result["parents"]
        stats_acc.update(result["components"], result["train_mask"])
        raw_paths[shard_idx] = result["raw_path"]

    stats = stats_acc.finalize(names)

    motion_files = []
    root_cond_files = []
    for shard_idx, raw_path in enumerate(raw_paths):
        if raw_path is None:
            raise RuntimeError(f"Missing raw shard path for shard {shard_idx}")
        raw_x = np.load(raw_path, mmap_mode="r")
        norm_x = ((raw_x.astype(np.float32) - stats.offset) / stats.scale).astype(np.float32)
        motion = norm_x[:, args.root_cond_dim :]
        root_cond = norm_x[:, : args.root_cond_dim]
        motion_rel, root_rel = _save_shard_arrays(args.output, shard_idx, motion, root_cond)
        motion_files.append(motion_rel)
        root_cond_files.append(root_rel)

    stats_payload = serialize_motion_feature_stats(stats, names=names, parents=parents, joint_subset=joint_subset)
    metadata = {
        "motion_files": np.asarray(motion_files, dtype=object),
        "root_cond_files": np.asarray(root_cond_files, dtype=object),
        "train_windows": split_windows["train_windows"],
        "val_windows": split_windows["val_windows"],
        "test_windows": split_windows["test_windows"],
        "range_names": np.asarray([spec["range_name"] for spec in shard_specs], dtype=object),
        "range_mirror": np.asarray([spec["mirror"] for spec in shard_specs], dtype=bool),
        "window_size": np.asarray(args.window_size, dtype=np.int32),
        "root_cond_dim": np.asarray(args.root_cond_dim, dtype=np.int32),
        "motion_dim": np.asarray(stats.offset.shape[0] - args.root_cond_dim, dtype=np.int32),
        "full_motion_dim": np.asarray(stats.offset.shape[0], dtype=np.int32),
    }
    metadata.update(stats_payload)
    np.savez(args.output / "metadata.npz", **metadata)

    print(f"saved={args.output}")
    print(f"motion_files={len(motion_files)}")
    print(f"train_windows={len(split_windows['train_windows'])}")
    print(f"val_windows={len(split_windows['val_windows'])}")
    print(f"test_windows={len(split_windows['test_windows'])}")


if __name__ == "__main__":
    main()
