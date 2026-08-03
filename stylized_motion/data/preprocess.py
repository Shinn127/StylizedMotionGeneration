from collections import deque
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import hashlib
import json
from itertools import islice
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import torch
from tqdm import tqdm

from stylized_motion.anim import bvh, quat
from stylized_motion.anim.features import MotionFeatureStats, build_motion_feature_components, serialize_motion_feature_stats
from stylized_motion.data.feature_data import canonical_json_bytes, open_feature_store, sha256_file
from stylized_motion.data.sampling import SplitManifest, build_split_manifest
from stylized_motion.data.trajectory_data import open_trajectory_store
from stylized_motion.data.token_data import open_token_store
from stylized_motion.util.paths import DATA_DIR
from stylized_motion.data.preprocess_worker import process_motion_pair


RAW_DIR = DATA_DIR / "raw"

LAFAN_SOURCE = RAW_DIR / "lafan"
STYLE100_SOURCE = RAW_DIR / "100style"

STYLE100_CLIPS = ["BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"]
FINGER_TOKENS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _mirror_bones(names):
    mirrored = []
    for idx, name in enumerate(names):
        if "Right" in name and name.replace("Right", "Left") in names:
            mirrored.append(names.index(name.replace("Right", "Left")))
        elif "Left" in name and name.replace("Left", "Right") in names:
            mirrored.append(names.index(name.replace("Left", "Right")))
        else:
            mirrored.append(idx)
    return np.array(mirrored)


def _should_drop_joint(name, prune_ends_and_fingers):
    if not prune_ends_and_fingers:
        return False
    if name.endswith("End"):
        return True
    return "Hand" in name and any(token in name for token in FINGER_TOKENS)


def _prune_skeleton(names, parents, positions, rotations, prune_ends_and_fingers):
    if not prune_ends_and_fingers:
        return names, parents, positions, rotations

    keep_mask = np.array([not _should_drop_joint(name, prune_ends_and_fingers) for name in names], dtype=bool)
    keep_indices = np.nonzero(keep_mask)[0]
    old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(keep_indices.tolist())}

    pruned_names = [names[idx] for idx in keep_indices]
    pruned_parents = []
    for old_idx in keep_indices:
        old_parent = int(parents[old_idx])
        if old_parent == -1:
            pruned_parents.append(-1)
        else:
            pruned_parents.append(old_to_new[old_parent])

    return (
        pruned_names,
        np.asarray(pruned_parents, dtype=np.int32),
        positions[:, keep_mask].copy(),
        rotations[:, keep_mask].copy(),
    )


def _compute_simulation_root(rotations, positions, names, parents):
    global_rotations, global_positions = quat.fk(rotations, positions, parents)

    sim_position_joint = names.index("Spine2")
    sim_rotation_joint = names.index("Hips")

    sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:, sim_position_joint : sim_position_joint + 1]
    sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode="interp")

    sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(
        global_rotations[:, sim_rotation_joint : sim_rotation_joint + 1], np.array([0.0, 0.0, 1.0])
    )
    sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[..., np.newaxis]
    sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode="interp")
    sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[..., np.newaxis]
    sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

    positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
    rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])

    positions = np.concatenate([sim_position, positions], axis=1)
    rotations = np.concatenate([sim_rotation, rotations], axis=1)
    bone_parents = np.concatenate([[-1], parents + 1])
    bone_names = ["Simulation"] + names
    return rotations, positions, bone_parents, bone_names


def _compute_velocities(rotations, positions, bone_parents):
    velocities = np.empty_like(positions)
    velocities[1:-1] = (
        0.5 * (positions[2:] - positions[1:-1]) * 60.0 + 0.5 * (positions[1:-1] - positions[:-2]) * 60.0
    )
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0
        + 0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0
    )
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    _, _, global_velocities, _ = quat.fk_vel(rotations, positions, velocities, angular_velocities, bone_parents)
    return velocities, angular_velocities, global_velocities


def _compute_contacts(global_velocities, bone_names):
    contact_velocity_threshold = 0.15
    contact_velocity = np.sqrt(
        np.sum(
            global_velocities[:, np.array([bone_names.index("LeftToeBase"), bone_names.index("RightToeBase")])] ** 2,
            axis=-1,
        )
    )
    contacts = contact_velocity < contact_velocity_threshold
    for ci in range(contacts.shape[1]):
        contacts[:, ci] = ndimage.median_filter(contacts[:, ci], size=6, mode="nearest")
    return contacts


def _process_motion_data(bvh_data, mirror, prune_ends_and_fingers=False):
    positions = bvh_data["positions"].astype(np.float32) * 0.01
    rotations = quat.unroll(quat.from_euler(np.radians(bvh_data["rotations"]), order=bvh_data["order"])).astype(np.float32)
    names, parents, positions, rotations = _prune_skeleton(
        bvh_data["names"],
        bvh_data["parents"],
        positions,
        rotations,
        prune_ends_and_fingers=prune_ends_and_fingers,
    )

    if mirror:
        mirror_bones = _mirror_bones(names)
        global_rotations, global_positions = quat.fk(rotations, positions, parents)
        global_positions = np.array([-1, 1, 1]) * global_positions[:, mirror_bones]
        global_rotations = np.array([1, 1, -1, -1]) * global_rotations[:, mirror_bones]
        rotations, positions = quat.ik(global_rotations, global_positions, parents)

    rotations, positions, bone_parents, bone_names = _compute_simulation_root(
        rotations, positions, names, parents
    )
    velocities, angular_velocities, global_velocities = _compute_velocities(rotations, positions, bone_parents)
    contacts = _compute_contacts(global_velocities, bone_names)

    return {
        "positions": positions.astype(np.float32),
        "velocities": velocities.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "angular_velocities": angular_velocities.astype(np.float32),
        "contacts": contacts.astype(np.uint8),
        "parents": bone_parents.astype(np.int32),
        "names": bone_names,
    }


_process_motion_pair = process_motion_pair


def iter_motion_pairs(bvh_paths, prune_ends_and_fingers, workers, desc="Processing motions"):
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    tasks = [(path, prune_ends_and_fingers) for path in bvh_paths]
    if workers == 1:
        for task in tqdm(tasks, desc=desc):
            yield _process_motion_pair(task)
        return

    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        task_iter = iter(tasks)
        pending = deque(executor.submit(_process_motion_pair, task) for task in islice(task_iter, workers))
        with tqdm(total=len(tasks), desc=desc) as progress:
            while pending:
                yield pending.popleft().result()
                progress.update()
                task = next(task_iter, None)
                if task is not None:
                    pending.append(executor.submit(_process_motion_pair, task))


class MotionDatabaseWriter:
    def __init__(self, output_path, total_frames, tags_data, prune_ends_and_fingers):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_frames = int(total_frames)
        self.tags_by_range = {}
        for range_name, tag, start, stop in tags_data:
            self.tags_by_range.setdefault(range_name, []).append((tag, start, stop))

        self.prune_ends_and_fingers = bool(prune_ends_and_fingers)
        self._temp_dir = TemporaryDirectory(prefix=".database-", dir=self.output_path.parent)
        self._arrays = None
        self.offset = 0
        self.range_starts = []
        self.range_stops = []
        self.range_names = []
        self.range_mirror = []
        self.tag_range_starts = []
        self.tag_range_stops = []
        self.tag_range_names = []
        self.tag_tags = []
        self.tag_mirror = []
        self.bone_parents = None
        self.bone_names = None

    def _allocate(self, motion):
        num_joints = motion["positions"].shape[1]
        root = Path(self._temp_dir.name)
        self._arrays = {
            "positions": np.lib.format.open_memmap(
                root / "positions.npy", mode="w+", dtype=np.float32, shape=(self.total_frames, num_joints, 3)
            ),
            "velocities": np.lib.format.open_memmap(
                root / "velocities.npy", mode="w+", dtype=np.float32, shape=(self.total_frames, num_joints, 3)
            ),
            "rotations": np.lib.format.open_memmap(
                root / "rotations.npy", mode="w+", dtype=np.float32, shape=(self.total_frames, num_joints, 4)
            ),
            "angular_velocities": np.lib.format.open_memmap(
                root / "angular_velocities.npy",
                mode="w+",
                dtype=np.float32,
                shape=(self.total_frames, num_joints, 3),
            ),
            "contacts": np.lib.format.open_memmap(
                root / "contacts.npy", mode="w+", dtype=np.uint8, shape=(self.total_frames, 2)
            ),
        }

    def add(self, range_name, mirror, motion):
        if self.bone_parents is None:
            self.bone_parents = motion["parents"]
            self.bone_names = motion["names"]
            self._allocate(motion)
        elif not np.array_equal(self.bone_parents, motion["parents"]) or self.bone_names != motion["names"]:
            raise ValueError(f"Skeleton mismatch while processing {range_name} (mirror={mirror})")

        nframes = len(motion["positions"])
        stop = self.offset + nframes
        if stop > self.total_frames:
            raise ValueError(f"Motion stream exceeds declared frame count {self.total_frames}")
        for key, array in self._arrays.items():
            array[self.offset:stop] = motion[key]

        self.range_starts.append(self.offset)
        self.range_stops.append(stop)
        self.range_names.append(range_name)
        self.range_mirror.append(mirror)

        for tag, tag_start, tag_stop in self.tags_by_range.get(range_name, []):
            tag_stop = nframes if tag_stop is None else min(tag_stop, nframes)
            self.tag_range_starts.append(self.offset + tag_start)
            self.tag_range_stops.append(self.offset + tag_stop)
            self.tag_range_names.append(range_name)
            self.tag_tags.append(tag)
            self.tag_mirror.append(mirror)
        self.offset = stop

    def save(self):
        if self._arrays is None or self.offset != self.total_frames:
            raise ValueError(f"Expected {self.total_frames} frames, wrote {self.offset}")
        for array in self._arrays.values():
            array.flush()
        np.savez(
            self.output_path,
            positions=self._arrays["positions"],
            velocities=self._arrays["velocities"],
            rotations=self._arrays["rotations"],
            angular_velocities=self._arrays["angular_velocities"],
            parents=self.bone_parents.astype(np.int32),
            names=self.bone_names,
            range_starts=np.asarray(self.range_starts, dtype=np.int32),
            range_stops=np.asarray(self.range_stops, dtype=np.int32),
            range_mirror=np.asarray(self.range_mirror, dtype=bool),
            range_names=np.asarray(self.range_names, dtype=object),
            contacts=self._arrays["contacts"],
            tag_range_starts=np.asarray(self.tag_range_starts, dtype=np.int32),
            tag_range_stops=np.asarray(self.tag_range_stops, dtype=np.int32),
            tag_range_names=np.asarray(self.tag_range_names, dtype=object),
            tag_tags=np.asarray(self.tag_tags, dtype=object),
            tag_mirror=np.asarray(self.tag_mirror, dtype=bool),
            joint_subset=np.asarray(
                "prune_ends_and_fingers" if self.prune_ends_and_fingers else "full",
                dtype=object,
            ),
        )
        self._arrays = None
        self._temp_dir.cleanup()


def build_lafan_tags():
    tags = []
    for path in sorted(LAFAN_SOURCE.glob("*.bvh")):
        tags.append((path.stem, "all", 0, None))
    return tags


def build_100style_tags(style_filter=None, max_styles=None):
    frame_cuts = STYLE100_SOURCE / "Frame_Cuts.csv"
    tags = []
    seen_styles = []
    with frame_cuts.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            style_name = row["STYLE_NAME"].strip()
            if style_filter is not None and style_name not in style_filter:
                continue
            if max_styles is not None and style_name not in seen_styles:
                if len(seen_styles) >= max_styles:
                    continue
                seen_styles.append(style_name)
            for clip in STYLE100_CLIPS:
                start_key = f"{clip}_START"
                stop_key = f"{clip}_STOP"
                start = row.get(start_key, "N/A")
                stop = row.get(stop_key, "N/A")
                if start == "N/A" or stop == "N/A":
                    continue
                range_name = f"{style_name}_{clip}"
                tags.append((range_name, "all", int(start), int(stop)))
                tags.append((range_name, style_name, int(start), int(stop)))
                tags.append((range_name, clip, int(start), int(stop)))
    return tags


def source_path_for(dataset_name, range_name):
    if dataset_name == "lafan":
        return LAFAN_SOURCE / f"{range_name}.bvh"
    if dataset_name == "100style":
        style_name, _sep, clip = range_name.rpartition("_")
        return STYLE100_SOURCE / style_name / f"{range_name}.bvh"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class _FeatureStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.x_sum: np.ndarray | None = None
        self.x_sumsq: np.ndarray | None = None
        self.ref_pos_sum: np.ndarray | None = None

    def update(self, components: Any, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(components.x),):
            raise ValueError("Feature statistics mask does not match shard frame count")
        if not np.any(mask):
            return
        values = np.asarray(components.x[mask], dtype=np.float64)
        value_sum = values.sum(axis=0, dtype=np.float64)
        value_squared = np.square(values).sum(axis=0, dtype=np.float64)
        self.x_sum = value_sum if self.x_sum is None else self.x_sum + value_sum
        self.x_sumsq = value_squared if self.x_sumsq is None else self.x_sumsq + value_squared
        positions = np.asarray(components.positions[mask], dtype=np.float64).sum(axis=0)
        self.ref_pos_sum = positions if self.ref_pos_sum is None else self.ref_pos_sum + positions
        self.count += int(mask.sum())

    def finalize(self, names: list[str]) -> MotionFeatureStats:
        if self.count <= 0 or self.x_sum is None or self.x_sumsq is None or self.ref_pos_sum is None:
            raise ValueError("No training frames available for feature statistics")
        mean = self.x_sum / self.count
        variance = np.maximum(self.x_sumsq / self.count - np.square(mean), 1e-8)
        std = np.sqrt(variance).astype(np.float32)
        nbones = len(names)
        rotation_stop = 9 + (nbones - 1) * 6
        hip_velocity_stop = rotation_stop + 3
        angular_stop = hip_velocity_stop + (nbones - 1) * 3
        scale = np.concatenate((
            np.full(3, std[0:3].mean(), dtype=np.float32),
            np.full(3, std[3:6].mean(), dtype=np.float32),
            np.full(3, std[6:9].mean(), dtype=np.float32),
            np.full(rotation_stop - 9, std[9:rotation_stop].mean(), dtype=np.float32),
            np.full(3, std[rotation_stop:hip_velocity_stop].mean(), dtype=np.float32),
            np.full(angular_stop - hip_velocity_stop, std[hip_velocity_stop:angular_stop].mean(), dtype=np.float32),
            np.full(2, std[angular_stop:].mean(), dtype=np.float32),
        ))
        scale = np.maximum(scale, 1e-8).astype(np.float32)
        from stylized_motion.anim.features import default_joint_weights

        joint_weights = default_joint_weights(names)
        weights = np.concatenate((
            np.ones(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            joint_weights[1:].repeat(6).astype(np.float32) * (nbones - 1),
            np.ones(3, dtype=np.float32),
            joint_weights[1:].repeat(3).astype(np.float32) * (nbones - 1),
            np.ones(2, dtype=np.float32),
        ))
        return MotionFeatureStats(
            offset=mean.astype(np.float32),
            scale=scale,
            dist=(std / scale).astype(np.float32),
            weights=weights.astype(np.float32),
            ref_pos=(self.ref_pos_sum / self.count).astype(np.float32),
        )


def _parse_style_filter(styles_arg: str | None) -> set[str] | None:
    if not styles_arg:
        return None
    return {item.strip() for item in styles_arg.split(",") if item.strip()}


def _build_shard_specs(dataset_name: str, styles_arg: str | None, max_styles: int | None) -> tuple[list[dict[str, object]], list[tuple[Any, ...]], list[Path]]:
    tags_data = build_lafan_tags() if dataset_name == "lafan" else build_100style_tags(
        style_filter=_parse_style_filter(styles_arg), max_styles=max_styles
    )
    bvh_paths = list(dict.fromkeys(
        source_path_for(dataset_name, str(range_name))
        for range_name, tag, _start, _stop in tags_data
        if tag == "all"
    ))
    if not bvh_paths:
        raise FileNotFoundError(f"No BVH files found for dataset {dataset_name}")
    specs = [
        {"range_name": path.stem, "mirror": bool(mirror), "nframes": int(bvh.read_frame_count(path))}
        for path in bvh_paths
        for mirror in (False, True)
    ]
    return specs, tags_data, bvh_paths


def _clip_labels(name: str) -> dict[str, str]:
    style, separator, action = str(name).rpartition("_")
    return {"style": style if separator else "", "action": action if separator else str(name)}


def _feature_split_manifest(specs: list[dict[str, object]], seed: int) -> tuple[SplitManifest, dict[str, int]]:
    source_names = sorted({str(spec["range_name"]) for spec in specs})
    split = build_split_manifest(
        source_names,
        seed=seed,
        stratify_keys=("style", "action"),
        labels={name: _clip_labels(name) for name in source_names},
    )
    source_ids = {name: index for index, name in enumerate(split.source_clip_names)}
    return split, source_ids


def _normalize_motion_shard(path: Path, stats: MotionFeatureStats, chunk_size: int = 16384) -> None:
    motion = np.load(path, mmap_mode="r+", allow_pickle=False)
    if motion.dtype != np.float32 or motion.ndim != 2 or motion.shape[1] != len(stats.offset):
        raise ValueError(f"Unexpected feature shard shape at {path}: {motion.shape}")
    for start in range(0, len(motion), int(chunk_size)):
        stop = min(start + int(chunk_size), len(motion))
        motion[start:stop] = (motion[start:stop] - stats.offset) / stats.scale
    motion.flush()


def _save_motion_shard(output_dir: Path, shard_idx: int, motion: np.ndarray) -> str:
    relative = Path("motion") / f"shard_{int(shard_idx):05d}.npy"
    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(motion, dtype=np.float32))
    return relative.as_posix()


def _publish_store(staging: Path, output: Path, overwrite: bool) -> None:
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Store already exists: {output}; pass overwrite=True to replace it")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    os.replace(staging, output)


def build_motion_database(
    dataset_name: str,
    output: str | Path,
    *,
    styles: str | None = None,
    max_styles: int | None = None,
    prune_ends_and_fingers: bool = False,
    workers: int = 1,
) -> Path:
    """Build the raw animation database used by trajectory preprocessing."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    specs, tags_data, bvh_paths = _build_shard_specs(dataset_name, styles, max_styles)
    writer = MotionDatabaseWriter(
        output,
        total_frames=sum(int(spec["nframes"]) for spec in specs),
        tags_data=tags_data,
        prune_ends_and_fingers=prune_ends_and_fingers,
    )
    shard_idx = 0
    for range_name, motions in iter_motion_pairs(bvh_paths, prune_ends_and_fingers, workers, desc="Building motion database"):
        for mirror, motion in motions:
            spec = specs[shard_idx]
            if (str(range_name), bool(mirror)) != (str(spec["range_name"]), bool(spec["mirror"])):
                raise ValueError("Processed motion order does not match shard specification")
            if len(motion["positions"]) != int(spec["nframes"]):
                raise ValueError("Processed motion frame count differs from BVH frame count")
            writer.add(str(range_name), bool(mirror), motion)
            shard_idx += 1
    if shard_idx != len(specs):
        raise ValueError(f"Expected {len(specs)} motion shards, wrote {shard_idx}")
    writer.save()
    return output


def build_feature_database(
    dataset_name: str,
    output: str | Path,
    *,
    styles: str | None = None,
    max_styles: int | None = None,
    prune_ends_and_fingers: bool = False,
    seed: int = 3407,
    workers: int = 1,
    overwrite: bool = False,
) -> Path:
    """Build and atomically publish a canonical FeatureStore."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"FeatureStore already exists: {output}")
    specs, tags_data, bvh_paths = _build_shard_specs(dataset_name, styles, max_styles)
    split_manifest, source_ids = _feature_split_manifest(specs, seed)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    raw_database = output.parent / "database.npz"
    writer = MotionDatabaseWriter(
        raw_database,
        total_frames=sum(int(spec["nframes"]) for spec in specs),
        tags_data=tags_data,
        prune_ends_and_fingers=prune_ends_and_fingers,
    )
    train_masks = [
        np.full(int(spec["nframes"]), split_manifest.split_by_source_clip[str(spec["range_name"])] == "train", dtype=bool)
        for spec in specs
    ]
    stats_accumulator = _FeatureStatsAccumulator()
    names: list[str] | None = None
    parents: np.ndarray | None = None
    motion_files: list[str] = []
    try:
        for shard_idx, (range_name, motions) in enumerate(iter_motion_pairs(
            bvh_paths, prune_ends_and_fingers, workers, desc="Building feature database"
        )):
            for mirror, motion in motions:
                spec_idx = shard_idx * 2 + int(bool(mirror))
                spec = specs[spec_idx]
                if str(range_name) != str(spec["range_name"]) or bool(mirror) != bool(spec["mirror"]):
                    raise ValueError("Processed motion order does not match feature shard specification")
                if names is None:
                    names = list(motion["names"])
                    parents = np.asarray(motion["parents"], dtype=np.int32)
                elif names != list(motion["names"]) or not np.array_equal(parents, motion["parents"]):
                    raise ValueError("Motion shards do not share one skeleton schema")
                components = build_motion_feature_components(motion)
                stats_accumulator.update(components, train_masks[spec_idx])
                motion_files.append(_save_motion_shard(staging, spec_idx, components.x))
                writer.add(str(range_name), bool(mirror), motion)
        if names is None or parents is None:
            raise ValueError("No motion shards were processed")
        stats = stats_accumulator.finalize(names)
        if stats.offset.shape != (230,):
            raise ValueError(f"Canonical FeatureStore requires motion_dim=230, got {stats.offset.shape}")
        for relative in motion_files:
            _normalize_motion_shard(staging / relative, stats)
        writer.save()
        range_names = [str(spec["range_name"]) for spec in specs]
        source_clip_names = list(split_manifest.source_clip_names)
        style_names = sorted({_clip_labels(name)["style"] for name in range_names})
        action_names = sorted({_clip_labels(name)["action"] for name in range_names})
        style_to_id = {name: idx for idx, name in enumerate(style_names)}
        action_to_id = {name: idx for idx, name in enumerate(action_names)}
        style_ids = np.asarray([style_to_id[_clip_labels(name)["style"]] for name in range_names], dtype=np.int32)
        action_ids = np.asarray([action_to_id[_clip_labels(name)["action"]] for name in range_names], dtype=np.int32)
        split_ids = np.asarray(
            [
                {"train": 0, "val": 1, "test": 2}[split_manifest.split_by_source_clip[str(spec["range_name"])] ]
                for spec in specs
            ],
            dtype=np.uint8,
        )
        source_clip_ids = np.asarray([source_ids[str(spec["range_name"])] for spec in specs], dtype=np.int32)
        clip_ids = np.arange(len(specs), dtype=np.int32)
        shard_frames = np.asarray([int(spec["nframes"]) for spec in specs], dtype=np.int64)
        range_starts = np.zeros(len(specs), dtype=np.int64)
        range_stops = shard_frames.copy()
        stats_payload = serialize_motion_feature_stats(stats, names=names, parents=parents, joint_subset=("prune_ends_and_fingers" if prune_ends_and_fingers else "full"))
        np.savez(
            staging / "index.npz",
            shard_num_frames=shard_frames,
            clip_ids=clip_ids,
            source_clip_ids=source_clip_ids,
            range_shard_indices=np.arange(len(specs), dtype=np.int32),
            range_starts=range_starts,
            range_stops=range_stops,
            range_mirror=np.asarray([bool(spec["mirror"]) for spec in specs], dtype=bool),
            split_ids=split_ids,
            style_ids=style_ids,
            action_ids=action_ids,
            **{key: np.asarray(value) for key, value in stats_payload.items() if key in {"offset", "scale", "dist", "weights", "ref_pos"}},
        )
        names_sha256 = hashlib.sha256(canonical_json_bytes(names)).hexdigest()
        stats_sha256 = hashlib.sha256()
        for key in ("offset", "scale", "weights", "ref_pos"):
            stats_sha256.update(key.encode("ascii"))
            stats_sha256.update(np.asarray(stats_payload[key], dtype=np.float32).tobytes())
        stats_hash = stats_sha256.hexdigest()
        schema_payload = {
            "name": "motion_feature_v2",
            "motion_dim": 230,
            "joint_subset": "prune_ends_and_fingers" if prune_ends_and_fingers else "full",
            "names_sha256": names_sha256,
            "stats_sha256": stats_hash,
        }
        feature_schema_hash = hashlib.sha256(canonical_json_bytes(schema_payload)).hexdigest()
        manifest = {
            "data_schema_version": 3,
            "store_type": "feature",
            "frame_rate": 60,
            "num_shards": len(motion_files),
            "shard_files": motion_files,
            "shard_sha256": [sha256_file(staging / relative) for relative in motion_files],
            "split_manifest_hash": split_manifest.split_manifest_hash,
            "feature_schema_hash": feature_schema_hash,
            "created_by": "stylized_motion.data.preprocess",
            "motion_dim": 230,
            "range_names": range_names,
            "source_clip_names": source_clip_names,
            "style_names": style_names,
            "action_names": action_names,
            "split_manifest": {**split_manifest.as_dict(), "split_manifest_hash": split_manifest.split_manifest_hash},
            "feature_schema": {
                **schema_payload,
                "names": names,
                "parents": parents.tolist(),
            },
            "normalization_train_frames": int(stats_accumulator.count),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        validate_data(feature_database=staging, full=True)
        _publish_store(staging, output, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def _encode_feature_shard(
    encoder: Any,
    motion: np.ndarray,
    *,
    chunk_size: int,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if motion.ndim != 2 or motion.dtype != np.float32:
        raise ValueError("Feature shards must be float32 [N,D]")
    num_frames = int(motion.shape[0])
    num_coordinates = int(encoder.num_coordinates)
    indices_out = np.empty((num_frames, num_coordinates), dtype=np.uint8)
    codes_out = np.empty((num_frames, num_coordinates), dtype=np.float16)
    receptive_field = int(getattr(encoder, "receptive_field"))
    lookahead_frames = int(getattr(encoder, "lookahead_frames"))
    if receptive_field <= 0 or lookahead_frames < 0 or lookahead_frames >= receptive_field:
        raise ValueError("Token encoder has invalid receptive_field/lookahead_frames metadata")
    history_frames = receptive_field - 1 - lookahead_frames
    with torch.inference_mode():
        for start in range(0, num_frames, int(chunk_size)):
            stop = min(num_frames, start + int(chunk_size))
            read_start = max(0, start - history_frames)
            values = torch.from_numpy(np.asarray(motion[read_start:stop], dtype=np.float32).copy()).unsqueeze(0).to(device)
            values = input_adapter(values) if input_adapter is not None else values
            codes, indices = encoder.encode_to_codes(values)
            offset = start - read_start
            length = stop - start
            indices_out[start:stop] = indices[0, offset : offset + length].detach().cpu().numpy().astype(np.uint8)
            codes_out[start:stop] = codes[0, offset : offset + length].detach().cpu().numpy().astype(np.float16)
    return indices_out, codes_out


def build_token_database(
    feature_database: str | Path,
    output: str | Path,
    *,
    encoder: Any,
    checkpoint_sha256: str,
    model_family_legacy: str | None = None,
    device: torch.device | str = "cpu",
    chunk_size: int = 1024,
    save_codes: bool = False,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    overwrite: bool = False,
) -> Path:
    """Encode feature shards with an injected TokenEncoderProtocol implementation."""
    if not checkpoint_sha256:
        raise ValueError("checkpoint_sha256 is required for a TokenStore")
    device = torch.device(device)
    feature_store = open_feature_store(feature_database)
    metadata = encoder.representation_metadata()
    if not isinstance(metadata, Mapping):
        raise ValueError("Token encoder representation_metadata() must return a mapping")
    if (
        int(encoder.num_coordinates) != 40
        or int(encoder.num_levels) != 9
        or int(encoder.receptive_field) != 64
        or int(encoder.lookahead_frames) != 0
    ):
        raise ValueError("Token encoder does not satisfy the canonical 40x9 causal contract")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"TokenStore already exists: {output}")
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        index_files: list[str] = []
        code_files: list[str] = []
        for shard_idx, motion_path in enumerate(feature_store.motion_files):
            motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
            indices, codes = _encode_feature_shard(
                encoder,
                motion,
                chunk_size=chunk_size,
                device=device,
                input_adapter=input_adapter,
            )
            index_relative = Path("indices") / f"shard_{shard_idx:05d}.npy"
            index_path = staging / index_relative
            index_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(index_path, indices)
            index_files.append(index_relative.as_posix())
            if save_codes:
                code_relative = Path("codes") / f"shard_{shard_idx:05d}.npy"
                code_path = staging / code_relative
                code_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(code_path, codes)
                code_files.append(code_relative.as_posix())
        np.savez(
            staging / "index.npz",
            shard_num_frames=feature_store.shard_num_frames.astype(np.int64),
            clip_ids=feature_store.clip_ids.astype(np.int32),
            source_clip_ids=feature_store.source_clip_ids.astype(np.int32),
            range_shard_indices=feature_store.range_shard_indices.astype(np.int32),
            range_starts=feature_store.range_starts.astype(np.int64),
            range_stops=feature_store.range_stops.astype(np.int64),
            range_mirror=feature_store.range_mirror.astype(bool),
            split_ids=feature_store.split_ids.astype(np.uint8),
            style_ids=feature_store.style_ids.astype(np.int32),
            action_ids=feature_store.action_ids.astype(np.int32),
        )
        representation = dict(metadata)
        legacy_family = model_family_legacy or {
            "flat_fsq": "fsq",
            "part_fsq": "part_fsq",
            "residual_part_fsq": "residual_part_fsq",
            "latent_residual_fsq": "latent_residual_part_fsq",
        }.get(str(representation.get("family", "")), "")
        manifest = {
            "data_schema_version": 3,
            "store_type": "token",
            "frame_rate": 60,
            "num_shards": len(index_files),
            "shard_files": index_files,
            "shard_sha256": [sha256_file(staging / relative) for relative in index_files],
            "split_manifest_hash": feature_store.split_manifest_hash,
            "feature_schema_hash": feature_store.feature_schema_hash,
            "created_by": "stylized_motion.data.preprocess",
            "feature_schema": feature_store.feature_schema(),
            "range_names": list(feature_store.range_names),
            "source_clip_names": list(feature_store.source_clip_names),
            "style_names": list(feature_store.style_names),
            "action_names": list(feature_store.action_names),
            "representation": representation,
            "representation_family": str(representation.get("family", "")),
            "representation_variant": str(representation.get("variant", "")),
            "representation_id": str(representation.get("representation_id", "")),
            "model_family_legacy": str(legacy_family),
            "checkpoint_sha256": str(checkpoint_sha256),
            "motion_dim": 230,
            "num_coordinates": 40,
            "num_levels": 9,
            "coordinate_order": list(representation.get("coordinate_order", [])),
            "coordinate_counts": dict(representation.get("coordinate_counts", {})),
            "temporal_downsample": int(representation.get("temporal_downsample", 1)),
            "receptive_field": int(representation.get("receptive_field", 64)),
            "lookahead_frames": int(representation.get("lookahead_frames", 0)),
            "decoder_passes_inference": int(representation.get("decoder_passes_inference", 1)),
        }
        if save_codes:
            manifest["code_shard_files"] = code_files
            manifest["code_shard_sha256"] = [sha256_file(staging / relative) for relative in code_files]
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        validate_data(feature_database=feature_database, token_database=staging, full=True)
        _publish_store(staging, output, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        feature_store.close()
    return output


def _trajectory_input_ranges(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("database_range_names", "database_range_mirror", "database_range_starts", "database_range_stops")
    if not all(key in data for key in required):
        raise ValueError("Trajectory inputs lack database range metadata")
    return (
        np.asarray(data[required[0]], dtype=object),
        np.asarray(data[required[1]], dtype=bool),
        np.asarray(data[required[2]], dtype=np.int64),
        np.asarray(data[required[3]], dtype=np.int64),
    )


def build_trajectory_inputs(
    database: str | Path,
    output: str | Path,
    *,
    future_frames: Iterable[int] = (20, 40, 60),
) -> Path:
    """Materialize raw root-local future controls with range-local offsets."""
    database = Path(database)
    output = Path(output)
    with np.load(database, allow_pickle=True) as npz:
        data = {key: np.asarray(npz[key]) for key in npz.files}
    range_names = np.asarray(data["range_names"], dtype=object)
    range_mirror = np.asarray(data["range_mirror"], dtype=bool)
    range_starts = np.asarray(data["range_starts"], dtype=np.int64)
    range_stops = np.asarray(data["range_stops"], dtype=np.int64)
    frames = np.asarray(tuple(int(value) for value in future_frames), dtype=np.int32)
    if frames.ndim != 1 or len(frames) == 0 or np.any(frames <= 0):
        raise ValueError("future_frames must be a non-empty sequence of positive offsets")
    root_positions = np.asarray(data["positions"], dtype=np.float32)[:, 0]
    root_rotations = np.asarray(data["rotations"], dtype=np.float32)[:, 0]
    root_directions = quat.mul_vec(root_rotations, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    indices_out: list[np.ndarray] = []
    values_out: list[np.ndarray] = []
    range_out: list[np.ndarray] = []
    mirror_out: list[np.ndarray] = []
    max_future = int(frames.max())
    for range_idx, (start, stop) in enumerate(zip(range_starts.tolist(), range_stops.tolist())):
        indices = np.arange(int(start) + 1, int(stop) - max_future, dtype=np.int64)
        if len(indices) == 0:
            continue
        future = indices[:, None] + frames[None, :]
        local_pos = quat.inv_mul_vec(
            root_rotations[indices][:, None], root_positions[future] - root_positions[indices][:, None]
        )
        local_dir = quat.inv_mul_vec(root_rotations[indices][:, None], root_directions[future])
        values = np.concatenate((local_pos.reshape(len(indices), -1), local_dir.reshape(len(indices), -1)), axis=-1)
        indices_out.append(indices)
        values_out.append(values.astype(np.float32))
        range_out.append(np.full(len(indices), range_idx, dtype=np.int32))
        mirror_out.append(np.full(len(indices), bool(range_mirror[range_idx]), dtype=bool))
    if not values_out:
        raise ValueError("No valid trajectory input frames were found")
    indices = np.concatenate(indices_out)
    values = np.concatenate(values_out)
    range_indices = np.concatenate(range_out)
    mirror = np.concatenate(mirror_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        indices=indices.astype(np.int64),
        T=values,
        future_frames=frames,
        sample_range_indices=range_indices,
        sample_local_indices=(indices - range_starts[range_indices]).astype(np.int64),
        sample_mirror=mirror,
        database_range_names=range_names,
        database_range_mirror=range_mirror,
        database_range_starts=range_starts,
        database_range_stops=range_stops,
        trajectory_feature_order=np.asarray("pos(future_frames),dir(future_frames)", dtype=object),
    )
    return output


def _match_trajectory_ranges(
    source_names: np.ndarray,
    source_mirror: np.ndarray,
    source_starts: np.ndarray,
    source_stops: np.ndarray,
    token_store: Any,
) -> np.ndarray:
    mapping = np.full(len(source_names), -1, dtype=np.int32)
    source_groups: dict[tuple[str, bool, int], list[int]] = {}
    target_groups: dict[tuple[str, bool, int], list[int]] = {}
    for index, values in enumerate(zip(source_names.tolist(), source_mirror.tolist(), source_starts.tolist(), source_stops.tolist())):
        name, mirror, start, stop = values
        source_groups.setdefault((str(name), bool(mirror), int(stop) - int(start)), []).append(index)
    for shard_idx, values in enumerate(zip(token_store.range_names, token_store.range_mirror, token_store.shard_num_frames.tolist())):
        name, mirror, length = values
        target_groups.setdefault((str(name), bool(mirror), int(length)), []).append(shard_idx)
    for key, target_indices in target_groups.items():
        source_indices = source_groups.get(key, [])
        if len(source_indices) != len(target_indices):
            raise ValueError(f"Trajectory source ranges do not match token ranges at {key!r}")
        for source_idx, shard_idx in zip(source_indices, target_indices):
            mapping[source_idx] = int(shard_idx)
    if np.any(mapping < 0):
        raise ValueError("Some trajectory source ranges could not be mapped to TokenStore shards")
    return mapping


def build_trajectory_database(
    token_database: str | Path,
    trajectory_input: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Align trajectory samples to the inherited TokenStore frame index."""
    token_store = open_token_store(token_database)
    trajectory_input = Path(trajectory_input)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"TrajectoryStore already exists: {output}")
    with np.load(trajectory_input, allow_pickle=True) as npz:
        data = {key: np.asarray(npz[key]) for key in npz.files}
    if "T" not in data:
        raise ValueError("Trajectory input must contain T")
    trajectory = np.asarray(data["T"], dtype=np.float32)
    if trajectory.ndim != 2 or len(trajectory) == 0 or not np.isfinite(trajectory).all():
        raise ValueError("Trajectory T must be finite float32 [N,C]")
    source_names, source_mirror, source_starts, source_stops = _trajectory_input_ranges(data)
    if "sample_range_indices" not in data or "sample_local_indices" not in data:
        raise ValueError("Trajectory input must contain range-local sample indices")
    source_range_indices = np.asarray(data["sample_range_indices"], dtype=np.int32)
    source_local_indices = np.asarray(data["sample_local_indices"], dtype=np.int64)
    if len(source_range_indices) != len(trajectory) or len(source_local_indices) != len(trajectory):
        raise ValueError("Trajectory samples and local index arrays have inconsistent lengths")
    if np.any(source_range_indices < 0) or np.any(source_range_indices >= len(source_names)):
        raise ValueError("Trajectory sample range index is invalid")
    source_to_shard = _match_trajectory_ranges(source_names, source_mirror, source_starts, source_stops, token_store)
    sample_shards = source_to_shard[source_range_indices]
    if np.any(source_local_indices < 0):
        raise ValueError("Trajectory local indices must be non-negative")
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    values_by_shard: list[np.ndarray] = []
    valid_by_shard: list[np.ndarray] = []
    trajectory_files: list[str] = []
    valid_files: list[str] = []
    try:
        for shard_idx, frames in enumerate(token_store.shard_num_frames.tolist()):
            values = np.zeros((int(frames), trajectory.shape[1]), dtype=np.float32)
            valid = np.zeros((int(frames),), dtype=bool)
            selected = np.flatnonzero(sample_shards == shard_idx)
            if len(selected):
                local = source_local_indices[selected]
                if np.any(local >= int(frames)):
                    raise ValueError(f"Trajectory local indices exceed token shard {shard_idx}")
                if np.any(valid[local]):
                    raise ValueError(f"Trajectory input contains duplicate samples in token shard {shard_idx}")
                values[local] = trajectory[selected]
                valid[local] = True
            values_by_shard.append(values)
            valid_by_shard.append(valid)
            value_relative = Path("trajectory") / f"shard_{shard_idx:05d}.npy"
            valid_relative = Path("valid") / f"shard_{shard_idx:05d}.npy"
            value_path = staging / value_relative
            valid_path = staging / valid_relative
            value_path.parent.mkdir(parents=True, exist_ok=True)
            valid_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(value_path, values)
            np.save(valid_path, valid)
            trajectory_files.append(value_relative.as_posix())
            valid_files.append(valid_relative.as_posix())
        total = np.zeros(trajectory.shape[1], dtype=np.float64)
        total_squared = np.zeros_like(total)
        count = 0
        for range_idx, split_id in enumerate(token_store.split_ids.tolist()):
            if int(split_id) != 0:
                continue
            shard_idx = int(token_store.range_shard_indices[range_idx])
            start = int(token_store.range_starts[range_idx])
            stop = int(token_store.range_stops[range_idx])
            selected = valid_by_shard[shard_idx][start:stop]
            values = values_by_shard[shard_idx][start:stop][selected].astype(np.float64)
            if len(values):
                total += values.sum(axis=0)
                total_squared += np.square(values).sum(axis=0)
                count += len(values)
        if count <= 0:
            raise ValueError("No valid trajectory frames are present in the train split")
        mean = total / count
        std = np.sqrt(np.maximum(total_squared / count - np.square(mean), 1e-12)).astype(np.float32)
        future_frames = np.asarray(data.get("future_frames", []), dtype=np.int32)
        feature_order = str(np.asarray(data.get("trajectory_feature_order", ""), dtype=object).item())
        np.savez(
            staging / "index.npz",
            shard_num_frames=token_store.shard_num_frames.astype(np.int64),
            clip_ids=token_store.clip_ids.astype(np.int32),
            source_clip_ids=token_store.source_clip_ids.astype(np.int32),
            range_shard_indices=token_store.range_shard_indices.astype(np.int32),
            range_starts=token_store.range_starts.astype(np.int64),
            range_stops=token_store.range_stops.astype(np.int64),
            range_mirror=token_store.range_mirror.astype(bool),
            split_ids=token_store.split_ids.astype(np.uint8),
            style_ids=token_store.style_ids.astype(np.int32),
            action_ids=token_store.action_ids.astype(np.int32),
            normalization_mean=mean.astype(np.float32),
            normalization_std=std,
        )
        manifest = {
            "data_schema_version": 3,
            "store_type": "trajectory",
            "frame_rate": 60,
            "num_shards": len(trajectory_files),
            "shard_files": trajectory_files,
            "shard_sha256": [sha256_file(staging / relative) for relative in trajectory_files],
            "valid_shard_files": valid_files,
            "valid_shard_sha256": [sha256_file(staging / relative) for relative in valid_files],
            "split_manifest_hash": token_store.split_manifest_hash,
            "feature_schema_hash": token_store.feature_schema_hash,
            "created_by": "stylized_motion.data.preprocess",
            "range_names": list(token_store.range_names),
            "source_clip_names": list(token_store.source_clip_names),
            "style_names": list(token_store.style_names),
            "action_names": list(token_store.action_names),
            "trajectory_dim": int(trajectory.shape[1]),
            "future_frames": future_frames.tolist(),
            "feature_order": feature_order,
            "checkpoint_sha256": token_store.checkpoint_sha256,
            "normalization_valid_frames": int(count),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        validate_data(token_database=token_database, trajectory_database=staging, full=True)
        _publish_store(staging, output, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        token_store.close()
    return output


def _assert_source_split_disjoint(store: Any) -> None:
    source_ids = np.asarray(store.source_clip_ids, dtype=np.int32)
    split_ids = np.asarray(store.split_ids, dtype=np.uint8)
    for source_id in sorted(set(source_ids.tolist())):
        values = set(split_ids[source_ids == source_id].tolist())
        if len(values) != 1:
            raise ValueError(f"Source clip {source_id} appears in multiple splits")


def validate_data(
    *,
    feature_database: str | Path | None = None,
    token_database: str | Path | None = None,
    trajectory_database: str | Path | None = None,
    full: bool = False,
) -> dict[str, object]:
    """Run runtime validation, and optional value/checksum validation."""
    feature_store = open_feature_store(feature_database) if feature_database is not None else None
    token_store = open_token_store(token_database) if token_database is not None else None
    trajectory_store = open_trajectory_store(trajectory_database, token_store=token_store) if trajectory_database is not None else None
    try:
        stores = [store for store in (feature_store, token_store, trajectory_store) if store is not None]
        for store in stores:
            _assert_source_split_disjoint(store)
        if feature_store is not None and token_store is not None:
            if feature_store.feature_schema_hash != token_store.feature_schema_hash or feature_store.split_manifest_hash != token_store.split_manifest_hash:
                raise ValueError("FeatureStore and TokenStore contract hashes differ")
            for key in ("shard_num_frames", "clip_ids", "source_clip_ids", "range_shard_indices", "range_starts", "range_stops", "range_mirror", "split_ids"):
                if not np.array_equal(getattr(feature_store, key), getattr(token_store, key)):
                    raise ValueError(f"FeatureStore and TokenStore are not aligned at {key}")
        if full:
            for store in stores:
                for relative, expected in zip(store.manifest["shard_files"], store.manifest["shard_sha256"]):
                    path = store.database / str(relative)
                    if sha256_file(path) != str(expected):
                        raise ValueError(f"Shard checksum mismatch: {path}")
                if store.manifest.get("store_type") == "token" and store.manifest.get("code_shard_files") is not None:
                    for relative, expected in zip(store.manifest["code_shard_files"], store.manifest["code_shard_sha256"]):
                        path = store.database / str(relative)
                        if sha256_file(path) != str(expected):
                            raise ValueError(f"Code shard checksum mismatch: {path}")
                if store.manifest.get("store_type") == "trajectory":
                    for relative, expected in zip(store.manifest["valid_shard_files"], store.manifest["valid_shard_sha256"]):
                        path = store.database / str(relative)
                        if sha256_file(path) != str(expected):
                            raise ValueError(f"Trajectory valid shard checksum mismatch: {path}")
            if feature_store is not None:
                if "normalization_train_frames" in feature_store.manifest and int(feature_store.manifest["normalization_train_frames"]) <= 0:
                    raise ValueError("FeatureStore has no recorded training statistics coverage")
                if (
                    not np.isfinite(feature_store.stats.offset).all()
                    or not np.isfinite(feature_store.stats.scale).all()
                    or not np.isfinite(feature_store.stats.dist).all()
                    or not np.isfinite(feature_store.stats.weights).all()
                    or not np.isfinite(feature_store.stats.ref_pos).all()
                    or np.any(feature_store.stats.scale <= 0)
                ):
                    raise ValueError("Feature normalization statistics are invalid")
                for path in feature_store.motion_files:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Feature shard contains non-finite values: {path}")
            if token_store is not None:
                for path in token_store.token_files:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                    if np.any(values >= token_store.num_levels):
                        raise ValueError(f"Token shard contains an out-of-range index: {path}")
            if trajectory_store is not None:
                for path, valid_path in zip(trajectory_store.trajectory_files, trajectory_store.valid_files):
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
                    if not np.isfinite(values[valid]).all():
                        raise ValueError(f"Trajectory shard contains non-finite valid values: {path}")
        return {
            "feature": feature_store is not None,
            "token": token_store is not None,
            "trajectory": trajectory_store is not None,
            "full": bool(full),
        }
    finally:
        for store in (feature_store, token_store, trajectory_store):
            if store is not None:
                store.close()


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate schema-v3 motion data stores.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    motion = subparsers.add_parser("motion-database")
    motion.add_argument("--dataset", choices=["lafan", "100style"], required=True)
    motion.add_argument("--output", type=Path, required=True)
    motion.add_argument("--styles", default=None)
    motion.add_argument("--max-styles", type=int, default=None)
    motion.add_argument("--prune-ends-and-fingers", action="store_true")
    motion.add_argument("--workers", type=int, default=1)
    feature = subparsers.add_parser("feature-database")
    feature.add_argument("--dataset", choices=["lafan", "100style"], required=True)
    feature.add_argument("--output", type=Path, required=True)
    feature.add_argument("--styles", default=None)
    feature.add_argument("--max-styles", type=int, default=None)
    feature.add_argument("--prune-ends-and-fingers", action="store_true")
    feature.add_argument("--seed", type=int, default=3407)
    feature.add_argument("--workers", type=int, default=1)
    feature.add_argument("--overwrite", action="store_true")
    token = subparsers.add_parser("token-database")
    token.add_argument("--feature-database", type=Path, required=True)
    token.add_argument("--output", type=Path, required=True)
    token.add_argument("--checkpoint", type=Path, required=True)
    token.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    token.add_argument("--chunk-size", type=int, default=1024)
    token.add_argument("--save-codes", action="store_true")
    token.add_argument("--overwrite", action="store_true")
    inputs = subparsers.add_parser("trajectory-inputs")
    inputs.add_argument("--database", type=Path, required=True)
    inputs.add_argument("--output", type=Path, required=True)
    inputs.add_argument("--future-frames", default="20,40,60")
    trajectory = subparsers.add_parser("trajectory-database")
    trajectory.add_argument("--token-database", type=Path, required=True)
    trajectory.add_argument("--trajectory-input", type=Path, required=True)
    trajectory.add_argument("--output", type=Path, required=True)
    trajectory.add_argument("--overwrite", action="store_true")
    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--feature-database", type=Path, default=None)
    validate.add_argument("--token-database", type=Path, default=None)
    validate.add_argument("--trajectory-database", type=Path, default=None)
    validate.add_argument("--full", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_cli_parser().parse_args(argv)
    if args.command == "motion-database":
        build_motion_database(args.dataset, args.output, styles=args.styles, max_styles=args.max_styles, prune_ends_and_fingers=args.prune_ends_and_fingers, workers=args.workers)
    elif args.command == "feature-database":
        build_feature_database(args.dataset, args.output, styles=args.styles, max_styles=args.max_styles, prune_ends_and_fingers=args.prune_ends_and_fingers, seed=args.seed, workers=args.workers, overwrite=args.overwrite)
    elif args.command == "token-database":
        raise RuntimeError("token-database requires an injected encoder; call run.py or build_token_database()")
    elif args.command == "trajectory-inputs":
        frames = [int(value) for value in args.future_frames.split(",") if value.strip()]
        build_trajectory_inputs(args.database, args.output, future_frames=frames)
    elif args.command == "trajectory-database":
        build_trajectory_database(args.token_database, args.trajectory_input, args.output, overwrite=args.overwrite)
    elif args.command == "validate-data":
        print(json.dumps(validate_data(feature_database=args.feature_database, token_database=args.token_database, trajectory_database=args.trajectory_database, full=args.full), indent=2))


__all__ = [
    "build_feature_database",
    "build_motion_database",
    "build_token_database",
    "build_trajectory_database",
    "build_trajectory_inputs",
    "validate_data",
]


if __name__ == "__main__":
    main()
