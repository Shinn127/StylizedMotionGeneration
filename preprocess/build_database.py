from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import islice
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from tqdm import tqdm

from preprocess import bvh, quat


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

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


def _process_motion_pair(task):
    path, prune_ends_and_fingers = task
    bvh_data = bvh.load(path.as_posix())
    motions = []
    for mirror in [False, True]:
        motion = _process_motion_data(bvh_data, mirror, prune_ends_and_fingers=prune_ends_and_fingers)
        motions.append((mirror, motion))
    return path.stem, motions


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
