import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from tqdm import tqdm

import bvh
import quat


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

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


def _process_motion(path, mirror, prune_ends_and_fingers=False):
    bvh_data = bvh.load(path.as_posix())
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
    motions = []
    for mirror in [False, True]:
        motion = _process_motion(path, mirror, prune_ends_and_fingers=prune_ends_and_fingers)
        motions.append((mirror, motion))
    return path.stem, motions


def _process_all_motion_pairs(bvh_paths, prune_ends_and_fingers, workers):
    workers = max(1, int(workers))
    tasks = [(path, prune_ends_and_fingers) for path in bvh_paths]
    if workers == 1:
        results = (_process_motion_pair(task) for task in tqdm(tasks, desc="Processing motions"))
        return list(results)

    context = mp.get_context("fork")
    chunksize = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = executor.map(_process_motion_pair, tasks, chunksize=chunksize)
        return list(tqdm(results, total=len(tasks), desc="Processing motions"))

def build_lafan_tags():
    tags = []
    for path in sorted(LAFAN_SOURCE.glob("*.bvh")):
        tags.append((path.stem, "all", 0, None))
    return tags


def _parse_style_filter(styles_arg):
    if not styles_arg:
        return None
    return {style.strip() for style in styles_arg.split(",") if style.strip()}


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


def generate_database(dataset_name, output_dir, styles_arg=None, max_styles=None, prune_ends_and_fingers=False, workers=1):
    output_dir.mkdir(parents=True, exist_ok=True)
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

    bone_positions = []
    bone_velocities = []
    bone_rotations = []
    bone_angular_velocities = []
    contact_states = []

    range_starts = []
    range_stops = []
    range_names = []
    range_mirror = []

    tag_range_starts = []
    tag_range_stops = []
    tag_range_names = []
    tag_tags = []
    tag_mirror = []

    bone_parents = None
    bone_names = None

    motion_pairs = _process_all_motion_pairs(bvh_paths, prune_ends_and_fingers, workers=workers)

    for range_name, motions in motion_pairs:
        for mirror, motion in motions:
            if bone_parents is None:
                bone_parents = motion["parents"]
                bone_names = motion["names"]

            offset = 0 if not range_starts else range_stops[-1]
            nframes = len(motion["positions"])

            bone_positions.append(motion["positions"])
            bone_velocities.append(motion["velocities"])
            bone_rotations.append(motion["rotations"])
            bone_angular_velocities.append(motion["angular_velocities"])
            contact_states.append(motion["contacts"])

            range_starts.append(offset)
            range_stops.append(offset + nframes)
            range_names.append(range_name)
            range_mirror.append(mirror)

            for tag_range_name, tag, tag_start_in_bvh, tag_stop_in_bvh in tags_data:
                if tag_range_name != range_name:
                    continue
                if tag_stop_in_bvh is None:
                    tag_stop_in_bvh = nframes
                tag_range_starts.append(offset + tag_start_in_bvh)
                tag_range_stops.append(offset + min(tag_stop_in_bvh, nframes))
                tag_range_names.append(tag_range_name)
                tag_tags.append(tag)
                tag_mirror.append(mirror)

    np.savez(
        output_dir / "database.npz",
        positions=np.concatenate(bone_positions, axis=0).astype(np.float32),
        velocities=np.concatenate(bone_velocities, axis=0).astype(np.float32),
        rotations=np.concatenate(bone_rotations, axis=0).astype(np.float32),
        angular_velocities=np.concatenate(bone_angular_velocities, axis=0).astype(np.float32),
        parents=bone_parents.astype(np.int32),
        names=bone_names,
        range_starts=np.array(range_starts).astype(np.int32),
        range_stops=np.array(range_stops).astype(np.int32),
        range_mirror=np.array(range_mirror).astype(bool),
        range_names=np.array(range_names, dtype=object),
        contacts=np.concatenate(contact_states, axis=0).astype(np.uint8),
        tag_range_starts=np.array(tag_range_starts).astype(np.int32),
        tag_range_stops=np.array(tag_range_stops).astype(np.int32),
        tag_range_names=np.array(tag_range_names, dtype=object),
        tag_tags=np.array(tag_tags, dtype=object),
        tag_mirror=np.array(tag_mirror).astype(bool),
        joint_subset=np.array(
            "prune_ends_and_fingers" if prune_ends_and_fingers else "full",
            dtype=object,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Build local motion database from linked raw datasets.")
    parser.add_argument("--dataset", choices=["lafan", "100style"], required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory.")
    parser.add_argument(
        "--styles",
        type=str,
        default=None,
        help="Comma-separated 100style subset, e.g. Aeroplane,Akimbo,Angry.",
    )
    parser.add_argument(
        "--max-styles",
        type=int,
        default=None,
        help="Use only the first N styles from 100style's Frame_Cuts.csv order.",
    )
    parser.add_argument(
        "--prune-ends-and-fingers",
        action="store_true",
        help="Exclude all *End terminal joints and all hand finger joint chains before building the database.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for BVH preprocessing. Use 1 for serial processing.",
    )
    args = parser.parse_args()

    default_output = PROCESSED_DIR / args.dataset
    output_dir = args.output or default_output
    generate_database(
        args.dataset,
        output_dir,
        styles_arg=args.styles,
        max_styles=args.max_styles,
        prune_ends_and_fingers=args.prune_ends_and_fingers,
        workers=args.workers,
    )
    print(f"Saved database to {output_dir / 'database.npz'}")


if __name__ == "__main__":
    main()
