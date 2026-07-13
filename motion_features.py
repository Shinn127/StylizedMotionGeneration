from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from preprocess import quat


SPLIT_NAMES = ("train", "val", "test")


@dataclass
class MotionFeatureStats:
    offset: np.ndarray
    scale: np.ndarray
    dist: np.ndarray
    weights: np.ndarray
    ref_pos: np.ndarray


@dataclass
class MotionState:
    local_positions: np.ndarray
    local_rotations: np.ndarray
    local_velocities: np.ndarray
    local_angular_velocities: np.ndarray
    contacts: np.ndarray
    root_positions: np.ndarray
    root_rotations: np.ndarray
    global_positions: np.ndarray | None = None
    global_rotations: np.ndarray | None = None


@dataclass
class MotionFeatureComponents:
    names: list[str]
    positions: np.ndarray
    x: np.ndarray
    x_rvel: np.ndarray
    x_rang: np.ndarray
    x_hip_pos: np.ndarray
    x_rot_6d: np.ndarray
    x_hip_vel: np.ndarray
    x_ang_local: np.ndarray
    x_contacts: np.ndarray


def load_database(database_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(database_path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def joint_feature_dim(nbones: int) -> int:
    return 3 + 3 + 3 + (nbones - 1) * 6 + 3 + (nbones - 1) * 3 + 2


WEIGHTS_MESH = {
    "Simulation": 0.00000000,
    "Hips": 0.27088639,
    "Spine": 0.12776886,
    "Spine1": 0.10730254,
    "Spine2": 0.08733685,
    "Spine3": 0.07508411,
    "Neck": 0.00838600,
    "Neck1": 0.00639638,
    "Head": 0.00515253,
    "HeadEnd": 0.00063045,
    "RightShoulder": 0.02654437,
    "RightArm": 0.02060832,
    "RightForeArm": 0.00825604,
    "RightHand": 0.00213240,
    "RightHandThumb1": 0.00073802,
    "RightHandThumb2": 0.00066565,
    "RightHandThumb3": 0.00063558,
    "RightHandThumb4": 0.00063045,
    "RightHandIndex1": 0.00070377,
    "RightHandIndex2": 0.00064898,
    "RightHandIndex3": 0.00063289,
    "RightHandIndex4": 0.00063045,
    "RightHandMiddle1": 0.00072178,
    "RightHandMiddle2": 0.00065547,
    "RightHandMiddle3": 0.00063321,
    "RightHandMiddle4": 0.00063045,
    "RightHandRing1": 0.00070793,
    "RightHandRing2": 0.00065231,
    "RightHandRing3": 0.00063322,
    "RightHandRing4": 0.00063045,
    "RightHandPinky1": 0.00067184,
    "RightHandPinky2": 0.00063829,
    "RightHandPinky3": 0.00063110,
    "RightHandPinky4": 0.00063045,
    "RightForeArmEnd": 0.00063045,
    "RightArmEnd": 0.00063045,
    "LeftShoulder": 0.02739252,
    "LeftArm": 0.02113067,
    "LeftForeArm": 0.00849728,
    "LeftHand": 0.00210641,
    "LeftHandThumb1": 0.00071845,
    "LeftHandThumb2": 0.00065790,
    "LeftHandThumb3": 0.00063489,
    "LeftHandThumb4": 0.00063045,
    "LeftHandIndex1": 0.00069211,
    "LeftHandIndex2": 0.00064446,
    "LeftHandIndex3": 0.00063293,
    "LeftHandIndex4": 0.00063045,
    "LeftHandMiddle1": 0.00071069,
    "LeftHandMiddle2": 0.00065042,
    "LeftHandMiddle3": 0.00063314,
    "LeftHandMiddle4": 0.00063045,
    "LeftHandRing1": 0.00070524,
    "LeftHandRing2": 0.00065236,
    "LeftHandRing3": 0.00063302,
    "LeftHandRing4": 0.00063045,
    "LeftHandPinky1": 0.00067250,
    "LeftHandPinky2": 0.00064092,
    "LeftHandPinky3": 0.00063160,
    "LeftHandPinky4": 0.00063045,
    "LeftForeArmEnd": 0.00063045,
    "LeftArmEnd": 0.00063045,
    "RightUpLeg": 0.05690333,
    "RightLeg": 0.02043630,
    "RightFoot": 0.00305942,
    "RightToeBase": 0.00080056,
    "RightToeBaseEnd": 0.00063045,
    "RightLegEnd": 0.00063045,
    "RightUpLegEnd": 0.00063045,
    "LeftUpLeg": 0.05668447,
    "LeftLeg": 0.02033588,
    "LeftFoot": 0.00289429,
    "LeftToeBase": 0.00078392,
    "LeftToeBaseEnd": 0.00063045,
    "LeftLegEnd": 0.00063045,
    "LeftUpLegEnd": 0.00063045,
}


def default_joint_weights(names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in WEIGHTS_MESH]
    if missing:
        raise KeyError(f"Missing weights_mesh entries for joints: {missing}")
    return np.asarray([WEIGHTS_MESH[name] for name in names], dtype=np.float32)


def compute_feature_stats(
    x: np.ndarray,
    x_rvel: np.ndarray,
    x_rang: np.ndarray,
    x_hip_pos: np.ndarray,
    x_rot_6d: np.ndarray,
    x_hip_vel: np.ndarray,
    x_ang: np.ndarray,
    x_contacts: np.ndarray,
    names: list[str],
) -> MotionFeatureStats:
    nbones = len(names)

    offset = x.mean(axis=0)
    scale = np.concatenate(
        [
            np.full(3, x_rvel.std(axis=0).mean(), dtype=np.float32),
            np.full(3, x_rang.std(axis=0).mean(), dtype=np.float32),
            np.full(3, x_hip_pos.std(axis=0).mean(), dtype=np.float32),
            np.full((nbones - 1) * 6, x_rot_6d.std(axis=0).mean(), dtype=np.float32),
            np.full(3, x_hip_vel.std(axis=0).mean(), dtype=np.float32),
            np.full((nbones - 1) * 3, x_ang.std(axis=0).mean(), dtype=np.float32),
            np.full(2, x_contacts.std(axis=0).mean(), dtype=np.float32),
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

    x_norm = (x - offset) / scale
    dist = x_norm.std(axis=0).astype(np.float32)

    return MotionFeatureStats(
        offset=offset.astype(np.float32),
        scale=scale.astype(np.float32),
        dist=dist,
        weights=weights,
        ref_pos=np.asarray(0.0, dtype=np.float32),
    )


def build_motion_feature_components(database: dict[str, np.ndarray]) -> MotionFeatureComponents:
    names = list(database["names"])
    x_pos = database["positions"].astype(np.float32)
    x_rot = database["rotations"].astype(np.float32)
    x_vel = database["velocities"].astype(np.float32)
    x_ang = database["angular_velocities"].astype(np.float32)
    x_contacts = database["contacts"].astype(np.float32)
    nbones = x_pos.shape[1]

    x_rvel = quat.inv_mul_vec(x_rot[:, 0], x_vel[:, 0]).astype(np.float32)
    x_rang = quat.inv_mul_vec(x_rot[:, 0], x_ang[:, 0]).astype(np.float32)
    x_hip_pos = x_pos[:, 1].astype(np.float32)
    x_rot_6d = quat.to_xform_xy(x_rot).reshape(len(x_rot), nbones, 6)[:, 1:].reshape(len(x_rot), (nbones - 1) * 6).astype(np.float32)
    x_hip_vel = x_vel[:, 1].astype(np.float32)
    x_ang_local = x_ang[:, 1:].reshape(len(x_ang), (nbones - 1) * 3).astype(np.float32)

    x = np.concatenate(
        [
            x_rvel,
            x_rang,
            x_hip_pos,
            x_rot_6d,
            x_hip_vel,
            x_ang_local,
            x_contacts,
        ],
        axis=-1,
    ).astype(np.float32)

    return MotionFeatureComponents(
        names=names,
        positions=x_pos,
        x=x,
        x_rvel=x_rvel,
        x_rang=x_rang,
        x_hip_pos=x_hip_pos,
        x_rot_6d=x_rot_6d,
        x_hip_vel=x_hip_vel,
        x_ang_local=x_ang_local,
        x_contacts=x_contacts,
    )


def build_motion_features(
    database: dict[str, np.ndarray],
    stat_frame_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, MotionFeatureStats]:
    components = build_motion_feature_components(database)
    names = components.names
    x = components.x
    nbones = components.positions.shape[1]

    if stat_frame_indices is None:
        stat_frame_indices = np.arange(len(x), dtype=np.int32)
    else:
        stat_frame_indices = np.asarray(stat_frame_indices, dtype=np.int32)
    if stat_frame_indices.ndim != 1 or len(stat_frame_indices) == 0:
        raise ValueError(f"Expected non-empty stat_frame_indices with shape [N], got {stat_frame_indices.shape}")

    stats = compute_feature_stats(
        x=components.x[stat_frame_indices],
        x_rvel=components.x_rvel[stat_frame_indices],
        x_rang=components.x_rang[stat_frame_indices],
        x_hip_pos=components.x_hip_pos[stat_frame_indices],
        x_rot_6d=components.x_rot_6d[stat_frame_indices],
        x_hip_vel=components.x_hip_vel[stat_frame_indices],
        x_ang=components.x_ang_local[stat_frame_indices],
        x_contacts=components.x_contacts[stat_frame_indices],
        names=names,
    )
    stats.ref_pos = components.positions[stat_frame_indices].mean(axis=0).astype(np.float32)

    expected_dim = joint_feature_dim(nbones)
    if x.shape[1] != expected_dim:
        raise ValueError(f"Unexpected motion feature dim {x.shape[1]} for {nbones} bones, expected {expected_dim}")

    return x, stats


def serialize_motion_feature_stats(
    stats: MotionFeatureStats,
    names: list[str] | None = None,
    parents: np.ndarray | None = None,
    joint_subset: str | None = None,
) -> dict[str, np.ndarray]:
    payload = {
        "offset": stats.offset.astype(np.float32),
        "scale": stats.scale.astype(np.float32),
        "dist": stats.dist.astype(np.float32),
        "weights": stats.weights.astype(np.float32),
        "ref_pos": stats.ref_pos.astype(np.float32),
    }
    if names is not None:
        payload["names"] = np.asarray(names, dtype=object)
    if parents is not None:
        payload["parents"] = np.asarray(parents, dtype=np.int32)
    if joint_subset is not None:
        payload["joint_subset"] = np.asarray(joint_subset, dtype=object)
    return payload


def deserialize_motion_feature_stats(payload: dict[str, np.ndarray]) -> tuple[MotionFeatureStats, dict[str, object]]:
    stats = MotionFeatureStats(
        offset=np.asarray(payload["offset"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
        dist=np.asarray(payload["dist"], dtype=np.float32),
        weights=np.asarray(payload["weights"], dtype=np.float32),
        ref_pos=np.asarray(payload["ref_pos"], dtype=np.float32),
    )
    metadata: dict[str, object] = {}
    if "names" in payload:
        metadata["names"] = np.asarray(payload["names"], dtype=object).tolist()
    if "parents" in payload:
        metadata["parents"] = np.asarray(payload["parents"], dtype=np.int32)
    if "joint_subset" in payload:
        metadata["joint_subset"] = str(np.asarray(payload["joint_subset"], dtype=object).item())
    return stats, metadata


def normalize_motion_features(x: np.ndarray, stats: MotionFeatureStats) -> np.ndarray:
    return ((x - stats.offset) / stats.scale).astype(np.float32)


def denormalize_motion_features(x: np.ndarray, stats: MotionFeatureStats) -> np.ndarray:
    return (x * stats.scale + stats.offset).astype(np.float32)


def unpack_motion_features(x: np.ndarray, nbones: int) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2:
        raise ValueError(f"Expected motion features with shape [T, D] or [D], got {x.shape}")

    expected_dim = joint_feature_dim(nbones)
    if x.shape[-1] != expected_dim:
        raise ValueError(f"Unexpected motion feature dim {x.shape[-1]} for {nbones} bones, expected {expected_dim}")

    rot6_end = 9 + (nbones - 1) * 6
    hvel_end = rot6_end + 3
    ang_end = hvel_end + (nbones - 1) * 3

    x_rvel = x[:, 0:3]
    x_rang = x[:, 3:6]
    x_hip = x[:, 6:9]
    x_rot_6d = x[:, 9:rot6_end].reshape(len(x), nbones - 1, 3, 2)
    x_hvel = x[:, rot6_end:hvel_end]
    x_ang = x[:, hvel_end:ang_end].reshape(len(x), nbones - 1, 3)
    x_contacts = x[:, ang_end:ang_end + 2]

    return {
        "root_linear_velocity_local": x_rvel.astype(np.float32),
        "root_angular_velocity_local": x_rang.astype(np.float32),
        "hips_position_local": x_hip.astype(np.float32),
        "joint_rotations_6d": x_rot_6d.astype(np.float32),
        "hips_velocity_local": x_hvel.astype(np.float32),
        "joint_angular_velocities_local": x_ang.astype(np.float32),
        "contacts": x_contacts.astype(np.float32),
    }


def reconstruct_motion_state_from_features(
    x: np.ndarray,
    stats: MotionFeatureStats,
    parents: np.ndarray | None = None,
    dt: float = 1.0 / 60.0,
    root_position0: np.ndarray | None = None,
    root_rotation0: np.ndarray | None = None,
    normalized: bool = True,
    contact_threshold: float | None = 0.5,
) -> MotionState:
    if normalized:
        x = denormalize_motion_features(x, stats)
    else:
        x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2:
        raise ValueError(f"Expected motion features with shape [T, D] or [D], got {x.shape}")

    ref_pos = np.asarray(stats.ref_pos, dtype=np.float32)
    if ref_pos.ndim != 2 or ref_pos.shape[1] != 3:
        raise ValueError(f"Expected stats.ref_pos with shape [J, 3], got {ref_pos.shape}")

    nbones = ref_pos.shape[0]
    parts = unpack_motion_features(x, nbones)
    nframes = len(x)

    root_positions = np.zeros((nframes, 3), dtype=np.float32)
    root_rotations = np.zeros((nframes, 4), dtype=np.float32)
    root_positions[0] = np.zeros(3, dtype=np.float32) if root_position0 is None else np.asarray(root_position0, dtype=np.float32)
    root_rotations[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) if root_rotation0 is None else np.asarray(root_rotation0, dtype=np.float32)
    root_rotations[0] = quat.normalize(root_rotations[0])

    root_linear_velocity_world = np.zeros((nframes, 3), dtype=np.float32)
    root_angular_velocity_world = np.zeros((nframes, 3), dtype=np.float32)
    root_linear_velocity_world[0] = quat.mul_vec(root_rotations[0], parts["root_linear_velocity_local"][0])
    root_angular_velocity_world[0] = quat.mul_vec(root_rotations[0], parts["root_angular_velocity_local"][0])

    for frame in range(1, nframes):
        prev_rot = root_rotations[frame - 1]
        root_linear_velocity_world[frame] = quat.mul_vec(prev_rot, parts["root_linear_velocity_local"][frame])
        root_angular_velocity_world[frame] = quat.mul_vec(prev_rot, parts["root_angular_velocity_local"][frame])
        root_positions[frame] = root_positions[frame - 1] + dt * root_linear_velocity_world[frame]
        root_rotations[frame] = quat.mul(
            quat.from_scaled_angle_axis(dt * root_angular_velocity_world[frame]),
            prev_rot,
        )
        root_rotations[frame] = quat.normalize(root_rotations[frame])

    nonroot_rotations = quat.from_xform_xy(parts["joint_rotations_6d"]).astype(np.float32)

    local_positions = np.repeat(ref_pos[None], nframes, axis=0).astype(np.float32)
    local_positions[:, 0] = root_positions
    local_positions[:, 1] = parts["hips_position_local"]

    local_rotations = np.zeros((nframes, nbones, 4), dtype=np.float32)
    local_rotations[:, 0] = root_rotations
    local_rotations[:, 1:] = nonroot_rotations

    local_velocities = np.zeros((nframes, nbones, 3), dtype=np.float32)
    local_velocities[:, 0] = root_linear_velocity_world
    local_velocities[:, 1] = parts["hips_velocity_local"]

    local_angular_velocities = np.zeros((nframes, nbones, 3), dtype=np.float32)
    local_angular_velocities[:, 0] = root_angular_velocity_world
    local_angular_velocities[:, 1:] = parts["joint_angular_velocities_local"]

    contacts = parts["contacts"]
    if contact_threshold is not None:
        contacts = (contacts > float(contact_threshold)).astype(np.uint8)

    global_rotations = None
    global_positions = None
    if parents is not None:
        global_rotations, global_positions = quat.fk(
            local_rotations,
            local_positions,
            np.asarray(parents, dtype=np.int32),
        )
        global_rotations = global_rotations.astype(np.float32)
        global_positions = global_positions.astype(np.float32)

    return MotionState(
        local_positions=local_positions,
        local_rotations=local_rotations,
        local_velocities=local_velocities,
        local_angular_velocities=local_angular_velocities,
        contacts=contacts.astype(np.float32) if contact_threshold is None else contacts,
        root_positions=root_positions,
        root_rotations=root_rotations,
        global_positions=global_positions,
        global_rotations=global_rotations,
    )


def reconstruct_motion_state_from_normalized_features(
    x_norm: np.ndarray,
    stats: MotionFeatureStats,
    parents: np.ndarray | None = None,
    dt: float = 1.0 / 60.0,
    root_position0: np.ndarray | None = None,
    root_rotation0: np.ndarray | None = None,
    contact_threshold: float | None = 0.5,
) -> MotionState:
    return reconstruct_motion_state_from_features(
        x=x_norm,
        stats=stats,
        parents=parents,
        dt=dt,
        root_position0=root_position0,
        root_rotation0=root_rotation0,
        normalized=True,
        contact_threshold=contact_threshold,
    )


def split_groups(group_names: list[str], seed: int = 3407) -> dict[str, set[str]]:
    unique_groups = np.array(sorted(set(group_names)), dtype=object)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_groups))
    unique_groups = unique_groups[perm]

    n = len(unique_groups)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val

    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = max(n_val, 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(n_train - 1, 1)

    train_groups = set(unique_groups[:n_train].tolist())
    val_groups = set(unique_groups[n_train : n_train + n_val].tolist())
    test_groups = set(unique_groups[n_train + n_val :].tolist())
    return {"train": train_groups, "val": val_groups, "test": test_groups}


def build_range_split_masks(
    range_names: np.ndarray,
    range_mirror: np.ndarray,
    seed: int = 3407,
) -> dict[str, np.ndarray]:
    base_groups = [str(name) for name in range_names.tolist()]
    split_to_groups = split_groups(base_groups, seed=seed)
    masks = {}
    for split_name in SPLIT_NAMES:
        groups = split_to_groups[split_name]
        masks[split_name] = np.array([str(name) in groups for name in range_names.tolist()], dtype=bool)
    return masks
