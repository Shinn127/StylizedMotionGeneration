from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preprocess import quat


DEFAULT_PRUNED_DATABASE = Path("data/processed/100style_test5_pruned/database.npz")
DEFAULT_FULL_DATABASE = Path("data/processed/100style_test5/database.npz")
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class MotionFeatureStats:
    offset: np.ndarray
    scale: np.ndarray
    dist: np.ndarray
    weights: np.ndarray
    ref_pos: np.ndarray


def resolve_database_path(use_full_skeleton: bool = False, database_path: str | Path | None = None) -> Path:
    if database_path is not None:
        return Path(database_path)
    return DEFAULT_FULL_DATABASE if use_full_skeleton else DEFAULT_PRUNED_DATABASE


def load_database(database_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(database_path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def joint_feature_dim(nbones: int) -> int:
    return 3 + 3 + 3 + (nbones - 1) * 6 + 3 + (nbones - 1) * 3 + 2


def default_joint_weights(names: list[str]) -> np.ndarray:
    weights = np.ones(len(names), dtype=np.float32)
    for idx, name in enumerate(names):
        if name == "Simulation":
            weights[idx] = 1.0
        elif "Spine" in name or "Neck" in name or name == "Head":
            weights[idx] = 1.5
        elif "Shoulder" in name or "Arm" in name or "Hand" in name:
            weights[idx] = 1.75
        elif "UpLeg" in name or name.endswith("Leg"):
            weights[idx] = 2.0
        elif "Foot" in name or "ToeBase" in name:
            weights[idx] = 2.25
        elif name == "Hips":
            weights[idx] = 2.0
    return weights


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


def build_motion_features(database: dict[str, np.ndarray]) -> tuple[np.ndarray, MotionFeatureStats]:
    names = database["names"].tolist()
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

    stats = compute_feature_stats(
        x=x,
        x_rvel=x_rvel,
        x_rang=x_rang,
        x_hip_pos=x_hip_pos,
        x_rot_6d=x_rot_6d,
        x_hip_vel=x_hip_vel,
        x_ang=x_ang_local,
        x_contacts=x_contacts,
        names=names,
    )
    stats.ref_pos = x_pos.mean(axis=0).astype(np.float32)

    expected_dim = joint_feature_dim(nbones)
    if x.shape[1] != expected_dim:
        raise ValueError(f"Unexpected motion feature dim {x.shape[1]} for {nbones} bones, expected {expected_dim}")

    return x, stats


def normalize_motion_features(x: np.ndarray, stats: MotionFeatureStats) -> np.ndarray:
    return ((x - stats.offset) / stats.scale).astype(np.float32)


def denormalize_motion_features(x: np.ndarray, stats: MotionFeatureStats) -> np.ndarray:
    return (x * stats.scale + stats.offset).astype(np.float32)


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
