"""BONES-SEED SOMA preprocessing contracts: static-root drop and derived feature width.

SOMA BVHs pin a ``Root`` joint at the world origin while the pelvis translation
flows through ``Hips``. The preprocess pipeline must drop that node so SOMA
databases share the ``Simulation -> Hips`` convention of Geno/100STYLE — the
motion-feature layout hardcodes index 1 as the hips slot, and the feature
width is derived from the pruned skeleton instead of the Geno-specific 230.
"""

import numpy as np
import pytest

from stylized_motion.anim import bvh, quat
from stylized_motion.anim.features import (
    build_motion_feature_components,
    build_motion_features,
    deserialize_motion_feature_stats,
    joint_feature_dim,
    reconstruct_motion_state_from_features,
    serialize_motion_feature_stats,
)
from stylized_motion.data.preprocess import _normalize_to_60fps, _process_motion_data
from stylized_motion.util.paths import SOMA_RESOURCE_DIR

SOMA_BIND = SOMA_RESOURCE_DIR / "SOMA_bind.bvh"


def _tiled_soma_bvh(frames=160, walk_translation=False):
    # The bind BVH is a pose file with a placeholder 24 fps frame time, so the
    # fixture adopts the 120 fps rate of the real soma_uniform motion clips;
    # the pipeline decimates it to 60 fps, and `frames` counts pre-decimation
    # rows (half of them survive into the processed database).
    if not SOMA_BIND.exists():
        pytest.skip("SOMA assets have not been generated")
    data = bvh.load(SOMA_BIND)
    reps = int(np.ceil(frames / data["positions"].shape[0]))
    data["positions"] = np.repeat(data["positions"], reps, axis=0)[:frames].astype(np.float32)
    data["rotations"] = np.repeat(data["rotations"], reps, axis=0)[:frames]
    data["frametime"] = 0.008333
    if walk_translation:
        hips = [str(n) for n in data["names"]].index("Hips")
        # Constant-velocity pelvis translation (cm/frame); savgol reproduces
        # linear signals exactly, keeping the round-trip assertions tight.
        data["positions"][:, hips, 2] += 0.3 * np.arange(frames)
    return data


def test_joint_feature_dim_matches_legacy_geno_width():
    # 25 bones (Simulation + 24 pruned Geno/100STYLE joints) -> the historical 230.
    assert joint_feature_dim(25) == 230


def test_normalize_to_60fps_decimates_120fps_input():
    bvh_data = _tiled_soma_bvh(frames=8)  # fixture adopts the clips' 0.008333 s frame time
    normalized = _normalize_to_60fps(bvh_data)

    assert len(normalized["positions"]) == 4
    np.testing.assert_allclose(normalized["positions"], bvh_data["positions"][::2])
    np.testing.assert_allclose(normalized["rotations"], bvh_data["rotations"][::2])
    assert abs(normalized["frametime"] - 1.0 / 60.0) < 1e-5
    # Joint-level metadata is untouched by frame decimation.
    assert normalized["names"] is bvh_data["names"]
    assert normalized["parents"] is bvh_data["parents"]


def test_normalize_to_60fps_passthrough_for_60fps_and_missing_rate():
    bvh_data = _tiled_soma_bvh(frames=8)
    bvh_data["frametime"] = 0.016666  # 60 fps with header rounding
    assert _normalize_to_60fps(bvh_data) is bvh_data

    del bvh_data["frametime"]
    assert _normalize_to_60fps(bvh_data) is bvh_data


def test_normalize_to_60fps_rejects_non_integer_multiples():
    bvh_data = _tiled_soma_bvh(frames=8)
    bvh_data["frametime"] = 1.0 / 90.0
    with pytest.raises(ValueError, match="integer multiple"):
        _normalize_to_60fps(bvh_data)


def test_process_motion_data_decimates_soma_120fps_like_the_bvh_script():
    data = _tiled_soma_bvh(frames=160)
    even_frames = {
        **data,
        "positions": data["positions"][::2],
        "rotations": data["rotations"][::2],
        "frametime": 1.0 / 60.0,
    }

    processed = _process_motion_data(data, mirror=False, prune_ends_and_fingers=True)
    reference = _process_motion_data(even_frames, mirror=False, prune_ends_and_fingers=True)

    # Processing the native 120 fps clip must match processing the even-frame
    # 60 fps slice — i.e. the in-pipeline decimation agrees verbatim with
    # scripts/downsample_bvh.py's keep-every-Nth-frame semantics.
    assert len(processed["positions"]) == 80
    for key in ("positions", "rotations", "velocities", "angular_velocities"):
        np.testing.assert_allclose(processed[key], reference[key], atol=1e-6)


def test_process_motion_data_drops_soma_static_rig_root():
    database = _process_motion_data(_tiled_soma_bvh(), mirror=False)

    names = [str(n) for n in database["names"]]
    assert "Root" not in names
    assert names[0] == "Simulation"
    assert names[1] == "Hips"
    assert database["parents"][1] == 0


def test_soma_pruned_feature_width_is_derived_from_skeleton():
    database = _process_motion_data(_tiled_soma_bvh(), mirror=False, prune_ends_and_fingers=True)

    nbones = len(database["names"])
    assert nbones == 27  # 26 SOMA joints after the Root/ends/fingers prune + Simulation
    components = build_motion_feature_components(database)
    assert components.x.shape[1] == joint_feature_dim(nbones)
    assert components.x.shape[1] == 248  # SOMA has its own width, not Geno's 230


def test_soma_feature_round_trip_preserves_root_translation():
    database = _process_motion_data(_tiled_soma_bvh(walk_translation=True), mirror=False, prune_ends_and_fingers=True)
    names = [str(n) for n in database["names"]]
    parents = np.asarray(database["parents"], dtype=np.int32)

    x, stats = build_motion_features(database)
    payload = serialize_motion_feature_stats(stats, names=names, parents=parents, joint_subset="prune_ends_and_fingers")
    decoded_stats, metadata = deserialize_motion_feature_stats(payload)
    state = reconstruct_motion_state_from_features(
        x=x,
        stats=decoded_stats,
        parents=np.asarray(metadata["parents"], dtype=np.int32),
        normalized=False,
        # Features are expressed in the simulation-root frame; reconstruction
        # needs the clip's initial root pose to recover the world trajectory.
        root_position0=database["positions"][0, 0],
        root_rotation0=database["rotations"][0, 0],
    )

    _, decoded_global = quat.fk(state.local_rotations, state.local_positions, parents)
    _, database_global = quat.fk(database["rotations"], database["positions"], parents)

    # The walking pelvis must survive the encode -> reconstruct round trip;
    # with the Root node kept in the database the character would stay pinned
    # at the origin and the root translation would be lost entirely.
    hips = names.index("Hips")
    error = np.abs(decoded_global[:, hips] - database_global[:, hips])
    assert error[10:-10].max() < 0.01
    travel = np.linalg.norm(decoded_global[-1, hips] - decoded_global[0, hips])
    assert travel > 0.15
