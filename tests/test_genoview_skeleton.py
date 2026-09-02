"""Contract tests for the GenoView-style character skeleton overlay."""

import inspect
from pathlib import Path

import numpy as np

from stylized_motion.anim import quat
from stylized_motion.anim.genoview import (
    GENO_RIG,
    GenoView,
    GenoViewCompare,
    skeleton_axis_endpoints,
    skeleton_overlay_pose,
)
from stylized_motion.anim.somaview import SOMA_RIG
from stylized_motion.util.paths import RESOURCE_DIR


def _database(nframes=3):
    names = ("Simulation", "Hips", "Spine", "Spine2")
    count = len(names)
    return {
        "positions": np.zeros((nframes, count, 3), dtype=np.float32),
        "rotations": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (nframes, count, 1)),
        "velocities": np.zeros((nframes, count, 3), dtype=np.float32),
        "angular_velocities": np.zeros((nframes, count, 4), dtype=np.float32),
        "contacts": np.zeros((nframes, 2), dtype=np.uint8),
        "parents": np.asarray([-1, 0, 1, 2], dtype=np.int32),
        "names": np.asarray(names, dtype=object),
        "range_starts": np.asarray([0], dtype=np.int32),
        "range_stops": np.asarray([nframes], dtype=np.int32),
        "range_names": np.asarray(["test"], dtype=object),
        "range_mirror": np.asarray([False], dtype=bool),
        "joint_subset": np.asarray("full", dtype=object),
        "frame_time": np.asarray(1.0 / 60.0, dtype=np.float32),
    }


def test_skeleton_axis_endpoints_follow_wxyz_quaternion_convention():
    position = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    quarter_turn = quat.from_angle_axis(np.pi / 2.0, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    x_axis, y_axis, z_axis = skeleton_axis_endpoints(position, quarter_turn, scale=0.5)

    # A +90 degree turn about +Y maps local +X to world -Z and keeps +Y upright,
    # so the overlay frames stay aligned with the pose driving the mesh.
    assert np.allclose(x_axis - position, [0.0, 0.0, -0.5], atol=1e-6)
    assert np.allclose(y_axis - position, [0.0, 0.5, 0.0], atol=1e-6)
    assert np.allclose(z_axis - position, [0.5, 0.0, 0.0], atol=1e-6)


def test_skeleton_axis_endpoints_match_fk_quaternion_convention():
    rng = np.random.default_rng(7)
    position = np.asarray([-0.3, 1.1, 2.7], dtype=np.float32)
    for _ in range(8):
        quaternion = quat.normalize(rng.normal(size=4).astype(np.float32))
        endpoints = skeleton_axis_endpoints(position, quaternion, scale=0.37)
        for axis in range(3):
            rotated_axis = quat.mul_vec(quaternion, np.eye(3, dtype=np.float32)[axis])
            assert np.allclose(endpoints[axis], position + 0.37 * rotated_axis, atol=1e-5)


def test_skeleton_overlay_pose_keeps_only_the_character_root_subtree():
    # SOMA-style chain: Simulation -> Root (pinned at the world origin) -> Hips,
    # with a helper node hanging off Root behind Hips in joint order.
    global_pos = np.asarray(
        [
            [0.0, 0.0, 0.0],  # Simulation
            [0.0, 0.0, 0.0],  # Root (pinned)
            [0.0, 1.0, 0.1],  # Hips (character root)
            [5.0, 0.0, 0.0],  # Helper (child of Root, after Hips)
            [0.0, 1.2, 0.1],  # Spine (child of Hips)
        ],
        dtype=np.float32,
    )
    global_rot = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (5, 1))
    parents = np.asarray([-1, 0, 1, 1, 2], dtype=np.int32)

    positions, rotations, overlay_parents = skeleton_overlay_pose(global_pos, global_rot, parents, root_index=2)

    # Hips and its subtree survive; the pinned Root, the simulation root, and
    # the helper hanging off Root are all cut, so nothing touches the origin.
    assert np.allclose(positions, [global_pos[2], global_pos[4]])
    assert rotations.shape == (2, 4)
    assert overlay_parents.tolist() == [-1, 0]


def test_skeleton_overlay_defaults_off_and_keeps_rig_last_default():
    parameters = list(inspect.signature(GenoView.__init__).parameters)
    assert "draw_skeleton" in parameters
    # Existing consumers (test_somaview) rely on GENO_RIG being the final default.
    assert parameters[-1] == "rig"

    view = GenoView(database=_database(), trajectory_path=None, resources_root=RESOURCE_DIR)
    assert view.skeleton_enabled is False
    enabled = GenoView(database=_database(), trajectory_path=None, resources_root=RESOURCE_DIR, draw_skeleton=True)
    assert enabled.skeleton_enabled is True


def test_compare_view_accepts_skeleton_flag():
    database = _database()
    view = GenoViewCompare(
        left_database=database,
        right_database=database,
        resources_root=RESOURCE_DIR,
        draw_skeleton=True,
    )
    assert view.compare_mode is True
    assert view.skeleton_enabled is True


def test_skeleton_flag_registered_in_both_viewers():
    import stylized_motion.anim.genoview as genoview
    import stylized_motion.anim.somaview as somaview

    for module in (genoview, somaview):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert 'add_argument("--skeleton"' in source
        assert "draw_skeleton=args.skeleton" in source


def test_rigs_root_the_overlay_at_the_character_hips():
    assert GENO_RIG.skeleton_root_joint == "Hips"
    assert SOMA_RIG.skeleton_root_joint == "Hips"


def test_renderer_schedules_skeleton_overlay():
    import stylized_motion.anim.renderer as renderer_module

    source = Path(renderer_module.__file__).read_text(encoding="utf-8")
    assert "draw_skeleton" in source
    assert "skeleton_overlay_pose" in source
    assert "view.skeleton_enabled" in source
    # The overlay starts at the rig's character root joint, so static rig
    # nodes pinned at the world origin never reach the drawn skeleton.
    assert "view.skeleton_root_index" in source
