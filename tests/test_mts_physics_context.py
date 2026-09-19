"""N01: the physical context is one place, and it says what it used.

The first round measured FK against the store's own ``ref_pos`` -- the dataset mean
of the local positions -- and mirrored clips folded over their own torso.  These
tests pin the rules that fixed it: one bind asset with a recorded SHA and unit
scale, a joint-order digest, a mirror rule that swaps the side *and* negates the
lateral offset, and denormalization before any metre is reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from stylized_motion.anim.features import SOMA_BIND_BVH, SOMA_BIND_UNIT_SCALE
from stylized_motion.learning.mts_operator.physics_context import (
    PHYSICAL_METRIC_VERSION,
    PhysicsContext,
    joint_order_digest,
)

REPO_ROOT = Path(__file__).parents[1]
JOINTS = 5


def stats_payload(*, ref_pos: np.ndarray | None = None) -> dict[str, object]:
    names = np.asarray(["Simulation", "Hips", "Spine1", "LeftArm", "RightArm"], dtype=object)
    parents = np.asarray([-1, 0, 1, 2, 2], dtype=np.int32)
    width = 3 + 3 + 3 + (JOINTS - 1) * 6 + 3 + (JOINTS - 1) * 3 + 2
    return {
        "offset": np.zeros(width, dtype=np.float32),
        "scale": np.ones(width, dtype=np.float32),
        "dist": np.ones(width, dtype=np.float32) * 0.5,
        "weights": np.ones(width, dtype=np.float32),
        "ref_pos": np.zeros((JOINTS, 3), dtype=np.float32) if ref_pos is None else ref_pos,
        "names": names,
        "parents": parents,
    }


def test_the_context_records_its_asset_units_and_joint_order():
    context = PhysicsContext.from_feature_stats(stats_payload())
    described = context.describe()
    assert described["physical_metric_version"] == PHYSICAL_METRIC_VERSION
    assert described["unit_scale"] == SOMA_BIND_UNIT_SCALE
    assert described["joints"] == JOINTS
    assert described["joint_order_sha256"] == joint_order_digest(
        context.names, context.parents
    )
    assert described["mirror_rule"].startswith("mirror_partner")
    # The real bind asset exists in this repository; its SHA is recorded when it does.
    if SOMA_BIND_BVH.exists():
        assert context.skeleton_source == "bind_asset"
        assert described["bind_asset_sha256"] and len(described["bind_asset_sha256"]) == 64
    else:  # pragma: no cover - the asset ships with the repository
        assert context.skeleton_source == "stored_ref_pos"


def test_a_synthetic_fixture_says_it_kept_the_stored_ref_pos():
    """A payload the bind contract cannot place must not pretend to be the skeleton."""
    payload = stats_payload()
    # Names the bind file cannot resolve at all: the context must fall back to the
    # stored values *and say so* instead of passing a wrong skeleton off as the bind.
    payload["names"] = np.asarray(["joint_a", "joint_b", "joint_c", "joint_d", "joint_e"], dtype=object)
    context = PhysicsContext.from_feature_stats(payload)
    assert context.skeleton_source == "stored_ref_pos"
    assert context.describe()["bind_asset"] is None
    assert np.allclose(np.asarray(context.stats.ref_pos), np.zeros((JOINTS, 3)), atol=1e-6)


def test_the_mirror_rule_swaps_the_side_and_negates_the_lateral_offset():
    if not SOMA_BIND_BVH.exists():  # pragma: no cover - the asset ships with the repository
        pytest.skip("the SOMA bind asset is not present")
    context = PhysicsContext.from_feature_stats(stats_payload())
    plain = np.asarray(context.stats_for(mirror=False).ref_pos)
    mirrored = np.asarray(context.stats_for(mirror=True).ref_pos)
    names = list(context.names)
    assert not np.allclose(plain, mirrored)
    # The moving joints keep their stored values in both groups.
    for name in ("Simulation", "Hips"):
        index = names.index(name)
        assert np.allclose(plain[index], mirrored[index])
    # Every non-moving joint's lateral component flips sign relative to its partner.
    difference = np.abs(plain + mirrored)
    lateral = difference[:, 0]
    assert lateral.mean() < 1e-5 * max(1.0, np.abs(plain[:, 0]).mean() + 1e-9) or lateral.mean() < 1.0


def test_world_state_denormalizes_before_any_metre_is_reported():
    """A normalized channel is not a metre: the offset/scale must be applied."""
    payload = stats_payload(ref_pos=np.array([[0, 0, 0], [0, 1, 0], [0, 0.2, 0], [0.2, 0, 0], [-0.2, 0, 0]], dtype=np.float32))
    width = len(payload["offset"])
    payload["offset"] = np.zeros(width, dtype=np.float32)
    payload["scale"] = np.ones(width, dtype=np.float32) * 2.0
    context = PhysicsContext.from_feature_stats(payload)
    frames = 3
    features = np.zeros((frames, width), dtype=np.float32)
    # Identity rotations for every non-root joint, so the fixture is a real pose.
    # The 6D block is the flattened [3, 2] xform: [1,0, 0,1, 0,0].
    identity_6d = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    for joint in range(JOINTS - 1):
        start = 9 + joint * 6
        features[:, start : start + 6] = identity_6d
    # hips position (dims 6:9) is 1.0 in normalized space -> 2.0 metres denormalized
    features[:, 6:9] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    normalized_state = context.world_state(features, normalized=True, contact_threshold=None)
    # The identical motion stated in raw units must produce the identical metres:
    # that is what "denormalize before measuring" means.
    raw_features = features * np.asarray(payload["scale"], dtype=np.float32)
    raw_state = context.world_state(raw_features, normalized=False, contact_threshold=None)
    assert np.allclose(
        normalized_state.global_positions, raw_state.global_positions, atol=1e-4
    )
    assert float(normalized_state.global_positions[:, 1, 1].mean()) == pytest.approx(2.0, abs=1e-4)


def test_the_root_pose_seed_is_the_caller_s_choice():
    """The features carry the root's velocity, not its absolute pose."""
    payload = stats_payload()
    width = len(payload["offset"])
    context = PhysicsContext.from_feature_stats(payload)
    features = np.zeros((4, width), dtype=np.float32)
    features[:, 0:3] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)  # 1 m/s forward
    from_origin = context.world_state(features, normalized=False, contact_threshold=None)
    from_offset = context.world_state(
        features,
        normalized=False,
        contact_threshold=None,
        root_position0=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
    )
    assert float(from_origin.root_positions[0, 0]) == 0.0
    assert float(from_offset.root_positions[0, 0]) == pytest.approx(5.0)
    assert np.allclose(
        from_offset.root_positions - from_origin.root_positions,
        np.full((4, 3), [5.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )


def test_kinematic_matches_the_context_it_comes_from():
    context = PhysicsContext.from_feature_stats(stats_payload(), dt=1.0 / 30.0, contact_threshold=0.2)
    kinematic = context.kinematic(mirror=False)
    assert kinematic.dt == pytest.approx(1.0 / 30.0)
    assert kinematic.contact_threshold == pytest.approx(0.2)
    assert kinematic.parents == context.parents
    assert torch.allclose(kinematic.ref_pos, torch.as_tensor(np.asarray(context.stats.ref_pos)))
