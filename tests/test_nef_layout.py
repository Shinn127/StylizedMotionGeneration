"""NEF-FSQ layout contract: stream partition, token slices and skeleton validation."""

from __future__ import annotations

import glob

import pytest
import torch

from stylized_motion.learning.nef_layout import (
    GENO_SKELETON,
    NEF_EDIT_PARTS,
    NEF_STREAM_COORDINATES,
    NEF_STREAM_NAMES,
    NEFLayout,
    SOMA_SKELETON,
    nef_edit_streams,
)


def skeleton_from_spec(spec) -> tuple[list[str], list[int]]:
    """Flattens the recorded chains into a parent-ordered names/parents pair."""
    index: dict[str, int] = {}
    names: list[str] = []
    parents: list[int] = []
    for chain in spec.chains:
        for position, name in enumerate(chain):
            if name not in index:
                index[name] = len(names)
                names.append(name)
                parents.append(-1 if position == 0 else index[chain[position - 1]])
            elif position > 0:
                assert parents[index[name]] == index[chain[position - 1]]
    return names, parents


def geno_layout() -> NEFLayout:
    names, parents = skeleton_from_spec(GENO_SKELETON)
    return NEFLayout.from_skeleton(names, parents)


def soma_layout() -> NEFLayout:
    names, parents = skeleton_from_spec(SOMA_SKELETON)
    return NEFLayout.from_skeleton(names, parents)


def test_geno_and_soma_layouts_expose_the_designed_streams():
    for layout, skeleton, num_joints, motion_dim in (
        (geno_layout(), "geno", 25, 230),
        (soma_layout(), "soma", 27, 248),
    ):
        assert layout.skeleton == skeleton
        assert layout.num_joints == num_joints
        assert 9 * layout.num_joints + 5 == motion_dim
        assert layout.num_coordinates == 40
        assert tuple(NEF_STREAM_NAMES) == layout.coordinate_order
        assert sum(layout.coordinate_counts.values()) == 40
        assert len(NEF_STREAM_NAMES) == 13
        assert [layout.stream_slices[stream] for stream in NEF_STREAM_NAMES] == [
            slice(0, 4), slice(4, 8), slice(8, 10), slice(10, 14), slice(14, 18),
            slice(18, 22), slice(22, 26), slice(26, 30), slice(30, 32), slice(32, 34),
            slice(34, 36), slice(36, 38), slice(38, 40),
        ]


def test_geno_stream_membership_matches_the_design_tables():
    layout = geno_layout()
    expected = {
        "torso_node": ["Spine", "Spine1", "Spine2", "Spine3"],
        "head_node": ["Neck1", "Head"],
        "left_arm_node": ["LeftArm", "LeftForeArm", "LeftHand"],
        "right_arm_node": ["RightArm", "RightForeArm", "RightHand"],
        "left_leg_node": ["LeftLeg", "LeftFoot", "LeftToeBase"],
        "right_leg_node": ["RightLeg", "RightFoot", "RightToeBase"],
        "hips_edge": ["Hips"],
        "head_edge": ["Neck"],
        "left_shoulder_edge": ["LeftShoulder"],
        "right_shoulder_edge": ["RightShoulder"],
        "left_leg_edge": ["LeftUpLeg"],
        "right_leg_edge": ["RightUpLeg"],
    }
    for stream, joint_names in expected.items():
        assert [layout.names[index] for index in layout.stream_joints(stream)] == joint_names
    feature_indices = layout.feature_indices(230)
    assert {stream: int(feature_indices[stream].numel()) for stream in NEF_STREAM_NAMES} == {
        "global": 8,
        "torso_node": 36,
        "head_node": 18,
        "left_arm_node": 27,
        "right_arm_node": 27,
        "left_leg_node": 27,
        "right_leg_node": 27,
        "hips_edge": 15,
        "head_edge": 9,
        "left_shoulder_edge": 9,
        "right_shoulder_edge": 9,
        "left_leg_edge": 9,
        "right_leg_edge": 9,
    }


def test_soma_stream_membership_matches_the_design_tables():
    layout = soma_layout()
    expected = {
        "torso_node": ["Spine1", "Spine2", "Chest"],
        "head_node": ["Neck2", "Head", "Jaw", "LeftEye", "RightEye"],
        "left_arm_node": ["LeftArm", "LeftForeArm", "LeftHand"],
        "right_arm_node": ["RightArm", "RightForeArm", "RightHand"],
        "left_leg_node": ["LeftShin", "LeftFoot", "LeftToeBase"],
        "right_leg_node": ["RightShin", "RightFoot", "RightToeBase"],
        "hips_edge": ["Hips"],
        "head_edge": ["Neck1"],
        "left_shoulder_edge": ["LeftShoulder"],
        "right_shoulder_edge": ["RightShoulder"],
        "left_leg_edge": ["LeftLeg"],
        "right_leg_edge": ["RightLeg"],
    }
    for stream, joint_names in expected.items():
        assert [layout.names[index] for index in layout.stream_joints(stream)] == joint_names
    feature_indices = layout.feature_indices(248)
    assert int(feature_indices["head_node"].numel()) == 45
    assert NEF_STREAM_COORDINATES["head_node"] == 2  # 45D head over 2 coordinates
    assert int(feature_indices["torso_node"].numel()) == 27


def test_feature_indices_are_a_disjoint_complete_partition():
    for layout, motion_dim in ((geno_layout(), 230), (soma_layout(), 248)):
        feature_indices = layout.feature_indices(motion_dim)
        assert list(feature_indices) == list(NEF_STREAM_NAMES)
        flattened = torch.cat([feature_indices[stream] for stream in NEF_STREAM_NAMES])
        assert flattened.numel() == motion_dim
        torch.testing.assert_close(torch.sort(flattened).values, torch.arange(motion_dim))
        global_indices = set(feature_indices["global"].tolist())
        hips_velocity = 9 + (layout.num_joints - 1) * 6
        assert {6, 7, 8, hips_velocity, hips_velocity + 1, hips_velocity + 2}.isdisjoint(global_indices)
        assert global_indices == {0, 1, 2, 3, 4, 5, 9 * layout.num_joints + 3, 9 * layout.num_joints + 4}
        assert set(feature_indices["hips_edge"].tolist()) == {
            6, 7, 8, hips_velocity, hips_velocity + 1, hips_velocity + 2,
            *range(9, 15),
            *range(12 + (layout.num_joints - 1) * 6, 15 + (layout.num_joints - 1) * 6),
        }


def test_partition_and_assemble_reproduces_the_input_exactly():
    torch.manual_seed(5)
    for layout, motion_dim in ((geno_layout(), 230), (soma_layout(), 248)):
        motion = torch.randn(2, 64, motion_dim)
        feature_indices = layout.feature_indices(motion_dim)
        streams = {stream: motion.index_select(-1, feature_indices[stream]) for stream in NEF_STREAM_NAMES}
        assembled = torch.zeros_like(motion)
        for stream in NEF_STREAM_NAMES:
            assembled[..., feature_indices[stream]] = streams[stream]
        torch.testing.assert_close(assembled, motion, rtol=0.0, atol=0.0)


def test_layout_rejects_wrong_skeletons_topology_and_motion_dim():
    names, parents = skeleton_from_spec(GENO_SKELETON)
    with pytest.raises(ValueError, match="joint 1 to be 'Hips'"):
        NEFLayout.from_skeleton(["Simulation", "Pelvis"] + list(names[2:]), parents)
    with pytest.raises(ValueError, match="Unknown skeleton"):
        NEFLayout.from_skeleton(names[:-1] + ["LeftHandExtra"], parents)
    with pytest.raises(ValueError, match="Unknown skeleton"):
        NEFLayout.from_skeleton(["Simulation", "Hips", "Spine"], [-1, 0, 1])
    duplicated = names[:-1] + [names[-2]]
    with pytest.raises(ValueError, match="duplicated joint names"):
        NEFLayout.from_skeleton(duplicated, parents)
    with pytest.raises(ValueError, match="parent-ordered"):
        NEFLayout.from_skeleton(["Simulation", "Hips", "Spine"], [-1, 0, 1 + 1])
    wrong_parents = list(parents)
    wrong_parents[names.index("LeftShoulder")] = names.index("Spine2")
    with pytest.raises(ValueError, match="chain"):
        NEFLayout.from_skeleton(names, wrong_parents)

    layout = geno_layout()
    with pytest.raises(ValueError, match="motion_dim=230"):
        layout.validate_motion_dim(248)
    with pytest.raises(ValueError, match="motion_dim=230"):
        layout.feature_indices(231)


def test_soma_layout_is_not_resolved_from_geno_numeric_indices():
    geno, soma = geno_layout(), soma_layout()
    # Same-named joints belong to different streams, so Geno indices are unusable for SOMA.
    assert [geno.names[index] for index in geno.stream_joints("left_leg_node")] == [
        "LeftLeg", "LeftFoot", "LeftToeBase"
    ]
    assert [soma.names[index] for index in soma.stream_joints("left_leg_edge")] == ["LeftLeg"]
    assert [soma.names[index] for index in soma.stream_joints("torso_node")] == ["Spine1", "Spine2", "Chest"]
    assert [geno.names[index] for index in geno.stream_joints("torso_node")] == [
        "Spine", "Spine1", "Spine2", "Spine3"
    ]
    # A mis-wired SOMA skeleton is rejected instead of being remapped onto Geno.
    names, parents = skeleton_from_spec(SOMA_SKELETON)
    broken = list(parents)
    broken[names.index("LeftShin")] = names.index("Hips")
    with pytest.raises(ValueError, match="chain"):
        NEFLayout.from_skeleton(names, broken)


def test_strict_and_full_part_edits_select_the_designed_streams():
    assert nef_edit_streams("left_arm", full_part=False) == ("left_arm_node",)
    assert nef_edit_streams("left_arm", full_part=True) == ("left_arm_node", "left_shoulder_edge")
    assert nef_edit_streams("left_leg", full_part=True) == ("left_leg_node", "left_leg_edge")
    assert nef_edit_streams("head", full_part=True) == ("head_node", "head_edge")
    assert nef_edit_streams("torso", full_part=True) == ("torso_node", "hips_edge")
    with pytest.raises(ValueError, match="Unknown NEF edit part"):
        nef_edit_streams("left_hand", full_part=False)
    assert set(NEF_EDIT_PARTS) == {"torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"}


def test_layout_metadata_persists_skeleton_and_ownership_contract():
    layout = soma_layout()
    metadata = layout.to_dict()
    assert metadata["skeleton"] == "soma"
    assert metadata["names"] == list(layout.names)
    assert metadata["parents"] == list(layout.parents)
    assert metadata["stream_order"] == list(NEF_STREAM_NAMES)
    assert metadata["stream_slices"]["left_arm_node"] == [10, 14]
    assert metadata["stream_joint_indices"]["head_edge"] == [layout.names.index("Neck1")]
    assert metadata["stream_coordinates"] == dict(NEF_STREAM_COORDINATES)
    assert "skeleton_sha256" not in metadata
    assert metadata["names"] != geno_layout().to_dict()["names"]


def test_recorded_tables_match_the_pruned_dataset_skeletons():
    """The shipped tables must match the real pruned Geno/SOMA feature schemas."""
    from stylized_motion.anim import bvh
    from stylized_motion.data.preprocess import _process_motion_data

    geno_paths = sorted(glob.glob("data/raw/100style/*/*.bvh"))
    soma_paths = sorted(glob.glob("data/raw/bones_seed/bvh/soma_uniform/**/*.bvh", recursive=True))
    cases = ((GENO_SKELETON, geno_paths[:1], "geno"), (SOMA_SKELETON, soma_paths[:1], "soma"))
    skipped = [skeleton for _spec, paths, skeleton in cases if not paths]
    if len(skipped) == len(cases):
        pytest.skip("raw Geno/SOMA BVH data is not available")
    for spec, paths, skeleton in cases:
        if not paths:
            continue
        processed = _process_motion_data(bvh.load(paths[0]), mirror=False, prune_ends_and_fingers=True)
        layout = NEFLayout.from_skeleton(processed["names"], processed["parents"])
        assert layout.skeleton == skeleton
        assert set(layout.names) == set(spec.joints)
        assert 9 * layout.num_joints + 5 == (230 if skeleton == "geno" else 248)
