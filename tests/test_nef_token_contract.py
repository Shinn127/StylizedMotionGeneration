"""NEF research token contract: layout metadata, region masks and alias identity.

These tests pin the read-only surface the MTS-FSQ operator work builds on:
``layout_hash``, ``stream_graph``, ``coordinate_metadata``, ``region_streams``,
``make_region_mask`` and the ``encode_indices`` / ``decode_indices`` aliases.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from stylized_motion.learning.nef_fsq import NEFMotionAutoencoder
from stylized_motion.learning.nef_layout import (
    GENO_SKELETON,
    NEF_EDIT_PARTS,
    NEF_STREAM_COORDINATES,
    NEF_STREAM_FAMILY,
    NEF_STREAM_NAMES,
    NEF_WHOLE_BODY_REGION,
    NEFLayout,
    SOMA_SKELETON,
    nef_edit_streams,
)
from stylized_motion.learning.representation import NEF_FSQ_FAMILY, build_representation


# Pinned so an accidental change to the persisted layout payload is caught: the
# hash is part of the contract that token stores and operators fingerprint.
GENO_LAYOUT_HASH = "51d5c9b447d3755b174dfa92d697669ac4a85d5285b4987ff02444e1656648b8"
SOMA_LAYOUT_HASH = "f2f14fd21ac0dfdd6f718f6fce8614ef767362c6b30eaeca6a02eec729daa0f5"

EXPECTED_STREAM_GRAPH = (
    ("global", "hips_edge", "global_to_root"),
    ("hips_edge", "torso_node", "edge_to_child"),
    ("torso_node", "head_edge", "node_to_edge"),
    ("head_edge", "head_node", "edge_to_child"),
    ("torso_node", "left_shoulder_edge", "node_to_edge"),
    ("left_shoulder_edge", "left_arm_node", "edge_to_child"),
    ("torso_node", "right_shoulder_edge", "node_to_edge"),
    ("right_shoulder_edge", "right_arm_node", "edge_to_child"),
    ("hips_edge", "left_leg_edge", "edge_to_edge"),
    ("left_leg_edge", "left_leg_node", "edge_to_child"),
    ("hips_edge", "right_leg_edge", "edge_to_edge"),
    ("right_leg_edge", "right_leg_node", "edge_to_child"),
)


def skeleton_from_spec(spec) -> tuple[list[str], list[int]]:
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


def layout_for(spec) -> NEFLayout:
    names, parents = skeleton_from_spec(spec)
    return NEFLayout.from_skeleton(names, parents)


def model_for(spec=GENO_SKELETON, *, stream_dim: int = 16, seed: int = 5) -> NEFMotionAutoencoder:
    torch.manual_seed(seed)
    names, parents = skeleton_from_spec(spec)
    return NEFMotionAutoencoder(names, parents, stream_dim=stream_dim).eval()


def test_layout_hash_is_stable_and_covers_the_persisted_payload():
    for spec, expected in ((GENO_SKELETON, GENO_LAYOUT_HASH), (SOMA_SKELETON, SOMA_LAYOUT_HASH)):
        layout = layout_for(spec)
        assert layout.layout_hash() == expected
        # Recomputed from the same payload: a stable, self-describing digest.
        payload = json.dumps(layout.to_dict(), sort_keys=True, separators=(",", ":"))
        assert layout.layout_hash() == hashlib.sha256(payload.encode()).hexdigest()
        assert layout_for(spec).layout_hash() == expected
    assert GENO_LAYOUT_HASH != SOMA_LAYOUT_HASH


def test_stream_graph_matches_the_designed_relations_for_both_skeletons():
    allowed = {"global_to_root", "node_to_edge", "edge_to_child", "edge_to_edge", "node_to_node"}
    for spec in (GENO_SKELETON, SOMA_SKELETON):
        graph = layout_for(spec).stream_graph()
        assert graph == EXPECTED_STREAM_GRAPH
        for parent, child, relation in graph:
            assert parent in NEF_STREAM_NAMES and child in NEF_STREAM_NAMES
            assert parent != child
            assert relation in allowed
    # The helper that names the relation is not a causal claim: it never returns
    # a same-kind pair for this design, but the vocabulary allows one.
    assert all(relation != "node_to_node" for *_, relation in EXPECTED_STREAM_GRAPH)


def test_coordinate_metadata_covers_every_coordinate_in_canonical_order():
    for spec, motion_dim in ((GENO_SKELETON, 230), (SOMA_SKELETON, 248)):
        layout = layout_for(spec)
        metadata = layout.coordinate_metadata()
        assert len(metadata) == layout.num_coordinates == 40
        assert [record["coordinate"] for record in metadata] == list(range(40))
        slices = layout.stream_slices
        feature_indices = layout.feature_indices(motion_dim)
        per_stream_features = {}
        for record in metadata:
            stream = record["stream"]
            assert record["stream_slice"] == [slices[stream].start, slices[stream].stop]
            assert record["family"] == NEF_STREAM_FAMILY[stream]
            assert record["coordinate_in_stream"] == record["coordinate"] - slices[stream].start
            assert record["joints"] == [layout.names[joint] for joint in layout.stream_joints(stream)]
            assert record["feature_count"] == int(feature_indices[stream].numel())
            per_stream_features[stream] = record["feature_count"]
        assert per_stream_features == {
            stream: int(feature_indices[stream].numel()) for stream in NEF_STREAM_NAMES
        }
        assert sum(per_stream_features.values()) == motion_dim
        json.dumps(metadata)  # must stay JSON-serializable for checkpoints


def test_region_radius_zero_and_one_match_the_designed_parts():
    for spec in (GENO_SKELETON, SOMA_SKELETON):
        layout = layout_for(spec)
        for part, (node_stream, edge_stream) in NEF_EDIT_PARTS.items():
            assert layout.region_streams(part, graph_radius=0) == (node_stream,)
            assert layout.region_streams(part, graph_radius=1) == nef_edit_streams(part, full_part=True)
        assert layout.region_streams(NEF_WHOLE_BODY_REGION, graph_radius=0) == NEF_STREAM_NAMES
        # Radius 2 expands through the graph rather than inventing new streams.
        arm = layout.region_streams("left_arm", graph_radius=2)
        assert arm == ("torso_node", "left_arm_node", "left_shoulder_edge")
        assert set(arm) <= set(NEF_STREAM_NAMES)


def test_region_streams_rejects_unknown_regions_and_negative_radius():
    layout = layout_for(GENO_SKELETON)
    with pytest.raises(ValueError, match="Unknown NEF region"):
        layout.region_streams("left_hand")
    with pytest.raises(ValueError, match="graph_radius"):
        layout.region_streams("left_arm", graph_radius=-1)
    with pytest.raises(ValueError, match="at least one"):
        layout.region_streams([])


def test_region_mask_selects_exactly_the_owned_coordinates_and_frames():
    layout = layout_for(GENO_SKELETON)
    slices = layout.stream_slices
    strict = layout.make_region_mask(["left_arm"], graph_radius=0, length=8)
    assert strict.shape == (8, 40) and strict.dtype == torch.bool
    assert set(torch.nonzero(strict.any(0)).flatten().tolist()) == {10, 11, 12, 13}
    full = layout.make_region_mask(["left_arm"], graph_radius=1, length=8)
    assert set(torch.nonzero(full.any(0)).flatten().tolist()) == set(
        slices["left_arm_node"].start + offset for offset in range(4)
    ) | set(slices["left_shoulder_edge"].start + offset for offset in range(2))
    windowed = layout.make_region_mask(
        ["left_arm"], graph_radius=1, frame_range=(2, 5), length=8
    )
    assert torch.equal(windowed.any(-1), torch.tensor([False, False, True, True, True, False, False, False]))
    assert torch.equal(windowed.any(0), full.any(0))
    whole = layout.make_region_mask(NEF_WHOLE_BODY_REGION, length=4)
    assert bool(whole.all())
    left = layout.make_region_mask(["left_arm"], graph_radius=1, length=4)
    right = layout.make_region_mask(["right_arm"], graph_radius=1, length=4)
    assert not bool((left & right).any())
    assert torch.equal(left | right, layout.make_region_mask(["left_arm", "right_arm"], graph_radius=1, length=4))


def test_region_mask_validates_frame_range_length_and_device():
    layout = layout_for(GENO_SKELETON)
    with pytest.raises(ValueError, match="half-open"):
        layout.make_region_mask(["left_arm"], frame_range=(5, 5), length=8)
    with pytest.raises(ValueError, match="half-open"):
        layout.make_region_mask(["left_arm"], frame_range=(0, 9), length=8)
    with pytest.raises(ValueError, match="length must be positive"):
        layout.make_region_mask(["left_arm"], length=0)
    mask = layout.make_region_mask(["left_arm"], length=4, device=torch.device("cpu"))
    assert mask.device.type == "cpu"
    assert int(mask.sum()) == 4 * NEF_STREAM_COORDINATES["left_arm_node"]


def test_encode_and_decode_aliases_are_bitwise_identical_and_gradient_free():
    torch.manual_seed(7)
    model = model_for()
    motion = torch.randn(2, 64, model.motion_dim)
    canonical_indices = model.encode_to_indices(motion)
    alias_indices = model.encode_indices(motion)
    torch.testing.assert_close(alias_indices, canonical_indices, rtol=0.0, atol=0.0)
    assert alias_indices.requires_grad is False
    assert model.encode_indices(motion, lengths=torch.tensor([64, 64])).shape == alias_indices.shape
    canonical_recon = model.decode_from_indices(canonical_indices)
    alias_recon = model.decode_indices(canonical_indices)
    torch.testing.assert_close(alias_recon, canonical_recon, rtol=0.0, atol=0.0)
    assert alias_recon.requires_grad is False


def test_alias_lengths_are_validated_but_never_truncate():
    model = model_for()
    indices = torch.zeros(2, 64, 40, dtype=torch.long)
    with pytest.raises(ValueError, match="must have 2 entries"):
        model.decode_indices(indices, lengths=torch.tensor([64]))
    with pytest.raises(ValueError, match="padded frames stay"):
        model.decode_indices(indices, lengths=torch.tensor([64, 0]))
    with pytest.raises(ValueError, match="padded frames stay"):
        model.decode_indices(indices, lengths=torch.tensor([64, 65]))
    # A legal lengths vector changes nothing about the returned tensor.
    torch.testing.assert_close(
        model.decode_indices(indices, lengths=torch.tensor([64, 32])),
        model.decode_from_indices(indices),
        rtol=0.0,
        atol=0.0,
    )


def test_layout_accessor_is_read_only_and_adapter_persists_the_hash():
    model = model_for()
    assert model.get_token_layout() is model.layout
    config = {
        "representation": {
            "family": NEF_FSQ_FAMILY,
            "variant": "independent",
            "config": {
                "motion_dim": 230,
                "stream_dim": 16,
                "num_levels": 9,
                "names": list(model.layout.names),
                "parents": list(model.layout.parents),
            },
        }
    }
    adapter = build_representation(config)
    layout = adapter.token_layout()
    assert layout is not None and layout.skeleton == model.layout.skeleton
    assert layout.layout_hash() == GENO_LAYOUT_HASH
    metadata = adapter.representation_metadata()
    assert metadata["nef_layout_hash"] == GENO_LAYOUT_HASH
    assert metadata["nef_layout"] == layout.to_dict()
    assert metadata["representation_id"] == "nef_fsq_independent_40x9"


def test_flat_representation_has_no_token_layout():
    config = {
        "representation": {
            "family": "flat_fsq",
            "variant": "flat",
            "config": {"motion_dim": 230, "code_dim": 8, "width": 8},
        }
    }
    adapter = build_representation(config)
    assert adapter.token_layout() is None
