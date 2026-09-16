"""NEF level-geometry, locality and temporal-influence probes."""

from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from stylized_motion.learning.nef_fsq import NEFMotionAutoencoder
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout
from stylized_motion.learning.nef_probe import (
    CSV_COLUMNS,
    KinematicContext,
    LevelGeometryProbe,
    adjacent_perturbations,
    far_perturbations,
    json_dumps,
    kinematic_descendants,
    locality_report,
    probe_csv_rows,
    read_probe_window,
    temporal_influence_width,
    write_probe_csv,
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


def geno_layout() -> NEFLayout:
    names, parents = skeleton_from_spec(GENO_SKELETON)
    return NEFLayout.from_skeleton(names, parents)


def geno_model(*, stream_dim: int = 16, seed: int = 5) -> NEFMotionAutoencoder:
    torch.manual_seed(seed)
    names, parents = skeleton_from_spec(GENO_SKELETON)
    return NEFMotionAutoencoder(names, parents, stream_dim=stream_dim).eval()


def feature_stats(model: NEFMotionAutoencoder) -> dict[str, object]:
    motion_dim = model.motion_dim
    return {
        "offset": np.zeros(motion_dim, dtype=np.float32),
        "scale": np.full(motion_dim, 0.3, dtype=np.float32),
        "ref_pos": np.tile(
            np.array([0.0, 0.9, 0.0], dtype=np.float32), (model.layout.num_joints, 1)
        ),
        "parents": np.asarray(model.layout.parents),
        "names": list(model.layout.names),
    }


class _AnalyticDecoder(nn.Module):
    """Decoder whose output is a fixed linear map of the coordinate levels.

    Every coordinate owns a constant weight on its own stream's features, so the
    decoded distance of a level shift is analytically known.
    """

    def __init__(self, layout: NEFLayout, motion_dim: int) -> None:
        super().__init__()
        self.layout = layout
        self.num_levels = 9
        self.motion_dim = int(motion_dim)
        feature_indices = layout.feature_indices(self.motion_dim)
        weights = torch.zeros(layout.num_coordinates, self.motion_dim)
        for coordinate, record in enumerate(layout.coordinate_metadata()):
            stream = str(record["stream"])
            weight = 1.0 + 0.01 * coordinate
            weights[coordinate, feature_indices[stream]] = weight
        self.register_buffer("weights", weights)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.float() @ self.weights


def test_adjacent_perturbations_stay_legal_and_flag_the_boundary():
    indices = torch.tensor([[[0, 4, 8, 3]]], dtype=torch.long)
    plus, minus = adjacent_perturbations(indices, 1, num_levels=9)
    assert plus.tokens[0, 0].tolist() == [0, 5, 8, 3]
    assert bool(plus.valid.all()) and float(plus.valid.float().mean()) == 1.0
    assert minus.tokens[0, 0].tolist() == [0, 3, 8, 3]

    _, low = adjacent_perturbations(indices, 0, num_levels=9)
    assert low.tokens[0, 0, 0].item() == 0  # clamped to a legal level
    assert bool(low.valid.item()) is False
    high, _ = adjacent_perturbations(indices, 2, num_levels=9)
    assert high.tokens[0, 0, 2].item() == 8
    assert bool(high.valid.item()) is False
    with pytest.raises(ValueError, match="outside"):
        adjacent_perturbations(torch.full((1, 1, 2), 9), 0, num_levels=9)


def test_far_perturbations_respect_the_minimum_distance():
    generator = torch.Generator().manual_seed(11)
    indices = torch.full((2, 5, 3), 4, dtype=torch.long)
    far = far_perturbations(
        indices, 0, num_levels=9, min_distance=4, samples=8, generator=generator
    )
    assert len(far) == 8
    for perturbation in far:
        # Every frame gets its own far jump; each one must clear the distance,
        # and frames whose jump would leave the level range are flagged invalid.
        assert bool((perturbation.offsets.abs() >= 4).all())
        shifted = indices[..., 0] + perturbation.offsets
        assert torch.equal(perturbation.valid, (shifted >= 0) & (shifted <= 8))
        assert perturbation.valid.any()
    with pytest.raises(ValueError, match="min_distance"):
        far_perturbations(indices, 0, num_levels=9, min_distance=9)
    with pytest.raises(ValueError, match="samples"):
        far_perturbations(indices, 0, num_levels=9, samples=0)


def test_level_probe_recovers_analytic_geometry_of_a_linear_decoder():
    layout = geno_layout()
    model = _AnalyticDecoder(layout, 9 * layout.num_joints + 5)
    probe = LevelGeometryProbe(model, far_samples=4, far_min_distance=4)
    indices = torch.full((2, 32, 40), 4, dtype=torch.long)
    report = probe.run(indices, generator=torch.Generator().manual_seed(3))
    assert report["summary"]["coordinates"] == 40
    assert report["summary"]["ordinal_geometry_supported"] is True
    for record in report["per_coordinate"]:
        coordinate = record["coordinate"]
        weight = 1.0 + 0.01 * coordinate
        features = int(layout.feature_indices(model.motion_dim)[record["stream"]].numel())
        # feature_l1 is a mean over all 230 features, not a sum:
        # |offset| * weight * (stream feature count / motion_dim).
        assert record["adjacent_distance"] == pytest.approx(
            weight * features / model.motion_dim, rel=1e-5
        )
        assert record["far_distance"] > record["adjacent_distance"] * 3.9
        assert record["adjacent_to_far_ratio"] < 0.26
        assert record["direction_consistency"] == pytest.approx(1.0, abs=1e-6)
        assert record["adjacent"]["offtarget_feature_max"] == pytest.approx(0.0)
    assert report["summary"]["mean_direction_consistency"] == pytest.approx(1.0, abs=1e-6)
    assert report["summary"]["kinematics"] is False


def test_level_probe_reports_kinematics_and_rejects_bad_inputs():
    torch.manual_seed(13)
    model = geno_model()
    kinematic = KinematicContext.from_feature_stats(feature_stats(model))
    probe = LevelGeometryProbe(model, kinematic=kinematic, far_samples=2)
    indices = torch.randint(0, 9, (1, 64, 40), generator=torch.Generator().manual_seed(17))
    report = probe.run(indices, coordinates=[0, 10], generator=torch.Generator().manual_seed(5))
    assert [record["coordinate"] for record in report["per_coordinate"]] == [0, 10]
    assert report["summary"]["kinematics"] is True
    for record in report["per_coordinate"]:
        for kind in ("adjacent", "far"):
            assert "fk_owned_mean" in record[kind]
            assert "fk_influence_mean" in record[kind]
            assert "owns_joints" in record[kind]
            assert "contact_flip_rate" in record[kind]
            assert record[kind]["root_pos_change"] >= 0.0
    # A local stream never moves joints it neither owns nor descends from.
    arm = report["per_coordinate"][1]
    assert arm["stream"] == "left_arm_node"
    assert arm["adjacent"]["owns_joints"] == 1.0
    assert arm["adjacent"]["fk_offtarget_max"] == pytest.approx(0.0, abs=0.0)
    assert arm["adjacent"]["fk_owned_mean"] > 0.0
    # The global stream owns no bones: a root/contact edit moves the whole body,
    # so its influence is reported over every joint and no leakage is claimed.
    global_record = report["per_coordinate"][0]
    assert global_record["stream"] == "global"
    assert global_record["adjacent"]["owns_joints"] == 0.0
    assert global_record["adjacent"]["fk_influence_mean"] > 0.0
    assert global_record["adjacent"]["fk_offtarget_max"] == 0.0
    with pytest.raises(ValueError, match="outside the token width"):
        probe.run(indices, coordinates=[40])
    with pytest.raises(ValueError, match="Expected tokens"):
        probe.run(indices[:, :, :39])
    with pytest.raises(ValueError, match="valid_mask"):
        probe.run(indices, valid_mask=torch.ones(2, 64, dtype=torch.bool))


def test_locality_report_keeps_stream_swaps_inside_their_own_features():
    torch.manual_seed(19)
    model = geno_model()
    layout = model.layout
    feature_indices = layout.feature_indices(model.motion_dim)
    streams = ("left_arm_node", "left_shoulder_edge")
    support = torch.cat([feature_indices[stream] for stream in streams]).tolist()
    indices = torch.randint(0, 9, (2, 64, 40), generator=torch.Generator().manual_seed(23))
    donor = torch.randint(0, 9, indices.shape, generator=torch.Generator().manual_seed(29))
    report = locality_report(
        model,
        indices,
        donor,
        slices=[layout.stream_slices[stream] for stream in streams],
        start=16,
        stop=32,
        target_joints=layout.stream_joints("left_arm_node"),
        feature_support=support,
        kinematic=KinematicContext.from_feature_stats(feature_stats(model)),
    )
    assert report["support_coordinates"] == 6
    assert report["support_fraction"] == pytest.approx(6 / 40)
    assert report["support_token_change_fraction"] > 0.0
    assert report["off_target_feature_max"] == pytest.approx(0.0, abs=0.0)
    assert report["off_target_feature_mean"] == pytest.approx(0.0, abs=0.0)
    assert report["edit_feature_mean"] > 0.0
    assert report["pre_edit_unchanged"] is True
    assert report["kinematics"]["non_target_joint_change_max"] == pytest.approx(0.0, abs=0.0)
    assert report["kinematics"]["target_joint_change"] > 0.0
    assert report["kinematics"]["contact_flip_rate"] == pytest.approx(0.0, abs=0.0)
    assert report["changed_frame_span"][0] == 16

    whole_body = locality_report(
        model,
        indices,
        donor,
        slices=[slice(0, 40)],
        start=16,
        stop=32,
        target_joints=layout.stream_joints("left_arm_node"),
        feature_support=list(range(model.motion_dim)),
    )
    assert whole_body["support_fraction"] == 1.0
    assert whole_body["off_target_feature_max"] == pytest.approx(0.0, abs=0.0)

    # An empty target-joint set means "the whole body is the target": every
    # joint is measured as a target and nothing is reported as off-target.
    all_joints = locality_report(
        model,
        indices,
        donor,
        slices=[slice(0, 40)],
        start=16,
        stop=32,
        target_joints=[],
        feature_support=list(range(model.motion_dim)),
        kinematic=KinematicContext.from_feature_stats(feature_stats(model)),
    )
    kinematics = all_joints["kinematics"]
    assert kinematics["target_joint_change"] > 0.0
    assert kinematics["non_target_joints"] == []
    assert kinematics["non_target_joint_change_max"] == pytest.approx(0.0, abs=0.0)

    with pytest.raises(ValueError, match="edit interval"):
        locality_report(
            model, indices, donor, slices=[slice(0, 4)], start=32, stop=16, target_joints=[2]
        )


def test_temporal_influence_width_matches_the_causal_decoder_contract():
    torch.manual_seed(31)
    model = geno_model()
    indices = torch.randint(0, 9, (1, 96, 40), generator=torch.Generator().manual_seed(37))
    report = temporal_influence_width(model, indices, stream="left_arm_node", frame=40)
    assert report["first_changed_frame"] == 40
    assert report["frames_before"] == 0
    assert report["frames_after"] == model.decoder_receptive_field - 1 == 33
    assert report["within_contract"] is True
    clamped = temporal_influence_width(
        model, indices, stream="global", frame=40, replacement=0
    )
    assert clamped["frames_after"] <= model.decoder_receptive_field - 1
    with pytest.raises(ValueError, match="frame must be"):
        temporal_influence_width(model, indices, stream="global", frame=96)
    with pytest.raises(ValueError, match="replacement"):
        temporal_influence_width(model, indices, stream="global", frame=0, replacement=9)


def test_kinematic_context_and_descendants():
    model = geno_model()
    kinematic = KinematicContext.from_feature_stats(feature_stats(model), dt=1.0 / 30.0)
    assert kinematic.toe_indices == (
        model.layout.names.index("LeftToeBase"),
        model.layout.names.index("RightToeBase"),
    )
    assert kinematic.dt == pytest.approx(1.0 / 30.0)
    assert kinematic.to(torch.device("cpu")).names == kinematic.names
    stats = dict(feature_stats(model))
    stats["names"] = ["Simulation", "Hips"]
    assert KinematicContext.from_feature_stats(stats).toe_indices is None

    parents = model.layout.parents
    head = model.layout.names.index("Head")
    left_arm = model.layout.names.index("LeftArm")
    assert kinematic_descendants(parents, [left_arm]) == {
        model.layout.names.index("LeftForeArm"),
        model.layout.names.index("LeftHand"),
    }
    assert kinematic_descendants(parents, [head]) == set()


def test_probe_reports_are_json_and_csv_serializable(tmp_path: Path):
    model = geno_model()
    probe = LevelGeometryProbe(model, far_samples=1, far_min_distance=4, decode_rows=16)
    indices = torch.randint(0, 9, (1, 16, 40), generator=torch.Generator().manual_seed(41))
    report = probe.run(indices, coordinates=[0, 1, 10], generator=torch.Generator().manual_seed(43))
    text = json_dumps(report)
    reloaded = json.loads(text)
    assert reloaded["summary"]["coordinates"] == 3
    rows = probe_csv_rows(report)
    assert list(rows[0]) == list(CSV_COLUMNS)
    path = write_probe_csv(tmp_path / "probe.csv", report)
    with path.open(encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert len(read) == 3
    assert read[2]["stream"] == "left_arm_node"
    assert float(read[0]["adjacent_distance"]) >= 0.0


class _PackedStoreStub:
    """Minimal schema-v4 reader surface used by read_probe_window."""

    def __init__(self, path: Path, clips: list[tuple[int, int, int]]) -> None:
        self.shard_files = [path]
        self.clip_shard = np.asarray([clip[0] for clip in clips], dtype=np.int32)
        self.clip_offset = np.asarray([clip[1] for clip in clips], dtype=np.int64)
        self.clip_length = np.asarray([clip[2] for clip in clips], dtype=np.int64)

    def read_frames(self, shard_idx: int, start: int, frames: int) -> np.ndarray:
        values = np.load(self.shard_files[shard_idx], mmap_mode="r", allow_pickle=False)
        return np.ascontiguousarray(values[start : start + frames], dtype=np.float32)


def test_read_probe_window_pads_at_the_clip_start_and_rejects_overruns(tmp_path: Path):
    shard = np.arange(24, dtype=np.float32).reshape(12, 2)
    path = tmp_path / "shard_00000.npy"
    np.save(path, shard)
    store = _PackedStoreStub(path, clips=[(0, 4, 8)])

    request = types.SimpleNamespace(variant_idx=0, target_start=4, target_frames=4)
    window = read_probe_window(store, request, history=3)
    assert window.shape == (7, 2)
    # Three padded frames copy the clip's first stored frame, then the clip runs.
    assert window[:3].tolist() == [shard[4].tolist()] * 3
    assert window[3:].tolist() == shard[4:8].tolist()

    request = types.SimpleNamespace(variant_idx=0, target_start=6, target_frames=4)
    window = read_probe_window(store, request, history=3)
    assert window.shape == (7, 2)
    # Padding never reaches outside the clip, so it repeats the clip's first frame.
    assert window[:1].tolist() == [shard[4].tolist()]
    assert window[1:].tolist() == shard[4:10].tolist()

    shards: dict[int, np.ndarray] = {}
    cached = read_probe_window(store, request, history=3, shards=shards)
    assert 0 in shards and np.array_equal(cached, window)

    bad = types.SimpleNamespace(variant_idx=0, target_start=9, target_frames=4)
    with pytest.raises(IndexError, match="leaves clip"):
        read_probe_window(store, bad, history=3)
    with pytest.raises(IndexError, match="Invalid clip index"):
        read_probe_window(store, types.SimpleNamespace(variant_idx=5, target_start=4, target_frames=4), history=0)
