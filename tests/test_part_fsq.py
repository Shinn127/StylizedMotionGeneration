import torch

from datasets.feature_dataset import _fixed_windows_from_intervals
from models.part_fsq import HierarchicalPartFSQMotionAutoencoder
from models.part_fsq_losses import adaptive_part_fsq_reuse_loss
from models.part_layout import GROUP_NAMES, PartFSQLayout


def _skeleton() -> tuple[list[str], list[int]]:
    names = [
        "Simulation",
        "Hips",
        "Spine",
        "LeftUpLeg",
        "LeftToeBase",
        "RightUpLeg",
        "RightToeBase",
        "LeftShoulder",
        "LeftHand",
        "RightShoulder",
        "RightHand",
    ]
    parents = [-1, 0, 1, 1, 3, 1, 5, 2, 7, 2, 9]
    return names, parents


def _model() -> HierarchicalPartFSQMotionAutoencoder:
    names, parents = _skeleton()
    return HierarchicalPartFSQMotionAutoencoder(names, parents, stream_dim=16)


def test_interval_metadata_adapter_keeps_the_old_fixed_64_frame_contract():
    intervals = torch.tensor([[3, 10, 154, 7]], dtype=torch.int32).numpy()
    windows = _fixed_windows_from_intervals(intervals, window_size=64)

    assert [(window.shard_idx, window.start_idx, window.end_idx, window.range_idx) for window in windows] == [
        (3, 10, 74, 7),
        (3, 74, 138, 7),
        (3, 90, 154, 7),
    ]


def test_part_layout_partitions_current_motion_features_and_exposes_40_coordinates():
    names, parents = _skeleton()
    layout = PartFSQLayout.from_skeleton(names, parents)

    assert layout.num_joints == len(names)
    assert layout.num_coordinates == 40
    assert layout.hips_index == 1
    assert layout.toe_indices == (4, 6)
    assert layout.group_slices["global"] == slice(0, 6)
    assert layout.group_slices["sync"] == slice(6, 10)
    assert layout.group_slices["right_arm"] == slice(35, 40)

    feature_indices = layout.feature_indices(9 * len(names) + 5)
    assert feature_indices["global"].numel() == 14
    assert feature_indices["left_leg"].numel() == 18
    assert feature_indices["right_arm"].numel() == 18
    all_features = torch.cat([feature_indices[group] for group in ("global", "torso", "left_leg", "right_leg", "left_arm", "right_arm")])
    torch.testing.assert_close(torch.sort(all_features).values, torch.arange(9 * len(names) + 5))


def test_part_fsq_roundtrips_codes_and_indices_without_statistics_by_default():
    torch.manual_seed(13)
    model = _model().eval()
    motion = torch.randn(2, 64, model.motion_dim)
    with torch.no_grad():
        output = model(motion)
        from_indices = model.decode_from_indices(output["indices"])
        from_codes = model.decode_from_codes(output["fsq_codes"])

    assert output["recon_state"].shape == motion.shape
    assert output["fsq_codes"].shape == (2, 64, 40)
    assert output["indices"].shape == (2, 64, 40)
    assert output["indices"].dtype == torch.long
    assert "tuple_unique_ratio" not in output
    assert int(output["indices"].min()) >= 0
    assert int(output["indices"].max()) <= 8
    assert tuple(output["group_codes"]) == GROUP_NAMES
    assert model.group_quantizers["leg"] is model.group_quantizers[model._quantizer_key("right_leg")]
    assert model.group_quantizers["arm"] is model.group_quantizers[model._quantizer_key("right_arm")]
    torch.testing.assert_close(from_indices, output["recon_state"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(from_codes, output["recon_state"], rtol=0.0, atol=0.0)


def test_part_fsq_statistics_are_available_on_request_and_codes_have_ste_gradients():
    torch.manual_seed(23)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim, requires_grad=True)
    output = model(motion, collect_metrics=True)

    assert output["group_coordinate_change_rates"].shape == (len(GROUP_NAMES),)
    output["fsq_codes"].sum().backward()
    assert motion.grad is not None
    assert torch.count_nonzero(motion.grad) > 0


def test_part_fsq_is_strictly_causal_with_a_64_frame_receptive_field():
    torch.manual_seed(17)
    model = _model().eval()
    motion = torch.randn(1, 96, model.motion_dim)
    changed = motion.clone()
    changed[:, 0] += 100.0
    changed[:, 65:] += 100.0
    with torch.no_grad():
        expected = model(motion)["recon_state"][:, 64]
        actual = model(changed)["recon_state"][:, 64]

    assert (model.receptive_field, model.context_left, model.lookahead_frames) == (64, 63, 0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_left_leg_code_has_no_direct_readout_path_to_other_parts():
    torch.manual_seed(29)
    model = _model().eval()
    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        indices = model.encode_to_indices(motion)
        changed = indices.clone()
        left_leg_codes = model.layout.group_slices["left_leg"]
        changed[..., left_leg_codes] = 8 - changed[..., left_leg_codes]
        expected = model.decode_from_indices(indices)
        actual = model.decode_from_indices(changed)

    feature_indices = model.layout.feature_indices(model.motion_dim)
    for group in ("global", "torso", "right_leg", "left_arm", "right_arm"):
        torch.testing.assert_close(actual[..., feature_indices[group]], expected[..., feature_indices[group]], rtol=0.0, atol=0.0)
    assert not torch.equal(actual[..., feature_indices["left_leg"]], expected[..., feature_indices["left_leg"]])


def test_reuse_disables_global_sync_and_the_transitioning_leg_only():
    model = _model()
    motion = torch.zeros(1, 3, model.motion_dim)
    motion[:, 1:, -2] = 1.0
    codes = torch.zeros(1, 3, model.num_coordinates)
    codes[:, 1:] = 1.0

    losses = adaptive_part_fsq_reuse_loss(codes, motion, model.layout, thresholds=1.0)
    group_losses = dict(zip(GROUP_NAMES, losses.group_losses.tolist()))
    group_gates = dict(zip(GROUP_NAMES, losses.group_gate_mean.tolist()))

    assert group_losses["global"] == 0.0
    assert group_losses["sync"] == 0.0
    assert group_losses["left_leg"] == 0.0
    assert group_gates["global"] < group_gates["right_leg"]
    assert group_gates["sync"] < group_gates["right_leg"]
    assert group_gates["left_leg"] < group_gates["right_leg"]
    assert group_losses["right_leg"] > 0.0
    assert group_losses["torso"] > 0.0
