import torch

from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.residual_part_fsq import GROUP_NAMES, ResidualPartFSQMotionAutoencoder


def _skeleton():
    names = [
        "Simulation", "Hips", "Spine", "LeftUpLeg", "LeftToeBase", "RightUpLeg", "RightToeBase",
        "LeftShoulder", "LeftHand", "RightShoulder", "RightHand",
    ]
    parents = [-1, 0, 1, 1, 3, 1, 5, 2, 7, 2, 9]
    return names, parents


def _model(dropout: float = 0.0):
    names, parents = _skeleton()
    return ResidualPartFSQMotionAutoencoder(
        names,
        parents,
        base_code_dim=32,
        base_width=32,
        part_state_dim=16,
        residual_decoder_dim=32,
        residual_decoder_width=32,
        residual_hidden_dim=16,
        residual_group_dropout=dropout,
    )


def test_residual_part_fsq_exposes_fixed_40_coordinate_layout():
    model = _model()
    assert model.num_coordinates == 40
    assert model.group_slices["base"] == slice(0, 20)
    assert model.group_slices["torso"] == slice(20, 26)
    assert model.group_slices["right_arm"] == slice(37, 40)
    assert tuple(model.group_slices) == GROUP_NAMES
    assert model.part_quantizers["leg"] is model._quantizer_for_group("right_leg")
    assert model.part_quantizers["arm"] is model._quantizer_for_group("right_arm")


def test_zero_initialized_residual_heads_start_from_complete_base_reconstruction():
    torch.manual_seed(3)
    model = _model().eval()
    motion = torch.randn(2, 64, model.motion_dim)
    with torch.no_grad():
        output = model(motion)
    torch.testing.assert_close(output["recon_state"], output["base_recon_state"], rtol=0.0, atol=0.0)
    for part in PART_NAMES:
        torch.testing.assert_close(output["part_residuals"][part], torch.zeros_like(output["part_residuals"][part]))


def test_codes_and_indices_roundtrip_without_source_motion():
    torch.manual_seed(5)
    model = _model().eval()
    motion = torch.randn(2, 64, model.motion_dim)
    with torch.no_grad():
        output = model(motion, collect_metrics=True)
        from_indices = model.decode_from_indices(output["indices"])
        from_codes = model.decode_from_codes(output["fsq_codes"])
    assert output["indices"].shape == (2, 64, 40)
    assert output["fsq_codes"].shape == (2, 64, 40)
    assert output["group_coordinate_change_rates"].shape == (len(GROUP_NAMES),)
    torch.testing.assert_close(from_indices, output["recon_state"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(from_codes, output["recon_state"], rtol=0.0, atol=0.0)


def test_all_fsq_groups_receive_ste_gradients():
    torch.manual_seed(7)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim, requires_grad=True)
    output = model(motion)
    output["fsq_codes"].sum().backward()
    assert motion.grad is not None
    assert torch.count_nonzero(motion.grad) > 0


def test_residual_part_fsq_is_strictly_causal_with_rf64():
    torch.manual_seed(11)
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


def test_part_index_change_only_writes_its_anatomical_features():
    torch.manual_seed(13)
    model = _model().eval()
    torch.nn.init.normal_(model.residual_output_heads["left_leg"].weight, std=0.1)
    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        indices = model.encode_to_indices(motion)
        changed = indices.clone()
        group_slice = model.group_slices["left_leg"]
        changed[..., group_slice] = 8 - changed[..., group_slice]
        expected = model.decode_from_indices(indices)
        actual = model.decode_from_indices(changed)
    feature_indices = model.layout.feature_indices(model.motion_dim)
    for group in ("global", "torso", "right_leg", "left_arm", "right_arm"):
        torch.testing.assert_close(actual[..., feature_indices[group]], expected[..., feature_indices[group]], rtol=0.0, atol=0.0)
    assert not torch.equal(actual[..., feature_indices["left_leg"]], expected[..., feature_indices["left_leg"]])


def test_representation_specific_base_loss_is_exposed_through_model_hook():
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim)
    output = model(motion)
    losses = model.compute_representation_losses(output, {"motion": motion})
    assert "base_reuse" in losses
    assert losses["base_reuse"].ndim == 0
