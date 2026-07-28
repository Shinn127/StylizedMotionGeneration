import torch

from models.latent_residual_part_fsq import LatentResidualPartFSQMotionAutoencoder
from models.part_layout import PART_NAMES
from models.residual_part_fsq import GROUP_NAMES


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


def _model(dropout: float = 0.0) -> LatentResidualPartFSQMotionAutoencoder:
    names, parents = _skeleton()
    return LatentResidualPartFSQMotionAutoencoder(
        names,
        parents,
        base_code_dim=32,
        base_width=32,
        part_state_dim=16,
        latent_fusion_hidden_dim=24,
        residual_group_dropout=dropout,
    )


def _activate_latent_residuals(model: LatentResidualPartFSQMotionAutoencoder) -> None:
    torch.manual_seed(101)
    for projector in model.latent_residual_fuse.values():
        torch.nn.init.normal_(projector[-1].weight, std=0.05)
        torch.nn.init.normal_(projector[-1].bias, std=0.01)


def test_latent_residual_part_fsq_keeps_the_40_coordinate_contract():
    model = _model()
    assert model.num_coordinates == 40
    assert tuple(model.group_slices) == GROUP_NAMES
    assert model.group_slices["base"] == slice(0, 20)
    assert model.group_slices["right_arm"] == slice(37, 40)
    assert not hasattr(model, "residual_decoder")
    assert model.decoder is model.base_decoder


def test_zero_initialized_latent_fusion_starts_from_the_base_reconstruction():
    torch.manual_seed(3)
    model = _model().eval()
    motion = torch.randn(2, 64, model.motion_dim)
    with torch.no_grad():
        output = model(motion, decode_base=True)

    torch.testing.assert_close(output["recon_state"], output["base_recon_state"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output["latent_residual_energy"], torch.zeros(()), rtol=0.0, atol=0.0)
    for part in PART_NAMES:
        torch.testing.assert_close(
            output["part_latent_residuals"][part],
            torch.zeros_like(output["part_latent_residuals"][part]),
            rtol=0.0,
            atol=0.0,
        )


def test_codes_and_indices_roundtrip_with_one_inference_decoder_pass():
    torch.manual_seed(5)
    model = _model().eval()
    _activate_latent_residuals(model)
    motion = torch.randn(2, 64, model.motion_dim)
    decoder_batches: list[int] = []
    hook = model.decoder.register_forward_hook(
        lambda _module, inputs, _output: decoder_batches.append(int(inputs[0].shape[0]))
    )
    try:
        with torch.no_grad():
            output = model(motion, collect_metrics=True)
            assert decoder_batches == [2]
            decoder_batches.clear()
            from_indices = model.decode_from_indices(output["indices"])
            assert decoder_batches == [2]
            decoder_batches.clear()
            from_codes = model.decode_from_codes(output["fsq_codes"])
            assert decoder_batches == [2]
    finally:
        hook.remove()

    assert output["indices"].shape == (2, 64, 40)
    assert output["fsq_codes"].shape == (2, 64, 40)
    assert output["group_coordinate_change_rates"].shape == (len(GROUP_NAMES),)
    torch.testing.assert_close(from_indices, output["recon_state"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(from_codes, output["recon_state"], rtol=0.0, atol=0.0)


def test_training_base_and_final_reconstruction_share_one_batched_decoder_call():
    torch.manual_seed(7)
    model = _model().eval()
    _activate_latent_residuals(model)
    motion = torch.randn(3, 64, model.motion_dim)
    decoder_batches: list[int] = []
    hook = model.decoder.register_forward_hook(
        lambda _module, inputs, _output: decoder_batches.append(int(inputs[0].shape[0]))
    )
    try:
        with torch.no_grad():
            output = model(motion, decode_base=True)
    finally:
        hook.remove()

    assert decoder_batches == [6]
    assert output["base_recon_state"].shape == motion.shape
    assert not torch.equal(output["recon_state"], output["base_recon_state"])


def test_part_index_edit_keeps_base_fixed_but_can_change_the_holistic_decode():
    torch.manual_seed(11)
    model = _model().eval()
    _activate_latent_residuals(model)
    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        indices = model.encode_to_indices(motion)
        edited_indices = indices.clone()
        part_slice = model.group_slices["left_leg"]
        edited_indices[..., part_slice] = 8 - edited_indices[..., part_slice]
        source_base = model.decode_base_from_indices(indices)
        edited_base = model.decode_base_from_indices(edited_indices)
        source = model.decode_from_indices(indices)
        edited = model.decode_from_indices(edited_indices)

    torch.testing.assert_close(edited_base, source_base, rtol=0.0, atol=0.0)
    assert not torch.equal(edited, source)
    assert torch.isfinite((edited - source).abs().mean())


def test_latent_residual_part_fsq_is_strictly_causal_with_rf64():
    torch.manual_seed(13)
    model = _model().eval()
    _activate_latent_residuals(model)
    motion = torch.randn(1, 96, model.motion_dim)
    changed = motion.clone()
    changed[:, 0] += 100.0
    changed[:, 65:] += 100.0
    with torch.no_grad():
        expected = model(motion)["recon_state"][:, 64]
        actual = model(changed)["recon_state"][:, 64]

    assert (model.receptive_field, model.context_left, model.lookahead_frames) == (64, 63, 0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_all_fsq_groups_keep_straight_through_gradients():
    torch.manual_seed(17)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim, requires_grad=True)
    output = model(motion)
    output["fsq_codes"].sum().backward()
    assert motion.grad is not None
    assert torch.count_nonzero(motion.grad) > 0
