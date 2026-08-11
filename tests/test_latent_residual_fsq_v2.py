import torch

from stylized_motion.learning.latent_residual_fsq_v2 import LatentResidualPartFSQV2MotionAutoencoder
from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.residual_part_fsq import GROUP_NAMES


def _skeleton():
    names = [
        "Simulation", "Hips", "Spine", "LeftUpLeg", "LeftToeBase", "RightUpLeg",
        "RightToeBase", "LeftShoulder", "LeftHand", "RightShoulder", "RightHand",
    ]
    parents = [-1, 0, 1, 1, 3, 1, 5, 2, 7, 2, 9]
    return names, parents


def _model():
    names, parents = _skeleton()
    return LatentResidualPartFSQV2MotionAutoencoder(
        names, parents, base_code_dim=32, base_width=32, part_state_dim=16,
        part_predictor_hidden_dim=24, part_encoder_width=24,
    )


def test_v2_is_independent_and_uses_dense_full_latent_projectors():
    model = _model()
    assert model.num_coordinates == 40
    assert tuple(model.group_slices) == GROUP_NAMES
    assert not hasattr(model, "part_latent_slices")
    assert not hasattr(model, "part_latent_dims")
    for part in PART_NAMES:
        projector = model.latent_residual_projectors[part]
        assert projector.in_features == model.part_state_dim
        assert projector.out_features == model.base_code_dim
    assert model._part_quantizer("left_leg") is model._part_quantizer("right_leg")
    assert model._part_quantizer("left_arm") is model._part_quantizer("right_arm")


def test_v2_roundtrips_codes_and_indices_with_one_decoder_pass():
    torch.manual_seed(5)
    model = _model().eval()
    motion = torch.randn(2, 64, model.motion_dim)
    calls = []
    hook = model.decoder.register_forward_hook(lambda _m, inputs, _out: calls.append(inputs[0].shape[0]))
    try:
        with torch.no_grad():
            output = model(motion)
            assert calls == [2]
            calls.clear()
            from_indices = model.decode_from_indices(output["indices"])
            assert calls == [2]
            calls.clear()
            from_codes = model.decode_from_codes(output["codes"])
            assert calls == [2]
    finally:
        hook.remove()
    assert output["indices"].shape == (2, 64, 40)
    torch.testing.assert_close(from_indices, output["recon_state"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(from_codes, output["recon_state"], rtol=0.0, atol=0.0)


def test_v2_training_variants_share_one_batched_decoder_call():
    model = _model().eval()
    motion = torch.randn(3, 64, model.motion_dim)
    calls = []
    hook = model.decoder.register_forward_hook(lambda _m, inputs, _out: calls.append(inputs[0].shape[0]))
    try:
        with torch.no_grad():
            output = model(
                motion, decode_base=True, edit_part="left_leg",
                donor_permutation=torch.tensor([1, 2, 0]),
            )
    finally:
        hook.remove()
    assert calls == [9]
    assert output["base_recon_state"].shape == motion.shape
    assert output["edit_recon_state"].shape == motion.shape


def test_v2_compensated_index_edit_keeps_target_base_codes_fixed():
    model = _model().eval()
    target = torch.randn(2, 64, model.motion_dim)
    donor = torch.randn(2, 64, model.motion_dim)
    with torch.no_grad():
        target_indices = model.encode_to_indices(target)
        donor_indices = model.encode_to_indices(donor)
        base_before = model.decode_base_from_indices(target_indices)
        edited = model.decode_from_indices_with_part_edit(target_indices, donor_indices, "left_arm")
        base_after = model.decode_base_from_indices(target_indices)
    assert edited.shape == target.shape
    torch.testing.assert_close(base_after, base_before, rtol=0.0, atol=0.0)


def test_v2_is_strictly_causal_with_rf64():
    model = _model().eval()
    motion = torch.randn(1, 96, model.motion_dim)
    changed = motion.clone()
    changed[:, 0] += 100.0
    changed[:, 65:] += 100.0
    with torch.no_grad():
        expected = model(motion)["recon_state"][:, 64]
        actual = model(changed)["recon_state"][:, 64]
    assert (model.receptive_field, model.lookahead_frames) == (64, 0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
