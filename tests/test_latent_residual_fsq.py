import torch

from stylized_motion.learning.latent_residual_fsq import LatentResidualPartFSQMotionAutoencoder
from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.residual_part_fsq import GROUP_NAMES


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


def _model() -> LatentResidualPartFSQMotionAutoencoder:
    names, parents = _skeleton()
    return LatentResidualPartFSQMotionAutoencoder(
        names,
        parents,
        base_code_dim=32,
        base_width=32,
        part_state_dim=16,
        part_predictor_hidden_dim=24,
        latent_projector_hidden_dim=24,
        part_latent_dims=(10, 6, 6, 5, 5),
    )


def test_latent_residual_fsq_keeps_the_40_coordinate_contract():
    model = _model()
    assert model.num_coordinates == 40
    assert tuple(model.group_slices) == GROUP_NAMES
    assert model.group_slices["base"] == slice(0, 20)
    assert model.group_slices["right_arm"] == slice(37, 40)
    assert model.part_latent_slices == {
        "torso": slice(0, 10),
        "left_leg": slice(10, 16),
        "right_leg": slice(16, 22),
        "left_arm": slice(22, 27),
        "right_arm": slice(27, 32),
    }
    assert not hasattr(model, "residual_decoder")
    assert model.decoder is model.base_decoder


def test_part_latent_residuals_have_disjoint_support():
    torch.manual_seed(3)
    model = _model().eval()
    motion = torch.randn(2, 64, model.motion_dim)
    with torch.no_grad():
        output = model(motion)

    for part in PART_NAMES:
        residual = output["part_latent_residuals"][part]
        part_slice = model.part_latent_slices[part]
        outside = torch.ones(model.base_code_dim, dtype=torch.bool)
        outside[part_slice] = False
        torch.testing.assert_close(residual[..., outside], torch.zeros_like(residual[..., outside]))
        assert torch.count_nonzero(residual[..., part_slice]) > 0


def test_codes_and_indices_roundtrip_with_one_inference_decoder_pass():
    torch.manual_seed(5)
    model = _model().eval()
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


def test_training_variants_share_one_batched_decoder_call():
    torch.manual_seed(7)
    model = _model().eval()
    motion = torch.randn(3, 64, model.motion_dim)
    permutation = torch.tensor([1, 2, 0])
    decoder_batches: list[int] = []
    hook = model.decoder.register_forward_hook(
        lambda _module, inputs, _output: decoder_batches.append(int(inputs[0].shape[0]))
    )
    try:
        with torch.no_grad():
            output = model(
                motion,
                decode_base=True,
                edit_part="left_leg",
                donor_permutation=permutation,
            )
    finally:
        hook.remove()

    assert decoder_batches == [9]
    assert output["base_recon_state"].shape == motion.shape
    assert output["recon_state"].shape == motion.shape
    assert output["edit_recon_state"].shape == motion.shape
    assert not torch.equal(output["recon_state"], output["base_recon_state"])


def test_part_index_edit_changes_only_its_latent_subspace_and_keeps_base_fixed():
    torch.manual_seed(11)
    model = _model().eval()
    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        indices = model.encode_to_indices(motion)
        edited_indices = indices.clone()
        group_slice = model.group_slices["left_leg"]
        edited_indices[..., group_slice] = 8 - edited_indices[..., group_slice]
        source_embeddings = model._decode_indices_to_embeddings(indices)
        edited_embeddings = model._decode_indices_to_embeddings(edited_indices)
        source_fused, _, _ = model._fuse_embeddings(source_embeddings)
        edited_fused, _, _ = model._fuse_embeddings(edited_embeddings)
        source_base = model.decode_base_from_indices(indices)
        edited_base = model.decode_base_from_indices(edited_indices)

    torch.testing.assert_close(edited_base, source_base, rtol=0.0, atol=0.0)
    latent_change = edited_fused - source_fused
    target_slice = model.part_latent_slices["left_leg"]
    outside = torch.ones(model.base_code_dim, dtype=torch.bool)
    outside[target_slice] = False
    torch.testing.assert_close(latent_change[..., outside], torch.zeros_like(latent_change[..., outside]))
    assert torch.count_nonzero(latent_change[..., target_slice]) > 0


def test_latent_residual_fsq_is_strictly_causal_with_rf64():
    torch.manual_seed(13)
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


def test_all_fsq_groups_keep_straight_through_gradients():
    torch.manual_seed(17)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim, requires_grad=True)
    output = model(motion)
    output["fsq_codes"].sum().backward()
    assert motion.grad is not None
    assert torch.count_nonzero(motion.grad) > 0


def test_latent_projectors_receive_reconstruction_gradients_on_the_first_step():
    torch.manual_seed(19)
    model = _model()
    motion = torch.randn(2, 64, model.motion_dim)
    output = model(motion)
    output["recon_state"].square().mean().backward()
    for part in PART_NAMES:
        output_layer = model.latent_residual_projectors[part][-1]
        assert output_layer.weight.grad is not None
        assert torch.count_nonzero(output_layer.weight.grad) > 0
