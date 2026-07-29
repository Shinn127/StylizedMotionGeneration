import torch

from evaluate_part_editing import (
    _accumulate_part_metrics,
    _finalize_part_metrics,
    _new_part_accumulator,
    build_multi_part_edits,
    edit_indices,
    model_feature_indices,
    model_group_slices,
)
from models.latent_residual_part_fsq import LatentResidualPartFSQMotionAutoencoder
from models.part_fsq import HierarchicalPartFSQMotionAutoencoder
from models.residual_part_fsq import ResidualPartFSQMotionAutoencoder


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


def test_edit_indices_replaces_only_the_requested_group():
    source = torch.zeros(2, 4, 40, dtype=torch.long)
    donor = torch.ones_like(source)
    edited = edit_indices(source, donor, slice(20, 26))

    torch.testing.assert_close(edited[..., :20], source[..., :20])
    torch.testing.assert_close(edited[..., 20:26], donor[..., 20:26])
    torch.testing.assert_close(edited[..., 26:], source[..., 26:])


def test_multi_part_edits_keep_each_variant_independent():
    source = torch.zeros(1, 4, 40, dtype=torch.long)
    donor = torch.ones_like(source)
    slices = {
        "torso": slice(20, 26),
        "left_leg": slice(26, 30),
        "right_leg": slice(30, 34),
        "left_arm": slice(34, 37),
        "right_arm": slice(37, 40),
    }
    edits = build_multi_part_edits(source, donor, tuple(slices), slices)

    assert edits.shape == (1, 5, 4, 40)
    for part_index, part in enumerate(slices):
        torch.testing.assert_close(edits[:, part_index, :, slices[part]], torch.ones(1, 4, slices[part].stop - slices[part].start, dtype=torch.long))
        untouched = torch.ones(40, dtype=torch.bool)
        untouched[slices[part]] = False
        torch.testing.assert_close(edits[:, part_index, :, untouched], torch.zeros(1, 4, int(untouched.sum()), dtype=torch.long))


def test_part_edit_metrics_use_sample_weighting_and_the_donor_baseline():
    accumulator = _new_part_accumulator()
    feature_index = torch.tensor([0])
    feature_indices = {
        group: feature_index
        for group in ("global", "torso", "left_leg", "right_leg", "left_arm", "right_arm")
    }

    source_recon = torch.zeros(2, 1, 1)
    source_target = torch.zeros_like(source_recon)
    donor_target = torch.ones_like(source_recon)
    edited = torch.full_like(source_recon, 0.5)
    _accumulate_part_metrics(
        accumulator,
        source_recon,
        edited,
        source_target,
        donor_target,
        feature_index,
        feature_index,
        None,
        None,
        None,
        feature_indices,
    )

    _accumulate_part_metrics(
        accumulator,
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1),
        feature_index,
        feature_index,
        None,
        None,
        None,
        feature_indices,
    )
    metrics = _finalize_part_metrics(accumulator, 3)

    assert abs(metrics["target_response"] - 2.0 / 3.0) < 1e-6
    assert metrics["source_target_reconstruction_error"] == 0.0
    assert metrics["source_target_donor_error"] == 1.0
    assert abs(metrics["target_transfer_gain"] - 2.0 / 3.0) < 1e-6


def test_part_fsq_part_edit_has_no_feature_space_leakage():
    torch.manual_seed(41)
    names, parents = _skeleton()
    model = HierarchicalPartFSQMotionAutoencoder(names, parents, stream_dim=16).eval()
    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        source_indices = model.encode_to_indices(motion)
        donor_indices = source_indices.clone()
        donor_indices[..., model_group_slices(model)["left_leg"]] = 8 - donor_indices[..., model_group_slices(model)["left_leg"]]
        source_recon = model.decode_from_indices(source_indices)
        edited = model.decode_from_indices(
            edit_indices(source_indices, donor_indices, model_group_slices(model)["left_leg"])
        )
    features = model_feature_indices(model)
    for group in ("global", "torso", "right_leg", "left_arm", "right_arm"):
        torch.testing.assert_close(edited[..., features[group]], source_recon[..., features[group]], rtol=0.0, atol=0.0)
    assert not torch.equal(edited[..., features["left_leg"]], source_recon[..., features["left_leg"]])


def test_residual_part_edit_keeps_base_path_unchanged():
    torch.manual_seed(43)
    names, parents = _skeleton()
    model = ResidualPartFSQMotionAutoencoder(
        names,
        parents,
        base_code_dim=32,
        base_width=32,
        part_state_dim=16,
        residual_decoder_dim=32,
        residual_decoder_width=32,
        residual_hidden_dim=16,
    ).eval()
    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        source_indices = model.encode_to_indices(motion)
        donor_indices = source_indices.clone()
        donor_indices[..., model.group_slices["left_arm"]] = 8 - donor_indices[..., model.group_slices["left_arm"]]
        edited_indices = edit_indices(source_indices, donor_indices, model.group_slices["left_arm"])
        source_base = model.decode_base_from_indices(source_indices)
        edited_base = model.decode_base_from_indices(edited_indices)
    torch.testing.assert_close(edited_base, source_base, rtol=0.0, atol=0.0)


def test_latent_residual_part_edit_keeps_base_fixed_and_reports_soft_leakage():
    torch.manual_seed(47)
    names, parents = _skeleton()
    model = LatentResidualPartFSQMotionAutoencoder(
        names,
        parents,
        base_code_dim=32,
        base_width=32,
        part_state_dim=16,
        part_predictor_hidden_dim=24,
        latent_projector_hidden_dim=24,
        part_latent_dims=(10, 6, 6, 5, 5),
    ).eval()
    for projector in model.latent_residual_projectors.values():
        torch.nn.init.normal_(projector[-1].weight, std=0.05)

    motion = torch.randn(1, 64, model.motion_dim)
    with torch.no_grad():
        source_indices = model.encode_to_indices(motion)
        donor_indices = source_indices.clone()
        part_slice = model.group_slices["left_arm"]
        donor_indices[..., part_slice] = 8 - donor_indices[..., part_slice]
        edited_indices = edit_indices(source_indices, donor_indices, part_slice)
        source_base = model.decode_base_from_indices(source_indices)
        edited_base = model.decode_base_from_indices(edited_indices)
        source_recon = model.decode_from_indices(source_indices)
        edited = model.decode_from_indices(edited_indices)

    torch.testing.assert_close(edited_base, source_base, rtol=0.0, atol=0.0)
    response = (edited - source_recon).abs()
    features = model_feature_indices(model)
    target_response = response.index_select(-1, features["left_arm"]).mean()
    non_target = torch.ones(model.motion_dim, dtype=torch.bool)
    non_target[features["left_arm"]] = False
    non_target_response = response[..., non_target].mean()
    leakage_ratio = non_target_response / target_response.clamp_min(1e-7)
    assert target_response > 0.0
    assert torch.isfinite(leakage_ratio)
