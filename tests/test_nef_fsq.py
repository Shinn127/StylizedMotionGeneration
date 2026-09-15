"""NEF-FSQ tokenizer contract: routing, causality, receptive field and objective."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from stylized_motion.learning.losses import compute_motion_reconstruction_losses
from stylized_motion.learning.nef_eval import (
    _accumulate_token_statistics,
    _contacts_from_toe_motion,
    _rotation_angle_error,
    _token_utilization,
    swap_stream_tokens,
)
from stylized_motion.learning.nef_fsq import NEFMotionAutoencoder
from stylized_motion.learning.nef_layout import (
    GENO_SKELETON,
    NEF_FAMILY_STREAMS,
    NEF_STREAM_FAMILY,
    NEF_STREAM_NAMES,
    SOMA_SKELETON,
)
from stylized_motion.learning.runner import build_loss_context, load_experiment_config


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


def _model(spec=GENO_SKELETON, stream_dim: int = 16, seed: int = 13) -> NEFMotionAutoencoder:
    torch.manual_seed(seed)
    names, parents = skeleton_from_spec(spec)
    return NEFMotionAutoencoder(names, parents, stream_dim=stream_dim).eval()


def _affected_frames(output: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    difference = (output - baseline).abs()
    difference = difference.amax(dim=tuple(index for index in range(difference.ndim) if index != 1))
    return torch.nonzero(difference > 0).flatten()


def test_geno_and_soma_roundtrip_reconstruction_and_tokens():
    for spec, motion_dim in ((GENO_SKELETON, 230), (SOMA_SKELETON, 248)):
        model = _model(spec)
        motion = torch.randn(2, 64, motion_dim)
        with torch.no_grad():
            output = model(motion)
            from_indices = model.decode_from_indices(output["indices"])
            from_codes = model.decode_from_codes(output["fsq_codes"])
        assert model.motion_dim == motion_dim
        assert model.receptive_field == 64 and model.lookahead_frames == 0
        assert output["recon_state"].shape == motion.shape
        assert output["fsq_codes"].shape == (2, 64, 40)
        assert output["indices"].shape == (2, 64, 40)
        assert output["indices"].dtype == torch.long
        assert int(output["indices"].min()) >= 0 and int(output["indices"].max()) <= 8
        assert float(output["commit_loss"]) == 0.0
        assert tuple(output["stream_codes"]) == NEF_STREAM_NAMES
        assert tuple(output["stream_indices"]) == NEF_STREAM_NAMES
        torch.testing.assert_close(from_indices, output["recon_state"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(from_codes, output["recon_state"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.decode_from_codes(model.encode_to_codes(motion)[0]), output["recon_state"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.encode_to_indices(motion), output["indices"], rtol=0.0, atol=0.0)


def test_one_input_stream_only_changes_its_own_indices():
    torch.manual_seed(17)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim)
    indices = model.layout.stream_slices
    with torch.no_grad():
        baseline = model.encode_to_indices(motion)
        for stream in NEF_STREAM_NAMES:
            perturbed = motion.clone()
            perturbed[..., model._feature_index(stream)] += 5.0
            values = model.encode_to_indices(perturbed)
            changed = [
                other
                for other in NEF_STREAM_NAMES
                if not torch.equal(values[..., indices[other]], baseline[..., indices[other]])
            ]
            assert changed == [stream], f"{stream} changed {changed}"


def test_one_token_slice_only_changes_its_decoder_owned_features():
    torch.manual_seed(23)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim)
    feature_indices = model.layout.feature_indices(model.motion_dim)
    slices = model.layout.stream_slices
    with torch.no_grad():
        indices = model.encode_to_indices(motion)
        baseline = model.decode_from_indices(indices)
        for stream in NEF_STREAM_NAMES:
            edited = indices.clone()
            edited[..., slices[stream]] = 8 - edited[..., slices[stream]]
            values = model.decode_from_indices(edited)
            for other in NEF_STREAM_NAMES:
                if other == stream:
                    continue
                torch.testing.assert_close(
                    values[..., feature_indices[other]], baseline[..., feature_indices[other]], rtol=0.0, atol=0.0
                )
            assert not torch.equal(values[..., feature_indices[stream]], baseline[..., feature_indices[stream]])


def test_left_right_streams_share_family_modules_and_keep_distinct_embeddings():
    model = _model()
    for family, streams in NEF_FAMILY_STREAMS.items():
        assert all(NEF_STREAM_FAMILY[stream] == family for stream in streams)
    assert NEF_STREAM_FAMILY["left_arm_node"] == NEF_STREAM_FAMILY["right_arm_node"] == "arm_node"
    assert NEF_STREAM_FAMILY["left_leg_edge"] == NEF_STREAM_FAMILY["right_leg_edge"] == "leg_edge"
    # One projection/head/quantizer object per family serves both sides.
    assert len(model.input_projections) == len(model.output_heads) == len(model.stream_quantizers) == 9
    assert model.stream_quantizers["arm_node"] is model.stream_quantizers[NEF_STREAM_FAMILY["right_arm_node"]]
    index_of = {stream: index for index, stream in enumerate(NEF_STREAM_NAMES)}
    for left, right in (
        ("left_arm_node", "right_arm_node"),
        ("left_leg_node", "right_leg_node"),
        ("left_shoulder_edge", "right_shoulder_edge"),
    ):
        assert not torch.equal(
            model.stream_embeddings[index_of[left]], model.stream_embeddings[index_of[right]]
        )


def test_small_batch_backward_is_finite_and_ste_reaches_the_encoder():
    torch.manual_seed(29)
    model = _model()
    motion = torch.randn(2, 64, model.motion_dim, requires_grad=True)
    output = model(motion)
    output["fsq_codes"].sum().backward()
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all()
    assert torch.count_nonzero(motion.grad) > 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert any(parameter.grad is not None for parameter in model.stream_encoder.parameters())


def test_encoder_decoder_and_composed_receptive_fields():
    torch.manual_seed(31)
    model = _model()
    assert (model.encoder_receptive_field, model.decoder_receptive_field) == (31, 34)
    assert model.receptive_field == 64
    motion = torch.randn(1, 96, model.motion_dim)
    with torch.no_grad():
        encoded = model._encode_streams(motion)
        perturbed = motion.clone()
        perturbed[:, 0] += 50.0
        affected = _affected_frames(model._encode_streams(perturbed), encoded)
        assert int(affected.max()) + 1 == 31

        latents = {
            stream: torch.randn(1, 64, model.stream_dim) for stream in NEF_STREAM_NAMES
        }
        baseline = model._decode_embeddings(latents)
        changed = {key: value.clone() for key, value in latents.items()}
        changed["left_arm_node"][:, 0] += 50.0
        affected = _affected_frames(model._decode_embeddings(changed), baseline)
        assert int(affected.max()) + 1 == 34


def test_future_frames_do_not_change_the_current_reconstruction():
    torch.manual_seed(37)
    model = _model()
    motion = torch.randn(1, 96, model.motion_dim)
    changed = motion.clone()
    changed[:, 0] += 100.0
    changed[:, 65:] += 100.0
    with torch.no_grad():
        expected = model(motion)["recon_state"][:, 64]
        actual = model(changed)["recon_state"][:, 64]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_chunked_causal_encoding_matches_the_full_sequence_in_the_valid_interval():
    torch.manual_seed(41)
    model = _model()
    motion = torch.randn(1, 192, model.motion_dim)
    with torch.no_grad():
        full = model.encode_to_indices(motion)
        second = model.encode_to_indices(motion[:, 96:])
    history = model.receptive_field - 1
    torch.testing.assert_close(second[:, history:], full[:, 96 + history :], rtol=0.0, atol=0.0)


def test_token_swap_influence_stays_inside_start_to_end_plus_lookahead():
    torch.manual_seed(43)
    model = _model()
    motion = torch.randn(1, 127, model.motion_dim)
    donor = torch.randn(1, 127, model.motion_dim)
    start, stop = 64, 80
    with torch.no_grad():
        baseline = model(motion)["recon_state"]
        edited = swap_stream_tokens(
            model.encode_to_indices(motion),
            model.encode_to_indices(donor),
            model.layout,
            ("left_arm_node",),
            start,
            stop,
        )
        values = model.decode_from_indices(edited)
    changed = torch.nonzero((values - baseline).abs().amax(dim=(0, 2)) > 0).flatten()
    # RF=34 and lookahead=0: the last edited token at stop-1 reaches at most stop+32.
    assert int(changed.min()) == start
    assert int(changed.max()) == stop + model.decoder_receptive_field - 2
    assert int(changed.max()) == stop + 32


def test_donor_swap_preserves_non_target_stream_features_exactly():
    torch.manual_seed(47)
    model = _model()
    target = torch.randn(1, 96, model.motion_dim)
    donor = torch.randn(1, 96, model.motion_dim)
    feature_indices = model.layout.feature_indices(model.motion_dim)
    with torch.no_grad():
        indices = torch.cat((model.encode_to_indices(target), model.encode_to_indices(donor)), dim=0)
        edited_indices = swap_stream_tokens(
            indices[0:1], indices[1:2], model.layout, ("left_arm_node",), 32, 64
        )
        # Target and edited rows share one decode call so the comparison is exact.
        values = model.decode_from_indices(torch.cat((indices[0:1], edited_indices), dim=0))
    baseline, edited = values[0:1], values[1:2]
    for stream in NEF_STREAM_NAMES:
        if stream == "left_arm_node":
            continue
        torch.testing.assert_close(
            edited[..., feature_indices[stream]], baseline[..., feature_indices[stream]], rtol=0.0, atol=0.0
        )
    assert not torch.equal(
        edited[..., feature_indices["left_arm_node"]], baseline[..., feature_indices["left_arm_node"]]
    )


def test_forward_metrics_cover_every_stream_and_the_shared_contract():
    torch.manual_seed(53)
    model = _model()
    motion = torch.randn(1, 64, model.motion_dim)
    output = model(motion, collect_metrics=True)
    for key in (
        "level_perplexity", "level_usage", "tuple_unique_ratio", "tuple_change_rate",
        "coordinate_change_rate", "stream_coordinate_change_rates",
    ):
        assert key in output
    assert output["stream_coordinate_change_rates"].shape == (len(NEF_STREAM_NAMES),)
    assert model.compute_representation_losses(output, {"motion": motion}) == {}
    assert model.num_coordinates == 40 and model.num_levels == 9


def test_nef_config_disables_every_extra_loss_and_keeps_delta_only():
    config = load_experiment_config(Path(__file__).parents[1] / "data" / "configs" / "nef_fsq_40x9.yaml")
    training = config["training"]
    assert training["delta_weight"] == 3.0
    for key in (
        "root_pos_weight", "root_rot_weight", "joint_weight", "contact_weight",
        "foot_slide_weight", "foot_height_weight", "reuse_weight", "base_reuse_weight",
        "latent_energy_weight", "base_recon_weight", "edit_weight", "edit_preserve_weight",
    ):
        assert float(training[key]) == 0.0, key


def test_weighted_recon_and_delta_match_the_hand_computed_denominator():
    """The denominator counts masked elements; it never divides by the weight sum."""
    motion = torch.zeros(1, 3, 4)
    recon = torch.zeros(1, 3, 4)
    recon[0, 0] = 1.0
    recon[0, 2] = torch.tensor([2.0, 0.0, 2.0, 0.0])
    weights = torch.tensor([1.0, 2.0, 0.5, 0.0])
    common = dict(
        batch_motion=motion,
        feature_weights=weights,
        feature_offset=torch.zeros(4),
        feature_scale=torch.ones(4),
        delta_weight=3.0,
        commit_weight=0.0,
        root_pos_weight=0.0,
        root_rot_weight=0.0,
        root_dt=1.0 / 60.0,
    )
    masked = compute_motion_reconstruction_losses(
        output={"recon_state": recon, "commit_loss": torch.zeros(())}, **common
    )
    assert float(masked.recon) == pytest.approx(6.5 / 12.0)
    assert float(masked.delta) == pytest.approx(6.5 / 8.0)
    assert float(masked.loss) == pytest.approx(6.5 / 12.0 + 3.0 * 6.5 / 8.0)

    padded = compute_motion_reconstruction_losses(
        output={"recon_state": recon, "commit_loss": torch.zeros(())},
        loss_mask=torch.tensor([[True, True, False]]),
        **common,
    )
    assert float(padded.recon) == pytest.approx(3.5 / 8.0)
    assert float(padded.delta) == pytest.approx(3.5 / 4.0)
    assert float(padded.loss) == pytest.approx(3.5 / 8.0 + 3.0 * 3.5 / 4.0)


def test_total_loss_is_recon_plus_three_times_delta_and_delta_is_finite_without_pairs():
    config = load_experiment_config(Path(__file__).parents[1] / "data" / "configs" / "nef_fsq_40x9.yaml")
    context = build_loss_context(
        config,
        type(
            "Store",
            (),
            {
                "stats": type(
                    "Stats",
                    (),
                    {
                        "offset": torch.zeros(230).numpy(),
                        "scale": torch.ones(230).numpy(),
                        "ref_pos": torch.zeros(25, 3).numpy(),
                    },
                )(),
                "model_feature_weights": lambda self: torch.ones(230).numpy(),
                "names": ["Simulation"] + [f"Joint{index}" for index in range(24)],
                "parents": torch.tensor([-1] + [0] * 24),
            },
        )(),
        torch.device("cpu"),
    )
    common = dict(
        feature_weights=torch.ones(230),
        feature_offset=torch.zeros(230),
        feature_scale=torch.ones(230),
        delta_weight=context["delta_weight"],
        commit_weight=0.0,
        root_pos_weight=context["root_pos_weight"],
        root_rot_weight=context["root_rot_weight"],
        root_dt=context["root_dt"],
        joint_weight=context["joint_weight"],
        contact_weight=context["contact_weight"],
        foot_slide_weight=context["foot_slide_weight"],
        foot_height_weight=context["foot_height_weight"],
    )
    motion = torch.zeros(2, 64, 230)
    recon = torch.zeros(2, 64, 230)
    recon[:, 1:, :3] = 1.0
    values = compute_motion_reconstruction_losses(
        batch_motion=motion, output={"recon_state": recon, "commit_loss": torch.zeros(())}, **common
    )
    torch.testing.assert_close(
        values.loss, values.recon + 3.0 * values.delta, rtol=0.0, atol=0.0
    )
    assert float(values.recon) > 0.0 and float(values.delta) > 0.0
    for name in ("root_pos", "root_rot", "joint", "contact", "foot_slide", "foot_height", "commit"):
        assert float(getattr(values, name)) == 0.0, name

    single = compute_motion_reconstruction_losses(
        batch_motion=motion[:, :1], output={"recon_state": recon[:, :1], "commit_loss": torch.zeros(())}, **common
    )
    assert torch.isfinite(single.delta) and float(single.delta) == 0.0
    torch.testing.assert_close(single.loss, single.recon, rtol=0.0, atol=0.0)


def test_rotation_error_uses_denormalized_rotation_features():
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    quarter_turn = torch.tensor([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]])
    offset = torch.tensor([[0.5, 0.2], [0.1, 0.4], [0.3, 0.2]])
    scale = 0.3
    normalized_identity = (identity - offset) / scale
    normalized_quarter_turn = (quarter_turn - offset) / scale

    raw_error = _rotation_angle_error(
        normalized_identity * scale + offset,
        normalized_quarter_turn * scale + offset,
    )
    assert float(raw_error) == pytest.approx(torch.pi / 2, abs=1e-6)
    assert not torch.isclose(
        _rotation_angle_error(normalized_identity, normalized_quarter_turn), raw_error
    )


def test_token_utilization_is_aggregated_across_windows_before_summarizing():
    accumulator = {}
    for level in (0, 1):
        values = torch.full((1, 64, 2), level, dtype=torch.long)
        _accumulate_token_statistics(accumulator, {"head_node": values}, num_levels=9)
    summary = _token_utilization(accumulator)["head_node"]
    assert summary["level_usage"] == pytest.approx(2.0 / 9.0)
    assert summary["level_perplexity"] == pytest.approx(2.0)
    assert summary["coordinate_change_rate"] == 0.0


def test_contacts_are_inferred_from_reconstructed_toe_motion():
    positions = torch.zeros(1, 4, 3, 3)
    positions[:, :, 1, 0] = torch.tensor([0.0, 0.001, 0.002, 0.003])
    positions[:, :, 2, 0] = torch.tensor([0.0, 0.01, 0.02, 0.03])
    contacts = _contacts_from_toe_motion(positions, (1, 2), dt=0.1)
    assert contacts.tolist() == [[[True, True], [True, True], [True, True], [True, True]]]

    positions[:, 2:, 2, 0] += 0.1
    contacts = _contacts_from_toe_motion(positions, (1, 2), dt=0.1)
    assert contacts[0, 2, 1].item() is False
