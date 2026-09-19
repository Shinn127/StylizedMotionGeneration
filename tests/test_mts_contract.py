"""MTS operator contract: token identity, masks, shapes and fingerprints."""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator import (
    MASK_KINDS,
    LayoutAdapter,
    OperatorBatch,
    TokenSpec,
    TransportOutput,
    masked_cross_entropy,
    masked_mean,
    masked_nll_from_probs,
    operator_metadata,
    tokenizer_fingerprint,
    validate_operator_metadata,
)
from stylized_motion.learning.mts_operator.contract import (
    fingerprint_hash,
    normalize_mask_mixture,
    require_matching_tokenizer,
)
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEF_STREAM_NAMES, NEFLayout


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
    return NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON))


def adapter() -> LayoutAdapter:
    return LayoutAdapter(geno_layout())


def test_token_spec_carries_the_layout_identity():
    layout = geno_layout()
    spec = TokenSpec.from_layout(layout, representation_id="nef_fsq_independent_40x9")
    assert (spec.num_coordinates, spec.num_levels, spec.num_streams) == (40, 9, 13)
    assert spec.family == "nef_fsq" and spec.representation_id == "nef_fsq_independent_40x9"
    assert spec.layout_hash == layout.layout_hash()
    assert spec.fingerprint() == TokenSpec.from_layout(
        layout, representation_id="nef_fsq_independent_40x9"
    ).fingerprint()
    assert spec.fingerprint() != TokenSpec.from_layout(
        layout, representation_id="other"
    ).fingerprint()
    with pytest.raises(TypeError, match="layout_hash"):
        TokenSpec.from_layout(object())
    with pytest.raises(ValueError, match="positive"):
        TokenSpec(num_coordinates=0)


def test_token_and_mask_validators_reject_padding_disguised_as_levels():
    spec = adapter().token_spec()
    tokens = torch.zeros(2, 8, 40, dtype=torch.long)
    assert spec.validate_tokens(tokens).shape == (2, 8, 40)
    assert spec.validate_tokens(tokens.to(torch.uint8)).dtype == torch.long
    with pytest.raises(ValueError, match="outside"):
        spec.validate_tokens(torch.full((2, 8, 40), 9))
    with pytest.raises(ValueError, match="outside"):
        spec.validate_tokens(torch.full((2, 8, 40), -1))
    with pytest.raises(ValueError, match=r"\[B, T, 40\]"):
        spec.validate_tokens(torch.zeros(2, 8, 39, dtype=torch.long))
    with pytest.raises(ValueError, match="integer"):
        spec.validate_tokens(torch.zeros(2, 8, 40))

    mask = torch.ones(2, 8, 40, dtype=torch.bool)
    assert spec.validate_mask(mask, name="visible_mask", batch=2, frames=8) is mask
    with pytest.raises(ValueError, match="boolean"):
        spec.validate_mask(mask.float(), name="visible_mask")
    with pytest.raises(ValueError, match="batch"):
        spec.validate_mask(mask, name="visible_mask", batch=3)
    frames = torch.ones(2, 8, dtype=torch.bool)
    assert spec.validate_frame_mask(frames, batch=2, frames=8) is frames
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        spec.validate_frame_mask(torch.ones(2, dtype=torch.bool), batch=2, frames=8)


def test_logits_and_probability_validators():
    spec = adapter().token_spec()
    logits = torch.randn(2, 8, 40, 9)
    assert spec.validate_logits(logits).shape == logits.shape
    with pytest.raises(ValueError, match=r"\[B, T, 40, 9\]"):
        spec.validate_logits(torch.randn(2, 8, 40, 8))
    with pytest.raises(ValueError, match="float"):
        spec.validate_logits(torch.zeros(2, 8, 40, 9, dtype=torch.long))
    probs = logits.softmax(-1)
    assert spec.validate_probs(probs).shape == probs.shape
    with pytest.raises(ValueError, match="sum to one"):
        spec.validate_probs(probs * 2.0)
    negative = torch.zeros_like(probs)
    negative[..., 0] = -0.5
    negative[..., 1] = 1.5
    with pytest.raises(ValueError, match="non-negative"):
        spec.validate_probs(negative)


def test_transport_output_checks_axes_and_exposes_masks():
    spec = adapter().token_spec()
    logits = torch.randn(2, 8, 40, 9)
    hidden = torch.randn(2, 8, 13, 16)
    output = TransportOutput(logits=logits, stream_hidden=hidden, spec=spec)
    assert output.frame_mask().all()
    torch.testing.assert_close(output.probabilities.sum(-1), torch.ones(2, 8, 40), rtol=1e-5, atol=1e-5)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    output = TransportOutput(logits=logits, valid_mask=mask, spec=spec)
    assert torch.equal(output.frame_mask(), mask)
    with pytest.raises(ValueError, match="share batch and frame axes"):
        TransportOutput(logits=logits, stream_hidden=torch.randn(2, 7, 13, 8), spec=spec)
    with pytest.raises(ValueError, match=r"\[B, T, 13, D\]"):
        TransportOutput(logits=logits, stream_hidden=torch.randn(2, 8, 12, 8), spec=spec)
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        TransportOutput(logits=logits, valid_mask=torch.ones(2, 7, dtype=torch.bool), spec=spec)


def test_masked_cross_entropy_ignores_padding_and_empty_supervision():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 3, 9)
    targets = torch.randint(0, 9, (2, 4, 3))
    full = masked_cross_entropy(logits, targets)
    reference = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 9), targets.reshape(-1)
    )
    assert float(full) == pytest.approx(float(reference), rel=1e-6)
    # A frame mask that drops rows must equal the CE over just those rows.
    frame_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    partial = masked_cross_entropy(logits, targets, valid_mask=frame_mask)
    selected = frame_mask.unsqueeze(-1).expand_as(targets)
    expected = torch.nn.functional.cross_entropy(
        logits[selected].reshape(-1, 9), targets[selected].reshape(-1)
    )
    assert float(partial) == pytest.approx(float(expected), rel=1e-6)
    # A coordinate mask narrows it further and never divides by zero.
    coordinate_mask = torch.zeros(2, 4, 3, dtype=torch.bool)
    coordinate_mask[:, :, 1] = True
    only_middle = masked_cross_entropy(logits, targets, coordinate_mask=coordinate_mask)
    expected_middle = torch.nn.functional.cross_entropy(
        logits[..., 1, :].reshape(-1, 9), targets[..., 1].reshape(-1)
    )
    assert float(only_middle) == pytest.approx(float(expected_middle), rel=1e-6)
    empty = masked_cross_entropy(logits, targets, coordinate_mask=torch.zeros_like(coordinate_mask))
    assert float(empty) == 0.0 and torch.isfinite(empty)
    # An empty selection still connects to the graph, so backward stays defined.
    grad_logits = logits.detach().clone().requires_grad_(True)
    empty_grad = masked_cross_entropy(
        grad_logits, targets, coordinate_mask=torch.zeros_like(coordinate_mask)
    )
    assert empty_grad.requires_grad
    empty_grad.backward()
    torch.testing.assert_close(grad_logits.grad, torch.zeros_like(grad_logits), rtol=0.0, atol=0.0)
    # ``sum`` is the accumulation primitive: it must equal mean * count exactly.
    selected = frame_mask.unsqueeze(-1).expand_as(targets)
    chosen = logits[selected].reshape(-1, 9)
    reduced = masked_cross_entropy(
        logits, targets, valid_mask=frame_mask, coordinate_mask=coordinate_mask, reduction="sum"
    )
    picked = coordinate_mask[selected].reshape(-1)
    expected_sum = torch.nn.functional.cross_entropy(
        chosen[picked].reshape(-1, 9), targets[selected][picked].reshape(-1), reduction="sum"
    )
    assert float(reduced) == pytest.approx(float(expected_sum), rel=1e-6)
    with pytest.raises(ValueError, match="reduction"):
        masked_cross_entropy(logits, targets, reduction="none")
    with pytest.raises(ValueError, match="targets must be"):
        masked_cross_entropy(logits, targets[:, :3])


def test_masked_nll_from_probs_scores_probabilities_not_logits():
    """p_target = .99 must give -log(.99); the old probability-as-logits path gave 1.38."""
    probabilities = torch.full((1, 1, 40, 9), 0.01 / 8, dtype=torch.float32)
    probabilities[..., 0] = 0.99
    targets = torch.zeros(1, 1, 40, dtype=torch.long)
    assert float(probabilities[0, 0, 0].sum()) == pytest.approx(1.0, rel=1e-6)
    nll = masked_nll_from_probs(probabilities, targets)
    assert float(nll) == pytest.approx(0.01005033585, rel=1e-5)
    # Pushing the same distribution through the logits CE recomputed a different
    # quantity; that mismatch is the audit counterexample pinned down here.
    as_logits = masked_cross_entropy(probabilities, targets)
    assert float(as_logits) == pytest.approx(1.3803597688674927, rel=1e-5)
    assert abs(float(as_logits) - float(nll)) > 1.0

    uniform = torch.full((2, 3, 40, 9), 1.0 / 9.0)
    uniform_targets = torch.randint(0, 9, (2, 3, 40))
    assert float(masked_nll_from_probs(uniform, uniform_targets)) == pytest.approx(
        float(torch.log(torch.tensor(9.0))), rel=1e-6
    )


def test_masked_nll_from_probs_matches_softmax_ce_in_value_and_gradient():
    torch.manual_seed(17)
    logits = torch.randn(2, 5, 4, 9, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 9, (2, 5, 4))
    coordinate_mask = torch.zeros(2, 5, 4, dtype=torch.bool)
    coordinate_mask[:, :, 1::2] = True
    valid_mask = torch.ones(2, 5, dtype=torch.bool)
    valid_mask[1, 3:] = False

    logits_ce = logits.detach().clone().requires_grad_(True)
    ce = masked_cross_entropy(
        logits_ce, targets, valid_mask=valid_mask, coordinate_mask=coordinate_mask
    )
    ce.backward()
    logits_nll = logits.detach().clone().requires_grad_(True)
    nll = masked_nll_from_probs(
        torch.softmax(logits_nll, dim=-1),
        targets,
        valid_mask=valid_mask,
        coordinate_mask=coordinate_mask,
    )
    nll.backward()
    assert float(ce.detach()) == pytest.approx(float(nll.detach()), rel=1e-10)
    torch.testing.assert_close(logits_ce.grad, logits_nll.grad, rtol=1e-9, atol=1e-11)
    # A zero-probability target is clamped, not turned into +inf.
    degenerate = torch.zeros(1, 1, 1, 9)
    degenerate[..., 0] = 1.0
    clamped = masked_nll_from_probs(
        degenerate, torch.ones(1, 1, 1, dtype=torch.long), clamp_min=1e-12
    )
    assert float(clamped) == pytest.approx(27.6310211159, rel=1e-6)
    assert torch.isfinite(clamped)


def test_masked_nll_padding_and_supervision_are_filtered_before_gather():
    torch.manual_seed(23)
    probabilities = torch.softmax(torch.randn(2, 6, 3, 9, dtype=torch.float64), dim=-1)
    targets = torch.randint(0, 9, (2, 6, 3))
    # A sentinel far outside the alphabet sits at a position that is never
    # supervised, so the gather must not see it.
    targets[1, 4:, :] = 999
    coordinate_mask = torch.zeros(2, 6, 3, dtype=torch.bool)
    coordinate_mask[:, :4, 0] = True
    valid_mask = torch.ones(2, 6, dtype=torch.bool)
    nll = masked_nll_from_probs(
        probabilities, targets, valid_mask=valid_mask, coordinate_mask=coordinate_mask
    )
    selected = coordinate_mask[:, :4, 0]
    expected = -torch.log(probabilities[:, :4, 0, :].gather(-1, targets[:, :4, 0, None]))[selected]
    assert float(nll) == pytest.approx(float(expected.mean()), rel=1e-10)
    # The same check for the logits path.
    ce = masked_cross_entropy(
        probabilities.log(), targets, valid_mask=valid_mask, coordinate_mask=coordinate_mask
    )
    assert torch.isfinite(ce)
    # Supervised targets are still range-checked.
    bad = targets.clone()
    bad[0, 0, 0] = 9
    with pytest.raises(ValueError, match="targets"):
        masked_nll_from_probs(probabilities, bad, coordinate_mask=coordinate_mask)


def test_masked_nll_is_invariant_to_batch_partitioning_and_padding():
    torch.manual_seed(29)
    probabilities = torch.softmax(torch.randn(4, 5, 40, 9, dtype=torch.float64), dim=-1)
    targets = torch.randint(0, 9, (4, 5, 40))
    valid = torch.ones(4, 5, dtype=torch.bool)
    valid[0, 3:] = False
    valid[2, :2] = False
    supervision = torch.zeros(4, 5, 40, dtype=torch.bool)
    supervision[:, 1:4, ::3] = True

    def supervised_nll(p, t, v, s):
        value = masked_nll_from_probs(p, t, valid_mask=v, coordinate_mask=s, reduction="sum")
        count = int((s & v.unsqueeze(-1)).sum())
        return float(value), count

    whole, whole_count = supervised_nll(probabilities, targets, valid, supervision)
    split = 0.0
    count = 0
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        part, part_count = supervised_nll(
            probabilities[start:stop], targets[start:stop], valid[start:stop], supervision[start:stop]
        )
        split += part
        count += part_count
    assert count == whole_count and whole_count > 0
    assert split == pytest.approx(whole, rel=1e-12)

    # Appending an invalid padding frame changes neither total NLL nor count.
    padded = torch.cat(
        [probabilities, torch.softmax(torch.randn(4, 2, 40, 9, dtype=torch.float64), dim=-1)], dim=1
    )
    padded_targets = torch.cat([targets, torch.randint(0, 9, (4, 2, 40))], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(4, 2, dtype=torch.bool)], dim=1)
    padded_supervision = torch.cat([supervision, torch.ones(4, 2, 40, dtype=torch.bool)], dim=1)
    padded_nll, padded_count = supervised_nll(
        padded, padded_targets, padded_valid, padded_supervision
    )
    assert padded_count == whole_count
    assert padded_nll == pytest.approx(whole, rel=1e-12)


def test_operator_batch_has_one_edit_mask_definition():
    """``hard & ~visible & valid``, anchors observed, visible_mask never implicit."""
    spec = TokenSpec.from_layout(geno_layout(), representation_id="nef_fsq_independent_40x9")
    tokens = torch.randint(0, 9, (2, 4, 40), generator=torch.Generator().manual_seed(2))
    hard = torch.zeros(4, 40, dtype=torch.bool)
    hard[1:3, :10] = True
    visible = torch.zeros_like(tokens, dtype=torch.bool)
    visible[:, 1:3, :6] = True
    valid = torch.ones(2, 4, dtype=torch.bool)
    valid[0, 3] = False
    batch = OperatorBatch(
        target_tokens=tokens, visible_mask=visible, hard_mask=hard, target_valid_mask=valid
    )
    edit = batch.effective_edit_mask(spec)
    expected = hard.unsqueeze(0) & ~visible & valid.unsqueeze(-1)
    assert torch.equal(edit, expected)
    assert torch.equal(batch.supervision_mask(spec), edit)
    # An anchor inside the region stays observed.
    anchor = torch.zeros_like(tokens, dtype=torch.bool)
    anchor[:, 1, :10] = True
    anchored = OperatorBatch(
        target_tokens=tokens,
        visible_mask=visible,
        hard_mask=hard,
        target_valid_mask=valid,
        anchor_mask=anchor,
    )
    anchored_edit = anchored.effective_edit_mask(spec)
    assert not bool(anchored_edit[:, 1].any())
    assert torch.equal(anchored_edit[:, 2], edit[:, 2])
    # A missing visible mask is an error in the loss path...
    with pytest.raises(ValueError, match="visible_mask"):
        OperatorBatch(target_tokens=tokens, hard_mask=hard).effective_edit_mask(spec)
    with pytest.raises(ValueError, match="visible_mask"):
        OperatorBatch(target_tokens=tokens, hard_mask=hard).supervision_mask(spec)
    # ... and means "everything outside the hard mask is evidence" in generation.
    generated = OperatorBatch(target_tokens=tokens, hard_mask=hard).effective_edit_mask(
        spec, require_visible=False
    )
    assert torch.equal(generated, hard.unsqueeze(0).expand_as(tokens))
    full = OperatorBatch(target_tokens=tokens).effective_edit_mask(spec, require_visible=False)
    assert torch.equal(full, torch.ones_like(tokens, dtype=torch.bool))
    # Broadcast and shape rules are stated, not guessed.
    with pytest.raises(ValueError, match="anchor_mask"):
        OperatorBatch(
            target_tokens=tokens,
            visible_mask=visible,
            anchor_mask=torch.zeros(1, 4, 40, dtype=torch.bool),
        ).effective_edit_mask(spec)
    with pytest.raises(ValueError, match="hard_mask"):
        OperatorBatch(
            target_tokens=tokens, visible_mask=visible, hard_mask=torch.zeros(5, 40, dtype=torch.bool)
        ).effective_edit_mask(spec)
    with pytest.raises(ValueError, match="target_valid_mask"):
        OperatorBatch(
            target_tokens=tokens,
            visible_mask=visible,
            target_valid_mask=torch.ones(2, 5, dtype=torch.bool),
        ).effective_edit_mask(spec)
    # A batch-sized hard mask is allowed and is not silently broadcast away.
    per_sample = OperatorBatch(
        target_tokens=tokens,
        visible_mask=visible,
        hard_mask=hard.unsqueeze(0).expand_as(tokens).clone(),
    ).effective_edit_mask(spec)
    assert torch.equal(per_sample, hard.unsqueeze(0) & ~visible)


def test_masked_mean_has_a_zero_safe_denominator():
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.tensor([[True, False], [False, False]])
    assert float(masked_mean(values, mask)) == pytest.approx(1.0)
    assert float(masked_mean(values, torch.zeros_like(mask))) == 0.0
    wide = torch.randn(2, 4, 3)
    frame_mask = torch.tensor([[True, True, False, False], [False] * 4])
    assert float(masked_mean(wide, frame_mask)) == pytest.approx(float(wide[:1, :2].mean()))


def test_layout_adapter_indices_match_the_layout_tables():
    layout = geno_layout()
    view = adapter()
    metadata = layout.coordinate_metadata()
    stream_ids = view.coordinate_stream_ids()
    assert stream_ids.shape == (40,)
    for coordinate, record in enumerate(metadata):
        assert view.stream_of_coordinate(coordinate) == record["stream"]
        assert view.stream_index(record["stream"]) == int(stream_ids[coordinate])
        assert view.family_of_coordinate(coordinate) == record["family"]
        assert view.family_of_stream(record["stream"]) == record["family"]
        assert view.coordinate_slice(record["stream"]) == layout.stream_slices[record["stream"]]
    assert view.coordinate_indices(["left_arm_node"]).tolist() == [10, 11, 12, 13]
    assert view.coordinate_indices(["left_arm_node", "left_shoulder_edge"]).tolist() == [
        10, 11, 12, 13, 32, 33
    ]
    with pytest.raises(ValueError, match="Unknown NEF stream"):
        view.stream_index("left_hand")
    feature_ids = view.feature_of_stream(layout.num_joints * 9 + 5)
    assert feature_ids.numel() == 9 * layout.num_joints + 5
    assert int(feature_ids.min()) == 0 and int(feature_ids.max()) == 12
    for stream, index in layout.feature_indices(9 * layout.num_joints + 5).items():
        assert set(feature_ids[index].tolist()) == {view.stream_index(stream)}
    assert view.joint_owner()["Hips"] == "hips_edge"


def test_layout_adapter_graph_matches_stream_graph_and_is_consistent():
    layout = geno_layout()
    view = adapter()
    source, target = view.edge_index()
    type_ids, types = view.edge_type_ids()
    assert source.shape == target.shape == type_ids.shape
    assert len(layout.stream_graph()) == source.numel()
    assert types == ("global_to_root", "edge_to_child", "node_to_edge", "edge_to_edge")
    for position, (parent, child, relation) in enumerate(layout.stream_graph()):
        assert view.stream_names[int(source[position])] == parent
        assert view.stream_names[int(target[position])] == child
        assert types[int(type_ids[position])] == relation
    adjacency = view.adjacency()
    assert adjacency.shape == (13, 13)
    torch.testing.assert_close(adjacency, adjacency.T)
    assert bool((adjacency.diagonal() == 1.0).all())
    assert float(adjacency[view.stream_index("left_arm_node"), view.stream_index("left_shoulder_edge")]) == 1.0
    assert float(adjacency[view.stream_index("left_arm_node"), view.stream_index("right_arm_node")]) == 0.0
    assert float(view.adjacency(include_self=False).diagonal().sum()) == 0.0


def test_layout_adapter_regions_delegate_to_the_layout():
    layout = geno_layout()
    view = adapter()
    mask = view.hard_mask(["left_arm"], graph_radius=1, frame_range=(2, 5), length=8)
    torch.testing.assert_close(mask, layout.make_region_mask(["left_arm"], graph_radius=1, frame_range=(2, 5), length=8))
    assert view.region_streams("left_leg", graph_radius=0) == ("left_leg_node",)
    assert view.part_streams("left_leg", full_part=True) == ("left_leg_node", "left_leg_edge")
    assert set(view.part_names()) == {"torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"}
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    assert spec.layout_hash == view.layout_hash
    described = view.as_dict()
    assert described["num_streams"] == 13 and described["skeleton"] == "geno"
    assert described["relation_types"] == list(view.relation_types())


def test_operator_metadata_round_trips_and_rejects_mismatches():
    view = adapter()
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    tokenizer = {
        "family": "nef_fsq",
        "variant": "independent",
        "representation_id": "nef_fsq_independent_40x9",
        "num_coordinates": 40,
        "num_levels": 9,
        "coordinate_order": list(NEF_STREAM_NAMES),
        "coordinate_counts": view.layout.coordinate_counts,
        "receptive_field": 64,
        "lookahead_frames": 0,
        "architecture_version": 1,
        "nef_layout_hash": view.layout_hash,
        "feature_schema": {"motion_dim": 230, "joint_subset": "prune_ends_and_fingers"},
    }
    metadata = operator_metadata(
        token_spec=spec, tokenizer_metadata=tokenizer, model_config={"dim": 128}, extra={"kind": "transport"}
    )
    assert metadata["tokenizer_hash"] == fingerprint_hash(metadata["tokenizer"])
    assert metadata["kind"] == "transport"
    stored = validate_operator_metadata(metadata, token_spec=spec, tokenizer_metadata=tokenizer)
    assert stored["token_spec"] == spec.as_dict()

    with pytest.raises(ValueError, match="contract version"):
        validate_operator_metadata({**metadata, "mts_contract_version": 99}, token_spec=spec)
    with pytest.raises(ValueError, match="token_spec"):
        validate_operator_metadata(metadata, token_spec=TokenSpec(num_coordinates=40, num_levels=8))
    changed = dict(tokenizer)
    changed["nef_layout_hash"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_operator_metadata(metadata, tokenizer_metadata=changed)
    with pytest.raises(ValueError, match="tokenizer hash"):
        validate_operator_metadata({**metadata, "tokenizer_hash": "deadbeef"}, tokenizer_metadata=tokenizer)


def test_fingerprints_ignore_weights_and_normalization_but_not_layout():
    base = tokenizer_fingerprint(
        {"family": "nef_fsq", "representation_id": "x", "nef_layout_hash": "abc"},
        feature_schema={"motion_dim": 230, "joint_subset": "full", "stats_sha256": "one"},
    )
    other = tokenizer_fingerprint(
        {"family": "nef_fsq", "representation_id": "x", "nef_layout_hash": "abc"},
        feature_schema={"motion_dim": 230, "joint_subset": "full", "stats_sha256": "two"},
    )
    assert base == other  # statistics are not part of the alphabet
    require_matching_tokenizer(base, other)
    different = tokenizer_fingerprint(
        {"family": "nef_fsq", "representation_id": "x", "nef_layout_hash": "def"},
        feature_schema={"motion_dim": 230, "joint_subset": "full"},
    )
    with pytest.raises(ValueError, match="nef_layout_hash"):
        require_matching_tokenizer(base, different)


def test_mask_mixture_is_validated_and_normalized():
    mixture = normalize_mask_mixture(
        {"random_coordinate": 0.20, "stream": 0.25, "temporal_span": 0.20,
         "spatiotemporal_block": 0.20, "full_generation": 0.15}
    )
    assert sum(mixture.values()) == pytest.approx(1.0)
    assert set(mixture) == set(MASK_KINDS)
    assert normalize_mask_mixture({"stream": 2.0, "full_generation": 1.0}) == {
        "stream": pytest.approx(2 / 3),
        "full_generation": pytest.approx(1 / 3),
    }
    with pytest.raises(ValueError, match="Unknown mask kinds"):
        normalize_mask_mixture({"random": 1.0})
    with pytest.raises(ValueError, match="positive total"):
        normalize_mask_mixture({"stream": 0.0})
