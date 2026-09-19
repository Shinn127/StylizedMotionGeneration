"""MTS base transport: embeddings, masking, graph locality and the model contract."""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.contract import TokenSpec
from stylized_motion.learning.mts_operator.embeddings import (
    StreamLevelHead,
    StreamTokenEmbedding,
)
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder
from stylized_motion.learning.mts_operator.graph import StreamGraphBlock, StreamGraphNetwork
from stylized_motion.learning.mts_operator.masking import (
    MaskBatch,
    MaskConfig,
    MaskGenerator,
    apply_hard_support,
)
from stylized_motion.learning.mts_operator.transport import (
    ContentConditioner,
    MotionTransportTransformer,
)
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout


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


def adapter() -> LayoutAdapter:
    return LayoutAdapter(NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON)))


def tokens_and_mask(batch: int = 3, frames: int = 16, *, seed: int = 1, visible: bool = True):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 9, (batch, frames, 40), generator=generator)
    mask = torch.full((batch, frames, 40), bool(visible), dtype=torch.bool)
    return tokens, mask


def touched_streams(adapter: LayoutAdapter, difference: torch.Tensor) -> list[str]:
    per_coordinate = difference.amax(dim=(0, 1, -1))
    coordinates = torch.nonzero(per_coordinate > 0).flatten().tolist()
    return sorted({adapter.stream_of_coordinate(int(c)) for c in coordinates})


# ---------------------------------------------------------------------------
# embeddings


def test_stream_embedding_pools_coordinates_into_their_own_stream():
    view = adapter()
    module = StreamTokenEmbedding(view, 8).eval()
    tokens, mask = tokens_and_mask()
    hidden = module(tokens, mask)
    assert hidden.shape == (3, 16, 13, 8)
    with torch.no_grad():
        edited = tokens.clone()
        edited[..., 10:14] = 8 - edited[..., 10:14]
        changed = (module(edited, mask) - hidden).abs().amax(dim=(0, 1))  # [13, 8]
    stream = view.stream_index("left_arm_node")
    touched = torch.nonzero(changed.amax(dim=-1) > 0).flatten().tolist()
    assert touched == [stream]
    # The stream state is its own coordinates flattened in layout order, mapped by
    # the stream's own projection, plus the stream embedding.
    indices = module.coordinate_indices(stream)
    chunk = (module.level_embedding(tokens.index_select(-1, indices))
             + module.coordinate_embedding.weight[indices])
    expected = module.stream_projection[stream](
        chunk.reshape(tokens.shape[0], tokens.shape[1], -1)
    ) + module.stream_embedding.weight[stream]
    torch.testing.assert_close(hidden[:, :, stream], expected, rtol=1e-5, atol=1e-5)


def test_hidden_positions_use_the_mask_vector_not_a_level():
    view = adapter()
    module = StreamTokenEmbedding(view, 8).eval()
    tokens, _ = tokens_and_mask()
    visible = torch.ones(3, 16, 40, dtype=torch.bool)
    other = torch.randint(0, 9, tokens.shape)
    # Fully hidden: the token values cannot matter at all.
    torch.testing.assert_close(
        module(other, torch.zeros(3, 16, 40, dtype=torch.bool)),
        module(tokens, torch.zeros(3, 16, 40, dtype=torch.bool)),
    )
    # Hiding one stream's coordinates removes exactly that stream's dependence
    # on the token values; every other stream still changes.
    partial = visible.clone()
    partial[..., 10:14] = False
    difference = (module(other, partial) - module(tokens, partial)).abs()
    unchanged = torch.nonzero(difference.amax(dim=(0, 1, 3)) == 0).flatten().tolist()
    # Exactly the fully hidden stream stops depending on the token values.
    assert [view.stream_names[index] for index in unchanged] == ["left_arm_node"]
    assert not torch.allclose(module(other, visible), module(tokens, visible))


def test_stream_embedding_validates_shapes_and_levels():
    module = StreamTokenEmbedding(adapter(), 8).eval()
    with pytest.raises(ValueError, match="outside"):
        module(torch.full((2, 4, 40), 9), torch.ones(2, 4, 40, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"\[B, T, 40\]"):
        module(torch.zeros(2, 4, 39, dtype=torch.long), torch.ones(2, 4, 39, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        module(torch.zeros(2, 4, 40, dtype=torch.long), torch.ones(2, 4, 40))


# ---------------------------------------------------------------------------
# masking


def test_each_mask_kind_has_its_designed_shape():
    view = adapter()
    generator = torch.Generator().manual_seed(5)
    masks = MaskGenerator(MaskConfig(coordinate_ratio=0.25, stream_ratio=0.25, span_ratio=0.5,
                                     block_frames=4, block_coordinates=6))
    random_kind = masks.sample_kind("random_coordinate", 4, 32, adapter=view, generator=generator)
    assert random_kind.kind == "random_coordinate"
    fraction = float(random_kind.supervision_mask.float().mean())
    assert 0.15 < fraction < 0.35

    stream_mask = masks.sample_kind("stream", 4, 32, adapter=view, generator=generator)
    stream_ids = view.coordinate_stream_ids()
    assert bool(stream_mask.supervision_mask.any())
    for row in range(4):
        hidden_coordinates = stream_mask.supervision_mask[row, 0]
        hidden_streams = {view.stream_names[int(s)] for s in torch.unique(stream_ids[hidden_coordinates])}
        visible_streams = {
            view.stream_names[int(s)] for s in torch.unique(stream_ids[~hidden_coordinates])
        }
        assert hidden_streams and not (hidden_streams & visible_streams)
        assert len(hidden_streams) == round(0.25 * 13)

    span_mask = masks.sample_kind("temporal_span", 4, 32, adapter=view, generator=generator)
    for row in range(4):
        per_frame = span_mask.supervision_mask[row].all(dim=-1)
        frames = torch.nonzero(per_frame).flatten()
        assert frames.numel() == 16 and int(frames.max() - frames.min()) == 15

    block = masks.sample_kind("spatiotemporal_block", 4, 32, adapter=view, generator=generator)
    for row in range(4):
        assert int(block.supervision_mask[row].sum()) == 4 * 6

    full = masks.sample_kind("full_generation", 4, 32, adapter=view, generator=generator)
    assert not bool(full.visible_mask.any())
    assert float(full.summary()["hidden_fraction"]) == 1.0


def test_masks_never_leave_a_sample_unsupervised_and_follow_the_mixture():
    view = adapter()
    generous = MaskGenerator(MaskConfig(coordinate_ratio=1.0, stream_ratio=1.0, span_ratio=1.0))
    generator = torch.Generator().manual_seed(9)
    for kind in ("random_coordinate", "stream", "temporal_span", "spatiotemporal_block"):
        batch = generous.sample_kind(kind, 3, 16, adapter=view, generator=generator)
        assert bool(batch.supervision_mask.any(dim=(1, 2)).all())
    only_full = MaskGenerator({"full_generation": 1.0})
    for _ in range(5):
        assert only_full.sample(2, 8, adapter=view, generator=generator).kind == "full_generation"
    drawn = {only_full.sample(2, 8, adapter=view, generator=generator).kind for _ in range(5)}
    assert drawn == {"full_generation"}
    with pytest.raises(ValueError, match="Unknown mask kind"):
        only_full.sample_kind("random", 2, 8, adapter=view)
    with pytest.raises(ValueError, match="LayoutAdapter"):
        only_full.sample_kind("stream", 2, 8, spec=view.token_spec(), generator=generator)


def test_mask_config_accepts_the_plan_flat_keys_and_rejects_unknown_options():
    config = MaskConfig.from_mapping(
        {
            "random_coordinate": 0.20,
            "stream": 0.25,
            "temporal_span": 0.20,
            "spatiotemporal_block": 0.20,
            "full_generation": 0.15,
            "block_frames": 8,
        }
    )
    assert config.normalized_mixture()["stream"] == pytest.approx(0.25)
    assert config.block_frames == 8
    nested = MaskConfig.from_mapping({"mixture": {"stream": 1.0}, "span_ratio": 0.25})
    assert nested.normalized_mixture() == {"stream": 1.0} and nested.span_ratio == 0.25
    default_mixture = MaskConfig().normalized_mixture()
    assert sum(default_mixture.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unknown masking options"):
        MaskConfig.from_mapping({"mask_rate": 0.5})
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        MaskConfig(coordinate_ratio=0.0).validate()
    with pytest.raises(ValueError, match="positive"):
        MaskConfig(block_frames=0).validate()


def test_hard_support_restricts_and_expands_supervision():
    view = adapter()
    generator = torch.Generator().manual_seed(11)
    batch = MaskGenerator().sample_kind("random_coordinate", 2, 16, adapter=view, generator=generator)
    support = view.hard_mask(["left_arm"], graph_radius=1, length=16)
    restricted = apply_hard_support(batch, support, mode="restrict")
    assert not bool(restricted.supervision_mask[:, ~support].any())
    assert bool(restricted.supervision_mask[:, support].any())
    expanded = apply_hard_support(batch, support, mode="expand")
    # Preservation outside the support: everything there is hidden and supervised.
    assert bool(expanded.supervision_mask[:, ~support].all())
    assert not bool(expanded.visible_mask[:, ~support].any())
    assert bool(expanded.supervision_mask[:, support].any())
    with pytest.raises(ValueError, match="Unsupported support mode"):
        apply_hard_support(batch, support, mode="replace")
    with pytest.raises(ValueError, match="support must be"):
        apply_hard_support(batch, torch.ones(8, 40, dtype=torch.bool))
    everything = MaskBatch(visible_mask=torch.ones(1, 4, 40, dtype=torch.bool), kind="full_generation")
    with pytest.raises(ValueError, match="covers every position"):
        apply_hard_support(everything, torch.ones(4, 40, dtype=torch.bool), mode="expand")

    # The fallback position must come from the region the mode supervises: the old
    # code always hid a position outside the support, so ``restrict`` could
    # supervise exactly the token it promised not to touch.
    outside_only = torch.zeros(1, 16, 40, dtype=torch.bool)
    outside_only[0, 0, 0] = True  # the single supervised position lies outside
    support = view.hard_mask(["left_arm"], graph_radius=1, length=16)
    rescued = apply_hard_support(
        MaskBatch(visible_mask=~outside_only, kind="random_coordinate"), support, mode="restrict"
    )
    assert bool(rescued.supervision_mask.any())
    assert not bool(rescued.supervision_mask[:, ~support].any())
    assert bool(rescued.supervision_mask[:, support].any())


# ---------------------------------------------------------------------------
# graph


def test_message_edges_are_bidirectional_and_typed():
    view = adapter()
    source, target, type_ids, direction, types = view.message_edges()
    edges = len(view.layout.stream_graph())
    assert source.numel() == target.numel() == type_ids.numel() == direction.numel() == 2 * edges
    assert direction[:edges].eq(0).all() and direction[edges:].eq(1).all()
    assert torch.equal(source[edges:], target[:edges]) and torch.equal(target[edges:], source[:edges])
    assert types == view.relation_types()
    block = StreamGraphBlock(view, 8)
    assert block.edge_source.numel() == 2 * edges


def test_graph_blocks_are_hop_limited_and_identity_at_depth_zero():
    view = adapter()
    tokens, mask = tokens_and_mask()
    module = StreamTokenEmbedding(view, 8).eval()
    hidden = module(tokens, mask)
    edited_tokens = tokens.clone()
    edited_tokens[..., 10:14] = 8 - edited_tokens[..., 10:14]
    edited_hidden = module(edited_tokens, mask)

    identity = StreamGraphNetwork(view, 8, depth=0).eval()
    torch.testing.assert_close(identity(hidden), hidden)
    for depth, expected in (
        (0, ["left_arm_node"]),
        (1, ["left_arm_node", "left_shoulder_edge"]),
        (2, ["left_arm_node", "left_shoulder_edge", "torso_node"]),
    ):
        network = StreamGraphNetwork(view, 8, depth=depth).eval()
        with torch.no_grad():
            difference = network(edited_hidden) - network(hidden)
        stream_change = difference.abs().amax(dim=(0, 1, 3))  # [13]
        touched = sorted(
            view.stream_names[i] for i in torch.nonzero(stream_change > 0).flatten().tolist()
        )
        assert touched == sorted(expected)


def test_graph_block_validates_shape_and_aggregation():
    view = adapter()
    block = StreamGraphBlock(view, 8)
    with pytest.raises(ValueError, match=r"\[B, T, 13, D\]"):
        block(torch.randn(2, 4, 12, 8))
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        StreamGraphBlock(view, 8, aggregation="max")


# ---------------------------------------------------------------------------
# transport


def transport(adapter_view: LayoutAdapter | None = None, **kwargs) -> MotionTransportTransformer:
    view = adapter_view or adapter()
    options = dict(dim=32, depth=2, heads=4, graph_depth=1, dropout=0.0)
    options.update(kwargs)
    return MotionTransportTransformer(view, **options).eval()


def test_transport_output_contract_and_ownership_scope():
    view = adapter()
    model = transport(view)
    tokens, mask = tokens_and_mask()
    output = model(tokens, mask)
    assert output.logits.shape == (3, 16, 40, 9)
    assert output.stream_hidden.shape == (3, 16, 13, 32)
    assert output.valid_mask is None and bool(output.frame_mask().all())
    assert output.spec.num_coordinates == 40
    with torch.no_grad():
        base = model(tokens, mask).logits
        edited = tokens.clone()
        edited[..., 10:14] = 8 - edited[..., 10:14]
        difference = model(edited, mask).logits - base
    assert touched_streams(view, difference) == ["left_arm_node", "left_shoulder_edge"]
    # Without the graph the edit stays inside its own stream.
    plain = transport(view, graph_mode="none")
    with torch.no_grad():
        difference = plain(edited, torch.ones(3, 16, 40, dtype=torch.bool)).logits - plain(
            tokens, torch.ones(3, 16, 40, dtype=torch.bool)
        ).logits
    assert touched_streams(view, difference) == ["left_arm_node"]


def test_transport_never_reads_hidden_tokens():
    model = transport()
    tokens, _ = tokens_and_mask()
    other = torch.randint(0, 9, tokens.shape)
    hidden = torch.zeros(3, 16, 40, dtype=torch.bool)
    with torch.no_grad():
        first = model(tokens, hidden).logits
        second = model(other, hidden).logits
    torch.testing.assert_close(first, second)
    partial = torch.ones(3, 16, 40, dtype=torch.bool)
    partial[..., 10:14] = False
    masked_low = tokens.clone()
    masked_low[..., 10:14] = 0
    masked_high = tokens.clone()
    masked_high[..., 10:14] = 8
    with torch.no_grad():
        third = model(masked_low, partial).logits
        fourth = model(masked_high, partial).logits
    torch.testing.assert_close(third, fourth)  # hidden arm tokens cannot leak
    assert not torch.allclose(third, first)  # the visible tokens still matter


def test_causal_transport_does_not_look_ahead():
    view = adapter()
    model = transport(view, temporal_mode="causal", graph_depth=0)
    tokens, mask = tokens_and_mask(frames=24)
    edited = tokens.clone()
    edited[:, 12] = 8 - edited[:, 12]
    with torch.no_grad():
        base = model(tokens, mask).logits
        changed = model(edited, mask).logits
    difference = (changed - base).abs().amax(dim=(0, 2, 3))
    assert float(difference[:12].max()) == 0.0
    assert float(difference[12:].max()) > 0.0
    # The bidirectional variant is explicitly not causal.
    bidirectional = transport(view, graph_depth=0)
    with torch.no_grad():
        difference = (
            bidirectional(edited, mask).logits - bidirectional(tokens, mask).logits
        ).abs().amax(dim=(0, 2, 3))
    assert float(difference[:12].max()) > 0.0


def test_transport_respects_valid_mask_and_rejects_all_padding():
    view = adapter()
    model = transport(view)
    tokens, mask = tokens_and_mask()
    valid = torch.ones(3, 16, dtype=torch.bool)
    valid[:, 10:] = False
    output = model(tokens, mask, valid_mask=valid)
    torch.testing.assert_close(output.frame_mask(), valid)
    with pytest.raises(ValueError, match="at least one valid frame"):
        model(tokens, mask, valid_mask=torch.zeros(3, 16, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        model(tokens, mask, valid_mask=torch.ones(3, 15, dtype=torch.bool))


def test_content_conditioner_accepts_ids_and_features():
    view = adapter()
    with_ids = transport(view, content_classes=4)
    tokens, mask = tokens_and_mask()
    generator = torch.Generator().manual_seed(21)
    with torch.no_grad():
        by_id = with_ids(tokens, mask, content_condition=torch.randint(0, 4, (3,), generator=generator)).logits
        per_frame = with_ids(
            tokens, mask, content_condition=torch.randint(0, 4, (3, 16), generator=generator)
        ).logits
    assert by_id.shape == per_frame.shape == (3, 16, 40, 9)
    assert not torch.allclose(by_id, per_frame)
    with_features = transport(view, content_dim=6)
    with torch.no_grad():
        clip_level = with_features(tokens, mask, content_condition=torch.randn(3, 6)).logits
        per_frame_features = with_features(
            tokens, mask, content_condition=torch.randn(3, 16, 6)
        ).logits
    assert clip_level.shape == per_frame_features.shape == (3, 16, 40, 9)
    with pytest.raises(ValueError, match="content ids"):
        with_ids(tokens, mask, content_condition=torch.randint(0, 4, (3, 5, 1)))
    with pytest.raises(ValueError, match=r"in \[0, 3\]"):
        with_ids(tokens, mask, content_condition=torch.full((3,), 7))
    with pytest.raises(ValueError, match="content features"):
        with_features(tokens, mask, content_condition=torch.randn(3, 5))
    with pytest.raises(ValueError, match="content_classes"):
        ContentConditioner(8)
    with pytest.raises(ValueError, match="content_classes"):
        transport(view, content_dim=4)(tokens, mask, content_condition=torch.zeros(3, dtype=torch.long))


def test_transport_training_step_is_finite_and_deterministic():
    view = adapter()
    model = transport(view, content_classes=3)
    tokens, mask = tokens_and_mask()
    generator = MaskGenerator().sample(3, 16, adapter=view, generator=torch.Generator().manual_seed(31))
    condition = torch.tensor([0, 1, 2])
    output = model(tokens, mask, content_condition=condition)
    supervision = generator.supervision_mask
    from stylized_motion.learning.mts_operator.contract import masked_cross_entropy

    loss = masked_cross_entropy(output.logits, tokens, coordinate_mask=supervision)
    loss.backward()
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    model.eval()
    with torch.no_grad():
        first = model(tokens, mask, content_condition=condition).logits
        second = model(tokens, mask, content_condition=condition).logits
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_transport_rejects_invalid_configuration():
    view = adapter()
    with pytest.raises(ValueError, match="graph_mode"):
        transport(view, graph_mode="full")
    with pytest.raises(ValueError, match="temporal_mode"):
        transport(view, temporal_mode="recurrent")
    with pytest.raises(ValueError, match="divisible by heads"):
        transport(view, dim=30, heads=4)
    with pytest.raises(ValueError, match="graph_depth"):
        transport(view, graph_depth=-1)
    spec = TokenSpec()
    assert spec.validate_tokens(torch.zeros(1, 2, 40, dtype=torch.long)).shape == (1, 2, 40)


# ---------------------------------------------------------------------------
# R06: coordinate information must survive embedding and head


class _TwoStreamAdapter:
    """A minimal duck-typed adapter: 2 streams, 2+2 coordinates, 9 levels.

    Going through the adapter API (not a hand-written 40-coordinate grouping) is
    what keeps the test honest about "which coordinate belongs to which stream".
    """

    num_levels = 9
    num_coordinates = 4
    num_streams = 2
    stream_names = ("a_node", "b_node")

    def coordinate_stream_ids(self, *, device=None) -> torch.Tensor:
        return torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

    def stream_coordinate_indices(self, *, device=None) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor([0, 1], dtype=torch.long, device=device),
            torch.tensor([2, 3], dtype=torch.long, device=device),
        )

    def token_spec(self):
        from stylized_motion.learning.mts_operator.contract import TokenSpec

        return TokenSpec(num_coordinates=4, num_levels=9, num_streams=2)


def test_within_stream_coordinates_are_distinguishable_by_construction():
    """Deterministic weights: the pooled stream state is an exact flatten+project."""
    adapter = _TwoStreamAdapter()
    embed_dim = 4
    module = StreamTokenEmbedding(adapter, embed_dim, token_embed_dim=2).eval()
    with torch.no_grad():
        # level 0 -> e0, level 1 -> e1, everything else 0; coordinates identified
        # by their own one-hot so a swap cannot cancel.
        module.level_embedding.weight.zero_()
        module.level_embedding.weight[0, 0] = 1.0
        module.level_embedding.weight[1, 1] = 1.0
        module.coordinate_embedding.weight.zero_()
        module.mask_embedding.zero_()
        for projection in module.stream_projection:
            projection.weight.zero_()
            projection.bias.zero_()
        # Identity on the flattened stream block: the state *is* the flattened input.
        module.stream_projection[0].weight[:] = torch.eye(2 * 2)
        module.stream_embedding.weight.zero_()
    tokens = torch.zeros(1, 1, 4, dtype=torch.long)
    tokens[0, 0, 0] = 0
    tokens[0, 0, 1] = 1
    visible = torch.ones(1, 1, 4, dtype=torch.bool)
    with torch.no_grad():
        state = module(tokens, visible)
        expected = torch.tensor([1.0, 0.0, 0.0, 1.0])  # [level0+e0, level1+e0] flattened
        torch.testing.assert_close(state[0, 0, 0], expected)
        # Swapping the two levels swaps the two 2-vectors: a *different* state.
        swapped = tokens.clone()
        swapped[0, 0, 0], swapped[0, 0, 1] = 1, 0
        swapped_state = module(swapped, visible)
        torch.testing.assert_close(
            swapped_state[0, 0, 0], torch.tensor([0.0, 1.0, 1.0, 0.0])
        )
    assert not torch.equal(state[0, 0, 0], swapped_state[0, 0, 0])
    # The other stream is untouched by a swap inside stream 0.
    assert torch.equal(state[0, 0, 1], swapped_state[0, 0, 1])


def test_hidden_levels_are_invisible_but_the_mask_position_is_not():
    adapter = _TwoStreamAdapter()
    module = StreamTokenEmbedding(adapter, 4, token_embed_dim=2).eval()
    with torch.no_grad():
        module.mask_embedding.fill_(0.5)
        module.coordinate_embedding.weight.zero_()
        module.coordinate_embedding.weight[1, 1] = 1.0  # coordinate 1 has identity e1
    tokens = torch.zeros(1, 1, 4, dtype=torch.long)
    visible = torch.ones(1, 1, 4, dtype=torch.bool)
    visible[0, 0, 0] = False
    other = tokens.clone()
    other[0, 0, 0] = 8  # the hidden coordinate's level changes
    with torch.no_grad():
        first = module(tokens, visible)
        second = module(other, visible)
    # A hidden coordinate contributes the mask vector + its identity: its level is
    # unreadable, and that is exactly what "hidden" has to mean.
    assert torch.equal(first, second)
    # Hiding a *different* coordinate is distinguishable (identity survives masking).
    visible_other = torch.ones(1, 1, 4, dtype=torch.bool)
    visible_other[0, 0, 1] = False
    with torch.no_grad():
        third = module(tokens, visible_other)
    assert not torch.equal(first, third)


def test_per_stream_head_makes_coordinate_logits_context_dependent():
    """The a-b logit difference must move with the context, not only with the bias."""
    adapter = _TwoStreamAdapter()
    embed_dim = 4
    module = StreamTokenEmbedding(adapter, embed_dim, token_embed_dim=2).eval()
    head = StreamLevelHead(adapter, embed_dim, 9).eval()
    with torch.no_grad():
        module.level_embedding.weight.normal_(std=0.5, generator=torch.Generator().manual_seed(3))
        module.coordinate_embedding.weight.normal_(std=0.5, generator=torch.Generator().manual_seed(4))
        head.stream_heads[0].weight.normal_(std=0.5, generator=torch.Generator().manual_seed(5))
        head.stream_heads[0].bias.zero_()
        head.coordinate_bias.weight.zero_()
    tokens = torch.randint(0, 9, (1, 3, 4), generator=torch.Generator().manual_seed(6))
    full = torch.ones(1, 3, 4, dtype=torch.bool)
    masked = full.clone()
    masked[0, :, 3] = False  # hide a coordinate of the *other* stream
    with torch.no_grad():
        first = head(module(tokens, full))
        second = head(module(tokens, masked))
    assert torch.equal(head.coordinate_bias.weight, torch.zeros_like(head.coordinate_bias.weight))
    # With a zero static bias, any per-coordinate logit difference is dynamic.
    gap_first = (first[0, :, 0] - first[0, :, 1]).abs().max()
    gap_second = (second[0, :, 0] - second[0, :, 1]).abs().max()
    assert float(gap_first) > 1e-6 and float(gap_second) > 1e-6
    assert not torch.equal(first[0, :, 0], first[0, :, 1]), "coordinates must differ at all"


def test_transport_config_records_token_embed_dim_and_revision():
    view = adapter()
    model = MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0)
    config = model.config()
    assert config["architecture_revision"] == 2
    assert config["token_embed_dim"] == 16
    rebuilt = MotionTransportTransformer(view, **config)
    assert rebuilt.token_embed_dim == 16
    with pytest.raises(ValueError, match="architecture revision"):
        MotionTransportTransformer(view, dim=32, depth=1, heads=2, architecture_revision=1)
    # An old config (revision 1, no token_embed_dim) must not load silently.
    legacy = dict(config)
    legacy["architecture_revision"] = 1
    with pytest.raises(ValueError, match="architecture revision"):
        MotionTransportTransformer(view, **legacy)
    # Replaying a stored config is the path that must never mis-load: it carries
    # both fields, so a revision mismatch is caught before any state dict is read.
    stored = dict(config)
    stored["architecture_revision"] = 1
    with pytest.raises(ValueError, match="architecture revision"):
        MotionTransportTransformer(view, **stored)
    # The state dict of another revision cannot load either (and load_state_dict is
    # never called with strict=False anywhere in this code base).
    other = MotionTransportTransformer(view, dim=32, depth=1, heads=2, token_embed_dim=8)
    with pytest.raises(RuntimeError):
        MotionTransportTransformer(view, **config).load_state_dict(other.state_dict())


def test_per_stream_parameters_are_reported():
    view = adapter()
    model = MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0)
    embedding_params = sum(parameter.numel() for parameter in model.embedding.parameters())
    head_params = sum(parameter.numel() for parameter in model.head.parameters())
    # One projection per stream and one head per stream, each sized from the
    # layout's own stream slices.
    sizes = view.stream_sizes()
    token_embed_dim = model.token_embed_dim
    expected_embedding = (
        view.num_levels * token_embed_dim
        + view.num_coordinates * token_embed_dim
        + view.num_streams * 32
        + token_embed_dim
        + sum(size * token_embed_dim * 32 + 32 for size in sizes)
    )
    expected_head = sum(32 * size * 9 + size * 9 for size in sizes) + view.num_coordinates * 9
    assert embedding_params == expected_embedding
    assert head_params == expected_head
    assert token_embed_dim == 16 and len(sizes) == 13


# ---------------------------------------------------------------------------
# R07: explicit temporal position


def test_position_table_is_positional_long_and_dtype_exact():
    from stylized_motion.learning.mts_operator.temporal import (
        SinusoidalPositionEncoding,
        sinusoidal_positions,
    )

    table = sinusoidal_positions(8, 32)
    assert table.shape == (1, 8, 1, 32)
    # Different positions, and the same position always encodes the same way.
    assert not torch.allclose(table[0, 0, 0], table[0, 1, 0])
    torch.testing.assert_close(sinusoidal_positions(8, 32), table)
    # Longer than any training window: no fixed table to overflow.
    long_table = sinusoidal_positions(4096, 32)
    assert long_table.shape == (1, 4096, 1, 32)
    # dtype/device follow the caller, and the values are bounded.
    half = sinusoidal_positions(4, 32, dtype=torch.float16)
    assert half.dtype == torch.float16 and float(half.abs().max()) <= 1.0 + 1e-3
    module = SinusoidalPositionEncoding(32)
    hidden = torch.zeros(2, 5, 3, 32)
    torch.testing.assert_close(module(hidden), sinusoidal_positions(5, 32).expand_as(hidden))
    with pytest.raises(ValueError, match=r"\[B, T, S, 32\]"):
        module(torch.zeros(2, 5, 32))
    with pytest.raises(ValueError, match="position encoding"):
        SinusoidalPositionEncoding(32, kind="rope")
    with pytest.raises(ValueError, match="positive"):
        sinusoidal_positions(0, 32)


def test_all_hidden_frames_are_no_longer_structurally_equal():
    """Full masking used to make every frame's logits identical."""
    view = adapter()
    model = transport(view, graph_depth=0)
    tokens = torch.randint(0, 9, (1, 6, 40), generator=torch.Generator().manual_seed(2))
    hidden = torch.zeros(1, 6, 40, dtype=torch.bool)
    with torch.no_grad():
        logits = model(tokens, hidden).logits
        pairwise = max(
            float((logits[0, i] - logits[0, j]).abs().max()) for i in range(6) for j in range(6)
        )
        spread = float(logits[0].std(dim=0).mean())
    assert pairwise > 1e-3 and spread > 1e-4, "frames must differ by construction, not by noise"
    # ... while the hidden token values still cannot influence anything.
    with torch.no_grad():
        other = model(torch.zeros_like(tokens), hidden).logits
    torch.testing.assert_close(logits, other, rtol=0.0, atol=0.0)


def test_transport_is_no_longer_time_permutation_equivariant():
    view = adapter()
    model = transport(view, graph_depth=0)
    tokens = torch.randint(0, 9, (2, 6, 40), generator=torch.Generator().manual_seed(7))
    visible = torch.ones(2, 6, 40, dtype=torch.bool)
    permutation = torch.randperm(6, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        direct = model(tokens[:, permutation], visible).logits
        moved = model(tokens, visible).logits[:, permutation]
    assert float((direct - moved).abs().max()) > 1e-3


def test_positions_never_let_padding_into_valid_outputs():
    view = adapter()
    model = transport(view, graph_depth=0)
    tokens = torch.randint(0, 9, (2, 5, 40), generator=torch.Generator().manual_seed(11))
    visible = torch.ones(2, 5, 40, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)
    padded = torch.cat([tokens, torch.randint(0, 9, (2, 4, 40))], dim=1)
    padded_visible = torch.cat([visible, torch.ones(2, 4, 40, dtype=torch.bool)], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(2, 4, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        base = model(tokens, visible, valid_mask=valid).logits
        extended = model(padded, padded_visible, valid_mask=padded_valid).logits
    torch.testing.assert_close(extended[:, :5], base, rtol=0.0, atol=1e-5)


def test_causal_prefix_ignores_future_tokens_with_positions():
    view = adapter()
    model = transport(view, graph_depth=0, temporal_mode="causal")
    tokens = torch.randint(0, 9, (1, 8, 40), generator=torch.Generator().manual_seed(12))
    visible = torch.ones(1, 8, 40, dtype=torch.bool)
    edited = tokens.clone()
    edited[:, 4:] = (tokens[:, 4:] + 3) % 9
    with torch.no_grad():
        before = model(tokens, visible).logits
        after = model(edited, visible).logits
    torch.testing.assert_close(before[:, :4], after[:, :4], rtol=0.0, atol=1e-6)
    assert float((before[:, 4:] - after[:, 4:]).abs().max()) > 1e-6


def test_reference_descriptor_sees_temporal_order():
    """Reordering distinct frames must change the descriptor; equal frames may not."""
    view = adapter()
    torch.manual_seed(21)
    encoder = GlobalStyleEncoder(view, dim=32, depth=1, heads=2, dropout=0.0, output_dim=32).eval()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(4)) * 0.1)
    tokens = torch.randint(0, 9, (2, 8, 40), generator=torch.Generator().manual_seed(13))
    valid = torch.ones(2, 8, dtype=torch.bool)
    permutation = torch.randperm(8, generator=torch.Generator().manual_seed(14))
    with torch.no_grad():
        base = encoder(tokens, valid_mask=valid)
        permuted = encoder(tokens[:, permutation], valid_mask=valid)
        reordered_original = encoder(tokens[:, permutation][:, permutation.argsort()], valid_mask=valid)
    # The relative change is what matters: the descriptor's own scale is small.
    relative = float((base - permuted).abs().max()) / float(base.abs().mean())
    assert relative > 1e-3
    torch.testing.assert_close(base, reordered_original, rtol=1e-6, atol=1e-5)
    # Identical frames are the legal exception: no reordering can matter.
    flat = tokens[:, :1].expand(-1, 8, -1).contiguous()
    with torch.no_grad():
        flat_base = encoder(flat, valid_mask=valid)
        flat_permuted = encoder(flat[:, permutation], valid_mask=valid)
    torch.testing.assert_close(flat_base, flat_permuted, rtol=0.0, atol=1e-6)
    # Padding frames are excluded from the pooled descriptor.
    padded = torch.cat([tokens, torch.randint(0, 9, (2, 4, 40))], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(2, 4, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        padded_descriptor = encoder(padded, valid_mask=padded_valid)
    torch.testing.assert_close(padded_descriptor, base, rtol=0.0, atol=1e-5)
