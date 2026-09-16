"""MTS base transport: embeddings, masking, graph locality and the model contract."""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.contract import TokenSpec
from stylized_motion.learning.mts_operator.embeddings import StreamTokenEmbedding
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
    # Pooling is a mean over the stream's own coordinates plus its embedding.
    stream_ids = view.coordinate_stream_ids()
    selected = stream_ids == stream
    level = module.level_embedding(tokens[..., selected])
    identity = module.coordinate_embedding.weight[selected]
    expected = (level + identity).mean(dim=2) + module.stream_embedding.weight[stream]
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
