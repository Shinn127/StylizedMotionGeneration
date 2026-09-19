"""End-to-end MTS operator pipeline: style response, content preservation, locking.

The synthetic task is deliberately learnable: a content pattern plus a style
level offset.  A model that has learned the style must score the *correct*
reference better than a random one, and must not change tokens outside its
support.  That is the Phase 3/4 acceptance statement in its cheapest form.
"""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.masking import MaskGenerator
from stylized_motion.learning.mts_operator.sampling import CommonRandomNumbers
from stylized_motion.learning.mts_operator.model import MtsStyleOperator, OperatorBatch
from stylized_motion.learning.mts_operator.operators import (
    AdditiveLogitField,
    ArbitraryKernelOperator,
    BirthDeathCTMCOperator,
)
from stylized_motion.learning.mts_operator.style_encoder import (
    ConstantStyleEncoder,
    GlobalStyleEncoder,
    StyleIDEncoder,
)
from stylized_motion.learning.mts_operator.training import (
    OperatorTrainer,
    TrainerConfig,
    TransportTrainer,
)
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

LEVELS = 9
FRAMES = 6
STYLES = 3
CONTENTS = 4


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


def content_pattern(content: int, frames: int = FRAMES) -> torch.Tensor:
    """A deterministic per-content level pattern over [T, 40]."""
    base = torch.arange(frames).unsqueeze(-1) + torch.arange(40).unsqueeze(0)
    return (base + 2 * content) % LEVELS


def styled_batch(
    *,
    batch: int,
    style_ids: torch.Tensor,
    content_ids: torch.Tensor,
) -> torch.Tensor:
    tokens = torch.stack([content_pattern(int(c)) for c in content_ids])
    offsets = (style_ids * 3).view(-1, 1, 1)
    return (tokens + offsets) % LEVELS


def sample_batch(batch: int = 6, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    style_ids = torch.randint(0, STYLES, (batch,), generator=generator)
    content_ids = torch.randint(0, CONTENTS, (batch,), generator=generator)
    return styled_batch(batch=batch, style_ids=style_ids, content_ids=content_ids), style_ids, content_ids


def transport(view: LayoutAdapter, *, seed: int = 0) -> MotionTransportTransformer:
    torch.manual_seed(seed)
    return MotionTransportTransformer(
        view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0, content_classes=CONTENTS
    )


_TRANSPORT_CACHE: dict[int, MotionTransportTransformer] = {}


def train_transport(view: LayoutAdapter, *, steps: int = 40, seed: int = 0) -> MotionTransportTransformer:
    """Trains the frozen upstream once per test session (this CPU is slow)."""
    cached = _TRANSPORT_CACHE.get(steps)
    if cached is not None:
        return cached
    model = transport(view, seed=seed)
    trainer = TransportTrainer(
        model,
        adapter=view,
        mask_generator=MaskGenerator({"full_generation": 1.0}),
        device="cpu",
        config=TrainerConfig(epochs=1, lr=2e-2, log_every_steps=0),
    )
    tokens, _, content_ids = sample_batch(seed=1)
    for _ in range(steps):
        trainer.train_step(tokens, content_condition=content_ids)
    _TRANSPORT_CACHE[steps] = model
    return model


def perturb(module: torch.nn.Module, *, seed: int = 0, std: float = 0.2) -> torch.nn.Module:
    """Non-zero parameters, so a zero-initialized head cannot fake a response."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * std)
    return module


def build_model(
    view: LayoutAdapter,
    transport_model: MotionTransportTransformer,
    *,
    style_encoder: torch.nn.Module,
    operator: torch.nn.Module,
    freeze_transport: bool = True,
) -> MtsStyleOperator:
    return MtsStyleOperator(
        view,
        transport=transport_model,
        style_encoder=style_encoder,
        operator=operator,
        freeze_transport=freeze_transport,
    )


def operator_batch(view: LayoutAdapter, *, mask_kind: str = "full_generation", strength=1.0):
    tokens, style_ids, content_ids = sample_batch(seed=3)
    mask = MaskGenerator({mask_kind: 1.0}).sample_kind(
        mask_kind, tokens.shape[0], tokens.shape[1], adapter=view
    )
    return OperatorBatch(
        target_tokens=tokens,
        reference_tokens=tokens,
        visible_mask=mask.visible_mask,
        strength=strength,
        content_condition=content_ids,
        style_ids=style_ids,
        kind=mask_kind,
    )


def test_style_id_sandbox_separates_style_from_content():
    view = adapter()
    transport_model = train_transport(view)
    style_encoder = StyleIDEncoder(num_styles=STYLES, output_dim=32)
    operator = AdditiveLogitField(num_levels=LEVELS, hidden_dim=32, coordinate_dim=16, style_dim=32, stream_dim=32)
    model = build_model(view, transport_model, style_encoder=style_encoder, operator=operator)
    trainer = OperatorTrainer(
        model, adapter=view, device="cpu", config=TrainerConfig(epochs=1, lr=1e-2, log_every_steps=0)
    )
    tokens, style_ids, content_ids = sample_batch(seed=1)
    mask = MaskGenerator({"full_generation": 1.0}).sample_kind("full_generation", tokens.shape[0], FRAMES, adapter=view)
    batch = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=mask.visible_mask,
        content_condition=content_ids,
    )
    for _ in range(80):
        trainer.train_step(batch)

    def nll_with(ids: torch.Tensor) -> float:
        probe = OperatorBatch(
            target_tokens=tokens,
            style_ids=ids,
            visible_mask=mask.visible_mask,
            content_condition=content_ids,
        )
        with torch.no_grad():
            loss, _ = model.loss(probe)
        return float(loss)

    correct = nll_with(style_ids)
    wrong = nll_with((style_ids + 1) % STYLES)
    shuffled = nll_with(torch.tensor([7 % STYLES, 1, 2, 0, 2, 1][: tokens.shape[0]]))
    assert correct < wrong
    assert correct < shuffled

    # Same content, different style id: the styled distribution must move.
    def probabilities(ids: torch.Tensor) -> torch.Tensor:
        probe = OperatorBatch(
            target_tokens=tokens,
            style_ids=ids,
            visible_mask=mask.visible_mask,
            content_condition=content_ids,
        )
        with torch.no_grad():
            return model(probe).probabilities

    styled_a = probabilities(torch.full((tokens.shape[0],), 0, dtype=torch.long))
    styled_b = probabilities(torch.full((tokens.shape[0],), 2, dtype=torch.long))
    assert float((styled_a - styled_b).abs().max()) > 1e-3

    # strength 0 is exactly the base distribution, whatever the style says.
    zero = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=mask.visible_mask,
        strength=0.0,
        content_condition=content_ids,
    )
    with torch.no_grad():
        result = model(zero)
    assert torch.equal(result.probabilities, result.base_probabilities)


def test_reference_encoder_beats_a_random_reference():
    view = adapter()
    transport_model = train_transport(view)
    style_encoder = GlobalStyleEncoder(view, dim=32, depth=1, heads=4, graph_depth=0, output_dim=32)
    operator = AdditiveLogitField(num_levels=LEVELS, hidden_dim=32, coordinate_dim=16, style_dim=32, stream_dim=32)
    model = build_model(view, transport_model, style_encoder=style_encoder, operator=operator)
    trainer = OperatorTrainer(
        model, adapter=view, device="cpu", config=TrainerConfig(epochs=1, lr=1e-2, log_every_steps=0)
    )
    # Targets and references share the style but differ in content.
    tokens, style_ids, content_ids = sample_batch(seed=5)
    reference_content = (content_ids + 1) % CONTENTS
    reference_tokens = styled_batch(
        batch=tokens.shape[0], style_ids=style_ids, content_ids=reference_content
    )
    wrong_style_ids = (style_ids + 1) % STYLES
    wrong_reference = styled_batch(
        batch=tokens.shape[0], style_ids=wrong_style_ids, content_ids=reference_content
    )
    mask = MaskGenerator({"full_generation": 1.0}).sample_kind(
        "full_generation", tokens.shape[0], FRAMES, adapter=view
    )
    batch = OperatorBatch(
        target_tokens=tokens,
        reference_tokens=reference_tokens,
        visible_mask=mask.visible_mask,
        content_condition=content_ids,
    )
    for _ in range(90):
        trainer.train_step(batch)

    def nll(reference: torch.Tensor) -> float:
        probe = OperatorBatch(
            target_tokens=tokens,
            reference_tokens=reference,
            visible_mask=mask.visible_mask,
            content_condition=content_ids,
        )
        with torch.no_grad():
            loss, _ = model.loss(probe)
        return float(loss)

    correct = nll(reference_tokens)
    assert correct < nll(wrong_reference)
    # A random (unrelated) reference is the worst case.
    random_reference = torch.randint(
        0, LEVELS, tokens.shape, generator=torch.Generator().manual_seed(9)
    )
    assert correct < nll(random_reference)


@pytest.mark.parametrize(
    "operator_factory",
    [
        lambda: AdditiveLogitField(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
        lambda: ArbitraryKernelOperator(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
        lambda: BirthDeathCTMCOperator(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
    ],
)
def test_every_operator_family_trains_and_locks_its_support(operator_factory):
    view = adapter()
    transport_model = train_transport(view, steps=20)
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        operator=operator_factory(),
    )
    trainer = OperatorTrainer(
        model, adapter=view, device="cpu", config=TrainerConfig(epochs=1, lr=5e-3, log_every_steps=0)
    )
    support = view.hard_mask(["left_arm", "right_arm"], graph_radius=1, length=FRAMES)
    tokens, style_ids, content_ids = sample_batch(seed=7)
    mask = MaskGenerator({"random_coordinate": 1.0}).sample_kind(
        "random_coordinate", tokens.shape[0], FRAMES, adapter=view
    )
    batch = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=mask.visible_mask,
        hard_mask=support,
        content_condition=content_ids,
    )
    first = trainer.train_step(batch)
    for _ in range(30):
        last = trainer.train_step(batch)
    assert float(last["loss"]) < float(first["loss"])
    assert all(torch.isfinite(torch.tensor(value)) for value in last.values())

    # Locked edit: tokens outside the support are copied, never resampled.
    drawn = model.generate_edit(
        batch, generator=torch.Generator().manual_seed(13)
    )
    assert bool((drawn[:, ~support] == tokens[:, ~support]).all())
    assert int(drawn.min()) >= 0 and int(drawn.max()) <= LEVELS - 1
    # Latents (content) survive: the drawn support tokens are not simply copied.
    assert bool((drawn[:, support] != tokens[:, support]).any())


def test_content_weight_is_rejected_and_empty_supervision_reports_a_count():
    view = adapter()
    transport_model = train_transport(view, steps=20)
    tokens, style_ids, content_ids = sample_batch(seed=11)
    mask = MaskGenerator({"spatiotemporal_block": 1.0}).sample_kind(
        "spatiotemporal_block", tokens.shape[0], FRAMES, adapter=view
    )
    batch = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=mask.visible_mask,
        content_condition=content_ids,
    )
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        operator=AdditiveLogitField(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    loss, metrics = model.loss(batch, content_weight=0.0)
    assert torch.isfinite(loss.detach())
    assert {"nll", "nll_sum", "supervised_tokens", "correct_tokens"} <= set(metrics)
    # The frozen-base "content" term was measured against the base distribution
    # rather than against content, so revision 2 removed it and refuses to run
    # with a weight that would silently do nothing.
    with pytest.raises(ValueError, match="content_weight"):
        model.loss(batch, content_weight=0.5)
    # The supervision statement is explicit: only hidden, supported positions count.
    supervision = batch.supervision_mask(model.spec)
    assert bool((supervision & batch.visible_mask).sum() == 0)
    assert bool(supervision.any())
    # An all-visible batch supervises nothing: it reports a count and a
    # graph-connected zero instead of raising or inventing a flattering loss.
    empty = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=torch.ones_like(tokens, dtype=torch.bool),
        content_condition=content_ids,
    )
    empty_loss, empty_metrics = model.loss(empty)
    assert float(empty_loss.detach()) == 0.0
    assert empty_loss.requires_grad
    assert int(empty_metrics["supervised_tokens"]) == 0
    assert float(empty_metrics["nll_sum"]) == 0.0


def test_hidden_token_values_never_reach_the_base_distribution():
    """The transport sees the mask, not the answer: no target-token peeking."""
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=29)
    model = perturb(
        build_model(
            view,
            transport_model,
            style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
            operator=AdditiveLogitField(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
        ),
        seed=3,
    )
    region = view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES).unsqueeze(0).expand_as(tokens)
    assert bool(region.any()) and not bool(region.all())
    visible = ~region
    shuffled = tokens.clone()
    shuffled[region] = (tokens[region] + 3) % LEVELS

    def make(values: torch.Tensor) -> OperatorBatch:
        return OperatorBatch(
            target_tokens=values,
            style_ids=style_ids,
            visible_mask=visible,
            content_condition=content_ids,
        )

    with torch.no_grad():
        first = model(make(tokens))
        second = model(make(shuffled))
    # Hidden values are masked before the embedding, so nothing about them may
    # change the base the operator is applied to — or the styled output.
    assert torch.equal(first.base_probabilities, second.base_probabilities)
    assert torch.equal(first.probabilities, second.probabilities)
    # The supervision itself does use the true tokens, so the loss does differ.
    loss_a, _ = model.loss(make(tokens))
    loss_b, _ = model.loss(make(shuffled))
    assert float(loss_a.detach()) != float(loss_b.detach())


def test_style_and_strength_move_only_the_region():
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=31)
    model = perturb(
        build_model(
            view,
            transport_model,
            style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
            operator=BirthDeathCTMCOperator(
                num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32
            ),
        ),
        seed=5,
    )
    region = view.hard_mask(["right_leg"], graph_radius=1, length=FRAMES).unsqueeze(0).expand_as(tokens)
    valid = torch.ones(tokens.shape[0], FRAMES, dtype=torch.bool)
    valid[:, -2:] = False  # padding frames
    outside = ~region | ~valid.unsqueeze(-1)

    def run(strength: float, styles: torch.Tensor):
        batch = OperatorBatch(
            target_tokens=tokens,
            style_ids=styles,
            visible_mask=~region,
            target_valid_mask=valid,
            content_condition=content_ids,
            strength=strength,
        )
        with torch.no_grad():
            result = model(batch)
        return result

    base_result = run(1.0, style_ids)
    other_styles = (style_ids + 1) % STYLES
    other = run(1.0, other_styles)
    weak = run(0.25, style_ids)
    for result in (base_result, other, weak):
        # Whatever the style or strength, positions outside the region (including
        # padding) keep the base distribution bit for bit.
        torch.testing.assert_close(
            result.probabilities[outside], result.base_probabilities[outside], rtol=0.0, atol=0.0
        )
    assert float(
        (base_result.probabilities[region] - other.probabilities[region]).abs().max()
    ) > 1e-4
    assert float((base_result.probabilities[region] - weak.probabilities[region]).abs().max()) > 1e-4
    # A locked draw keeps the same positions and only varies inside the region.
    batch = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=~region,
        target_valid_mask=valid,
        content_condition=content_ids,
        strength=1.0,
    )
    crn = CommonRandomNumbers(seed=99)
    drawn = model.generate_edit(batch, crn=crn, sample_id=0)
    again = model.generate_edit(batch, crn=crn, sample_id=0)
    independent = model.generate_edit(batch, crn=crn, sample_id=1)
    assert torch.equal(drawn, again)
    assert len(crn.uniforms_cache) == 2
    for sample in (drawn, again, independent):
        assert torch.equal(sample[outside], tokens[outside])
    # Coupling is what makes the base-vs-styled comparison paired.
    with torch.no_grad():
        base_draw = model.generate_edit(
            OperatorBatch(
                target_tokens=tokens, style_ids=style_ids, visible_mask=~region,
                target_valid_mask=valid, content_condition=content_ids, strength=0.0,
            ),
            crn=crn, sample_id=0,
        )
    assert torch.equal(base_draw[outside], tokens[outside])


def test_generation_locks_anchors_and_an_empty_region_is_the_source():
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=17)
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        operator=AdditiveLogitField(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    support = view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES)
    region = support.unsqueeze(0).expand_as(tokens)
    batch = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        # Local edit: observed tokens are evidence, the region is authored.
        visible_mask=~region,
        hard_mask=support,
        content_condition=content_ids,
        strength=1.0,
    )
    drawn = model.generate_edit(batch, generator=torch.Generator().manual_seed(23))
    assert bool((drawn[~region] == tokens[~region]).all())
    # Anchors inside the region are locked as well, and only the rest is resampled.
    # Half of the region's frames are anchors: the same mask used by the loss.
    anchor = torch.zeros_like(tokens, dtype=torch.bool)
    anchor[:, : max(FRAMES // 2, 1)] = region[:, : max(FRAMES // 2, 1)]
    assert bool(anchor.any())
    anchored = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=~region,
        hard_mask=support,
        anchor_mask=anchor,
        content_condition=content_ids,
        strength=1.0,
    )
    anchored_edit = anchored.effective_edit_mask(model.spec)
    assert not bool((anchored_edit & anchor).any())
    assert bool(anchored_edit.any())
    anchored_draw = model.generate_edit(anchored, generator=torch.Generator().manual_seed(23))
    assert bool((anchored_draw[anchor] == tokens[anchor]).all())
    assert bool((anchored_draw[~region] == tokens[~region]).all())
    assert bool((anchored_draw[anchored_edit] ==
                 model.generate_edit(anchored, generator=torch.Generator().manual_seed(23))[anchored_edit]).all())
    # An empty region returns the source tokens exactly, not a sample of anything.
    empty = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        hard_mask=torch.zeros(FRAMES, 40, dtype=torch.bool),
        content_condition=content_ids,
        strength=1.0,
    )
    empty_draw = model.generate_edit(empty, generator=torch.Generator().manual_seed(23))
    assert torch.equal(empty_draw, tokens)
    # The generation path defaults its observed set to the complement of the hard
    # mask; the loss path refuses to guess at all.
    with pytest.raises(ValueError, match="visible_mask"):
        model.loss(OperatorBatch(target_tokens=tokens, style_ids=style_ids, hard_mask=support))


def test_operator_evaluate_aggregates_by_supervised_tokens_not_batch_means():
    """NLL of a uniform operator is log 9 whatever fraction of tokens is scored."""
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=5)
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        # Default (non identity_mix) kernel: at initialization the kernel is
        # exactly uniform, so every supervised token has NLL = log 9.
        operator=ArbitraryKernelOperator(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    trainer = OperatorTrainer(model, adapter=view, config=TrainerConfig(epochs=1, log_every_steps=0), device="cpu")
    expected = float(torch.log(torch.tensor(float(LEVELS))))

    def batch_with_fraction(fraction: float) -> OperatorBatch:
        visible = torch.rand(1, FRAMES, 40, generator=torch.Generator().manual_seed(3)) >= fraction
        return OperatorBatch(
            target_tokens=tokens[:1],
            style_ids=style_ids[:1],
            visible_mask=visible,
            content_condition=content_ids[:1],
        )

    sparse = batch_with_fraction(0.05)
    dense = batch_with_fraction(0.95)
    sparse_fraction = float(sparse.supervision_mask(model.spec).float().mean())
    dense_fraction = float(dense.supervision_mask(model.spec).float().mean())
    assert sparse_fraction < 0.1 and dense_fraction > 0.9
    for batch, fraction in ((sparse, sparse_fraction), (dense, dense_fraction)):
        report = trainer.evaluate([batch])
        if int(report["supervised_tokens"]) == 0:
            continue
        assert report["nll"] == pytest.approx(expected, rel=1e-5)
        # The removed implementation reported mean_batch_nll * supervision_fraction.
        assert report["nll"] != pytest.approx(expected * fraction, rel=1e-2)
    # Both batches together: still exactly log 9, never a mixture of the two means.
    combined = trainer.evaluate([sparse, dense, sparse])
    assert combined["nll"] == pytest.approx(expected, rel=1e-5)
    assert combined["supervised_tokens"] == (
        int(sparse.supervision_mask(model.spec).sum())
        + int(dense.supervision_mask(model.spec).sum())
        + int(sparse.supervision_mask(model.spec).sum())
    )


def test_operator_trainer_skips_a_batch_without_supervision():
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=7)
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        operator=AdditiveLogitField(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    trainer = OperatorTrainer(model, adapter=view, config=TrainerConfig(epochs=1, log_every_steps=0), device="cpu")
    before = [parameter.detach().clone() for parameter in model.trainable_parameters()]
    empty = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=torch.ones_like(tokens, dtype=torch.bool),
        content_condition=content_ids,
    )
    record = trainer.train_step(empty)
    assert record["loss"] is None and record["skipped"] == 1.0
    assert trainer.global_step == 0
    for original, current in zip(before, model.trainable_parameters()):
        torch.testing.assert_close(original, current.detach(), rtol=0.0, atol=0.0)
    assert not trainer.optimizer.state_dict()["state"]
    # A skipped batch must not dilute a real one, and the epoch log survives.
    lines: list[str] = []
    view_mask = MaskGenerator({"spatiotemporal_block": 1.0}).sample_kind(
        "spatiotemporal_block", tokens.shape[0], FRAMES, adapter=view
    )
    real = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=view_mask.visible_mask,
        content_condition=content_ids,
    )
    result = trainer.fit(lambda epoch: [empty, real, empty], epochs=1, log=lines.append)
    assert trainer.global_step == 1
    assert result["history"][0]["skipped_steps"] == pytest.approx(2.0)
    assert result["history"][0]["steps"] == pytest.approx(3.0)
    assert result["history"][0]["supervised_tokens"] > 0
    assert all("loss=" in line for line in lines)
    assert any("skipped=2" in line for line in lines)


def test_operator_evaluate_reports_none_when_nothing_is_supervised():
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=9)
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        operator=BirthDeathCTMCOperator(num_levels=LEVELS, hidden_dim=16, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    trainer = OperatorTrainer(model, adapter=view, config=TrainerConfig(epochs=1, log_every_steps=0), device="cpu")
    empty = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=torch.ones_like(tokens, dtype=torch.bool),
        content_condition=content_ids,
    )
    report = trainer.evaluate([empty, empty])
    assert report["loss"] is None and report["nll"] is None
    assert report["supervised_tokens"] == 0 and report["batches"] == 2
    assert trainer.evaluate([])["batches"] == 0


def test_operator_model_reports_its_configuration_and_requires_matching_alphabet():
    view = adapter()
    transport_model = train_transport(view, steps=10)
    model = build_model(
        view,
        transport_model,
        style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
        operator=BirthDeathCTMCOperator(num_levels=LEVELS, hidden_dim=16, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    described = model.describe()
    assert described["operator"]["name"] == "birth_death"
    assert described["freeze_transport"] is True
    assert described["uses_style_ids"] is True
    assert described["token_spec"]["num_coordinates"] == 40
    trainable = list(model.trainable_parameters())
    assert trainable, "the operator must expose trainable parameters"
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in model.transport.parameters())

    mismatched = BirthDeathCTMCOperator(num_levels=8, hidden_dim=8, coordinate_dim=8, style_dim=8, stream_dim=32)
    with pytest.raises(ValueError, match="levels"):
        MtsStyleOperator(
            view,
            transport=transport_model,
            style_encoder=StyleIDEncoder(num_styles=2, output_dim=8),
            operator=mismatched,
        )
    reference_model = build_model(
        view,
        transport_model,
        style_encoder=GlobalStyleEncoder(view, dim=16, depth=1, heads=2),
        operator=AdditiveLogitField(num_levels=LEVELS, hidden_dim=16, coordinate_dim=8, style_dim=16, stream_dim=32),
    )
    tokens, style_ids, content_ids = sample_batch(seed=13)
    observed = torch.zeros_like(tokens, dtype=torch.bool)
    with pytest.raises(ValueError, match="reference_tokens"):
        reference_model(OperatorBatch(target_tokens=tokens, style_ids=style_ids, visible_mask=observed))
    with pytest.raises(ValueError, match="style_ids"):
        model(OperatorBatch(target_tokens=tokens, visible_mask=observed))
    # A missing visible_mask is its own error now, not a silent all-visible default.
    with pytest.raises(ValueError, match="visible_mask"):
        model(OperatorBatch(target_tokens=tokens, style_ids=style_ids))


def test_iterative_filling_never_peeks_at_uncommitted_tokens():
    """R12: a position's value may not shape the draw before it is committed.

    Two batches that differ *only* at positions the schedule commits in the second
    step must produce step-1 tokens that are bitwise equal.  If the model could see
    the answer early, the first commit would move with it.
    """
    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=41)
    model = perturb(
        build_model(
            view,
            transport_model,
            style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
            operator=AdditiveLogitField(num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32),
        ),
        seed=7,
    )
    region = view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES).unsqueeze(0).expand_as(tokens)
    steps = 3
    from stylized_motion.learning.mts_operator.sampling import monotonic_fill_steps

    schedule = monotonic_fill_steps(region.clone(), steps)
    assert len(schedule) == steps and bool(schedule[0].any()) and bool(schedule[-1].any())
    later = schedule[-1]

    def make(values: torch.Tensor) -> OperatorBatch:
        return OperatorBatch(
            target_tokens=values,
            style_ids=style_ids,
            visible_mask=~region,
            hard_mask=region[0],
            content_condition=content_ids,
            strength=1.0,
        )

    altered = tokens.clone()
    altered[later] = (tokens[later] + 4) % LEVELS
    generator_a = torch.Generator(device="cpu").manual_seed(1234)
    generator_b = torch.Generator(device="cpu").manual_seed(1234)
    with torch.no_grad():
        drawn_a, commits_a = model.generate_edit(
            make(tokens), generator=generator_a, steps=steps, return_trace=True
        )
        drawn_b, commits_b = model.generate_edit(
            make(altered), generator=generator_b, steps=steps, return_trace=True
        )
    for step in range(steps):
        assert torch.equal(commits_a[step], commits_b[step]), f"step {step} committed differently"
    # Step 1 draws from a context that does not contain the altered positions, so it
    # must be identical; the later steps legitimately diverge.
    first_commit = schedule[0]
    assert torch.equal(drawn_a[first_commit], drawn_b[first_commit])
    # Reproducibility at each step count: same seed, same result.
    for count in (1, 2, 8):
        with torch.no_grad():
            one = model.generate_edit(
                make(tokens), generator=torch.Generator().manual_seed(99), steps=count
            )
            two = model.generate_edit(
                make(tokens), generator=torch.Generator().manual_seed(99), steps=count
            )
        assert torch.equal(one, two), f"steps={count} is not reproducible"
    # Outside the region nothing is ever resampled, whatever the step count.
    traced, _ = model.generate_edit(
        make(tokens), generator=torch.Generator().manual_seed(5), steps=8, return_trace=True
    )
    assert torch.equal(traced[~region], tokens[~region])
    # A step count above the number of editable positions is legal.
    many = model.generate_edit(
        make(tokens), generator=torch.Generator().manual_seed(5), steps=200
    )
    assert torch.equal(many[~region], tokens[~region])


# ---------------------------------------------------------------------------
# T03: the no-reference control reads no style input at all


def test_the_constant_descriptor_reads_no_style_input():
    """T03: the control is "the model never looks at the reference", not a shuffle.

    A shuffled or permuted reference still feeds a reference-dependent input, so a
    difference could come from the encoder reading order rather than from style.
    This control has no reference input at all: the probabilities must be identical
    for the true reference, a shuffled one, a zeroed one, or none, and identical
    for the true, wrong or absent style id.
    """
    view = adapter()
    transport_model = train_transport(view)
    torch.manual_seed(3)
    width = int(transport_model.dim)
    const_encoder = ConstantStyleEncoder(output_dim=width)
    id_encoder = StyleIDEncoder(num_styles=STYLES, output_dim=width)
    ref_encoder = GlobalStyleEncoder(
        view, dim=width, depth=1, heads=2, graph_depth=0, dropout=0.0, output_dim=width
    )
    def operator_for_this_transport():
        return AdditiveLogitField(
            num_levels=LEVELS,
            hidden_dim=width,
            coordinate_dim=8,
            style_dim=width,
            stream_dim=width,
        )

    constant_model = build_model(
        view, transport_model, style_encoder=const_encoder, operator=operator_for_this_transport()
    ).eval()
    id_model = build_model(
        view, transport_model, style_encoder=id_encoder, operator=operator_for_this_transport()
    ).eval()
    ref_model = build_model(
        view, transport_model, style_encoder=ref_encoder, operator=operator_for_this_transport()
    ).eval()

    batch = operator_batch(view)
    shuffled = OperatorBatch(
        target_tokens=batch.target_tokens,
        reference_tokens=batch.reference_tokens.flip(0),
        visible_mask=batch.visible_mask,
        strength=batch.strength,
        content_condition=batch.content_condition,
        style_ids=(batch.style_ids + 1) % STYLES,
        kind=batch.kind,
    )
    without_reference = OperatorBatch(
        target_tokens=batch.target_tokens,
        reference_tokens=None,
        visible_mask=batch.visible_mask,
        strength=batch.strength,
        content_condition=batch.content_condition,
        style_ids=None,
        kind=batch.kind,
    )
    with torch.no_grad():
        baseline = constant_model(batch).probabilities
        assert torch.equal(constant_model(shuffled).probabilities, baseline)
        assert torch.equal(constant_model(without_reference).probabilities, baseline)
    # The control's descriptor is a constant, and the parameter difference against
    # the arms it controls is reported rather than hidden.
    const_parameters = sum(p.numel() for p in const_encoder.parameters())
    assert const_parameters == const_encoder.output_dim
    assert const_parameters < sum(p.numel() for p in id_encoder.parameters())
    assert const_parameters < sum(p.numel() for p in ref_encoder.parameters())
    # The two *input-reading* arms do move when the style input changes: the
    # control is only meaningful because they are not constant.  Their style heads
    # need non-zero weights for that (a freshly built operator starts at a zero
    # effect, which is exactly the state the constant arm stays in).
    perturb(id_model.operator, seed=11)
    perturb(ref_model.operator, seed=12)
    with torch.no_grad():
        assert not torch.equal(id_model(shuffled).probabilities, id_model(batch).probabilities)
        assert not torch.equal(
            ref_model(shuffled).probabilities, ref_model(batch).probabilities
        )


def test_the_constant_control_can_still_learn():
    """T03: a no-reference control is only useful if it can still be trained."""
    view = adapter()
    transport_model = train_transport(view)
    torch.manual_seed(5)
    encoder = ConstantStyleEncoder(output_dim=int(transport_model.dim))
    model = build_model(
        view,
        transport_model,
        style_encoder=encoder,
        operator=AdditiveLogitField(
            num_levels=LEVELS,
            hidden_dim=int(transport_model.dim),
            coordinate_dim=8,
            style_dim=int(transport_model.dim),
            stream_dim=int(transport_model.dim),
        ),
    )
    batch = operator_batch(view)
    trainer = OperatorTrainer(
        model, adapter=view, device="cpu", config=TrainerConfig(epochs=1, lr=0.01)
    )
    before = encoder.descriptor.detach().clone()
    metrics = [trainer.train_step(batch) for _ in range(3)]
    assert all(value["loss"] is not None for value in metrics)
    assert not torch.equal(encoder.descriptor.detach(), before), "the constant must receive gradient"


def test_a_multi_step_styled_draw_is_not_silently_the_base_draw():
    """N03: at steps>1 the block being drawn must be *hidden* when it is sampled.

    The iterative path used to mark the current commit block visible before drawing
    it, so the operator (which only edits hidden positions) never styled it: a
    4-step styled draw came back bitwise equal to a 4-step base draw even though the
    two distributions differed (TV > 0).  The counterexample is the identity anchor
    at strength 0, where the two *are* equal by construction -- so a test that only
    compares them cannot tell the two situations apart, and both are asserted here.
    """
    from stylized_motion.learning.mts_operator.sampling import CommonRandomNumbers

    view = adapter()
    transport_model = train_transport(view, steps=10)
    tokens, style_ids, content_ids = sample_batch(seed=43)
    model = perturb(
        build_model(
            view,
            transport_model,
            style_encoder=StyleIDEncoder(num_styles=STYLES, output_dim=16),
            operator=AdditiveLogitField(
                num_levels=LEVELS, hidden_dim=24, coordinate_dim=8, style_dim=16, stream_dim=32
            ),
        ),
        seed=11,
    )
    region = view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES)
    batch = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=~region.unsqueeze(0).expand_as(tokens),
        hard_mask=region,
        content_condition=content_ids,
        strength=1.0,
    )
    crn = CommonRandomNumbers(seed=5)
    with torch.no_grad():
        base_draw = model.generate_edit(batch, crn=crn, sample_id=0, step_id=4, steps=4, use_base=True)
        styled_draw = model.generate_edit(batch, crn=crn, sample_id=0, step_id=4, steps=4, use_base=False)
        result = model(batch)
    tv = float(0.5 * (result.probabilities - result.base_probabilities).abs().sum(-1).mean())
    assert tv > 0.0, "the fixture must have a styled distribution that differs from the base"
    inside = region.unsqueeze(0).expand_as(base_draw)
    changed = float((styled_draw != base_draw)[inside].float().mean())
    assert changed > 0.0, (
        "a multi-step styled draw that is bitwise equal to the base draw means the "
        "block was visible while it was sampled"
    )
    # The anchor still holds: at strength 0 the two distributions coincide and the
    # coupled draws must be identical.
    anchored = OperatorBatch(
        target_tokens=tokens,
        style_ids=style_ids,
        visible_mask=~region.unsqueeze(0).expand_as(tokens),
        hard_mask=region,
        content_condition=content_ids,
        strength=0.0,
    )
    crn = CommonRandomNumbers(seed=5)
    with torch.no_grad():
        base_zero = model.generate_edit(anchored, crn=crn, sample_id=0, step_id=4, steps=4, use_base=True)
        styled_zero = model.generate_edit(anchored, crn=crn, sample_id=0, step_id=4, steps=4, use_base=False)
    assert torch.equal(base_zero, styled_zero)
