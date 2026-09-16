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
from stylized_motion.learning.mts_operator.model import MtsStyleOperator, OperatorBatch
from stylized_motion.learning.mts_operator.operators import (
    AdditiveLogitField,
    ArbitraryKernelOperator,
    BirthDeathCTMCOperator,
)
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder, StyleIDEncoder
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


def test_content_preservation_term_changes_the_loss():
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
    _, plain = model.loss(batch, content_weight=0.0)
    loss, weighted = model.loss(batch, content_weight=0.5)
    assert "base_nll" in weighted and "base_nll" not in plain
    assert float(loss.detach()) >= float(weighted["nll"])
    # The supervision statement is explicit: only hidden, supported positions count.
    supervision = batch.supervision_mask(model.spec)
    assert bool((supervision & batch.visible_mask).sum() == 0)
    assert bool(supervision.any())
    with pytest.raises(ValueError, match="supervises nothing"):
        empty = OperatorBatch(
            target_tokens=tokens,
            style_ids=style_ids,
            visible_mask=torch.ones_like(tokens, dtype=torch.bool),
            content_condition=content_ids,
        )
        model.loss(empty)


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
    with pytest.raises(ValueError, match="reference_tokens"):
        reference_model(OperatorBatch(target_tokens=tokens, style_ids=style_ids))
    with pytest.raises(ValueError, match="style_ids"):
        model(OperatorBatch(target_tokens=tokens))
