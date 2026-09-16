"""MTS metric functions: structure, zero-cases and the honesty of the report."""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.masking import MaskGenerator
from stylized_motion.learning.mts_operator.metrics import (
    aggregate,
    content_preservation,
    physics_metrics,
    reference_sensitivity,
    representation_metrics,
    strength_response,
    style_retrieval,
    support_locality,
    unavailable_metrics,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator, OperatorBatch
from stylized_motion.learning.mts_operator.operators import (
    AdditiveLogitField,
    BirthDeathCTMCOperator,
)
from stylized_motion.learning.mts_operator.style_encoder import StyleIDEncoder
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
from stylized_motion.learning.nef_probe import KinematicContext
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

LEVELS = 9
FRAMES = 6
STYLES = 3


def adapter() -> LayoutAdapter:
    index: dict[str, int] = {}
    names: list[str] = []
    parents: list[int] = []
    for chain in GENO_SKELETON.chains:
        for position, name in enumerate(chain):
            if name not in index:
                index[name] = len(names)
                names.append(name)
                parents.append(-1 if position == 0 else index[chain[position - 1]])
    return LayoutAdapter(NEFLayout.from_skeleton(names, parents))


def kinematic_context(view: LayoutAdapter) -> KinematicContext:
    motion_dim = view.layout.num_joints * 9 + 5
    return KinematicContext(
        feature_offset=torch.zeros(motion_dim),
        feature_scale=torch.full((motion_dim,), 0.3),
        ref_pos=torch.tile(torch.tensor([0.0, 0.9, 0.0]), (view.layout.num_joints, 1)),
        parents=tuple(int(value) for value in view.layout.parents),
        names=tuple(view.layout.names),
    )


def build_model(view: LayoutAdapter, *, operator=None, trained: bool = False) -> MtsStyleOperator:
    torch.manual_seed(0)
    transport = MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0)
    style_encoder = StyleIDEncoder(num_styles=STYLES, output_dim=32)
    operator = operator or AdditiveLogitField(
        num_levels=LEVELS, hidden_dim=32, coordinate_dim=16, style_dim=32, stream_dim=32
    )
    model = MtsStyleOperator(
        view, transport=transport, style_encoder=style_encoder, operator=operator, freeze_transport=True
    )
    if trained:
        from stylized_motion.learning.mts_operator.training import OperatorTrainer, TrainerConfig

        trainer = OperatorTrainer(
            model, adapter=view, device="cpu", config=TrainerConfig(epochs=1, lr=1e-2, log_every_steps=0)
        )
        for _ in range(40):
            trainer.train_step(fixed_batch(view, seed=2))
    model.eval()
    return model


def fixed_batch(view: LayoutAdapter, *, seed: int = 0, mask_kind: str = "full_generation") -> OperatorBatch:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, LEVELS, (4, FRAMES, 40), generator=generator)
    mask = MaskGenerator({mask_kind: 1.0}).sample_kind(mask_kind, 4, FRAMES, adapter=view)
    return OperatorBatch(
        target_tokens=tokens,
        style_ids=torch.randint(0, STYLES, (4,), generator=generator),
        visible_mask=mask.visible_mask,
        content_condition=torch.randint(0, 3, (4,), generator=generator),
        strength=1.0,
    )


def test_metric_documentation_is_explicit_about_what_is_not_computed():
    representation = representation_metrics()
    assert len(representation) == 10
    assert all(isinstance(value, str) and value for value in representation.values())
    unavailable = unavailable_metrics()
    assert set(unavailable) == {"motion_fid", "text_motion_alignment", "style_classifier_accuracy"}
    assert all(isinstance(value, str) and value for value in unavailable.values())


def test_strength_response_is_zero_at_zero_and_grows():
    view = adapter()
    model = build_model(view, trained=True)
    batch = fixed_batch(view, seed=3)
    curve = strength_response(model, batch, strengths=[0.0, 0.5, 1.0, 2.0])
    assert [point["strength"] for point in curve] == [0.0, 0.5, 1.0, 2.0]
    assert curve[0]["total_variation"] == pytest.approx(0.0, abs=1e-6)
    assert curve[0]["changed_token_ratio"] == 0.0
    assert curve[1]["total_variation"] >= curve[0]["total_variation"]
    assert curve[3]["total_variation"] >= curve[1]["total_variation"]
    for point in curve:
        assert 0.0 <= point["changed_token_ratio"] <= 1.0
        assert point["nll"] > 0.0
        assert point["styled_entropy"] > 0.0


def test_reference_sensitivity_reports_correct_wrong_and_random():
    view = adapter()
    model = build_model(view, trained=True)
    batch = fixed_batch(view, seed=5)
    generator = torch.Generator().manual_seed(6)
    wrong = torch.randint(0, LEVELS, batch.target_tokens.shape, generator=generator)
    random_reference = torch.randint(0, LEVELS, batch.target_tokens.shape, generator=generator)
    report = reference_sensitivity(model, batch, wrong_reference=wrong, random_reference=random_reference)
    for key in ("nll_correct", "nll_wrong", "nll_random", "nll_correct_minus_wrong"):
        assert key in report
        assert torch.isfinite(torch.tensor(report[key]))
    assert report["changed_token_ratio_wrong_reference"] >= 0.0
    assert report["total_variation_random_reference"] >= 0.0
    # Without comparisons the report is only the correct-reference NLL.
    plain = reference_sensitivity(model, batch)
    assert set(plain) == {"nll_correct"}
    assert plain["nll_correct"] == pytest.approx(report["nll_correct"])


def test_content_preservation_is_exact_when_nothing_was_edited():
    view = adapter()
    model = build_model(view, trained=True)
    batch = fixed_batch(view, seed=7)
    zero = OperatorBatch(
        target_tokens=batch.target_tokens,
        style_ids=batch.style_ids,
        visible_mask=batch.visible_mask,
        content_condition=batch.content_condition,
        strength=0.0,
    )
    report = content_preservation(model, zero)
    assert report["styled_nll"] == pytest.approx(report["base_nll"], rel=1e-6)
    assert report["argmax_change_ratio"] == 0.0
    edited = content_preservation(model, batch)
    assert edited["nll_increase"] >= -1e-6
    assert 0.0 <= edited["argmax_change_ratio"] <= 1.0


def test_support_locality_separates_inside_from_outside():
    view = adapter()
    model = build_model(view, trained=True)
    support = view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES)
    batch = OperatorBatch(
        target_tokens=torch.randint(0, LEVELS, (2, FRAMES, 40), generator=torch.Generator().manual_seed(8)),
        style_ids=torch.zeros(2, dtype=torch.long),
        hard_mask=support,
        strength=1.0,
    )
    report = support_locality(model, batch)
    assert report["support_fraction"] == pytest.approx(6 / 40)
    assert report["leakage"] == 0.0  # locked edit: outside the support nothing changes
    assert report["leaked_tokens"] == 0
    assert report["changed_token_ratio_inside"] >= 0.0
    assert "note" in support_locality(model, OperatorBatch(target_tokens=batch.target_tokens, style_ids=batch.style_ids))


def test_style_retrieval_ranks_the_matching_reference():
    view = adapter()
    model = build_model(view, trained=True)
    generator = torch.Generator().manual_seed(9)
    targets = torch.randint(0, LEVELS, (2, FRAMES, 40), generator=generator)
    references = [torch.randint(0, LEVELS, (FRAMES, 40), generator=generator) for _ in range(2)]
    batch = OperatorBatch(
        target_tokens=targets,
        style_ids=torch.zeros(2, dtype=torch.long),
        visible_mask=torch.zeros_like(targets, dtype=torch.bool),
    )
    report = style_retrieval(
        model,
        batch,
        candidate_sets=[references, references],
        correct_index=[0, 1],
    )
    assert report["candidates"] == 2.0 and report["targets"] == 2.0
    assert report["chance"] == pytest.approx(0.5)
    assert 0.0 <= report["top1_accuracy"] <= 1.0
    # A degenerate candidate set (only the correct one) always succeeds.
    single = style_retrieval(model, batch, candidate_sets=[[references[0]], [references[1]]], correct_index=[0, 0])
    assert single["top1_accuracy"] == 1.0
    with pytest.raises(ValueError, match="one candidate set"):
        style_retrieval(model, batch, candidate_sets=[references], correct_index=[0, 1])


def test_physics_metrics_are_zero_for_an_unchanged_motion_and_positive_otherwise():
    view = adapter()
    kinematic = kinematic_context(view)
    motion_dim = view.layout.num_joints * 9 + 5
    motion = torch.randn(1, 8, motion_dim, generator=torch.Generator().manual_seed(11)) * 0.1
    identical = physics_metrics(baseline_motion=motion, edited_motion=motion.clone(), kinematic=kinematic)
    for key in (
        "fk_change_mean", "fk_change_max", "root_position_change", "root_rotation_change",
        "contact_flip_rate", "foot_slide_edited", "foot_slide_baseline", "foot_skate_ratio",
    ):
        assert key in identical, key
    assert identical["fk_change_mean"] == 0.0
    assert identical["fk_change_max"] == 0.0
    assert identical["contact_flip_rate"] == 0.0

    changed = motion.clone()
    changed[:, :, 9:15] += 0.4
    report = physics_metrics(
        baseline_motion=motion, edited_motion=changed, kinematic=kinematic, edit_interval=(2, 6)
    )
    assert report["fk_change_mean"] > 0.0
    assert report["fk_change_max"] >= report["fk_change_mean"]
    assert "boundary_jerk_max" in report
    assert report["boundary_jerk_max"] > 0.0
    with pytest.raises(ValueError, match="share a shape"):
        physics_metrics(
            baseline_motion=motion, edited_motion=changed[:, :4], kinematic=kinematic
        )


def test_aggregate_summarizes_rows():
    rows = [{"a": 1.0, "b": 4.0}, {"a": 3.0, "b": 4.0}, {"a": 5.0}]
    summary = aggregate(rows)
    assert summary["a"]["mean"] == pytest.approx(3.0)
    assert summary["a"]["median"] == pytest.approx(3.0)
    assert summary["a"]["max"] == pytest.approx(5.0)
    assert summary["a"]["n"] == 3
    assert summary["b"]["n"] == 2
    assert aggregate([]) == {}
