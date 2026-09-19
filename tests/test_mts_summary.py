"""N00: the unified read-only summary and the counterexamples that created it.

Each test here freezes one of the failures the first round shipped:

* a conditioned arm scored with no condition at all (E04–E06 read the action
  vocabulary from a config key that never existed);
* three aggregations mixed under one unlabelled name (E05's "overall" was a row
  mean while the table said token-weighted, and the base appeared twice);
* an ordinal "adjacent fraction" that clamped the neighbour index and counted the
  boundary bin twice, reaching 2.0 at level 0;
* a NaN that reached the JSON because the writer's keys and the reader's keys
  disagreed;
* a verdict assembled from checks that were never executed.
"""

from __future__ import annotations

import json

import pytest
import torch

from stylized_motion.learning.mts_operator.metrics import ordinal_level_mass, wasserstein_1d
from stylized_motion.learning.mts_operator.summary import (
    AGGREGATIONS,
    OBJECTIVE_REPRODUCTION_TOLERANCE,
    ArmSpec,
    ReferencePool,
    assert_condition_present,
    check_status,
    condition_label,
    condition_vector,
    improvement,
    objective_reproduction,
    require_finite,
    stage_status,
    summarize_metric,
    transport_state_mismatches,
    write_summary,
)
from stylized_motion.learning.mts_operator.windows import ContentVocabulary

LEVELS = 9


class _Batch:
    """The two fields ``assert_condition_present`` reads, nothing else."""

    def __init__(self, content_condition=None):
        self.content_condition = content_condition


class _Arm:
    def __init__(self, name, vocabulary, encoder_kind="style_id"):
        self.spec = ArmSpec(name=name, path=None)  # type: ignore[arg-type]
        self.content_vocabulary = vocabulary
        self.encoder_kind = encoder_kind


# ---------------------------------------------------------------------------
# missing action


def test_a_conditioned_arm_without_a_condition_is_refused():
    """The E04–E06 bug: the vocabulary lookup missed and every arm was scored
    unconditioned.  Silence was the failure; the check must raise."""
    vocabulary = ContentVocabulary(kind="action_id", classes=("run", "walk"))
    arm = _Arm("style_id", vocabulary)
    with pytest.raises(ValueError, match="no content condition"):
        assert_condition_present(arm, _Batch(content_condition=None), omit_action=False)
    # With the condition present the same call passes.
    assert_condition_present(arm, _Batch(content_condition=torch.tensor([0])), omit_action=False)


def test_the_omitted_action_ablation_is_labelled_and_never_silent():
    """Omitting the action is allowed only as a *labelled* ablation."""
    vocabulary = ContentVocabulary(kind="action_id", classes=("run", "walk"))
    arm = _Arm("style_id", vocabulary, encoder_kind="reference")
    assert condition_label(arm, omit_action=False) == "action"
    assert condition_label(arm, omit_action=True) == "action_omitted"
    # The ablation passes the guard, because it is explicit.
    assert_condition_present(arm, _Batch(content_condition=None), omit_action=True)
    # An unconditional model has no condition to miss.
    unconditional = _Arm("plain", ContentVocabulary(kind="none"))
    assert condition_label(unconditional, omit_action=False) == "unconditional"
    assert_condition_present(unconditional, _Batch(), omit_action=False)


def test_condition_vector_uses_the_transport_vocabulary_and_errors_on_unknown_actions():
    vocabulary = ContentVocabulary(kind="action_id", classes=("run", "walk"))
    vector = condition_vector(vocabulary, [{"content": "walk"}, {"content": "run"}], omit_action=False, device="cpu")
    assert vector is not None and vector.tolist() == [1, 0]
    assert condition_vector(vocabulary, [{"content": "run"}], omit_action=True, device="cpu") is None
    with pytest.raises(ValueError, match="Unknown action"):
        condition_vector(vocabulary, [{"content": "cartwheel"}], omit_action=False, device="cpu")


# ---------------------------------------------------------------------------
# three aggregations


#: Two rows with deliberately unequal supervision and a protocol with two kinds:
#: every aggregation then gives a different, hand-checkable number.
_ROWS = (
    {"kind": "full_generation", "style": "a", "supervised_tokens": 10, "nll": 10.0},
    {"kind": "full_generation", "style": "a", "supervised_tokens": 30, "nll": 90.0},
    {"kind": "stream", "style": "b", "supervised_tokens": 60, "nll": 60.0},
)


def test_three_aggregations_are_computed_and_hand_checked():
    summary = summarize_metric(_ROWS, "nll", weights={"full_generation": 0.5, "stream": 0.5})
    # micro: total NLL over total tokens = (10 + 90 + 60) / 100
    assert summary["micro"]["aggregation"] == "micro"
    assert summary["micro"]["nll"] == pytest.approx(160.0 / 100.0)
    # macro_row: mean of the per-row means = (1.0 + 3.0 + 1.0) / 3
    assert summary["macro_row"]["aggregation"] == "macro_row"
    assert summary["macro_row"]["nll"] == pytest.approx(5.0 / 3.0)
    # protocol_weighted: token-weighted inside the kind, then the fixed weights:
    # 0.5 * (100/40) + 0.5 * (60/60)
    assert summary["protocol_weighted"]["aggregation"] == "protocol_weighted"
    assert summary["protocol_weighted"]["nll"] == pytest.approx(0.5 * 2.5 + 0.5 * 1.0)
    # The by-kind view is its own named aggregation, never folded into the others.
    assert summary["by_kind"]["full_generation"]["aggregation"] == "token_weighted_within_kind"
    assert summary["by_kind"]["full_generation"]["nll"] == pytest.approx(2.5)
    assert summary["by_style"]["b"]["nll"] == pytest.approx(1.0)
    # Every aggregation carries its label: a table cannot mix them by accident.
    assert set(AGGREGATIONS) <= set(summary)


def test_a_kind_without_a_protocol_weight_is_null_not_silently_dropped():
    summary = summarize_metric(_ROWS, "nll", weights={"full_generation": 1.0})
    assert summary["protocol_weighted"]["nll"] is None
    assert summary["protocol_weighted"]["missing_weights_for_kinds"] == ["stream"]


def test_improvement_is_positive_when_the_first_model_is_better():
    better = summarize_metric(_ROWS, "nll", weights={"full_generation": 0.5, "stream": 0.5})
    worse_rows = tuple(dict(row, nll=float(row["nll"]) + 2.0 * int(row["supervised_tokens"])) for row in _ROWS)
    worse = summarize_metric(worse_rows, "nll", weights={"full_generation": 0.5, "stream": 0.5})
    report = improvement(better, worse, label="wrong")
    for aggregation in AGGREGATIONS:
        assert report[aggregation]["nll_improvement_vs_wrong"] == pytest.approx(2.0)
        assert report[aggregation]["nll_improvement_vs_wrong"] > 0.0


def test_unscored_rows_are_counted_not_averaged_away():
    rows = _ROWS + ({"kind": "stream", "style": "b", "supervised_tokens": 0, "nll": 0.0},)
    summary = summarize_metric(rows, "nll", weights={"full_generation": 0.5, "stream": 0.5})
    assert summary["rows"] == 4 and summary["rows_scored"] == 3 and summary["rows_unscored"] == 1


# ---------------------------------------------------------------------------
# ordinal boundaries


def test_adjacent_mass_never_double_counts_the_boundary():
    """The E05 bug: ``clamp`` made level 0's neighbour level 0 itself, so the
    "fraction" was p(0)+2*p(1)+... and could reach 2."""
    probabilities = torch.zeros(1, 1, 1, LEVELS)
    probabilities[..., 0] = 1.0  # base mode is the boundary level
    base = probabilities.clone()
    report = ordinal_level_mass(probabilities, base)
    assert report["adjacent_mass_fraction"] == pytest.approx(1.0)
    assert report["adjacent_mass_fraction"] <= 1.0
    # Top boundary, same rule.
    top = torch.zeros(1, 1, 1, LEVELS)
    top[..., LEVELS - 1] = 1.0
    assert ordinal_level_mass(top, top)["adjacent_mass_fraction"] == pytest.approx(1.0)
    # In the interior the three bins are counted once each.
    middle = torch.zeros(1, 1, 1, LEVELS)
    middle[..., 3] = 0.5
    middle[..., 4] = 0.25
    middle[..., 5] = 0.25
    base_middle = torch.zeros(1, 1, 1, LEVELS)
    base_middle[..., 4] = 1.0
    interior = ordinal_level_mass(middle, base_middle)
    assert interior["adjacent_mass_fraction"] == pytest.approx(1.0)
    # 0.5 * |3 - 4| + 0.25 * |4 - 4| + 0.25 * |5 - 4| = 0.75
    assert interior["expected_level_displacement"] == pytest.approx(0.75)


def test_adjacent_mass_respects_supervision_and_reports_displacement():
    probabilities = torch.zeros(2, 1, 1, LEVELS)
    probabilities[0, ..., 4] = 1.0
    probabilities[1, ..., 0] = 1.0
    base = torch.zeros_like(probabilities)
    base[..., 4] = 1.0
    supervision = torch.tensor([[[True]], [[False]]])
    report = ordinal_level_mass(probabilities, base, supervision=supervision)
    assert report["supervised_positions"] == 1
    assert report["expected_level_displacement"] == pytest.approx(0.0)
    assert ordinal_level_mass(probabilities, base, supervision=torch.zeros(2, 1, 1, dtype=torch.bool))["adjacent_mass_fraction"] is None


def test_wasserstein_one_is_exact_for_a_one_level_shift():
    left = torch.zeros(1, LEVELS)
    left[..., 2] = 1.0
    right = torch.zeros(1, LEVELS)
    right[..., 3] = 1.0
    assert float(wasserstein_1d(left, right)) == pytest.approx(1.0)
    assert float(wasserstein_1d(left, left)) == pytest.approx(0.0)
    # A point mass at level 2 against the uniform distribution: |CDF gap| is
    # (1+2)/9 before the mass and (6+5+4+3+2+1)/9 after it.
    spread = torch.full((1, LEVELS), 1.0 / LEVELS)
    assert float(wasserstein_1d(left, spread)) == pytest.approx((1 + 2 + 6 + 5 + 4 + 3 + 2 + 1) / 9.0)


# ---------------------------------------------------------------------------
# NaN


def test_nan_and_inf_never_reach_a_summary(tmp_path):
    with pytest.raises(ValueError, match="not finite"):
        require_finite({"nll": float("nan")}, where="summary")
    with pytest.raises(ValueError, match="not finite"):
        require_finite({"rows": [{"a": 1.0}, {"a": float("inf")}]}, where="summary")
    # None is the way to say "not computed", and it is written as null.
    path = write_summary(tmp_path / "ok.json", {"nll": None, "rows": [1, 2]})
    assert json.loads(path.read_text())["nll"] is None
    with pytest.raises(ValueError, match="not finite"):
        write_summary(tmp_path / "bad.json", {"tv": float("nan")})
    assert not (tmp_path / "bad.json").exists()


# ---------------------------------------------------------------------------
# acceptance


def test_unexecuted_checks_are_never_counted_as_passes():
    report = check_status(
        {
            "ran_and_passed": {"passed": True},
            "ran_and_failed": {"passed": False},
            "never_run": {"passed": None},
        }
    )
    assert report["passed"] == ["ran_and_passed"]
    assert report["failed"] == ["ran_and_failed"]
    assert report["not_run"] == ["never_run"]
    assert report["all_passed"] is False


def test_stage_status_marks_every_unmeasured_label_not_evaluated():
    status = stage_status({"likelihood_signal": {"value": "weak_positive", "evidence": {"gap": 0.005}}})
    assert status["likelihood_signal"]["value"] == "weak_positive"
    for label in ("implementation_complete", "measurement_valid", "motion_quality", "style_fidelity", "generalization"):
        assert status[label]["value"] == "not_evaluated"
    with pytest.raises(ValueError, match="Unknown stage status"):
        stage_status({"everything_is_fine": "yes"})


# ---------------------------------------------------------------------------
# objective reproduction


def test_objective_reproduction_tolerance_is_the_fp32_one():
    assert OBJECTIVE_REPRODUCTION_TOLERANCE == 1e-5
    assert objective_reproduction(1.5381096229, 1.5381096)["passed"] is True
    failed = objective_reproduction(1.5381096229, 1.5382)
    assert failed["passed"] is False and "does not reproduce" in failed["reason"]
    assert objective_reproduction(None, 1.0)["passed"] is None  # nothing to compare against


# ---------------------------------------------------------------------------
# reference pool


def test_reference_pool_never_reuses_a_take_source_or_mirror():
    labels = {
        1: {"clip_id": 1, "action": "walk", "style": "neutral", "source_group": 10, "source_id": 100, "mirror": False},
        2: {"clip_id": 2, "action": "walk", "style": "injured leg", "source_group": 11, "source_id": 101, "mirror": False},
        3: {"clip_id": 3, "action": "walk", "style": "injured leg", "source_group": 10, "source_id": 100, "mirror": True},
        4: {"clip_id": 4, "action": "walk", "style": "injured leg", "source_group": 12, "source_id": 102, "mirror": True},
        5: {"clip_id": 5, "action": "run", "style": "injured leg", "source_group": 13, "source_id": 103, "mirror": False},
        6: {"clip_id": 6, "action": "walk", "style": "injured torso", "source_group": 14, "source_id": 104, "mirror": False},
    }
    pool = ReferencePool(labels=labels, by_action={"walk": [1, 2, 3, 4, 6], "run": [5]})
    # Clip 6 is the only candidate that is neither the excluded correct reference
    # (2), a same-take/same-source variant (3) nor a mirrored clip (4).
    pick = pool.pick(1, exclude=(2,))
    assert pick is not None and pick["clip_id"] == 6
    # A wrong-style reference must really be a different style.
    assert pool.pick(1, exclude=(2,), other_style=True)["style"] != labels[1]["style"]
    # No legal candidate in another action group: None, never a rolled tensor.
    assert pool.pick(5, exclude=()) is None


# ---------------------------------------------------------------------------
# transport identity


def test_transport_mismatch_is_named_tensor_by_tensor():
    class _Model:
        def __init__(self, state):
            self.transport = type("T", (), {"state_dict": lambda self, s=state: s})()

    left = _Model({"a": torch.zeros(2)})
    right = _Model({"a": torch.ones(2)})
    assert transport_state_mismatches(left, right) == ["a"]
    assert transport_state_mismatches(left, left) == []
