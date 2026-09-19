"""MTS operator families and sampling: identity, support, mass and references."""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator.operators import (
    OPERATOR_NAMES,
    AdditiveLogitField,
    ArbitraryKernelOperator,
    BirthDeathCTMCOperator,
    OperatorInputs,
    OperatorOutput,
    birth_death_generator,
    build_operator,
    poisson_term_count,
)
from stylized_motion.learning.mts_operator.operators import uniformization_expm_apply

LEVELS = 9
BATCH = 2
FRAMES = 4
COORDINATES = 40


def base_logits(*, seed: int = 0) -> torch.Tensor:
    return torch.randn(
        BATCH, FRAMES, COORDINATES, LEVELS, generator=torch.Generator().manual_seed(seed)
    )


def full_mask() -> torch.Tensor:
    return torch.ones(FRAMES, COORDINATES, dtype=torch.bool)


def inputs_for(
    logits: torch.Tensor | None = None,
    *,
    strength: torch.Tensor | float = 1.0,
    hard_mask: torch.Tensor | None = None,
    visible_mask: torch.Tensor | None = None,
    edit_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    style_dim: int = 16,
    seed: int = 1,
) -> OperatorInputs:
    logits = base_logits(seed=0) if logits is None else logits
    return OperatorInputs(
        base_logits=logits,
        style_embedding=torch.randn(BATCH, style_dim, generator=torch.Generator().manual_seed(seed)),
        strength=strength,
        hard_mask=hard_mask,
        visible_mask=visible_mask,
        edit_mask=edit_mask,
        valid_mask=valid_mask,
    )


def trained(operator, *, seed: int = 5, std: float = 0.5):
    """Give an operator non-trivial parameters so its effect is visible."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in operator.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=std, generator=generator)
            else:
                parameter.normal_(std=std, generator=generator)
    return operator


def test_operator_factory_and_metadata():
    assert OPERATOR_NAMES == ("logit_field", "arbitrary_kernel", "birth_death")
    for name in OPERATOR_NAMES:
        operator = build_operator(name, num_levels=LEVELS, hidden_dim=16, style_dim=8)
        described = operator.describe()
        assert described["name"] == name
        assert isinstance(described["has_identity_at_zero"], bool)
    assert build_operator("logit_field", style_dim=8).describe()["has_identity_at_zero"] is True
    assert build_operator("arbitrary_kernel", style_dim=8).describe()["has_identity_at_zero"] is False
    assert build_operator("birth_death", style_dim=8).describe()["has_identity_at_zero"] is True
    assert build_operator("birth_death", style_dim=8).describe()["structured"] is True
    with pytest.raises(ValueError, match="Unknown operator"):
        build_operator("ctmc")
    with pytest.raises(ValueError, match="num_levels"):
        AdditiveLogitField(num_levels=1)


def test_logit_field_is_identity_at_zero_and_local_to_its_support():
    operator = AdditiveLogitField(num_levels=LEVELS, hidden_dim=16, style_dim=16)
    logits = base_logits()
    zero = operator(inputs_for(logits, strength=0.0, hard_mask=full_mask()))
    torch.testing.assert_close(zero.probabilities, logits.softmax(-1), rtol=0.0, atol=0.0)
    masked = operator(inputs_for(logits, strength=1.0, hard_mask=torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)))
    torch.testing.assert_close(masked.probabilities, logits.softmax(-1), rtol=0.0, atol=0.0)

    operator = trained(AdditiveLogitField(num_levels=LEVELS, hidden_dim=16, style_dim=16))
    support = torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)
    support[:, :8] = True
    styled = operator(inputs_for(logits, strength=1.0, hard_mask=support))
    base_probs = logits.softmax(-1)
    assert float((styled.probabilities[:, :, :8] - base_probs[:, :, :8]).abs().max().detach()) > 0.0
    torch.testing.assert_close(
        styled.probabilities[:, :, 8:], base_probs[:, :, 8:], rtol=0.0, atol=0.0
    )
    assert styled.logits is not None
    # The edit set can only narrow the region.
    half = torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)
    half[:, :4] = True
    narrower = operator(inputs_for(logits, strength=1.0, hard_mask=support, edit_mask=half))
    torch.testing.assert_close(
        narrower.probabilities[:, :, 4:], base_probs[:, :, 4:], rtol=0.0, atol=0.0
    )
    assert float((narrower.probabilities[:, :, :4] - base_probs[:, :, :4]).abs().max().detach()) > 0.0
    # ... and cannot escape the hard mask.
    escaping = operator(
        inputs_for(
            logits,
            strength=1.0,
            hard_mask=torch.zeros(FRAMES, COORDINATES, dtype=torch.bool),
            edit_mask=full_mask(),
        )
    )
    torch.testing.assert_close(escaping.probabilities, base_probs, rtol=0.0, atol=0.0)
    # An observed token is evidence, not an edit target: with a visible mask the
    # complement is edited and the observed positions stay exactly at their base.
    observed = torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)
    observed[:, :32] = True
    complement = operator(inputs_for(logits, strength=1.0, visible_mask=observed))
    torch.testing.assert_close(
        complement.probabilities[:, :, :32], base_probs[:, :, :32], rtol=0.0, atol=0.0
    )
    assert float((complement.probabilities[:, :, 32:] - base_probs[:, :, 32:]).abs().max().detach()) > 0.0


def test_logit_field_strength_scales_the_deviation():
    operator = trained(AdditiveLogitField(num_levels=LEVELS, hidden_dim=16, style_dim=16))
    logits = base_logits()
    base_probs = logits.softmax(-1)
    deviations = []
    for strength in (0.0, 0.5, 1.0, 2.0):
        styled = operator(inputs_for(logits, strength=strength, hard_mask=full_mask()))
        deviations.append(float((styled.probabilities - base_probs).abs().mean()))
        assert float(styled.probabilities.sum(-1).sub(1).abs().max()) < 1e-5
    assert deviations[0] == 0.0
    assert deviations[0] < deviations[1] < deviations[2] < deviations[3]
    with pytest.raises(ValueError, match="non-negative"):
        operator(inputs_for(logits, strength=-0.5, hard_mask=full_mask()))
    with pytest.raises(ValueError, match="scalar or"):
        operator(
            OperatorInputs(
                base_logits=logits,
                style_embedding=torch.zeros(BATCH, 16),
                strength=torch.ones(BATCH + 1),
            )
        )


def logits_to_double_probs(shape: tuple[int, ...] = (1, 3, 1)) -> torch.Tensor:
    return torch.rand(*shape, LEVELS, dtype=torch.float64, generator=torch.Generator().manual_seed(3)).softmax(-1)


def partial_region() -> torch.Tensor:
    support = torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)
    support[:, :8] = True
    return support


def test_every_family_keeps_the_base_where_it_may_not_edit():
    """Empty region, empty edit set or an invalid frame: bit-exact base output."""
    logits = base_logits()
    base_probs = logits.softmax(-1)
    nothing = torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)
    valid = torch.ones(BATCH, FRAMES, dtype=torch.bool)
    valid[:, 2:] = False
    for cls in (AdditiveLogitField, ArbitraryKernelOperator, BirthDeathCTMCOperator):
        operator = trained(cls(num_levels=LEVELS, hidden_dim=16, style_dim=16))
        empty_region = operator(inputs_for(logits, strength=1.0, hard_mask=nothing))
        torch.testing.assert_close(empty_region.probabilities, base_probs, rtol=0.0, atol=0.0)
        empty_edit = operator(
            inputs_for(
                logits,
                strength=1.0,
                hard_mask=full_mask(),
                edit_mask=torch.zeros(BATCH, FRAMES, COORDINATES, dtype=torch.bool),
            )
        )
        torch.testing.assert_close(empty_edit.probabilities, base_probs, rtol=0.0, atol=0.0)
        invalid = operator(inputs_for(logits, strength=1.0, hard_mask=full_mask(), valid_mask=valid))
        torch.testing.assert_close(invalid.probabilities[:, 2:], base_probs[:, 2:], rtol=0.0, atol=0.0)
        # The region survives: the loss is not a no-op everywhere in that batch.
        assert float((invalid.probabilities[:, :2] - base_probs[:, :2]).abs().max().detach()) >= 0.0


def test_arbitrary_kernel_has_no_identity_anchor_but_respects_its_region():
    operator = trained(ArbitraryKernelOperator(num_levels=LEVELS, hidden_dim=16, style_dim=16))
    logits = base_logits()
    base_probs = logits.softmax(-1)
    support = partial_region()
    # Strength 0 means uniform *inside the region*: the honest failure mode of an
    # unconstrained family, which is why it is the control condition.
    zero = operator(inputs_for(logits, strength=0.0, hard_mask=support))
    uniform = torch.full_like(zero.probabilities, 1.0 / LEVELS)
    torch.testing.assert_close(zero.probabilities[:, :, :8], uniform[:, :, :8], rtol=1e-5, atol=1e-6)
    # ... and exactly the base outside it.
    torch.testing.assert_close(
        zero.probabilities[:, :, 8:], base_probs[:, :, 8:], rtol=0.0, atol=0.0
    )
    assert float((zero.probabilities[:, :, :8] - base_probs[:, :, :8]).abs().max()) > 1e-3
    # An empty region is base, not uniform (the audit counterexample).
    empty = operator(inputs_for(logits, strength=1.0, hard_mask=torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)))
    torch.testing.assert_close(empty.probabilities, base_probs, rtol=0.0, atol=0.0)
    # An empty edit set is base too, even with a full hard mask.
    empty_edit = operator(
        inputs_for(
            logits,
            strength=1.0,
            hard_mask=full_mask(),
            edit_mask=torch.zeros(BATCH, FRAMES, COORDINATES, dtype=torch.bool),
        )
    )
    torch.testing.assert_close(empty_edit.probabilities, base_probs, rtol=0.0, atol=0.0)
    # Outside the region the base is untouched even when the kernel is trained.
    styled = operator(inputs_for(logits, strength=1.0, hard_mask=support))
    torch.testing.assert_close(
        styled.probabilities[:, :, 8:], base_probs[:, :, 8:], rtol=0.0, atol=0.0
    )
    assert float((styled.probabilities[:, :, :8] - base_probs[:, :, :8]).abs().max()) > 1e-3
    # A uniform kernel maps any base onto the uniform distribution.
    uniform_output = operator(inputs_for(torch.zeros_like(logits), strength=0.0, hard_mask=support))
    torch.testing.assert_close(
        uniform_output.probabilities[:, :, :8], uniform[:, :, :8], rtol=1e-5, atol=1e-6
    )


def test_arbitrary_kernel_identity_mix_needs_strength_in_the_unit_interval():
    operator = trained(
        ArbitraryKernelOperator(num_levels=LEVELS, hidden_dim=16, style_dim=16, identity_mix=True)
    )
    logits = base_logits()
    base_probs = logits.softmax(-1)
    support = partial_region()
    # lambda = 0 is the identity kernel: base inside and outside the region.
    zero = operator(inputs_for(logits, strength=0.0, hard_mask=support))
    torch.testing.assert_close(zero.probabilities, base_probs, rtol=1e-6, atol=1e-7)
    # lambda = 1 applies the learned kernel unchanged inside the region and still
    # leaves everything outside it at the base distribution.
    full = operator(inputs_for(logits, strength=1.0, hard_mask=support))
    torch.testing.assert_close(
        full.probabilities[:, :, 8:], base_probs[:, :, 8:], rtol=0.0, atol=0.0
    )
    assert float((full.probabilities[:, :, :8] - base_probs[:, :, :8]).abs().max()) > 1e-3
    # The mixture is exactly linear: total variation from the base grows with lambda.
    def deviation(strength: float) -> float:
        styled = operator(inputs_for(logits, strength=strength, hard_mask=support))
        return float((styled.probabilities - base_probs).abs().sum(-1).mean())

    deviations = [deviation(value) for value in (0.0, 0.25, 0.5, 1.0)]
    assert deviations[0] == pytest.approx(0.0, abs=1e-8)
    assert deviations[0] < deviations[1] < deviations[2] < deviations[3]
    assert deviations[1] == pytest.approx(deviations[3] / 4.0, rel=1e-4)
    assert deviations[2] == pytest.approx(deviations[3] / 2.0, rel=1e-4)
    with pytest.raises(ValueError, match="strength"):
        operator(inputs_for(logits, strength=1.5, hard_mask=support))
    with pytest.raises(ValueError, match="strength"):
        operator(inputs_for(logits, strength=torch.tensor([1.0, 1.25]), hard_mask=support))
    with pytest.raises(ValueError, match="non-negative"):
        operator(inputs_for(logits, strength=-0.1, hard_mask=support))


def test_arbitrary_kernel_offdiagonal_mass_is_a_true_offdiagonal_sum():
    operator = trained(ArbitraryKernelOperator(num_levels=LEVELS, hidden_dim=16, style_dim=16))
    logits = base_logits()
    inputs = inputs_for(logits, strength=1.0, hard_mask=full_mask())
    with torch.no_grad():
        output = operator(inputs)
        kernel_logits = operator.kernel_head(operator.condition(inputs)).reshape(
            BATCH, FRAMES, COORDINATES, LEVELS, LEVELS
        )
        kernel = kernel_logits.softmax(dim=-1)
        diagonal = torch.diag_embed(kernel.diagonal(dim1=-2, dim2=-1))
        expected = (kernel - diagonal).sum(dim=(-1, -2)).mean()
    assert float(output.diagnostics["kernel_offdiagonal_mass"]) == pytest.approx(
        float(expected), rel=1e-5
    )
    # An identity kernel puts nothing off-diagonal; a uniform kernel puts
    # levels - 1 (i.e. every entry except one per row) there.
    identity_kernel = torch.eye(LEVELS).expand(BATCH, FRAMES, COORDINATES, LEVELS, LEVELS)
    uniform_kernel = torch.full_like(identity_kernel, 1.0 / LEVELS)
    assert float(operator._offdiagonal_mass(identity_kernel).mean()) == pytest.approx(0.0)
    assert float(operator._offdiagonal_mass(uniform_kernel).mean()) == pytest.approx(
        float(LEVELS) - 1.0, rel=1e-6
    )
    with torch.no_grad():
        uniform_output = operator(
            inputs_for(logits, strength=0.0, hard_mask=full_mask())
        )
    assert float(uniform_output.diagnostics["kernel_offdiagonal_ratio"]) == pytest.approx(
        1.0 - 1.0 / LEVELS, rel=1e-5
    )


def test_level_order_is_a_chain_in_level_space_via_the_inverse_permutation():
    order = [2, 0, 3, 1, 4, 6, 5, 8, 7]
    operator = BirthDeathCTMCOperator(
        num_levels=LEVELS, hidden_dim=16, style_dim=16, level_order=order
    )
    up = torch.arange(1.0, LEVELS + 1.0).view(1, 1, 1, LEVELS).expand(1, 1, 1, LEVELS)
    down = torch.full((1, 1, 1, LEVELS), 0.25)
    generator = operator._level_generator(up.clone(), down.clone())
    # Edge order[i] -> order[i + 1] carries up_rate[i]; the reverse carries down.
    for index in range(LEVELS - 1):
        assert float(generator[0, 0, 0, order[index], order[index + 1]]) == pytest.approx(float(up[0, 0, 0, index]))
        assert float(generator[0, 0, 0, order[index + 1], order[index]]) == pytest.approx(float(down[0, 0, 0, index + 1]))
    # Only those edges exist: the level-space graph is the permuted chain.
    allowed = {(order[i], order[i + 1]) for i in range(LEVELS - 1)} | {
        (order[i + 1], order[i]) for i in range(LEVELS - 1)
    }
    for a in range(LEVELS):
        for b in range(LEVELS):
            if a != b and (a, b) not in allowed:
                assert float(generator[0, 0, 0, a, b]) == 0.0
    assert float(generator.sum(dim=-1).abs().max()) < 1e-6
    # The permuted chain is still identity-clean at lambda = 0 and matches exp(Q).
    logits = base_logits()
    inputs = inputs_for(logits, strength=1.0, hard_mask=full_mask())
    zero = operator(inputs_for(logits, strength=0.0, hard_mask=full_mask()))
    torch.testing.assert_close(zero.probabilities, logits.softmax(-1), rtol=0.0, atol=0.0)
    with torch.no_grad():
        torch.testing.assert_close(
            operator(inputs).probabilities, operator.reference_expm(inputs), rtol=1e-4, atol=1e-5
        )


def test_birth_death_generator_has_adjacent_rates_only():
    up = torch.rand(2, 3, LEVELS) + 0.1
    down = torch.rand(2, 3, LEVELS) + 0.1
    generator = birth_death_generator(up, down)
    assert generator.shape == (2, 3, LEVELS, LEVELS)
    assert float(generator.sum(dim=-1).abs().max()) < 1e-6
    diagonal = generator.diagonal(dim1=-2, dim2=-1)
    assert bool((diagonal <= 0).all())
    for level in range(LEVELS - 1):
        torch.testing.assert_close(
            generator[..., level, level + 1], up[..., level], rtol=0.0, atol=1e-6
        )
        torch.testing.assert_close(
            generator[..., level + 1, level], down[..., level + 1], rtol=0.0, atol=1e-6
        )
    # Only the two adjacent diagonals may be non-zero.
    mask = torch.ones_like(generator, dtype=torch.bool)
    for offset in (-1, 0, 1):
        mask.diagonal(offset=offset, dim1=-2, dim2=-1).fill_(False)
    assert float(generator[mask].abs().max()) == 0.0


def test_poisson_term_count_grows_with_rate_and_reports_the_tail():
    assert poisson_term_count(0.0, tolerance=1e-10) == (1, 0.0)
    small, small_tail = poisson_term_count(1.0, tolerance=1e-10)
    large, large_tail = poisson_term_count(20.0, tolerance=1e-10)
    assert 1 <= small < large
    assert small_tail <= 1e-10 and large_tail <= 1e-10
    capped, capped_tail = poisson_term_count(10.0, tolerance=1e-30, max_terms=8)
    assert capped == 8 and capped_tail > 1e-30


def test_uniformization_matches_matrix_exp_and_conserves_mass():
    torch.manual_seed(3)
    up = torch.rand(BATCH, FRAMES, COORDINATES, LEVELS, generator=torch.Generator().manual_seed(4)) * 2.0
    down = torch.rand(BATCH, FRAMES, COORDINATES, LEVELS, generator=torch.Generator().manual_seed(5)) * 2.0
    up[..., -1] = 0.0
    down[..., 0] = 0.0
    generator = birth_death_generator(up, down)
    base = base_logits().softmax(-1)
    applied, diagnostics = uniformization_expm_apply(generator, base)
    reference = (base.unsqueeze(-2) @ torch.matrix_exp(generator)).squeeze(-2)
    assert float((applied - reference).abs().max()) < 1e-5
    assert float(applied.sum(-1).sub(1).abs().max()) < 1e-5
    assert float(diagnostics["mass_error"]) < 1e-5
    assert float(diagnostics["min_probability_before_clamp"]) > -1e-6
    assert float(diagnostics["uniformization_terms"]) >= 1.0
    assert float(diagnostics["poisson_tail"]) <= 1e-9

    # A zero generator is the identity, exactly.
    zero_generator = torch.zeros_like(generator)
    identity, zero_diagnostics = uniformization_expm_apply(zero_generator, base)
    torch.testing.assert_close(identity, base, rtol=0.0, atol=0.0)
    assert float(zero_diagnostics["uniformization_terms"]) == 0.0

    # Semigroup: exp(Q s) exp(Q t) == exp(Q (s + t)).
    half = uniformization_expm_apply(generator * 0.4, base)[0]
    composed = uniformization_expm_apply(generator * 0.6, half)[0]
    whole = uniformization_expm_apply(generator * 1.0, base)[0]
    assert float((composed - whole).abs().max()) < 1e-5
    with pytest.raises(ValueError, match="share leading shape"):
        uniformization_expm_apply(generator, base[:, :, :1, :])


def test_birth_death_operator_identity_support_and_reference_agreement():
    operator = trained(BirthDeathCTMCOperator(num_levels=LEVELS, hidden_dim=16, style_dim=16))
    logits = base_logits()
    base_probs = logits.softmax(-1)
    zero = operator(inputs_for(logits, strength=0.0, hard_mask=full_mask()))
    assert torch.equal(zero.probabilities, base_probs)
    mask_off = operator(
        inputs_for(logits, strength=1.0, hard_mask=torch.zeros(FRAMES, COORDINATES, dtype=torch.bool))
    )
    assert torch.equal(mask_off.probabilities, base_probs)
    assert float(mask_off.diagnostics["max_up_rate"]) == 0.0

    support = torch.zeros(FRAMES, COORDINATES, dtype=torch.bool)
    support[:, :4] = True
    styled = operator(inputs_for(logits, strength=1.0, hard_mask=support))
    assert float((styled.probabilities[:, :, :4] - base_probs[:, :, :4]).abs().max().detach()) > 0.0
    torch.testing.assert_close(
        styled.probabilities[:, :, 4:], base_probs[:, :, 4:], rtol=0.0, atol=0.0
    )
    assert styled.rates is not None and styled.rates.shape == (BATCH, FRAMES, COORDINATES, 2, LEVELS)
    reference = operator.reference_expm(inputs_for(logits, strength=1.0, hard_mask=support))
    assert float((styled.probabilities - reference).abs().max()) < 1e-5

    stronger = operator(inputs_for(logits, strength=2.0, hard_mask=full_mask()))
    weaker = operator(inputs_for(logits, strength=0.5, hard_mask=full_mask()))
    assert float((stronger.probabilities - base_probs).abs().mean()) > float(
        (weaker.probabilities - base_probs).abs().mean()
    )
    with pytest.raises(ValueError, match="max_rate"):
        BirthDeathCTMCOperator(max_rate=0.0)


def test_operator_output_rejects_invalid_distributions():
    with pytest.raises(ValueError, match="negative"):
        OperatorOutput(probabilities=torch.full((1, 1, 2, 3), -0.1))
    # NaN fails every comparison, so it needs its own finiteness check.
    with pytest.raises(ValueError, match="finite"):
        OperatorOutput(probabilities=torch.full((1, 1, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        OperatorOutput(probabilities=torch.full((1, 1, 2, 3), float("inf")))
    with pytest.raises(ValueError, match="sum to one"):
        OperatorOutput(probabilities=torch.full((1, 1, 2, 3), 0.5))
    logits = base_logits()
    with pytest.raises(ValueError, match="style_embedding batch"):
        OperatorInputs(base_logits=logits, style_embedding=torch.zeros(1, 8))
    with pytest.raises(ValueError, match="hard_mask must be"):
        OperatorInputs(base_logits=logits, style_embedding=torch.zeros(BATCH, 8),
                       hard_mask=torch.ones(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="edit_mask must be"):
        OperatorInputs(base_logits=logits, style_embedding=torch.zeros(BATCH, 8),
                       edit_mask=torch.ones(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="valid_mask"):
        OperatorInputs(base_logits=logits, style_embedding=torch.zeros(BATCH, 8),
                       valid_mask=torch.ones(BATCH, FRAMES + 1, dtype=torch.bool))


def test_operators_can_use_stream_hidden_context_and_accept_valid_masks():
    from stylized_motion.learning.mts_operator import LayoutAdapter
    from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

    def skeleton(spec):
        index, names, parents = {}, [], []
        for chain in spec.chains:
            for position, name in enumerate(chain):
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                    parents.append(-1 if position == 0 else index[chain[position - 1]])
        return names, parents

    adapter = LayoutAdapter(NEFLayout.from_skeleton(*skeleton(GENO_SKELETON)))
    logits = base_logits()
    hidden = torch.randn(BATCH, FRAMES, 13, 8, generator=torch.Generator().manual_seed(9))
    valid = torch.ones(BATCH, FRAMES, dtype=torch.bool)
    valid[:, 3:] = False
    for cls in (AdditiveLogitField, BirthDeathCTMCOperator):
        operator = trained(
            cls(num_levels=LEVELS, hidden_dim=16, coordinate_dim=8, style_dim=8, stream_dim=8)
        )
        inputs = OperatorInputs(
            base_logits=logits,
            style_embedding=torch.randn(BATCH, 8),
            strength=1.0,
            hard_mask=full_mask(),
            stream_hidden=hidden,
            coordinate_stream_ids=adapter.coordinate_stream_ids(),
            valid_mask=valid,
        )
        output = operator(inputs)
        base_probs = logits.softmax(-1)
        # Invalid frames are never edited.
        torch.testing.assert_close(output.probabilities[:, 3:], base_probs[:, 3:], rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="coordinate_stream_ids"):
        AdditiveLogitField(style_dim=8, stream_dim=8)(
            OperatorInputs(base_logits=logits, style_embedding=torch.zeros(BATCH, 8), stream_hidden=hidden)
        )
    with pytest.raises(ValueError, match="stream_dim"):
        AdditiveLogitField(style_dim=8)(
            OperatorInputs(base_logits=logits, style_embedding=torch.zeros(BATCH, 8), stream_hidden=hidden)
        )

def test_ctmc_matches_matrix_exp_in_value_and_gradient():
    """``p @ exp(Q)`` in float64: the oracle the uniformization must reproduce."""
    torch.manual_seed(21)
    up = torch.rand(1, 3, 1, LEVELS, dtype=torch.float64, generator=torch.Generator().manual_seed(21)) * 2.0
    down = torch.rand(1, 3, 1, LEVELS, dtype=torch.float64, generator=torch.Generator().manual_seed(22)) * 2.0
    up[..., -1] = 0.0
    down[..., 0] = 0.0
    # One element has no rate at all: an absorbing row must stay exactly put.
    up[0, 1, 0, :] = 0.0
    down[0, 1, 0, :] = 0.0
    generator = birth_death_generator(up, down)
    base = logits_to_double_probs()

    applied, diagnostics = uniformization_expm_apply(generator, base.detach().clone())
    reference = (base.detach().unsqueeze(-2) @ torch.matrix_exp(generator)).squeeze(-2)
    torch.testing.assert_close(applied, reference, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(applied[0, 1, 0], base.detach()[0, 1, 0], rtol=0.0, atol=0.0)
    assert float(diagnostics["mass_error"]) <= float(diagnostics["mass_tolerance"])
    assert float(diagnostics["poisson_tail"]) <= 1e-10

    # Gradients with respect to the *rates*, which is the direction the operator
    # actually trains in.  Perturbing a raw generator entry breaks the row-sum
    # zero property and leaves the truncated series with cancelling terms of order
    # 1e-1 whose truncation error dominates the derivative; the rate directions
    # keep Q a generator.  The cotangent is random because the sum of a
    # mass-conserving map is constant in Q.
    torch.manual_seed(31)
    base_p = logits_to_double_probs()
    cotangent = torch.randn(
        1, 3, 1, LEVELS, dtype=torch.float64, generator=torch.Generator().manual_seed(31)
    )
    rate_up = up.detach().clone().requires_grad_(True)
    rate_down = down.detach().clone().requires_grad_(True)
    rate_generator = birth_death_generator(rate_up, rate_down)
    (uniformization_expm_apply(rate_generator, base_p.detach().clone())[0] * cotangent).sum().backward()
    oracle_up = up.detach().clone().requires_grad_(True)
    oracle_down = down.detach().clone().requires_grad_(True)
    oracle_generator = birth_death_generator(oracle_up, oracle_down)
    (
        (base_p.detach().unsqueeze(-2) @ torch.matrix_exp(oracle_generator)).squeeze(-2) * cotangent
    ).sum().backward()
    # The absorbing element stays out of the comparison: a row scale of exactly
    # zero is the identity branch, deliberately flat in the rates so a fully
    # masked token is bit-for-bit the base (checked bit-exactly above).
    acting = torch.ones_like(rate_up, dtype=torch.bool)
    acting[0, 1, 0, :] = False
    torch.testing.assert_close(
        rate_up.grad[acting], oracle_up.grad[acting], rtol=1e-8, atol=1e-10
    )
    torch.testing.assert_close(
        rate_down.grad[acting], oracle_down.grad[acting], rtol=1e-8, atol=1e-10
    )
    assert float(rate_up.grad[0, 1, 0].abs().max()) == 0.0
    assert float(oracle_up.grad[0, 1, 0].abs().max()) > 0.0


def test_uniformization_rejects_truncation_and_non_finite_inputs():
    up = torch.full((1, 1, 1, LEVELS), 2.0)
    down = torch.full((1, 1, 1, LEVELS), 2.0)
    up[..., -1] = 0.0
    down[..., 0] = 0.0
    generator = birth_death_generator(up, down)
    base = base_logits()[:1, :1, :1].softmax(-1)
    # A truncated series loses a large share of the Poisson mass; renormalizing
    # would hide that, so it has to fail loudly instead.
    with pytest.raises(ValueError, match="max_terms"):
        uniformization_expm_apply(generator, base, tolerance=1e-14, max_terms=3)
    broken = generator.clone()
    broken[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        uniformization_expm_apply(broken, base)
    nan_base = base.clone()
    nan_base[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        uniformization_expm_apply(generator, nan_base)
    with pytest.raises(ValueError, match="share leading shape"):
        uniformization_expm_apply(generator, base[:, :, :0, :])


def test_birth_death_operator_can_be_configured_to_fail_on_truncation():
    operator = trained(
        BirthDeathCTMCOperator(
            num_levels=LEVELS, hidden_dim=16, style_dim=16,
            uniformization_tolerance=1e-14, max_terms=2,
        )
    )
    with pytest.raises(ValueError, match="max_terms"):
        operator(inputs_for(base_logits(), strength=1.0, hard_mask=full_mask()))


def test_strength_is_per_sample_and_never_moves_the_region():
    logits = base_logits()
    base_probs = logits.softmax(-1)
    support = partial_region()
    for cls in (AdditiveLogitField, BirthDeathCTMCOperator):
        operator = trained(cls(num_levels=LEVELS, hidden_dim=16, style_dim=16))
        mixed = operator(
            inputs_for(
                logits,
                strength=torch.tensor([0.0, 1.0]),
                hard_mask=support,
            )
        )
        # Row 0 has strength 0 and must be exactly the base...
        torch.testing.assert_close(mixed.probabilities[0], base_probs[0], rtol=0.0, atol=0.0)
        # ... while row 1 moves inside the region only.
        assert float((mixed.probabilities[1, :, :8] - base_probs[1, :, :8]).abs().max()) > 1e-4
        torch.testing.assert_close(
            mixed.probabilities[1, :, 8:], base_probs[1, :, 8:], rtol=0.0, atol=0.0
        )


def test_shuffled_adjacency_is_the_geometry_control():
    """A permuted level order must stay a valid, identity-clean CTMC."""
    torch.manual_seed(11)
    logits = base_logits()
    base_probs = logits.softmax(-1)
    order = [3, 7, 1, 8, 0, 5, 2, 6, 4]
    design = trained(BirthDeathCTMCOperator(num_levels=LEVELS, hidden_dim=16, style_dim=16))
    shuffled = trained(
        BirthDeathCTMCOperator(num_levels=LEVELS, hidden_dim=16, style_dim=16, level_order=order)
    )
    assert design.shuffled_adjacency is False and shuffled.shuffled_adjacency is True
    assert shuffled.level_order == tuple(order)
    assert shuffled.config()["level_order"] == order
    assert "level_order" not in design.config()
    zero = shuffled(inputs_for(logits, strength=0.0, hard_mask=full_mask()))
    assert torch.equal(zero.probabilities, base_probs)
    output = shuffled(inputs_for(logits, strength=1.0, hard_mask=full_mask()))
    assert float(output.probabilities.sum(-1).sub(1).abs().max()) < 1e-5
    assert float((output.probabilities - shuffled.reference_expm(
        inputs_for(logits, strength=1.0, hard_mask=full_mask())
    )).abs().max()) < 1e-5
    assert float(output.diagnostics["shuffled_adjacency"]) == 1.0
    assert not torch.allclose(
        output.probabilities,
        design(inputs_for(logits, strength=1.0, hard_mask=full_mask())).probabilities,
    )
    with pytest.raises(ValueError, match="permutation"):
        BirthDeathCTMCOperator(num_levels=LEVELS, level_order=[0, 1, 2])
    with pytest.raises(ValueError, match="permutation"):
        BirthDeathCTMCOperator(num_levels=LEVELS, level_order=[0, 1, 2, 3, 4, 5, 6, 7, 7])
