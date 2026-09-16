"""MTS sampling: inverse CDF, common random numbers and paired statistics."""

from __future__ import annotations

import pytest
import torch

from stylized_motion.learning.mts_operator.sampling import (
    CommonRandomNumbers,
    inverse_cdf_sample,
    paired_comparison,
    region_support_mask,
    sample_tokens,
)


def one_hot_probabilities(*levels: int, num_levels: int = 9) -> torch.Tensor:
    probabilities = torch.zeros(len(levels), num_levels)
    probabilities[torch.arange(len(levels)), torch.tensor(levels)] = 1.0
    return probabilities


def test_inverse_cdf_selects_the_interval_the_uniform_falls_into():
    probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    uniforms = torch.tensor([0.0, 0.24, 0.26, 0.51, 0.99])
    drawn = inverse_cdf_sample(probabilities.expand(len(uniforms), -1), uniforms)
    assert drawn.tolist() == [0, 0, 1, 2, 3]
    # A degenerate distribution always draws its only level.
    deterministic = one_hot_probabilities(7, 7, 7)
    uniform = torch.tensor([0.0, 0.5, 0.999999])
    # u = 0 lands on the first index by definition; every draw above zero picks
    # the degenerate level.
    assert inverse_cdf_sample(deterministic, uniform).tolist() == [0, 7, 7]
    # A uniform draw of exactly 1.0 still lands in range.
    assert int(inverse_cdf_sample(probabilities, torch.tensor([1.0])).item()) == 3
    with pytest.raises(ValueError, match="num_levels"):
        inverse_cdf_sample(probabilities, torch.tensor([0.5]), num_levels=3)
    with pytest.raises(ValueError, match="broadcast"):
        inverse_cdf_sample(probabilities, torch.zeros(2, 2))


def test_common_random_numbers_are_stable_and_per_shape():
    crn = CommonRandomNumbers(seed=17)
    first = crn.uniforms((2, 3))
    second = crn.uniforms((2, 3))
    assert torch.equal(first, second)  # same shape, same uniforms
    other = crn.uniforms((3, 2))
    assert other.shape == (3, 2) and not torch.equal(first.flatten(), other.flatten())
    values = crn.uniforms((2, 3))
    assert bool((values >= 0).all() and (values < 1).all())
    # A different seed gives a different draw of the same shape.
    fresh = CommonRandomNumbers(seed=18)
    assert not torch.equal(fresh.uniforms((2, 3)), first)
    crn.clear()
    assert crn.draws == 0 and crn.uniforms_cache == {}


def test_paired_sampling_makes_identical_distributions_agree_exactly():
    probabilities = torch.rand(4, 3, 5, generator=torch.Generator().manual_seed(2))
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    report = paired_comparison(probabilities, probabilities.clone())
    assert report["changed_token_ratio"] == 0.0
    assert report["total_variation"] == pytest.approx(0.0, abs=1e-6)
    assert report["coupled"] is True
    assert "coupling statistic" in report["note"]


def test_paired_comparison_reports_coupling_and_distribution_statistics():
    base = one_hot_probabilities(0, 0, 0, 0, num_levels=9)
    styled = one_hot_probabilities(5, 5, 5, 0, num_levels=9)
    report = paired_comparison(base, styled)
    assert report["changed_token_ratio"] == pytest.approx(0.75)
    assert report["changed_token_count"] == 3 and report["positions"] == 4
    assert report["total_variation"] == pytest.approx(0.75)  # three of four positions moved
    support = torch.ones(4, dtype=torch.bool)
    report = paired_comparison(base, styled, hard_mask=support)
    assert report["outside_support_changed"] == 0
    masked = paired_comparison(base, styled, hard_mask=torch.zeros(4, dtype=torch.bool))
    assert masked["outside_support_changed"] == 3
    with pytest.raises(ValueError, match="hard_mask must have shape"):
        paired_comparison(base, styled, hard_mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="share a shape"):
        paired_comparison(base, styled[:, :1])


def test_common_random_numbers_make_a_strength_sweep_a_paired_measurement():
    torch.manual_seed(5)
    logits = torch.randn(1, 6, 40, 9)
    base = logits.softmax(-1)
    crn = CommonRandomNumbers(seed=11)
    changed = []
    for strength in (0.0, 0.5, 1.0):
        styled = (logits * (1.0 + strength)).softmax(-1) if strength else base
        report = paired_comparison(base, styled, crn=crn)
        changed.append(report["changed_token_ratio"])
    assert changed[0] == 0.0
    assert changed[0] <= changed[1] <= changed[2]
    assert crn.draws == 3


def test_sample_tokens_uses_uniforms_or_fresh_randomness():
    probabilities = one_hot_probabilities(3, 3, 3)
    generator = torch.Generator().manual_seed(4)
    first = sample_tokens(probabilities, generator=generator)
    second = sample_tokens(probabilities, generator=generator)
    assert torch.equal(first, second)  # the same generator state reproduces a draw
    assert first.tolist() == [3, 3, 3]
    crn = CommonRandomNumbers(seed=2)
    coupled = sample_tokens(probabilities, crn=crn)
    assert coupled.tolist() == [3, 3, 3]
    assert crn.draws == 1


def test_region_support_mask_delegates_to_the_layout():
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
    mask = region_support_mask(adapter, ["left_arm"], graph_radius=1, length=8)
    assert mask.shape == (8, 40)
    assert sorted(torch.nonzero(mask.any(0)).flatten().tolist()) == [10, 11, 12, 13, 32, 33]
    windowed = region_support_mask(adapter, ["left_arm"], frame_range=(2, 4), length=8)
    assert int(windowed.sum()) == 2 * 4
