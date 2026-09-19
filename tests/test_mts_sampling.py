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
    # A uniform draw of exactly 1.0 is canonicalised into [0, 1) and stays in range.
    assert int(inverse_cdf_sample(probabilities, torch.tensor([1.0])).item()) == 3
    with pytest.raises(ValueError, match="num_levels"):
        inverse_cdf_sample(probabilities, torch.tensor([0.5]), num_levels=3)
    with pytest.raises(ValueError, match="broadcast"):
        inverse_cdf_sample(probabilities, torch.zeros(2, 2))


def test_inverse_cdf_never_selects_a_zero_mass_bin():
    """The search is for the first level whose CDF exceeds u, not ``cdf < u``."""
    deterministic = one_hot_probabilities(7, 7, 7, 7)
    uniform = torch.tensor([0.0, 0.5, 0.999999, 1.0])
    # u = 0 must not land on level 0 any more: level 0 has no mass at all.
    assert inverse_cdf_sample(deterministic, uniform).tolist() == [7, 7, 7, 7]
    holes = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
    draws = inverse_cdf_sample(
        holes.expand(4, -1), torch.tensor([0.0, 0.49, 0.51, 0.9999999])
    )
    assert draws.tolist() == [1, 1, 3, 3]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        inverse_cdf_sample(deterministic, torch.tensor([1.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        inverse_cdf_sample(deterministic, torch.tensor([-1e-6]))


def test_common_random_numbers_are_stable_and_keyed_by_sample_and_step():
    crn = CommonRandomNumbers(seed=17)
    first = crn.uniforms((2, 3))
    second = crn.uniforms((2, 3))
    assert torch.equal(first, second)  # same key, same uniforms
    assert len(crn.uniforms_cache) == 1
    other = crn.uniforms((3, 2))
    assert other.shape == (3, 2) and not torch.equal(first.flatten(), other.flatten())
    values = crn.uniforms((2, 3))
    assert bool((values >= 0).all() and (values < 1).all())
    # A different seed gives a different draw of the same shape.
    fresh = CommonRandomNumbers(seed=18)
    assert not torch.equal(fresh.uniforms((2, 3)), first)
    crn.clear()
    assert crn.draws == 0 and crn.uniforms_cache == {}


def test_common_random_numbers_do_not_depend_on_call_order_or_device_object():
    """The old cache key used ``id(device)`` and a seed from the insertion order."""
    shape = (2, 5, 3)
    ordered = CommonRandomNumbers(seed=23)
    sample_one_first = ordered.uniforms(shape, sample_id=1)
    sample_zero_second = ordered.uniforms(shape, sample_id=0)
    reversed_order = CommonRandomNumbers(seed=23)
    sample_zero_first = reversed_order.uniforms(shape, sample_id=0)
    sample_one_second = reversed_order.uniforms(shape, sample_id=1)
    assert torch.equal(sample_zero_first, sample_zero_second)
    assert torch.equal(sample_one_first, sample_one_second)
    assert not torch.equal(sample_zero_first, sample_one_first)
    # Passing a device (even a freshly constructed one) must not change the values.
    with_device = CommonRandomNumbers(seed=23)
    assert torch.equal(
        with_device.uniforms(shape, sample_id=0, device=torch.device("cpu")),
        with_device.uniforms(shape, sample_id=0, device="cpu"),
    )
    assert len(with_device.uniforms_cache) == 1
    # Steps are independent cells of the same sample.
    stepped = CommonRandomNumbers(seed=23)
    assert not torch.equal(
        stepped.uniforms(shape, sample_id=0, step_id=0),
        stepped.uniforms(shape, sample_id=0, step_id=1),
    )


def test_common_random_numbers_couple_identical_distributions_exactly():
    probabilities = torch.full((2, 6, 40, 9), 1.0 / 9.0)
    crn = CommonRandomNumbers(seed=12)
    first = crn.sample(probabilities)
    second = crn.sample(probabilities)
    assert torch.equal(first, second)
    assert len(crn.uniforms_cache) == 1
    # Four samples of one condition are four distinct draws with distinct keys.
    crn_four = CommonRandomNumbers(seed=12)
    draws = [crn_four.sample(probabilities, sample_id=index) for index in range(4)]
    assert len(crn_four.uniforms_cache) == 4
    assert len({tuple(draw.flatten().tolist()) for draw in draws}) > 1
    # ... and the same four are reproducible in a fresh instance.
    again = CommonRandomNumbers(seed=12)
    for index, draw in enumerate(draws):
        assert torch.equal(again.sample(probabilities, sample_id=index), draw)


def test_cpu_generator_with_fresh_randomness_is_reproducible_across_instances():
    probabilities = torch.full((2, 3, 4, 9), 1.0 / 9.0)
    first = sample_tokens(probabilities, generator=torch.Generator(device="cpu").manual_seed(5))
    second = sample_tokens(probabilities, generator=torch.Generator(device="cpu").manual_seed(5))
    assert torch.equal(first, second)
    # The CRN path keys the draw by sample/step, so the pairs stay coupled.
    crn = CommonRandomNumbers(seed=5)
    assert torch.equal(
        sample_tokens(probabilities, crn=crn, sample_id=2),
        sample_tokens(probabilities, crn=crn, sample_id=2),
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the device-mismatch guard")
def test_sampling_works_with_a_cpu_generator_on_cuda_tensors():
    """Regression guard: a CPU generator with CUDA probabilities must not crash.

    This is the production configuration (reproducible CPU generator, GPU model)
    and it was silently broken in three places: the mask generator, the transport
    generator loop and sample_tokens().
    """
    from stylized_motion.learning.mts_operator.sampling import sample_tokens
    from stylized_motion.learning.mts_operator.masking import MaskGenerator

    probabilities = one_hot_probabilities(3, 3).to("cuda")
    generator = torch.Generator(device="cpu").manual_seed(5)
    drawn = sample_tokens(probabilities, generator=generator)
    assert drawn.device.type == "cuda" and drawn.tolist() == [3, 3]
    masks = MaskGenerator({"random_coordinate": 1.0})
    batch = masks.sample_kind(
        "random_coordinate", 2, 8, generator=torch.Generator(device="cpu").manual_seed(1), device=torch.device("cuda")
    )
    assert batch.visible_mask.device.type == "cuda" and not bool(batch.visible_mask.all())


def test_generation_cli_region_contract():
    """``--no-locked-edit`` is only meaningful for a whole-body region."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "generate_mts_operator.py"
    spec = importlib.util.spec_from_file_location("generate_mts_operator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    frames, coordinates = 5, 40
    partial = torch.zeros(frames, coordinates, dtype=torch.bool)
    partial[:2, :6] = True
    region, visible, whole_body = module.resolve_generation_region(
        partial, locked_edit=True, frames=frames
    )
    assert region.shape == (1, frames, coordinates)
    assert not whole_body
    assert torch.equal(region[0], partial)
    assert torch.equal(visible[0], ~partial)
    with pytest.raises(ValueError, match="whole_body"):
        module.resolve_generation_region(partial, locked_edit=False, frames=frames)
    full = torch.ones(frames, coordinates, dtype=torch.bool)
    _, visible_full, whole_body = module.resolve_generation_region(
        full, locked_edit=False, frames=frames
    )
    assert whole_body and not bool(visible_full.any())


# ---------------------------------------------------------------------------
# C06: monotonic filling


def test_monotonic_fill_commits_every_position_exactly_once():
    from stylized_motion.learning.mts_operator.sampling import monotonic_fill_steps

    remaining = torch.zeros(2, 4, 5, dtype=torch.bool)
    remaining[0, :, :3] = True
    remaining[0, 1, 3] = True
    remaining[1, 3, 4] = True
    for steps in (1, 2, 3, 5, 40):
        commits = monotonic_fill_steps(remaining, steps)
        stacked = torch.stack(commits)
        assert int(stacked.sum(dim=0).max()) <= 1, steps
        assert torch.equal(stacked.any(dim=0), remaining), steps
        # Later commits never take more than earlier ones, per sample.
        counts = stacked.flatten(2).sum(dim=2)
        for row in range(2):
            per_step = counts[:, row].tolist()
            assert all(a >= b for a, b in zip(per_step, per_step[1:])), per_step
    with pytest.raises(ValueError, match="positive"):
        monotonic_fill_steps(remaining, 0)
    with pytest.raises(ValueError, match=r"\[B, T, K\]"):
        monotonic_fill_steps(remaining[0], 2)


def test_monotonic_fill_draws_are_shared_per_step_and_ordered():
    from stylized_motion.learning.mts_operator.sampling import (
        CommonRandomNumbers,
        fill_remaining,
    )

    probabilities = torch.full((1, 3, 4, 9), 1.0 / 9.0)
    remaining = torch.ones(1, 3, 4, dtype=torch.bool)
    crn = CommonRandomNumbers(seed=3)
    first, commits_first = fill_remaining(
        probabilities, remaining, steps=3, crn=crn, sample_id=1
    )
    again, commits_again = fill_remaining(
        probabilities, remaining, steps=3, crn=crn, sample_id=1
    )
    assert torch.equal(first, again)
    assert all(torch.equal(a, b) for a, b in zip(commits_first, commits_again))
    assert len(crn.uniforms_cache) == 3, "one CRN cell per step"
    # A different sample id is a different draw.
    other, _ = fill_remaining(probabilities, remaining, steps=3, crn=crn, sample_id=2)
    assert not torch.equal(first, other)
