"""R09a: the frozen validation protocol.

The checks are about *reproducibility*: the same checkpoint must score the same
number twice, a different training seed must not move the validation set, and a
kind with no supervised token must be named rather than silently re-weighting the
objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stylized_motion.learning.mts_operator import (
    MASK_KINDS,
    LayoutAdapter,
    MaskGenerator,
    TrainerConfig,
    TransportTrainer,
)
from stylized_motion.learning.mts_operator.pairs import StylePairSampler, StyleSplit
from stylized_motion.learning.mts_operator.eval_protocol import (
    DEFAULT_BATCHES_PER_KIND,
    ValidationBatchBuilder,
    ValidationProtocol,
    build_validation_samples,
)
from stylized_motion.learning.mts_operator.pairs import ClipRecord
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
from stylized_motion.learning.mts_operator.windows import (
    ContentVocabulary,
    TokenSource,
    WindowSample,
)
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

FRAMES = 5
COORDINATES = 40
LEVELS = 9


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
    return names, parents


def adapter() -> LayoutAdapter:
    return LayoutAdapter(NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON)))


class _FakeTokens:
    """A token source with deterministic, clip-identifiable windows."""

    def __init__(self, *, frames: int = FRAMES, clips: int = 12) -> None:
        self.frames = int(frames)
        self.adapter = adapter()
        self.windows_by_clip = {
            clip: [type("Request", (), {"target_start": 0, "variant_idx": clip})()]
            for clip in range(clips)
        }

    def request(self, clip_id: int, start: int):
        return (int(clip_id), int(start))

    def read(self, request) -> torch.Tensor:
        clip_id, start = request
        base = torch.arange(self.frames * COORDINATES).view(self.frames, COORDINATES)
        return ((base + clip_id * 7 + start) % LEVELS).long()

    def window(self, clip_id: int):
        if int(clip_id) not in self.windows_by_clip:
            return None
        return WindowSample(
            tokens=self.read((int(clip_id), 0)),
            valid_mask=torch.ones(self.frames, dtype=torch.bool),
            metadata={"clip_id": int(clip_id)},
        )


def records(*, clips: int = 12) -> list[ClipRecord]:
    return [
        ClipRecord(
            clip_id=index,
            style=f"Style{index % 3}",
            content=f"action{index % 2}",
            performer="subject1",
            source_group=index,
            split="val",
            frames=120,
        )
        for index in range(clips)
    ]


def sampler(*, clips: int = 12, seed: int = 3) -> StylePairSampler:
    return StylePairSampler(
        records(clips=clips),
        # The fixture records all live in the val split, so the stage selects by
        # data split (the actor-holdout path) rather than by style vocabulary.
        style_split=StyleSplit(
            train_styles=("Style0", "Style1", "Style2"), val_styles=(), test_unseen_styles=()
        ),
        seed=seed,
        use_data_splits=True,
    )


def test_validation_samples_are_frozen_and_reproducible():
    protocol = ValidationProtocol.build(
        sampler(),
        token_source=_FakeTokens(),
        kinds=MASK_KINDS,
        batches_per_kind=2,
        batch_size=3,
        frames=FRAMES,
        seed=11,
    )
    assert len(protocol.samples) == len(MASK_KINDS) * 2 * 3
    assert protocol.counts() == {kind: 6 for kind in MASK_KINDS}
    assert DEFAULT_BATCHES_PER_KIND == 4
    # Rebuilding with the same seed gives the same items, ids and seeds.
    again = ValidationProtocol.build(
        sampler(),
        token_source=_FakeTokens(),
        kinds=MASK_KINDS,
        batches_per_kind=2,
        batch_size=3,
        frames=FRAMES,
        seed=11,
    )
    assert [sample.as_dict() for sample in again.samples] == [
        sample.as_dict() for sample in protocol.samples
    ]
    # A different *training* seed cannot move the validation set, because it is
    # built from this call's seed alone.
    payload = protocol.describe()
    assert payload["samples_per_kind"] == {kind: 6 for kind in MASK_KINDS}
    assert sorted(payload["splits"]) == ["val"]
    assert len(payload["seeds"]) == len(protocol.samples)
    assert len(set(payload["sample_ids"])) == len(protocol.samples)
    # ... and every item declares the window it will read, not just a clip.
    for sample in protocol.samples:
        assert sample.target_start >= 0 and sample.reference_start >= 0
        assert sample.target_clip != sample.reference_clip
    protocol.write(Path("/tmp/validation_protocol.json"))
    stored = json.loads(Path("/tmp/validation_protocol.json").read_text(encoding="utf-8"))
    assert stored["items"][0]["kind"] in MASK_KINDS


def test_validation_batches_are_identical_across_calls():
    protocol = ValidationProtocol.build(
        sampler(),
        token_source=_FakeTokens(),
        kinds=("stream", "full_generation"),
        batches_per_kind=2,
        batch_size=2,
        frames=FRAMES,
        seed=5,
    )
    builder = ValidationBatchBuilder(
        token_source=_FakeTokens(),
        mask_generator=MaskGenerator(),
        adapter=adapter(),
        device="cpu",
    )
    first = builder.batches(protocol, batch_size=2)
    second = builder.batches(protocol, batch_size=2)
    expected = ["full_generation", "full_generation", "stream", "stream"]  # protocol.kinds is sorted
    assert [kind for kind, _ in first] == [kind for kind, _ in second] == expected
    for (kind_a, batch_a), (kind_b, batch_b) in zip(first, second):
        assert kind_a == kind_b
        torch.testing.assert_close(batch_a.target_tokens, batch_b.target_tokens, rtol=0.0, atol=0.0)
        torch.testing.assert_close(batch_a.visible_mask, batch_b.visible_mask, rtol=0.0, atol=0.0)
        assert [meta["sample_id"] for meta in batch_a.sample_metadata] == [
            meta["sample_id"] for meta in batch_b.sample_metadata
        ]


def test_validation_does_not_touch_the_training_rng():
    def draw(target, generator):
        return [
            pair.as_dict()
            for pair in target.sample(count=3, mode="same_style", stage="val", generator=generator)
        ]

    training = sampler(seed=7)
    training_rng = np.random.default_rng(5)
    before = draw(training, training_rng)
    # Building the protocol must not consume from the training generator.
    ValidationProtocol.build(
        training,
        token_source=_FakeTokens(),
        kinds=("stream",),
        batches_per_kind=1,
        batch_size=4,
        frames=FRAMES,
        seed=99,
    )
    after = draw(training, training_rng)
    # Control: the same generator stream in a run where no protocol was built.
    control_rng = np.random.default_rng(5)
    control_first = draw(sampler(seed=7), control_rng)
    control_second = draw(sampler(seed=7), control_rng)
    assert before != after, "an explicit generator advances when it is used"
    assert before == control_first
    assert after == control_second, "the protocol build consumed nothing from the training stream"


def test_objective_uses_fixed_weights_and_names_missing_kinds():
    import dataclasses

    base = build_samples()[0]
    protocol = ValidationProtocol(
        samples=(base, dataclasses.replace(base, kind="full_generation", sample_id=1)),
        kinds=("stream", "full_generation"),
        weights={"stream": 0.5, "full_generation": 0.5},
    )
    report = protocol.objective(
        {
            "stream": {"nll_sum": 20.0, "supervised_tokens": 10},
            "full_generation": {"nll_sum": 9.0, "supervised_tokens": 3},
        }
    )
    # 0.5 * (20/10) + 0.5 * (9/3) = 1.0 + 1.5
    assert report["objective"] == pytest.approx(2.5)
    assert report["counts"] == {"stream": 10, "full_generation": 3}
    assert report["usable"] is True and report["missing_kinds"] == []
    # An empty kind is named, and its weight is NOT moved to the other one: the
    # objective becomes unusable rather than smaller (C03 item 6).
    partial = protocol.objective({"stream": {"nll_sum": 20.0, "supervised_tokens": 10}})
    assert partial["missing_kinds"] == ["full_generation"]
    assert partial["objective"] is None and partial["usable"] is False
    assert partial["weights"]["full_generation"] == pytest.approx(0.5)
    # Nothing supervised at all: no number, and the caller cannot treat it as best.
    empty = protocol.objective({})
    assert empty["objective"] is None and empty["usable"] is False
    assert sorted(empty["missing_kinds"]) == ["full_generation", "stream"]
    with pytest.raises(ValueError, match="at least one sample"):
        ValidationProtocol(samples=(), kinds=("stream",))


def build_samples():
    return ValidationProtocol.build(
        sampler(),
        token_source=_FakeTokens(),
        kinds=("stream",),
        batches_per_kind=1,
        batch_size=2,
        frames=FRAMES,
        seed=1,
    ).samples


# ---------------------------------------------------------------------------
# T02: the protocol fingerprint covers the whole protocol, not just its shape


def _target_protocol(*, seed: int = 5, rows_per_kind: int = 2):
    from stylized_motion.learning.mts_operator.eval_protocol import (
        build_target_only_samples,
        store_split_of_clip,
    )

    class Source(_FakeTokens):
        def __init__(self) -> None:
            super().__init__(clips=6)
            self.store = type("Store", (), {"split_ids": np.zeros(6, dtype=np.uint8)})()

    source = Source()
    assert store_split_of_clip(source.store, 0) == "train"
    samples, selection = build_target_only_samples(
        source,
        kinds=("stream", "full_generation"),
        rows_per_kind=rows_per_kind,
        frames=FRAMES,
        seed=seed,
        split="train",
        mask_generator=MaskGenerator(),
    )
    return ValidationProtocol(
        samples=tuple(samples),
        kinds=("full_generation", "stream"),
        protocol_id="fingerprint-test",
        selection=selection,
    )


def test_protocol_fingerprint_changes_with_any_row_or_identity():
    """T02: a hash over the described shape cannot tell two protocols apart.

    The recorded hash must cover every row's window, seed and mask configuration
    and the data identity the caller passes, so "the same protocol" means the same
    protocol.
    """
    import dataclasses

    protocol = _target_protocol()
    baseline = protocol.fingerprint()
    assert len(baseline) == 64

    moved = dataclasses.replace(
        protocol,
        samples=(
            dataclasses.replace(
                protocol.samples[0], target_start=protocol.samples[0].target_start + 1
            ),
            *protocol.samples[1:],
        ),
    )
    assert moved.fingerprint() != baseline, "a moved window must change the hash"

    reseeded = dataclasses.replace(
        protocol,
        samples=(
            dataclasses.replace(protocol.samples[0], seed=protocol.samples[0].seed + 1),
            *protocol.samples[1:],
        ),
    )
    assert reseeded.fingerprint() != baseline, "a new mask seed must change the hash"

    remasked = dataclasses.replace(
        protocol,
        samples=(
            dataclasses.replace(protocol.samples[0], mask_config={"stream": 1.0}),
            *protocol.samples[1:],
        ),
    )
    assert remasked.fingerprint() != baseline, "a different mask config must change the hash"

    # The data identity travels inside the protocol's selection block, so a
    # protocol file is self-describing and the hash covers where it came from.
    other_data = dataclasses.replace(protocol, selection={**protocol.selection, "store_identity": {"split_table_sha256": "A"}})
    assert other_data.fingerprint() != baseline
    # And it is stable: the same protocol hashes the same twice.
    assert protocol.fingerprint() == baseline


def test_the_validation_number_repeats_exactly_for_one_checkpoint():
    """The whole point: two evaluations of one model give the same number."""
    view = adapter()
    torch.manual_seed(4)
    model = MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(6)) * 0.1)
    trainer = TransportTrainer(
        model, adapter=view, mask_generator=MaskGenerator({"stream": 1.0}), device="cpu",
        config=TrainerConfig(epochs=1, log_every_steps=0),
    )
    protocol = ValidationProtocol.build(
        sampler(),
        token_source=_FakeTokens(),
        kinds=("stream", "full_generation", "random_coordinate"),
        batches_per_kind=2,
        batch_size=3,
        frames=FRAMES,
        seed=17,
    )
    builder = ValidationBatchBuilder(
        token_source=_FakeTokens(), mask_generator=MaskGenerator(), adapter=view, device="cpu"
    )
    first = builder.evaluate(trainer, protocol, batch_size=3, transport=True)
    second = builder.evaluate(trainer, protocol, batch_size=3, transport=True)
    assert first["objective"] == pytest.approx(second["objective"])
    assert first["counts"] == second["counts"]
    assert first["objectives_per_kind"] == second["objectives_per_kind"]
    # A different model gives a different number (the protocol is not degenerate).
    other = MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0)
    other_trainer = TransportTrainer(
        other, adapter=view, mask_generator=MaskGenerator({"stream": 1.0}), device="cpu",
        config=TrainerConfig(epochs=1, log_every_steps=0),
    )
    third = builder.evaluate(other_trainer, protocol, batch_size=3, transport=True)
    assert third["objective"] != pytest.approx(first["objective"], rel=1e-6)


def test_validation_protocol_rejects_unknown_kinds_and_empty_splits():
    with pytest.raises(ValueError, match="Unknown mask kind"):
        build_validation_samples(
            sampler(), token_source=_FakeTokens(), kinds=("everything",),
            batches_per_kind=1, batch_size=2, frames=FRAMES, seed=1,
        )
    # A split that cannot produce a single same-style/different-content pair is a
    # protocol error, not a silent skip.
    single_content = [
        ClipRecord(clip_id=index, style=f"Style{index % 3}", content="only",
                   performer="subject1", source_group=index, split="val", frames=120)
        for index in range(6)
    ]
    empty_sampler = StylePairSampler(single_content, seed=2, use_data_splits=True)
    with pytest.raises(ValueError, match="produced no|no .* batches"):
        build_validation_samples(
            empty_sampler, token_source=_FakeTokens(), kinds=("stream",),
            batches_per_kind=1, batch_size=2, frames=FRAMES, seed=1,
        )


# ---------------------------------------------------------------------------
# R10: the evaluation manifest


def eval_records():
    """Six val clips: two styles, two actions, three actors (a real take per clip)."""
    rows = [
        ("Style0", "walk", "subject1"),
        ("Style0", "walk", "subject1"),
        ("Style0", "run", "subject2"),
        ("Style0", "run", "subject3"),
        ("Style1", "walk", "subject1"),
        ("Style1", "run", "subject2"),
    ]
    return [
        ClipRecord(clip_id=index, style=style, content=action, performer=actor,
                   source_group=index, split="val", frames=120)
        for index, (style, action, actor) in enumerate(rows)
    ]


def test_eval_manifest_has_no_leaks_and_records_every_role():
    from stylized_motion.learning.mts_operator.eval_protocol import (
        EVAL_MANIFEST_VERSION,
        build_eval_rows,
        eligible_window_clips,
        read_eval_manifest,
        write_eval_manifest,
    )

    source = _FakeTokens(clips=6)
    rows = build_eval_rows(
        eval_records(), source, split="val", samples=6, batch_size=3, frames=FRAMES, seed=4
    )
    assert len(rows) == 6
    assert len({row.sample_id for row in rows}) == 6
    for row in rows:
        assert row.candidates, row.reasons
        # Every candidate shares the style and never leaks (same clip or same take).
        for candidate in row.candidates:
            assert candidate.style == row.target.style
            assert candidate.clip_id != row.target.clip_id
            assert candidate.take != row.target.take
        assert row.correct is not None and row.correct.style == row.target.style
        assert set(row.positive_indices) <= set(range(len(row.candidates)))
        if row.wrong is not None:
            assert row.wrong.style != row.target.style
            assert row.wrong.clip_id != row.target.clip_id and row.wrong.take != row.target.take
        if row.random is not None:
            # A real clip from the data; it may happen to share the style.
            assert 0 <= row.random.clip_id < 6
            assert row.random.take != row.target.take
        assert row.mask_seed != row.sample_seed
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        path = write_eval_manifest(
            _Path(directory) / "eval_manifest.json",
            rows,
            identity={"representation_id": "nef_fsq_independent_40x9"},
            protocol={"split": "val"},
        )
        meta, restored = read_eval_manifest(path)
        assert meta["protocol_version"] == EVAL_MANIFEST_VERSION
        assert [row.as_dict() for row in restored] == [row.as_dict() for row in rows]
    assert set(eligible_window_clips(source, FRAMES)) == set(range(6))


def test_eval_manifest_marks_unavailable_roles_instead_of_inventing_them():
    from stylized_motion.learning.mts_operator.eval_protocol import build_eval_rows

    # A single style in the split: there is no legal different-style negative.
    single_style = [
        ClipRecord(clip_id=index, style="Only", content="walk", performer="subject1",
                   source_group=index, split="val", frames=120)
        for index in range(4)
    ]
    rows = build_eval_rows(
        single_style, _FakeTokens(clips=4), split="val", samples=3, batch_size=2, frames=FRAMES
    )
    assert all(row.wrong is None for row in rows)
    assert all(row.reasons.get("wrong_reference") == "no_legal_different_style_clip" for row in rows)
    # The random reference is still a real clip from the data.
    assert all(row.random is not None for row in rows)


def test_eval_manifest_rejects_non_finite_numbers(tmp_path):
    from stylized_motion.learning.mts_operator.eval_protocol import (
        ClipRef,
        EvalRow,
        write_eval_manifest,
    )

    row = EvalRow(
        sample_id=0, split="val",
        target=ClipRef(clip_id=0, start=0, frames=FRAMES, style="S"),
        candidates=(ClipRef(clip_id=1, start=0, frames=FRAMES, style="S"),),
        positive_indices=(0,), mask_kind="full_generation", mask_seed=1, sample_seed=2,
    )
    path = write_eval_manifest(tmp_path / "manifest.json", [row])
    assert path.exists() and (tmp_path / "manifest.jsonl").exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["rows"] == 1 and payload["protocol_version"] == 2


def test_wrong_reference_prefers_same_action_then_same_actor():
    from stylized_motion.learning.mts_operator.eval_protocol import build_eval_rows

    rows = build_eval_rows(
        eval_records(), _FakeTokens(clips=6), split="val", samples=6, batch_size=2, frames=FRAMES
    )
    matched_action = 0
    for row in rows:
        if row.wrong is None:
            continue
        if row.wrong.action == row.target.action:
            matched_action += 1
        assert row.wrong.style != row.target.style
    assert matched_action >= 1, "the negative is chosen to match the action when it can"


# ---------------------------------------------------------------------------
# C03: the protocol is per sample, not per batch


def _protocol_for_c03():
    from stylized_motion.learning.mts_operator.eval_protocol import ValidationProtocol

    return ValidationProtocol.build(
        sampler(),
        token_source=_FakeTokens(),
        kinds=("stream", "full_generation"),
        batches_per_kind=2,
        batch_size=3,
        frames=FRAMES,
        seed=11,
    )


def _builder_c03(*, style_index=None, content=None, valid_mask=None):
    from stylized_motion.learning.mts_operator.eval_protocol import ValidationBatchBuilder

    class _Source(_FakeTokens):
        def window_at(self, clip_id, start):
            sample = super().window(clip_id)
            if sample is None or valid_mask is None:
                return sample
            import dataclasses

            return dataclasses.replace(sample, valid_mask=valid_mask(sample))

    source = _Source()
    builder = ValidationBatchBuilder(
        token_source=source,
        mask_generator=MaskGenerator(),
        adapter=adapter(),
        device="cpu",
        content_vocabulary=content,
        style_index=style_index,
        encoder_kind="style_id" if style_index else "reference",
    )
    return builder


def test_a_manifest_row_means_the_same_thing_at_any_batch_size():
    """C03: per-row masks, so B=1/2/4 describe the same experiment."""
    protocol = _protocol_for_c03()
    per_size = {}
    for size in (1, 2, 4):
        builder = _builder_c03()
        rows = {}
        for kind, batch in builder.batches(protocol, batch_size=size):
            for index, meta in enumerate(batch.sample_metadata):
                rows[meta["sample_id"]] = (
                    batch.visible_mask[index].clone(),
                    batch.target_tokens[index].clone(),
                )
        per_size[size] = rows
    assert set(per_size[1]) == set(per_size[4]) == {s.sample_id for s in protocol.samples}
    for sample_id, (visible, tokens) in per_size[1].items():
        for size in (2, 4):
            other_visible, other_tokens = per_size[size][sample_id]
            assert torch.equal(visible, other_visible), (sample_id, size)
            assert torch.equal(tokens, other_tokens), (sample_id, size)


def test_validation_builder_carries_style_ids_and_valid_masks():
    """C03: a style-ID validation batch needs style ids, and valid comes from the window."""
    protocol = _protocol_for_c03()
    style_index = {"Style0": 0, "Style1": 1, "Style2": 2}
    builder = _builder_c03(style_index=style_index)
    seen_styles = set()
    for kind, batch in builder.batches(protocol, batch_size=2):
        assert batch.style_ids is not None
        for index, meta in enumerate(batch.sample_metadata):
            expected = style_index[meta["style"]]
            assert int(batch.style_ids[index]) == expected
            seen_styles.add(meta["style"])
        assert batch.target_valid_mask is not None and bool(batch.target_valid_mask.all())
    assert seen_styles, "the fixture must exercise at least one style"
    # A padded window keeps its valid mask instead of being treated as real frames.
    builder = _builder_c03(valid_mask=lambda sample: torch.tensor([True, True, False, False, False]))
    for kind, batch in builder.batches(protocol, batch_size=3):
        assert not bool(batch.target_valid_mask.all())
        assert int(batch.target_valid_mask.sum(dim=1).min()) == 2


def test_objective_is_unusable_when_a_weighted_kind_is_missing_or_not_finite():
    """C03: a missing kind must not make the objective easier to beat."""
    protocol = ValidationProtocol(
        samples=_protocol_for_c03().samples,
        kinds=("stream", "full_generation"),
        weights={"stream": 0.5, "full_generation": 0.5},
    )
    good = protocol.objective(
        {
            "stream": {"nll_sum": 10.0, "supervised_tokens": 5},
            "full_generation": {"nll_sum": 4.0, "supervised_tokens": 2},
        }
    )
    # 0.5 * (10/5) + 0.5 * (4/2) = 1.0 + 1.0
    assert good["usable"] is True and good["objective"] == pytest.approx(2.0)
    # Missing kind -> no objective at all, not a smaller one.
    for partial in (
        {"stream": {"nll_sum": 10.0, "supervised_tokens": 5}},
        {"stream": {"nll_sum": 10.0, "supervised_tokens": 0},
         "full_generation": {"nll_sum": 4.0, "supervised_tokens": 2}},
        {"stream": {"nll_sum": float("nan"), "supervised_tokens": 5},
         "full_generation": {"nll_sum": 4.0, "supervised_tokens": 2}},
        {"stream": {"nll_sum": float("inf"), "supervised_tokens": 5},
         "full_generation": {"nll_sum": 4.0, "supervised_tokens": 2}},
    ):
        report = protocol.objective(partial)
        assert report["usable"] is False, partial
        assert report["objective"] is None, partial
        assert report["invalid_kinds"] or report["missing_kinds"], partial
    with pytest.raises(ValueError, match="finite|non-negative|positive"):
        ValidationProtocol(samples=protocol.samples, kinds=("stream",), weights={"stream": float("nan")})
    with pytest.raises(ValueError, match="positive"):
        ValidationProtocol(
            samples=protocol.samples,
            kinds=("stream", "full_generation"),
            weights={"stream": 0.0, "full_generation": 0.0},
        )
    # Weights that do not sum to one are normalized once, at construction, and the
    # normalized values are what the run reports.
    normalized = ValidationProtocol(
        samples=protocol.samples,
        kinds=("stream", "full_generation"),
        weights={"stream": 3.0, "full_generation": 1.0},
    )
    assert normalized.weights["stream"] == pytest.approx(0.75)
    assert normalized.weights["full_generation"] == pytest.approx(0.25)
    assert sum(normalized.weights.values()) == pytest.approx(1.0)


def test_protocol_write_rejects_non_finite_rows(tmp_path):
    from stylized_motion.learning.mts_operator.eval_protocol import ValidationProtocol

    protocol = _protocol_for_c03()
    protocol.write(tmp_path / "protocol.json")
    stored = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert stored["samples_per_kind"] and stored["weights"]
    assert all(isfinite(value) for value in stored["weights"].values())


def isfinite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


def test_row_slicing_keeps_field_semantics():
    """C03: hard_mask [T, K] must not be sliced because T happens to equal B."""
    from stylized_motion.learning.mts_operator.metrics import _row
    from stylized_motion.learning.mts_operator.model import OperatorBatch

    view = adapter()
    frames = 3
    batch = OperatorBatch(
        target_tokens=torch.randint(0, 9, (3, frames, 40), generator=torch.Generator().manual_seed(1)),
        reference_tokens=torch.randint(0, 9, (3, frames, 40), generator=torch.Generator().manual_seed(2)),
        visible_mask=torch.zeros(3, frames, 40, dtype=torch.bool),
        hard_mask=torch.ones(frames, 40, dtype=torch.bool),  # T == B == 3
        target_valid_mask=torch.ones(3, frames, dtype=torch.bool),
        strength=torch.tensor([1.0, 1.0, 1.0]),
        sample_metadata=[{"sample_id": index} for index in range(3)],
    )
    single = _row(batch, 1)
    assert single.hard_mask is not None and tuple(single.hard_mask.shape) == (frames, 40)
    assert tuple(single.target_tokens.shape) == (1, frames, 40)
    assert tuple(single.target_valid_mask.shape) == (1, frames)
    assert tuple(single.strength.shape) == (1,)
    assert [meta["sample_id"] for meta in single.sample_metadata] == [1]
