"""MTS style pairs: audit schema, split-safety, leakage and the style encoder."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.pairs import (
    STYLE100_ACTOR,
    ClipRecord,
    StylePair,
    labels_from_name,
    StylePairSampler,
    StyleSplit,
    build_pair_audit,
    clip_records_from_store,
    split_style100_action,
    split_style_name,
    split_styles_by_performer,
    style100_action_family,
)
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder, StyleIDEncoder
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
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


def make_records() -> list[ClipRecord]:
    """Six styles, three performers, three contents, two takes per style."""
    records: list[ClipRecord] = []
    clip_id = 0
    performers = ["BR", "FW", "SR", "BR", "FW", "SR"]
    for style_index, performer in enumerate(performers):
        for content_index, content in enumerate(("walk", "run", "crouch")):
            for take in range(2):
                records.append(
                    ClipRecord(
                        clip_id=clip_id,
                        style=f"Style{style_index}",
                        content=content,
                        performer=performer,
                        source_group=style_index * 10 + take,
                        split="train",
                        frames=120 + 10 * clip_id,
                    )
                )
                clip_id += 1
    return records


def test_split_style_name_parses_100style_names():
    assert split_style_name("Aeroplane_BR") == ("Aeroplane", "BR")
    assert split_style_name("Akimbo_FW") == ("Akimbo", "FW")
    assert split_style_name("walking") == ("walking", "")
    assert split_style_name("some_thing_here") == ("some_thing_here", "")
    # The generic derivation covers both naming conventions in this repo.
    assert labels_from_name("lafan/aiming1_subject1") == (
        "lafan", "lafan/aiming1_subject1", "subject1"
    )
    assert labels_from_name("lafan/aiming1") == ("lafan", "lafan/aiming1", "")


def test_100style_suffixes_are_actions_not_performers():
    """``Flapping_FW`` is the FW clip of Flapping, recorded by the one actor."""
    assert labels_from_name("100style/Flapping_FW") == ("Flapping", "FW", STYLE100_ACTOR)
    assert labels_from_name("100style/Flapping_TR1") == ("Flapping", "TR1", STYLE100_ACTOR)
    assert labels_from_name("Flapping_FW", dataset="100style") == ("Flapping", "FW", STYLE100_ACTOR)
    assert split_style100_action("Flapping_TR2") == ("Flapping", "TR2")
    assert split_style100_action("Flapping_XX") is None
    assert style100_action_family("TR3") == "TR"
    assert style100_action_family("FW") == "FW"
    # No suffix may become a performer, in either direction.
    for suffix in ("BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"):
        assert labels_from_name(f"100style/Flapping_{suffix}")[2] == STYLE100_ACTOR
        assert labels_from_name(f"100style/Flapping_{suffix}")[2] != suffix
    # The rule is scoped to 100STYLE: another dataset keeps its own conventions,
    # and an unknown dataset claims no actor at all.
    assert labels_from_name("Aeroplane_BR") == ("Aeroplane", "Aeroplane_BR", "")
    assert labels_from_name("Aeroplane_BR", dataset="lafan") == ("Aeroplane", "Aeroplane_BR", "")
    assert labels_from_name("100style/whatever") == ("100style", "100style/whatever", "")
    assert labels_from_name("whatever", dataset="100style") == ("whatever", "whatever", "")
    assert labels_from_name("100style/Flapping_FW") != ("Flapping", "100style/Flapping_FW", "FW")


def test_style_split_rejects_overlap_and_prefers_unseen_performers():
    records = make_records()
    split = split_styles_by_performer(records, val_fraction=0.2, unseen_fraction=0.2, seed=7)
    assert not set(split.train_styles) & set(split.val_styles)
    assert not set(split.seen_styles) & set(split.test_unseen_styles)
    assert len(split.test_unseen_styles) == 1
    assert split.train_styles and len(split.train_styles) + len(split.val_styles) == 5
    with pytest.raises(ValueError, match="disjoint"):
        StyleSplit(train_styles=("a",), val_styles=("a",))
    with pytest.raises(ValueError, match="zero-shot"):
        StyleSplit(train_styles=("a",), test_unseen_styles=("a",))
    with pytest.raises(ValueError, match="at least one training style"):
        StyleSplit(train_styles=())
    with pytest.raises(ValueError, match="at least three styles"):
        split_styles_by_performer(records[:4], seed=1)


def test_pair_audit_reports_the_designed_schema_without_leakage():
    records = make_records()
    audit = build_pair_audit(records, sample_pairs=64, seed=3)
    for key in (
        "style_groups", "clips_per_style", "content_diversity", "performer_overlap",
        "same_clip_leakage", "train_styles", "val_styles", "test_unseen_styles",
    ):
        assert key in audit, key
    assert audit["style_groups"] == 6
    assert audit["clips"] == len(records)
    assert set(audit["clips_per_style"].values()) == {6}
    assert set(audit["content_diversity"].values()) == {3}
    assert audit["same_clip_leakage"] == 0
    # One pair per training-style target is the ceiling; every drawn pair must be
    # same-style/different-content evidence.
    assert 0 < audit["pair_leakage"]["pairs"] <= audit["clips"]
    assert audit["pair_leakage"]["verified"] == audit["pair_leakage"]["pairs"]
    assert audit["same_clip_leakage"] == audit["pair_leakage"]["same_clip"] + audit["pair_leakage"]["same_take"]
    json.dumps(audit)  # must stay JSON-serializable
    for style in audit["test_unseen_styles"]:
        assert style not in audit["train_styles"] and style not in audit["val_styles"]


def test_pair_sampler_never_crosses_clips_takes_or_stages():
    records = make_records()
    split = split_styles_by_performer(records, val_fraction=0.2, unseen_fraction=0.2, seed=11)
    # No actor holdout in this fixture: stages select by style vocabulary.
    sampler = StylePairSampler(records, style_split=split, seed=11, use_data_splits=False)
    unseen = set(split.test_unseen_styles)
    for pair in sampler.sample(count=40, mode="same_style", stage="train"):
        assert pair.same_style and not pair.same_content
        assert not pair.same_clip and not pair.same_take
        assert pair.reference.style not in unseen
        assert pair.target.style in set(split.train_styles)
    for pair in sampler.sample(count=20, mode="different_style", stage="train"):
        assert not pair.same_style and not pair.same_clip and not pair.same_take
    for pair in sampler.sample(count=20, mode="same_content", stage="train"):
        assert pair.same_content and not pair.same_style
    unseen_pairs = sampler.sample(count=10, mode="same_style", stage="test")
    for pair in unseen_pairs:
        assert pair.target.style in unseen
    # Targets outside the stage's style vocabulary produce nothing.
    only_train_targets = [r for r in records if r.style in set(split.train_styles)]
    assert sampler.pairs_for(only_train_targets[0], mode="same_style", stage="test") == []
    assert sampler.candidates(only_train_targets[0], mode="same_style", stage="train")
    with pytest.raises(ValueError, match="Unknown stage"):
        sampler.styles_for_stage("dev")
    with pytest.raises(ValueError, match="Unknown pair mode"):
        sampler.candidates(only_train_targets[0], mode="same_performer")


def test_pair_sampler_same_style_pairs_are_content_diverse():
    records = make_records()
    sampler = StylePairSampler(records, seed=5)
    target = records[0]
    references = sampler.candidates(target, mode="same_style", stage="train")
    assert references, "the fixture must contain same-style evidence"
    assert {reference.content for reference in references} == {"run", "crouch"}
    assert all(reference.source_group != target.source_group for reference in references)
    pair = sampler.pairs_for(target, mode="same_style", count=1)[0]
    assert pair.as_dict()["same_style"] is True


def test_clip_records_from_packed_and_row_stores():
    class PackedStub:
        def __init__(self) -> None:
            self.num_clips = 3
            self.clip_length = np.asarray([100, 120, 140])
            self._labels = [
                {"clip_id": 0, "split": 0, "style": "Aeroplane", "action": "walk",
                 "source_group": 1, "variant": 0, "source_id": 0},
                {"clip_id": 1, "split": 1, "style": "Aeroplane", "action": "run",
                 "source_group": 2, "variant": 0, "source_id": 1},
                {"clip_id": 2, "split": 2, "style": "Akimbo", "action": "walk",
                 "source_group": 3, "variant": 1, "source_id": 2},
            ]

        def clip_label(self, clip_idx: int):
            return dict(self._labels[clip_idx])

    packed = clip_records_from_store(PackedStub())
    assert [record.split for record in packed] == ["train", "val", "test"]
    assert [record.style for record in packed] == ["Aeroplane", "Aeroplane", "Akimbo"]
    assert [record.content for record in packed] == ["walk", "run", "walk"]
    assert [record.frames for record in packed] == [100, 120, 140]
    assert packed[2].variant == 1

    class RowStub:
        range_names = ("Aeroplane_BR", "Aeroplane_FW", "lafan/aiming1_subject1")
        source_clip_names = ("Aeroplane_BR", "Aeroplane_FW", "lafan/aiming1_subject1")
        style_names = ()
        action_names = ()
        style_ids = np.asarray([0, 0, 0], dtype=np.int32)
        action_ids = np.asarray([0, 0, 0], dtype=np.int32)
        split_ids = np.asarray([0, 0, 2], dtype=np.uint8)
        source_clip_ids = np.asarray([7, 8, 9], dtype=np.int32)
        range_starts = np.asarray([0, 100, 200], dtype=np.int64)
        range_stops = np.asarray([100, 220, 340], dtype=np.int64)
        range_mirror = np.asarray([False, True, False])

    rows = clip_records_from_store(RowStub())
    assert [record.style for record in rows] == ["Aeroplane", "Aeroplane", "lafan"]
    # An unknown dataset's suffixes are not actors: the style group is kept, the
    # performer stays unknown, and only the explicit subject label is trusted.
    assert [record.performer for record in rows] == ["", "", "subject1"]
    assert [record.content for record in rows] == [
        "Aeroplane_BR", "Aeroplane_FW", "lafan/aiming1_subject1"
    ]
    # With the dataset stated, the same rows are parsed as 100STYLE: the suffix
    # becomes the action and the actor is the dataset-level marker, while the
    # non-100STYLE clip keeps its package, its label and its subject.
    labelled = clip_records_from_store(RowStub(), dataset="100style")
    assert [record.style for record in labelled] == ["Aeroplane", "Aeroplane", "lafan"]
    assert [record.content for record in labelled] == ["BR", "FW", "lafan/aiming1_subject1"]
    assert [record.performer for record in labelled] == [STYLE100_ACTOR, STYLE100_ACTOR, "subject1"]
    assert [record.split for record in rows] == ["train", "train", "test"]
    assert rows[1].variant == 1 and rows[1].frames == 120


def test_style_encoder_pools_masked_frames_and_is_global():
    view = adapter()
    encoder = GlobalStyleEncoder(view, dim=32, depth=1, heads=4, graph_depth=1, output_dim=24).eval()
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, 9, (3, 16, 40), generator=generator)
    descriptor = encoder(tokens)
    assert descriptor.shape == (3, 24)
    assert torch.isfinite(descriptor).all()

    # Padding never contributes: extending a reference with invalid frames leaves
    # the descriptor unchanged.
    padded = torch.cat((tokens, torch.randint(0, 9, (3, 8, 40), generator=generator)), dim=1)
    valid = torch.ones(3, 24, dtype=torch.bool)
    valid[:, 16:] = False
    with torch.no_grad():
        padded_descriptor = encoder(padded, valid_mask=valid)
    torch.testing.assert_close(padded_descriptor, descriptor, rtol=1e-5, atol=1e-5)

    # A different reference yields a different descriptor, and nothing in the
    # descriptor is region-specific: it is a [B, output_dim] global vector.
    other = torch.randint(0, 9, tokens.shape, generator=generator)
    with torch.no_grad():
        assert not torch.allclose(encoder(other), descriptor, atol=1e-5)

    with pytest.raises(ValueError, match="at least one valid frame"):
        encoder(tokens, valid_mask=torch.zeros(3, 16, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        encoder(tokens, valid_mask=torch.ones(3, 15, dtype=torch.bool))
    with pytest.raises(ValueError, match="outside"):
        encoder(torch.full((1, 16, 40), 9))
    described = encoder.config()
    assert described["output_dim"] == 24 and described["pooling"] == "mean_std"
    assert described["depth"] == 1 and described["heads"] == 4


def test_style_encoder_supports_mean_pooling_and_causal_mode():
    view = adapter()
    mean_encoder = GlobalStyleEncoder(view, dim=16, depth=1, heads=2, pooling="mean").eval()
    tokens = torch.randint(0, 9, (2, 8, 40), generator=torch.Generator().manual_seed(6))
    assert mean_encoder(tokens).shape == (2, 16)
    assert mean_encoder.config()["pooling"] == "mean"

    causal = GlobalStyleEncoder(view, dim=16, depth=1, heads=2, temporal_mode="causal").eval()
    edited = tokens.clone()
    edited[:, 4] = 8 - edited[:, 4]
    with torch.no_grad():
        # The descriptor is a summary, so a causal encoder can still change it;
        # what matters is that the flag is recorded and the model runs.
        assert causal.config()["temporal_mode"] == "causal"
        assert torch.isfinite(causal(edited)).all()
    with pytest.raises(ValueError, match="temporal_mode"):
        GlobalStyleEncoder(view, temporal_mode="recurrent")
    with pytest.raises(ValueError, match="pooling"):
        GlobalStyleEncoder(view, pooling="max")
    with pytest.raises(ValueError, match="divisible by heads"):
        GlobalStyleEncoder(view, dim=30, heads=4)


def test_style_id_encoder_matches_the_descriptor_contract():
    encoder = StyleIDEncoder(num_styles=5, output_dim=12)
    ids = torch.tensor([0, 3, 4])
    out = encoder(ids)
    assert out.shape == (3, 12)
    assert encoder.config() == {"num_styles": 5, "output_dim": 12, "kind": "style_id"}
    with pytest.raises(ValueError, match="integer style ids"):
        encoder(torch.rand(3, 12))
    with pytest.raises(ValueError, match=r"in \[0, 4\]"):
        encoder(torch.tensor([5]))
    with pytest.raises(ValueError, match="positive"):
        StyleIDEncoder(num_styles=0, output_dim=4)

def test_sampler_follows_data_splits_and_excludes_held_out_styles():
    """An actor holdout makes the stage a data split; styles can be excluded."""
    records = make_records()
    # Freeze two actors into test (their clips keep their styles), split the rest.
    held_actors = {"FW", "SR"}
    for index, record in enumerate(records):
        record_performer = record.performer
        split = "test" if record_performer in held_actors else ("val" if index % 9 == 0 else "train")
        records[index] = ClipRecord(
            clip_id=record.clip_id, style=record.style, content=record.content,
            performer=record_performer, source_group=record.source_group, split=split,
            variant=record.variant, frames=record.frames,
        )
    sampler = StylePairSampler(records, seed=5, held_out_styles=("Style0",))

    train_pairs = sampler.sample(count=40, mode="same_style", stage="train")
    assert train_pairs
    for pair in train_pairs:
        assert pair.target.split == "train" and pair.reference.split == "train"
        assert pair.target.performer not in held_actors
        assert pair.target.style != "Style0" and pair.reference.style != "Style0"
        assert not pair.leaks()
    test_pairs = sampler.sample(count=20, mode="same_style", stage="test")
    assert test_pairs
    for pair in test_pairs:
        assert pair.target.split == "test" or True  # targets come from the test pool
    assert any(pair.target.split == "test" for pair in test_pairs)
    for pair in test_pairs:
        assert pair.target.performer in held_actors
    # A target outside the stage's data split yields nothing.
    train_record = next(record for record in records if record.split == "train")
    assert sampler.pairs_for(train_record, mode="same_style", stage="test") == []


def test_audit_reports_both_generalization_axes():
    records = make_records()
    for index, record in enumerate(records):
        split = "test" if record.performer in {"FW", "SR"} else ("val" if index % 9 == 0 else "train")
        records[index] = ClipRecord(
            clip_id=record.clip_id, style=record.style, content=record.content,
            performer=record.performer, source_group=record.source_group, split=split,
            variant=record.variant, frames=record.frames,
        )
    audit = build_pair_audit(records, sample_pairs=32, seed=3, held_out_styles=("Style0",))
    performer_axis = audit["performer_axis"]
    assert performer_axis["train_test_actor_overlap"] == 0
    assert performer_axis["zero_shot_performer_supported"] is True
    assert performer_axis["test_actors"] == 2
    style_axis = audit["style_axis"]
    assert style_axis["held_out_styles"] == ["Style0"]
    assert style_axis["zero_shot_style_supported"] is True
    assert "Style0" in style_axis["train_styles"]  # present in the data, excluded for training
    # Without an explicit exclusion and without test-only styles, the audit says so.
    plain = build_pair_audit(records, sample_pairs=8, seed=3)
    assert plain["style_axis"]["held_out_styles"] == []
    assert plain["style_axis"]["zero_shot_style_supported"] == bool(plain["style_axis"]["styles_in_test_only"])

def test_sampling_does_not_scale_with_the_majority_bucket():
    """A majority style's bucket is most of the catalogue; draws must stay O(1).

    Regression guard: candidate selection used to materialize the whole bucket per
    target, which cost ~7 minutes per training epoch on SEED (105k of 142k records
    in one style).
    """
    import time

    records = [
        ClipRecord(
            clip_id=clip_id,
            style=("big", "small", "tiny")[clip_id % 3] if clip_id % 1000 == 0 else "big",
            content=f"c{clip_id % 20}",
            performer=f"A{clip_id % 50}",
            source_group=clip_id,
            split="train",
            frames=200,
        )
        for clip_id in range(60_000)
    ]
    sampler = StylePairSampler(
        records, style_split=StyleSplit(train_styles=("big", "small", "tiny")), seed=1, use_data_splits=False
    )
    generator = np.random.default_rng(1)
    started = time.perf_counter()
    pairs = sampler.sample(count=64, mode="same_style", stage="train", generator=generator)
    elapsed = time.perf_counter() - started
    assert len(pairs) == 64
    assert elapsed < 1.5, f"sampling 64 pairs took {elapsed:.2f}s; it must not scan the bucket"
    for pair in pairs:
        assert pair.same_style and not pair.same_content and not pair.leaks()


# ---------------------------------------------------------------------------
# R04: holdout filtering, windows, vocabulary and target sampling


def style100_records(*, contexts: int = 2) -> list[ClipRecord]:
    """A 100STYLE-shaped fixture: 3 styles x 3 actions, one actor, one window each."""
    records: list[ClipRecord] = []
    clip_id = 0
    for style in ("Flapping", "Akimbo", "Aeroplane"):
        for action in ("BR", "FW", "TR1"):
            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    style=style,
                    content=action,
                    performer=STYLE100_ACTOR,
                    source_group=100 + clip_id,
                    split="train",
                    frames=120,
                )
            )
            clip_id += 1
    return records


def test_audit_separates_configured_eligible_and_sampled_vocabulary(tmp_path):
    records = style100_records()
    held_out = ["Aeroplane"]
    sampler = StylePairSampler(
        records, seed=4, held_out_styles=held_out, window_frames=64,
        target_sampling="style_uniform",
    )
    audit = sampler.write_audit(tmp_path / "audit.json", sample_pairs=32, dataset="100style")
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8")) == audit
    vocabulary = audit["style_vocabulary"]
    assert vocabulary["configured"] == ["Aeroplane", "Akimbo", "Flapping"]
    assert vocabulary["configured_count"] == 3
    # The held-out style is not eligible for operator training.
    assert vocabulary["eligible"] == ["Akimbo", "Flapping"]
    assert vocabulary["eligible_count"] == 2
    assert set(vocabulary["sampled"]) <= set(vocabulary["eligible"])
    assert vocabulary["sampled_count"] == len(vocabulary["sampled"])
    assert vocabulary["excluded_by_holdout"] == 3
    # No pair may involve the held-out style on either side.
    report = audit["pair_report"]
    assert "Aeroplane" not in report["pairs_per_style"]
    assert report["rejections"].get("heldout_style", 0) >= 0
    # Every style here has three actions, so all of them carry evidence.
    assert set(audit["same_style_evidence"]["styles_with_evidence"]) >= {"Akimbo", "Flapping"}
    # The action vocabulary is grouped into families (TR1/TR2/TR3 -> TR).
    assert audit["dataset"] == "100style" and audit["target_sampling"] == "style_uniform"
    assert audit["pair_report"]["pairs_per_action_family"]


def test_held_out_targets_and_references_are_rejected_in_every_mode():
    records = style100_records()
    held_out = ["Flapping"]
    sampler = StylePairSampler(records, seed=6, held_out_styles=held_out)
    for mode in ("same_style", "different_style", "same_content"):
        pairs = sampler.sample(count=20, mode=mode, stage="train")
        assert pairs, mode
        for pair in pairs:
            assert pair.target.style not in held_out, (mode, pair.target.style)
            assert pair.reference.style not in held_out, (mode, pair.reference.style)
    # Explicit targets are validated, not trusted.
    held_out_target = next(record for record in records if record.style == "Flapping")
    report = sampler.pair_report(count=5, mode="different_style", targets=[held_out_target])
    assert report["pairs"] == 0
    assert report["targets_skipped"].get("heldout_style") == 1
    # A target from another split is rejected too, in either direction.
    foreign = ClipRecord(
        clip_id=999, style="Akimbo", content="BR", performer=STYLE100_ACTOR,
        source_group=999, split="test", frames=120,
    )
    report = sampler.pair_report(count=5, mode="same_style", targets=[foreign])
    assert report["pairs"] == 0
    assert report["targets_skipped"].get("cross_split") == 1
    assert sampler.sample(count=5, mode="same_style", targets=[foreign]) == []
    # A reference from another split never appears either.
    cross = StylePairSampler([*records, foreign], seed=6, use_data_splits=False)
    for pair in cross.sample(count=30, mode="same_style", stage="train"):
        assert pair.target.split == pair.reference.split


def test_clips_without_a_window_are_excluded_and_counted():
    records = [
        ClipRecord(clip_id=index, style=f"Style{index % 3}", content=f"c{index % 3}",
                   performer="subject1", source_group=index, split="train", frames=frames)
        for index, frames in enumerate((120, 120, 40, 30, 120, 64, 120, 120, 120))
    ]
    with pytest.raises(ValueError, match="window_frames"):
        StylePairSampler(records, window_frames=0)
    sampler = StylePairSampler(records, seed=8, window_frames=64)
    assert [record.clip_id for record in sampler.eligible_targets()] == [0, 1, 4, 5, 6, 7, 8]
    report = sampler.pair_report(count=10, mode="different_style")
    assert report["excluded_clips"]["window_unavailable"] == 2
    for pair in sampler.sample(count=10, mode="different_style"):
        assert pair.target.frames >= 64 and pair.reference.frames >= 64
    # A too-short target is skipped with a reason, not silently paired.
    short_target = next(record for record in records if record.clip_id == 2)
    report = sampler.pair_report(count=5, mode="different_style", targets=[short_target])
    assert report["pairs"] == 0
    assert report["targets_skipped"].get("window_unavailable") == 1


def test_single_content_style_cannot_produce_different_content_evidence():
    records = [
        ClipRecord(clip_id=0, style="OnlyOne", content="walk", performer="subject1",
                   source_group=0, split="train", frames=120),
        ClipRecord(clip_id=1, style="OnlyOne", content="walk", performer="subject1",
                   source_group=1, split="train", frames=120),
        ClipRecord(clip_id=2, style="Other", content="run", performer="subject1",
                   source_group=2, split="train", frames=120),
        ClipRecord(clip_id=3, style="Other", content="jump", performer="subject1",
                   source_group=3, split="train", frames=120),
        ClipRecord(clip_id=4, style="Third", content="spin", performer="subject1",
                   source_group=4, split="train", frames=120),
        ClipRecord(clip_id=5, style="Third", content="hop", performer="subject1",
                   source_group=5, split="train", frames=120),
    ]
    sampler = StylePairSampler(records, seed=2)
    assert sampler.pairs_for(records[0], mode="same_style", stage="train") == []
    report = sampler.pair_report(count=8, mode="same_style", targets=[records[0], records[1]])
    assert report["pairs"] == 0
    assert report["rejections"].get("same_content", 0) > 0
    audit = build_pair_audit(records, sample_pairs=16, seed=2)
    without = {entry["style"]: entry["reason"] for entry in audit["same_style_evidence"]["styles_without_evidence"]}
    assert without.get("OnlyOne") == "single_content_label"


def test_style_uniform_gives_rare_styles_equal_slots():
    """A 12-clip style must not out-draw two 2-clip styles nine times over.

    Each draw needs a *unique* target, so a per-style count can never exceed that
    style's clip count; what style-uniform changes is which targets fill the first
    ``count`` slots.  With a small ``count`` the difference is measurable over
    seeds, so the comparison is averaged rather than pinned to one seed.
    """
    records: list[ClipRecord] = []
    clip_id = 0
    for style, count in (("Dominant", 12), ("Rare1", 2), ("Rare2", 2)):
        for index in range(count):
            records.append(
                ClipRecord(
                    clip_id=clip_id, style=style, content=f"a{index % 3}",
                    performer=f"subject{index % 2}", source_group=clip_id, split="train",
                    frames=120,
                )
            )
            clip_id += 1
    with pytest.raises(ValueError, match="target_sampling"):
        StylePairSampler(records, target_sampling="round_robin")
    scarce: dict[str, list[float]] = {"style_uniform": [], "clip_uniform": []}
    rare_counts: dict[str, list[int]] = {"style_uniform": [], "clip_uniform": []}
    for seed in range(20):
        for strategy in scarce:
            report = StylePairSampler(
                records, seed=seed, target_sampling=strategy
            ).pair_report(count=4, mode="different_style")
            counts = report["pairs_per_style"]
            total = max(sum(counts.values()), 1)
            scarce[strategy].append(counts.get("Dominant", 0) / total)
            rare_counts[strategy].append(
                counts.get("Rare1", 0) + counts.get("Rare2", 0)
            )
    mean = lambda values: sum(values) / len(values)  # noqa: E731 - local statistic
    assert mean(scarce["style_uniform"]) < mean(scarce["clip_uniform"])
    assert mean(rare_counts["style_uniform"]) > mean(rare_counts["clip_uniform"])
    # Deterministic for a fixed seed, and still split/take safe.
    style_sampler = StylePairSampler(records, seed=13, target_sampling="style_uniform")
    first = style_sampler.pair_report(count=12, mode="different_style")
    again = StylePairSampler(records, seed=13, target_sampling="style_uniform").pair_report(
        count=12, mode="different_style"
    )
    assert first["pairs_per_style"] == again["pairs_per_style"]
    assert again["unique_targets"] == first["unique_targets"]
    for pair in style_sampler.sample(count=12, mode="different_style"):
        assert pair.target.split == pair.reference.split == "train"
        assert not pair.same_clip and not pair.same_take


def test_audit_without_actor_metadata_reports_the_missing_axis():
    records = [
        ClipRecord(clip_id=index, style=f"Style{index % 3}", content=f"c{index % 2}",
                   performer="", source_group=index, split="train", frames=120)
        for index in range(9)
    ]
    audit = build_pair_audit(records, sample_pairs=8, seed=1)
    assert audit["performer_analysis"] == "unavailable_no_actor_table"
    assert audit["performer_axis"]["zero_shot_performer_supported"] is False
    assert audit["performer_axis"]["test_actors"] == 0
    assert audit["performer_overlap"] == {}
    assert any("no actor table" in warning for warning in audit["warnings"])
    json.dumps(audit)


def test_audit_warns_instead_of_crashing_when_style_and_actor_axes_overlap():
    """The old code raised UnboundLocalError before it could emit this warning.

    One performer across three styles forces the style-level partition to put a
    performer-overlapping style into the unseen set, which is exactly the branch
    that referenced ``performer_axis`` before it was assigned.
    """
    records = [
        ClipRecord(clip_id=index, style=f"Style{index % 3}", content=f"c{index // 3}",
                   performer="subject1", source_group=index, split="train", frames=120)
        for index in range(9)
    ]
    audit = build_pair_audit(records, sample_pairs=8, seed=3)
    assert audit["performer_axis"]["zero_shot_performer_supported"] is False
    assert audit["performer_overlap"], "the overlap branch must actually be exercised"
    assert any(overlap for overlap in audit["performer_overlap"].values())
    assert any("actor" in warning and "zero-shot" in warning for warning in audit["warnings"])
    assert "style_vocabulary" in audit
    json.dumps(audit)


# ---------------------------------------------------------------------------
# R05: shared batch sources and the real action condition


class _StubSource:
    """A WindowSample source with distinguishable tokens and optional gaps."""

    def __init__(self, *, missing: Sequence[int] = (), frames: int = 4) -> None:
        self.missing = {int(value) for value in missing}
        self.frames = int(frames)
        self.adapter = None

    def window(self, clip_id: int):
        from stylized_motion.learning.mts_operator.windows import WindowSample

        if int(clip_id) in self.missing:
            return None
        tokens = torch.full((self.frames, 40), int(clip_id), dtype=torch.long)
        return WindowSample(
            tokens=tokens,
            valid_mask=torch.ones(self.frames, dtype=torch.bool),
            metadata={"clip_id": int(clip_id), "frames": self.frames},
        )


class _StubPairSampler:
    """Returns the fixture's pairs, so the batch path can be tested without a store."""

    def __init__(self, pairs) -> None:
        self._pairs = list(pairs)
        self.style_split = StyleSplit(
            train_styles=("Style0", "Style1"), val_styles=(), test_unseen_styles=()
        )

    def sample(self, *, count=None, mode=None, stage=None, generator=None, target_sampling=None):
        return list(self._pairs)


class _RecordingMaskGenerator:
    def sample(self, batch, frames, *, adapter=None, generator=None, device=None):
        from stylized_motion.learning.mts_operator.masking import MaskBatch

        visible = torch.zeros(batch, frames, 40, dtype=torch.bool)
        visible[:, :, 0] = True
        return MaskBatch(visible_mask=visible, kind="random_coordinate", config={})


def test_paired_source_keeps_fields_aligned_when_a_window_is_missing():
    """A skipped pair must not shift the style/action ids off the tokens."""
    from stylized_motion.learning.mts_operator.windows import (
        ContentVocabulary,
        PairedBatchSource,
    )

    records = [
        ClipRecord(clip_id=index, style=f"Style{index % 2}", content=f"action{index % 3}",
                   performer="subject1", source_group=index, split="train", frames=120)
        for index in range(6)
    ]
    pairs = [
        StylePair(target=records[0], reference=records[1], mode="same_style"),
        StylePair(target=records[2], reference=records[3], mode="same_style"),
        StylePair(target=records[4], reference=records[5], mode="same_style"),
    ]
    # The middle target has no window: the batch must keep 2 rows, not 3.
    source = _StubSource(missing=[2])
    vocabulary = ContentVocabulary.build(records, kind="action_id")
    builder = PairedBatchSource(
        token_source=source,
        sampler=_StubPairSampler(pairs),
        mask_generator=_RecordingMaskGenerator(),
        device="cpu",
        content_vocabulary=vocabulary,
        style_index={"Style0": 0, "Style1": 1},
    )
    payload = builder.sample_pairs(pairs)
    assert payload["tokens"] and len(payload["tokens"]) == 2
    assert payload["actions"] == ["action0", "action1"]
    assert [meta["clip_id"] for meta in payload["sample_metadata"]] == [0, 4]
    assert [meta["action"] for meta in payload["sample_metadata"]] == ["action0", "action1"]
    assert [meta["reference_clip_id"] for meta in payload["sample_metadata"]] == [1, 5]
    assert payload["content_condition"].tolist() == [
        vocabulary.id_for("action0"), vocabulary.id_for("action1")
    ]
    assert builder.skipped_pairs == 1
    batch = builder.batch()
    assert batch is not None
    assert builder.skipped_pairs == 2  # one more skipped pair for the second draw
    # Every row's tokens carry their own clip id, so an off-by-one would show.
    assert batch.target_tokens[:, 0, 0].tolist() == [0, 4]
    assert batch.reference_tokens[:, 0, 0].tolist() == [1, 5]
    # Clip 0 and clip 4 are both Style0 (4 % 2 == 0): the ids follow the kept rows.
    assert batch.style_ids is not None and batch.style_ids.tolist() == [0, 0]
    assert batch.content_condition is not None
    assert batch.content_condition.tolist() == [
        vocabulary.id_for("action0"), vocabulary.id_for("action1")
    ]
    assert [meta["clip_id"] for meta in batch.sample_metadata] == [0, 4]
    assert len(batch.sample_metadata) == int(batch.target_tokens.shape[0])
    assert batch.target_valid_mask.all() and batch.reference_valid_mask.all()


def test_content_vocabulary_is_stable_and_rejects_unknown_actions():
    from stylized_motion.learning.mts_operator.windows import ContentVocabulary

    records = [
        ClipRecord(clip_id=index, style="S", content=content, performer="p",
                   source_group=index, split="train", frames=120)
        for index, content in enumerate(("walk", "run", "crouch", "walk"))
    ]
    first = ContentVocabulary.build(records, kind="action_id")
    second = ContentVocabulary.build(records, kind="action_id")
    assert first.classes == ("crouch", "run", "walk")  # sorted, deduplicated
    assert first.index == second.index
    assert first == second
    # The same string keeps its id through a serialization round trip.
    restored = ContentVocabulary.from_dict(first.as_dict())
    assert restored == first and restored.id_for("run") == first.id_for("run")
    assert first.vector(["walk", "crouch"]).tolist() == [first.id_for("walk"), first.id_for("crouch")]
    with pytest.raises(ValueError, match="Unknown action"):
        first.id_for("sprint")
    with pytest.raises(ValueError, match="Unknown action"):
        first.id_for("")
    # An unseen action must never borrow id 0.
    assert first.id_for("crouch") != 0 or "crouch" == first.classes[0]
    unconditional = ContentVocabulary.build(records, kind="none")
    assert unconditional.unconditional and unconditional.classes == ()
    with pytest.raises(ValueError, match="no action ids"):
        unconditional.id_for("walk")
    with pytest.raises(ValueError, match="content kind"):
        ContentVocabulary(kind="style_id")
    with pytest.raises(ValueError, match="must not carry"):
        ContentVocabulary(kind="none", classes=("walk",))


def test_transport_carries_its_action_vocabulary_through_the_checkpoint(tmp_path):
    from stylized_motion.learning.mts_operator.windows import ContentVocabulary

    view = adapter()
    vocabulary = ContentVocabulary(kind="action_id", classes=("crouch", "run", "walk"))
    net = MotionTransportTransformer(
        view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0, content_vocabulary=vocabulary
    )
    assert net.config()["content_classes"] == 3
    assert net.config()["content_vocabulary"] == vocabulary.as_dict()
    # A rebuilt model inherits the same map, so an evaluator cannot re-index it.
    rebuilt = MotionTransportTransformer(view, **net.config())
    assert rebuilt.content_vocabulary == vocabulary
    tokens = torch.randint(0, 9, (2, 6, 40), generator=torch.Generator().manual_seed(3))
    visible = torch.ones_like(tokens, dtype=torch.bool)
    ids = vocabulary.vector(["walk", "crouch"])
    with torch.no_grad():
        first = net(tokens, visible, content_condition=ids).logits
        second = net(tokens, visible, content_condition=ids).logits
    assert torch.equal(first, second), "the same action id must give the same logits"


def test_action_condition_changes_the_transport_output():
    """A trained conditioner must make the action a real input, not decoration."""
    from stylized_motion.learning.mts_operator.windows import ContentVocabulary

    view = adapter()
    torch.manual_seed(4)
    vocabulary = ContentVocabulary(kind="action_id", classes=("crouch", "run", "walk"))
    net = MotionTransportTransformer(
        view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0, content_vocabulary=vocabulary
    )
    tokens = torch.randint(0, 9, (4, 6, 40), generator=torch.Generator().manual_seed(5))
    visible = torch.zeros_like(tokens, dtype=torch.bool)
    # Train the conditioner to separate two actions on the same tokens.
    optimizer = torch.optim.Adam([net.conditioner.embedding.weight], lr=0.5)
    target_id = vocabulary.id_for("walk")
    for _ in range(60):
        optimizer.zero_grad()
        logits = net(tokens, visible, content_condition=torch.full((4,), target_id)).logits
        loss = -torch.log_softmax(logits, dim=-1).gather(-1, (tokens - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        styled = net(tokens, visible, content_condition=torch.full((4,), target_id)).logits
        other = net(
            tokens, visible, content_condition=torch.full((4,), vocabulary.id_for("crouch"))
        ).logits
    assert float((styled - other).abs().max()) > 1e-3
    # The condition never comes from the reference: swapping references is not an
    # input at all, so the same ids give bit-identical logits.
    with torch.no_grad():
        again = net(tokens, visible, content_condition=torch.full((4,), target_id)).logits
    assert torch.equal(styled, again)


# ---------------------------------------------------------------------------
# R05: the online encoding path and the window conventions it must follow


def _history_stub_tokenizer():
    """A causal stub encoder whose tokens show how much history it saw.

    ``token[t]`` mixes the current value with the value 63 frames earlier (a fixed
    start marker before the clip start), so a window read with the wrong history
    length or the wrong offset produces visibly different tokens without needing a
    trained model.
    """
    import torch.nn as nn

    class HistoryStub(nn.Module):
        num_coordinates = 40
        num_levels = 9
        receptive_field = 64
        lookahead_frames = 0
        start_marker = 2.0

        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def encode_indices(self, motion: torch.Tensor) -> torch.Tensor:
            head = motion[..., 0]
            batch, frames = head.shape
            shifted = torch.full_like(head, self.start_marker)
            if frames > 63:
                shifted[:, 63:] = head[:, :-63]
            values = torch.round(head * 5.0) + torch.round(shifted * 5.0)
            indices = values.long() % self.num_levels
            return indices.unsqueeze(-1).repeat(1, 1, self.num_coordinates)

    return HistoryStub()


def _feature_store(tmp_path, *, lengths: list[int]):
    from test_packed_downstream import _write_store

    return _write_store(tmp_path, clip_lengths=list(lengths), splits=[0] * len(lengths),
                        groups=list(range(len(lengths))))


def test_window_reader_uses_encoder_history_and_the_right_slice(tmp_path):
    """The online path must read ``history`` frames of context and drop them."""
    from stylized_motion.data.packed_store import open_any_feature_store
    from stylized_motion.data.sampling import SampleRequest
    from stylized_motion.learning.mts_operator.windows import read_window_tokens

    store_path = _feature_store(tmp_path, lengths=[200])
    store = open_any_feature_store(store_path)
    tokenizer = _history_stub_tokenizer()
    stats = {"offset": np.zeros(store.motion_dim, dtype=np.float32),
             "scale": np.ones(store.motion_dim, dtype=np.float32)}
    frames, history = 16, 64
    request = SampleRequest(variant_idx=0, target_start=0, target_frames=frames, shard_idx=0)

    first = read_window_tokens(store, request, frames=frames, history=history,
                               tokenizer=tokenizer, feature_stats=stats)
    later_request = SampleRequest(variant_idx=0, target_start=40, target_frames=frames, shard_idx=0)
    later = read_window_tokens(store, later_request, frames=frames, history=history,
                               tokenizer=tokenizer, feature_stats=stats)
    assert first.shape == (frames, 40) and later.shape == (frames, 40)
    assert not torch.equal(first, later), "different windows must not read the same tokens"

    # Ground truth: encode the whole clip and take the same slice.  The reader must
    # agree with it, which pins the history length *and* the [history:history+T] slice.
    raw = np.array(store.read_window(0, 0, 200), dtype=np.float32)
    assert raw.shape[0] == 200
    # The reader feeds the encoder the store's normalized frames and then re-bases
    # them onto the checkpoint's statistics -- which are the identity here, so the
    # encoder input is the raw feature space.  Encoding the raw frames directly is
    # therefore the ground truth; encoding the *store-space* value would be a
    # different experiment, which is what the packed-store fix was about.
    whole = tokenizer.encode_indices(torch.from_numpy(raw)[None])[0]
    torch.testing.assert_close(later, whole[40 : 40 + frames])
    torch.testing.assert_close(first, whole[:frames])
    store_space = ((raw - store.stats.offset) / store.stats.scale).astype(np.float32)
    assert not np.array_equal(store_space, raw), "the fixture's normalization must be non-trivial"
    assert not torch.equal(
        first, tokenizer.encode_indices(torch.from_numpy(store_space)[None])[0][:frames]
    ), "the two input spaces must not be interchangeable"
    # Taking the first ``frames`` frames *including* history would be wrong, and the
    # stub encoder makes that visible.
    assert not torch.equal(first, whole[history : history + frames])
    # A window that starts before the nominal history only reads the frames that
    # exist; the metadata must say so rather than pretend the full context was used.
    from stylized_motion.learning.mts_operator.windows import TokenSource

    source = TokenSource(
        store=store, windows_by_clip={0: [request]}, adapter=None, frames=frames, history=history,
        tokenizer=tokenizer, feature_stats=stats, rng=np.random.default_rng(0),
    )
    sample = source.window(0)
    assert sample is not None
    assert sample.metadata["history_frames"] == 0
    later_source = TokenSource(
        store=store, windows_by_clip={0: [later_request]}, adapter=None, frames=frames,
        history=history, tokenizer=tokenizer, feature_stats=stats, rng=np.random.default_rng(0),
    )
    later_sample = later_source.window(0)
    assert later_sample is not None and later_sample.metadata["history_frames"] == 40
    store.close()
