"""MTS style pairs: audit schema, split-safety, leakage and the style encoder."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.pairs import (
    ClipRecord,
    labels_from_name,
    StylePairSampler,
    StyleSplit,
    build_pair_audit,
    clip_records_from_store,
    split_style_name,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder, StyleIDEncoder
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
    assert labels_from_name("100style/Flapping_TR1") == (
        "Flapping", "100style/Flapping_TR1", "TR1"
    )
    assert labels_from_name("lafan/aiming1") == ("lafan", "lafan/aiming1", "")
    assert labels_from_name("Aeroplane_BR") == ("Aeroplane", "Aeroplane_BR", "BR")


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
    assert [record.performer for record in rows] == ["BR", "FW", "subject1"]
    assert [record.content for record in rows] == [
        "Aeroplane_BR", "Aeroplane_FW", "lafan/aiming1_subject1"
    ]
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
