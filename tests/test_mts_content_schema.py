"""N02: the content label map is explicit, versioned and never guesses.

The round-1 protocol's content labels leaked the style ("Basic Locomotion Styles"
was non-neutral in 143 of 143 rows, "Basic Locomotion Neutral" was neutral in 38 of
38).  These tests pin the three rules that keep the fix honest: only the two
declared labels are merged, an unmapped label is its own class rather than a guess,
and a label the schema was not built on is reported instead of absorbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stylized_motion.learning.mts_operator.content_schema import (
    CONTENT_SCHEMA_V1_MERGE,
    CONTENT_SCHEMA_V1_NOTE,
    ContentSchema,
    default_content_schema,
    identity_content_schema,
)
from stylized_motion.learning.mts_operator.pairs import ClipRecord
from stylized_motion.learning.mts_operator.windows import build_content_vocabulary

REPO_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPO_ROOT / "data" / "configs" / "mts_content_schema_v1.yaml"

TRAIN_ACTIONS = (
    "Advanced Locomotion",
    "Baseline",
    "Basic Locomotion Neutral",
    "Basic Locomotion Styles",
    "Sports",
)


def test_only_the_two_declared_labels_are_merged():
    schema = default_content_schema()
    assert schema.merge == CONTENT_SCHEMA_V1_MERGE
    assert schema.canonical("Basic Locomotion Styles") == "Basic Locomotion"
    assert schema.canonical("Basic Locomotion Neutral") == "Basic Locomotion"
    # Everything else is identity: merging "Sports" into a known class would be a
    # guess about a different motion vocabulary.
    assert schema.canonical("Sports") == "Sports"
    assert schema.canonical("Advanced Locomotion") == "Advanced Locomotion"
    resolved = schema.resolve("Basic Locomotion Styles")
    assert resolved == {
        "raw": "Basic Locomotion Styles",
        "canonical": "Basic Locomotion",
        "renamed": True,
        "declared": True,
    }


def test_a_label_outside_the_schema_is_reported_not_absorbed():
    schema = default_content_schema(declared=TRAIN_ACTIONS)
    # "Martial Arts" is a real label the training split never declared.
    resolved = schema.resolve("Martial Arts")
    assert resolved["canonical"] == "Martial Arts" and resolved["declared"] is False
    assert schema.undeclared(["Martial Arts", "Sports", "Martial Arts"]) == {"Martial Arts": 2}
    # A declared label is not reported, and neither is a mapped one.
    assert schema.undeclared(["Sports", "Basic Locomotion Styles"]) == {}


def test_the_schema_version_and_note_travel_with_the_map():
    schema = default_content_schema(declared=TRAIN_ACTIONS)
    payload = schema.as_dict()
    assert payload["version"] == 1 and payload["declared"] == list(TRAIN_ACTIONS)
    assert CONTENT_SCHEMA_V1_NOTE in payload["note"]
    assert ContentSchema.from_dict(payload).as_dict() == payload
    # A round trip through a file keeps the version: a checkpoint that records the
    # version can never be scored by a store labelled with another map.
    with pytest.raises(ValueError):
        ContentSchema.from_dict(None)


def test_a_self_mapping_is_refused_as_noise():
    with pytest.raises(ValueError, match="maps to itself"):
        ContentSchema(version=1, merge={"Sports": "Sports"})


def test_the_shipped_config_parses_and_matches_the_default_map():
    """The YAML config is what a run reads; it must be the audited map."""
    schema = ContentSchema.read(CONFIG_PATH)
    assert schema.version == 1
    assert schema.merge == CONTENT_SCHEMA_V1_MERGE
    assert "Basic Locomotion" in schema.note or CONTENT_SCHEMA_V1_NOTE[:20] in schema.note
    # The declared list is the training split's own action list.
    assert "Basic Locomotion Styles" in schema.declared
    assert "Martial Arts" not in schema.declared


def test_the_legacy_map_is_identity_and_keeps_the_raw_ids():
    legacy = identity_content_schema(declared=TRAIN_ACTIONS)
    assert legacy.version == 0 and legacy.merge == {}
    for label in TRAIN_ACTIONS:
        assert legacy.canonical(label) == label


def test_classes_and_vocabulary_come_from_the_canonical_labels():
    schema = default_content_schema()
    classes = schema.classes(TRAIN_ACTIONS)
    assert classes == tuple(sorted(set(classes)))
    assert "Basic Locomotion" in classes and "Basic Locomotion Styles" not in classes
    # The vocabulary built from canonical labels has one fewer class, and its ids
    # are positions in the sorted canonical list -- never a raw-label index.
    records = [ClipRecord(clip_id=index, style="neutral", content=label) for index, label in enumerate(TRAIN_ACTIONS)]
    canonical_records = schema.apply(records)
    vocabulary = build_content_vocabulary(canonical_records, kind="action_id")
    assert len(vocabulary["classes"]) == len(TRAIN_ACTIONS) - 1
    assert vocabulary["classes"] == list(classes)


def test_applying_the_schema_touches_only_the_content_label():
    records = [
        ClipRecord(
            clip_id=3,
            style="injured leg",
            content="Basic Locomotion Styles",
            performer="",
            source_group=17,
            split="val",
            variant=1,
            frames=64,
        )
    ]
    (updated,) = default_content_schema().apply(records)
    assert updated.content == "Basic Locomotion"
    assert updated.clip_id == 3 and updated.style == "injured leg"
    assert updated.source_group == 17 and updated.variant == 1 and updated.split == "val"
