"""The content label schema: raw action names in, one canonical label out.

The first round's operator validation had a short-cut built into its labels: in
the 213-row protocol, ``Basic Locomotion Styles`` was almost always a non-neutral
style and ``Basic Locomotion Neutral`` was always neutral, so a constant arm that
never saw the style could read the style off the *content* label.  This module
holds the versioned map that removes exactly that confusion and nothing else.

Rules, in order of importance:

* the map is *explicit* and versioned; a checkpoint records the version it was
  trained with, so a re-labelled store can never be scored by an old model
  silently;
* an unmapped label maps to itself -- that is identity, not a guess.  Merging two
  labels is a statement about those two labels only;
* a label the schema was not built on (not in ``declared``) is reported, never
  folded into a known class, so "Sports" cannot become "Basic Locomotion";
* the raw label is always carried beside the canonical one, because the trained
  round-1 checkpoints were conditioned on the raw ids and their scores are only
  interpretable with the raw label in hand.

Nothing here touches the tokenizer or the token bytes: this is a label map.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

CONTENT_SCHEMA_VERSION = 1
#: The first canonical map: merge the two labels whose names carry the style
#: switch.  Everything else is identity.
CONTENT_SCHEMA_V1_MERGE: dict[str, str] = {
    "Basic Locomotion Neutral": "Basic Locomotion",
    "Basic Locomotion Styles": "Basic Locomotion",
}
CONTENT_SCHEMA_V1_NOTE = (
    "v1 merges only Basic Locomotion Neutral and Basic Locomotion Styles into "
    "'Basic Locomotion', because 'Styles' vs 'Neutral' told the model which side of "
    "the style split a clip came from.  It does not claim the walk/run/turn mix, the "
    "speed range or the contact phases inside one label are matched."
)


@dataclass(frozen=True)
class ContentSchema:
    """A versioned ``raw -> canonical`` map, with the labels it was built on."""

    version: int = CONTENT_SCHEMA_VERSION
    merge: Mapping[str, str] = None  # type: ignore[assignment]
    declared: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        # Version 0 is the explicit legacy map: no merge at all, raw labels as they
        # are.  It is a version like any other, recorded so an old checkpoint's
        # conditioning can be named.
        if int(self.version) < 0:
            raise ValueError("A content schema version is a non-negative integer")
        mapping = {str(k): str(v) for k, v in dict(self.merge or {}).items()}
        for raw, canonical in mapping.items():
            if not raw or not canonical:
                raise ValueError(f"A content mapping must be name -> name, got {raw!r} -> {canonical!r}")
            if raw == canonical:
                raise ValueError(
                    f"{raw!r} maps to itself; an identity entry hides a label that the schema "
                    "means to rename, and it is noise otherwise"
                )
        object.__setattr__(self, "merge", mapping)
        object.__setattr__(self, "declared", tuple(str(value) for value in self.declared))

    # -- mapping ----------------------------------------------------------
    def canonical(self, raw: Any) -> str:
        """The canonical label of one raw action name (identity when unmapped)."""
        key = str(raw)
        if not key:
            raise ValueError("Empty action label: a missing label is not a class")
        return str(self.merge.get(key, key))

    def resolve(self, raw: Any) -> dict[str, Any]:
        """``canonical`` plus the audit trail: was it renamed, was it declared?"""
        key = str(raw)
        canonical = self.canonical(key)
        return {
            "raw": key,
            "canonical": canonical,
            "renamed": canonical != key,
            "declared": (not self.declared) or key in self.declared,
        }

    def classes(self, raw_labels: Sequence[str]) -> tuple[str, ...]:
        """The canonical classes of a raw label list, sorted and deduplicated.

        This is what a vocabulary is built from; the frozen id is the position in
        this tuple, so the same raw list always gives the same ids.
        """
        return tuple(sorted({self.canonical(label) for label in raw_labels}))

    def apply(self, records: Sequence[Any]) -> list[Any]:
        """Clip records with canonical content labels (everything else intact)."""
        return [replace(record, content=self.canonical(record.content)) for record in records]

    def undeclared(self, raw_labels: Sequence[str]) -> dict[str, int]:
        """Raw labels outside the training-declared set, with their counts."""
        counts: dict[str, int] = {}
        for label in raw_labels:
            if self.declared and str(label) not in self.declared:
                counts[str(label)] = counts.get(str(label), 0) + 1
        return dict(sorted(counts.items()))

    # -- (de)serialization ------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "merge": {str(k): str(v) for k, v in sorted(self.merge.items())},
            "declared": list(self.declared),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ContentSchema":
        if not payload:
            raise ValueError("A content schema payload is required")
        return cls(
            version=int(payload.get("version", CONTENT_SCHEMA_VERSION)),
            merge=payload.get("merge") or {},
            declared=tuple(payload.get("declared") or ()),
            note=str(payload.get("note", "")),
        )

    @classmethod
    def read(cls, path: str | Path) -> "ContentSchema":
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "content_schema" in payload:
            payload = payload["content_schema"]
        return cls.from_dict(payload)

    def write(self, path: str | Path) -> Path:
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump({"content_schema": self.as_dict()}, sort_keys=False), encoding="utf-8"
        )
        return path


def default_content_schema(*, declared: Sequence[str] = ()) -> ContentSchema:
    """The v1 schema: only the two Basic Locomotion labels are merged."""
    return ContentSchema(
        version=CONTENT_SCHEMA_VERSION,
        merge=dict(CONTENT_SCHEMA_V1_MERGE),
        declared=tuple(str(label) for label in declared),
        note=CONTENT_SCHEMA_V1_NOTE,
    )


def content_schema_from_config(
    content_config: Mapping[str, Any] | None, *, root: str | Path | None = None
) -> tuple["ContentSchema | None", dict[str, Any]]:
    """The schema a recipe's ``data.content`` block names, plus its audit trail.

    ``schema`` is either a path to a YAML/JSON schema file or an inline block.  A
    recipe without one keeps the legacy raw labels and says so, so a run that never
    saw a map cannot claim to have used one.

    ``strict: true`` makes an undeclared raw label an error instead of an extra
    identity class; a strict recipe cannot silently grow its content map.
    """
    block = dict(content_config or {})
    declared = block.get("schema") is not None
    if not declared:
        return None, {
            "schema": None,
            "source_path": None,
            "source_sha256": None,
            "strict": False,
            "note": "no content schema in the recipe: raw action labels are used as they are",
        }
    source = block.get("schema")
    source_path: str | None = None
    source_sha = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        if not path.exists():
            raise FileNotFoundError(f"The recipe's content schema does not exist: {path}")
        schema = ContentSchema.read(path)
        source_path = str(path)
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    elif isinstance(source, Mapping):
        schema = ContentSchema.from_dict(source)
    else:
        raise ValueError(
            f"data.content.schema must be a path or a mapping, got {type(source).__name__}"
        )
    return schema, {
        "schema": schema.as_dict(),
        "source_path": source_path,
        "source_sha256": source_sha,
        "strict": bool(block.get("strict", False)),
        "note": "canonical content labels; the raw label is kept in the audit report",
    }


def apply_content_schema(
    records: Sequence[Any],
    schema: "ContentSchema | None",
    *,
    strict: bool = False,
    strict_splits: Sequence[str] = ("train",),
) -> tuple[list[Any], dict[str, Any]]:
    """Clip records with canonical content labels, plus the audit report.

    The report carries the raw-label counts the schema did not declare, per split.
    An undeclared label is *reported*, never folded into a known class.  ``strict``
    applies to ``strict_splits`` (the training split by default): a *training* label
    the map was not built on stops the run, while a validation-only label outside
    the vocabulary is excluded from the protocol by the vocabulary filter -- the
    same rule the transport's validation already uses for ``Other``/``Sports``.
    """
    values = list(records)
    raw = [str(record.content) for record in values]
    if schema is None:
        return values, {
            "applied": False,
            "classes_before": len(set(raw)),
            "classes_after": len(set(raw)),
            "undeclared": {},
            "undeclared_by_split": {},
            "renamed_rows": 0,
        }
    undeclared_by_split: dict[str, dict[str, int]] = {}
    for record in values:
        if not schema.declared or str(record.content) in schema.declared:
            continue
        split = str(getattr(record, "split", ""))
        counts = undeclared_by_split.setdefault(split, {})
        counts[str(record.content)] = counts.get(str(record.content), 0) + 1
    undeclared_by_split = {
        split: dict(sorted(counts.items())) for split, counts in sorted(undeclared_by_split.items())
    }
    strict_names = {str(value) for value in strict_splits}
    strict_labels = {
        label: count
        for split, counts in undeclared_by_split.items()
        if split in strict_names
        for label, count in counts.items()
    }
    if strict and strict_labels:
        raise ValueError(
            f"The content schema {schema.version} was not built on these labels: {strict_labels}. "
            "A strict recipe refuses to grow its map on its training split; add the label to the "
            "schema or drop strict."
        )
    renamed = 0
    mapped: list[Any] = []
    for record in values:
        canonical = schema.canonical(record.content)
        renamed += int(canonical != str(record.content))
        mapped.append(replace(record, content=canonical))
    after = [str(record.content) for record in mapped]
    return mapped, {
        "applied": True,
        "version": int(schema.version),
        "classes_before": len(set(raw)),
        "classes_after": len(set(after)),
        "undeclared": dict(sorted(strict_labels.items())),
        "undeclared_by_split": undeclared_by_split,
        "undeclared_outside_the_strict_splits": {
            split: counts for split, counts in undeclared_by_split.items() if split not in strict_names
        },
        "renamed_rows": renamed,
        "identity_rows": len(values) - renamed,
    }


def identity_content_schema(*, declared: Sequence[str] = ()) -> ContentSchema:
    """The legacy map: nothing merged, every raw label is its own class."""
    return ContentSchema(
        version=0, merge={}, declared=tuple(str(label) for label in declared),
        note="legacy: raw action labels as they are (what the round-1 checkpoints trained on)",
    )


__all__ = [
    "CONTENT_SCHEMA_V1_MERGE",
    "apply_content_schema",
    "content_schema_from_config",
    "CONTENT_SCHEMA_V1_NOTE",
    "CONTENT_SCHEMA_VERSION",
    "ContentSchema",
    "default_content_schema",
    "identity_content_schema",
]
