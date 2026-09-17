"""Style pair audit and split-safe sampling.

The operator can only learn "style" if the data actually contains same-style /
different-content evidence and the reference/target pairs cannot leak.  This
module makes both auditable before any operator training (plan Phase 1):

* :func:`build_pair_audit` reports how many styles exist, how much content each
  style covers, where performers overlap between splits, and how many pairs
  would leak;
* :class:`StylePairSampler` draws references under explicit rules — same style or
  not, different content, never the same clip or the same take, and never a style
  held out for zero-shot evaluation.

The style split and the content split are separate objects on purpose: a model
may be trained on broad motion (content split) while its style vocabulary is
restricted to ``train_styles``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_NAMES = ("train", "val", "test")
PAIR_MODES = ("same_style", "different_style", "same_content")


@dataclass(frozen=True)
class ClipRecord:
    """Everything the audit needs to know about one logical clip."""

    clip_id: int
    style: str
    content: str
    performer: str = ""
    source_group: int = -1
    split: str = "train"
    variant: int = 0
    frames: int = 0

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {self.split!r}; expected {list(SPLIT_NAMES)}")


@dataclass(frozen=True)
class StylePair:
    """One reference/target pair plus the evidence it carries."""

    target: ClipRecord
    reference: ClipRecord
    mode: str

    @property
    def same_style(self) -> bool:
        return self.target.style == self.reference.style

    @property
    def same_content(self) -> bool:
        return self.target.content == self.reference.content

    @property
    def same_clip(self) -> bool:
        return self.target.clip_id == self.reference.clip_id

    @property
    def same_take(self) -> bool:
        return (
            self.target.source_group >= 0
            and self.target.source_group == self.reference.source_group
        )

    def leaks(self) -> bool:
        """A pair is unusable when it can be identical or merely cropped."""
        return self.same_clip or self.same_take

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_clip_id": self.target.clip_id,
            "reference_clip_id": self.reference.clip_id,
            "target_style": self.target.style,
            "reference_style": self.reference.style,
            "target_content": self.target.content,
            "reference_content": self.reference.content,
            "same_style": self.same_style,
            "same_content": self.same_content,
            "same_take": self.same_take,
        }


@dataclass
class StyleSplit:
    """Style vocabulary per stage; zero-shot styles never enter training."""

    train_styles: tuple[str, ...]
    val_styles: tuple[str, ...] = ()
    test_unseen_styles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap = set(self.train_styles) & set(self.val_styles)
        if overlap:
            raise ValueError(f"train and val styles must be disjoint, got {sorted(overlap)}")
        overlap = (set(self.train_styles) | set(self.val_styles)) & set(self.test_unseen_styles)
        if overlap:
            raise ValueError(
                f"zero-shot styles must not appear in train or val, got {sorted(overlap)}"
            )
        if not self.train_styles:
            raise ValueError("StyleSplit needs at least one training style")

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train_styles": list(self.train_styles),
            "val_styles": list(self.val_styles),
            "test_unseen_styles": list(self.test_unseen_styles),
        }

    @property
    def seen_styles(self) -> tuple[str, ...]:
        return tuple(self.train_styles) + tuple(self.val_styles)


def split_styles_by_performer(
    records: Sequence[ClipRecord],
    *,
    val_fraction: float = 0.2,
    unseen_fraction: float = 0.2,
    seed: int = 3407,
) -> StyleSplit:
    """Holds out whole styles, preferring ones whose performers are unseen.

    A style whose performer also appears in the training styles is not a
    zero-shot style: the model could recognise the performer instead of the
    style.  When the records carry no performer labels at all (some stores have
    no actor table) the split degrades to a style-only split and
    :func:`build_pair_audit` reports that the performer analysis was not
    possible — an honest "unknown" beats a fabricated overlap.
    """
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= unseen_fraction < 1.0:
        raise ValueError("val_fraction and unseen_fraction must be in [0, 1)")
    performers_of: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.performer:
            performers_of[record.style].add(record.performer)
    styles = sorted({record.style for record in records})
    if len(styles) < 3:
        raise ValueError(f"The audit needs at least three styles, found {len(styles)}")
    rng = np.random.default_rng(seed)
    shuffled = [str(style) for style in rng.permutation(styles)]
    unseen_count = max(1, int(round(unseen_fraction * len(styles))))
    unseen = shuffled[:unseen_count]
    remaining = shuffled[unseen_count:]
    unseen_performers = {performer for style in unseen for performer in performers_of[style]}
    # Validation prefers overlapping performers; the rest becomes training.
    overlapping = [style for style in remaining if performers_of[style] & unseen_performers]
    val_count = int(round(val_fraction * len(styles)))
    if overlapping:
        val = overlapping[:val_count]
    else:
        # No usable performer information: hold out whole styles directly.
        val = remaining[:val_count]
    train = [style for style in remaining if style not in val]
    if not train:
        raise ValueError("No styles left for training; lower val_fraction/unseen_fraction")
    return StyleSplit(
        train_styles=tuple(sorted(train)),
        val_styles=tuple(sorted(val)),
        test_unseen_styles=tuple(sorted(unseen)),
    )


def build_pair_audit(
    records: Sequence[ClipRecord],
    *,
    style_split: StyleSplit | None = None,
    sample_pairs: int = 512,
    seed: int = 3407,
    held_out_styles: Sequence[str] = (),
) -> dict[str, Any]:
    """The ``style_pair_audit.json`` payload of plan Phase 1.

    Reports the two generalization axes separately, because they are not the
    same experiment: *unseen performer* comes from the data split (a catalogue
    actor holdout freezes whole actors into test), while *unseen style* requires
    excluding a style from operator training outright.  A style that merely
    happens to be rare in the held-out actors is not an unseen style.
    """
    if not records:
        raise ValueError("Pair audit needs at least one clip record")
    split = style_split or split_styles_by_performer(records, seed=seed)
    clips_per_style: Counter[str] = Counter()
    contents_per_style: dict[str, set[str]] = defaultdict(set)
    performers_per_style: dict[str, set[str]] = defaultdict(set)
    frames_per_style: Counter[str] = Counter()
    for record in records:
        clips_per_style[record.style] += 1
        contents_per_style[record.style].add(record.content)
        if record.performer:
            # An empty label means "unknown"; treating it as a performer would
            # make the overlap analysis look measured when it is not.
            performers_per_style[record.style].add(record.performer)
        frames_per_style[record.style] += int(record.frames)

    performers_known = any(performers_per_style.get(style) for style in performers_per_style)
    performer_analysis = "overlap_reported" if performers_known else "unavailable_no_actor_table"
    train_performers = {
        performer
        for style in split.train_styles
        for performer in performers_per_style.get(style, set())
    }
    performer_overlap = (
        {
            style: sorted(performers_per_style.get(style, set()) & train_performers)
            for style in split.test_unseen_styles
        }
        if performers_known
        else {}
    )
    warnings: list[str] = []
    if not performers_known:
        warnings.append(
            "The store carries no actor table, so zero-shot styles cannot be checked "
            "for performer overlap; the style split is style-only."
        )
    total_clips = sum(clips_per_style.values())
    dominant_style, dominant_clips = clips_per_style.most_common(1)[0]
    dominant_share = dominant_clips / max(total_clips, 1)
    if dominant_share >= 0.5:
        warnings.append(
            f"Style {dominant_style!r} holds {dominant_share:.0%} of the clips; a single "
            "majority style dominates any style-conditional objective."
        )
    single_content_styles = sorted(
        style for style, contents in contents_per_style.items() if len(contents) <= 1
    )
    if single_content_styles:
        warnings.append(
            f"{len(single_content_styles)} of {len(contents_per_style)} styles have a single "
            f"content label, so same-style/different-content evidence is thin: {single_content_styles[:5]}"
        )
    if not split.val_styles:
        warnings.append("No validation styles were held out; model selection has no style split.")
    overlapping_unseen = {
        style: performers for style, performers in performer_overlap.items() if performers
    }
    if overlapping_unseen and performer_axis["zero_shot_performer_supported"] is False:
        # Only a warning when the *data* split is not actor-disjoint: if the store
        # already froze whole actors into test, a style-level partition that
        # overlaps performers is beside the point.
        warnings.append(
            "The style-level partition puts styles whose actors also appear in training "
            f"styles into the unseen set ({ {style: len(names) for style, names in overlapping_unseen.items()} }), "
            "and the data split is not actor-disjoint either: freeze whole actors into test at "
            "the catalogue level (seed-catalog --actor-holdout-ratio) before claiming zero-shot "
            "style transfer."
        )

    sampler = StylePairSampler(records, style_split=split, seed=seed)
    leakage = {"same_clip": 0, "same_take": 0, "pairs": 0, "verified": 0}
    for pair in sampler.sample(count=sample_pairs, mode="same_style"):
        leakage["pairs"] += 1
        leakage["verified"] += 1 if pair.same_style and not pair.same_content else 0
        leakage["same_clip"] += int(pair.same_clip)
        leakage["same_take"] += int(pair.same_take)

    styles_train = [
        style for style in split.train_styles if clips_per_style.get(style, 0) > 0
    ]
    split_styles: dict[str, set[str]] = defaultdict(set)
    split_actors: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_styles[record.split].add(record.style)
        if record.performer:
            split_actors[record.split].add(record.performer)
    held_out = {str(style) for style in held_out_styles}
    test_only_styles = sorted(split_styles["test"] - split_styles["train"] - split_styles["val"])
    performer_axis = {
        "train_actors": len(split_actors["train"]),
        "val_actors": len(split_actors["val"]),
        "test_actors": len(split_actors["test"]),
        "train_test_actor_overlap": len(split_actors["train"] & split_actors["test"]),
        "test_actor_examples": sorted(split_actors["test"])[:5],
        "zero_shot_performer_supported": len(split_actors["train"] & split_actors["test"]) == 0
        and len(split_actors["test"]) > 0,
    }
    style_axis = {
        "train_styles": sorted(split_styles["train"]),
        "styles_in_test_only": test_only_styles,
        "held_out_styles": sorted(held_out),
        "zero_shot_style_supported": bool(test_only_styles or held_out),
    }
    if not style_axis["zero_shot_style_supported"]:
        warnings.append(
            "No style is exclusive to the held-out actors, so this split supports the "
            "unseen-performer axis only: excluding styles from operator training is what "
            "creates an unseen-style axis (config data.pairs.held_out_styles)."
        )
    return {
        "clips": len(records),
        "style_groups": len(clips_per_style),
        "clips_per_style": {style: int(clips_per_style[style]) for style in sorted(clips_per_style)},
        "frames_per_style": {style: int(frames_per_style[style]) for style in sorted(frames_per_style)},
        "content_diversity": {
            style: len(contents) for style, contents in sorted(contents_per_style.items())
        },
        "content_entropy": {
            style: float(_entropy(sorted(contents))) for style, contents in sorted(contents_per_style.items())
        },
        "performer_analysis": performer_analysis,
        "performer_overlap": performer_overlap,
        "performer_axis": performer_axis,
        "style_axis": style_axis,
        "style_balance": {
            "dominant_style": dominant_style,
            "dominant_share": float(dominant_share),
            "style_entropy": float(
                _entropy([style for style, count in clips_per_style.items() for _ in range(count)])
            ),
        },
        "warnings": warnings,
        "same_clip_leakage": leakage["same_take"] + leakage["same_clip"],
        "pair_leakage": leakage,
        "train_styles": list(split.train_styles),
        "val_styles": list(split.val_styles),
        "test_unseen_styles": list(split.test_unseen_styles),
        "styles_with_same_style_evidence": styles_train,
        "styles_without_targets": sorted(
            style
            for style in split.train_styles
            if not any(record.style == style for record in records)
        ),
        "seed": int(seed),
        "notes": (
            "same_clip_leakage counts reference/target pairs that share a clip or a take; "
            "content_diversity counts distinct content labels per style."
        ),
    }


def _entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    probabilities = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    return float(-(probabilities * np.log(probabilities)).sum())


class StylePairSampler:
    """Draws reference clips under explicit, auditable rules."""

    def __init__(
        self,
        records: Sequence[ClipRecord],
        *,
        style_split: StyleSplit | None = None,
        seed: int = 3407,
        allow_same_take: bool = False,
        held_out_styles: Sequence[str] = (),
        use_data_splits: bool = True,
    ) -> None:
        if not records:
            raise ValueError("StylePairSampler needs at least one clip record")
        self.records = list(records)
        self.style_split = style_split or split_styles_by_performer(self.records, seed=seed)
        self.seed = int(seed)
        self.allow_same_take = bool(allow_same_take)
        #: Styles the operator must never train on (the unseen-style axis).
        self.held_out_styles = {str(style) for style in held_out_styles}
        #: When records carry a real split (an actor holdout), stage follows it.
        self.use_data_splits = bool(use_data_splits)
        self._by_style: dict[str, list[ClipRecord]] = defaultdict(list)
        for record in self.records:
            self._by_style[record.style].append(record)
        # Candidate lookup must not scan the whole catalogue: on SEED that is
        # 142k records per target, which dominated the operator's step time
        # (0.6 s/step at batch 8).  Index by (split, style) and by (split, content)
        # so the scan stays proportional to the candidate pool.
        self._by_split_style: dict[tuple[str, str], list[ClipRecord]] = defaultdict(list)
        self._by_split_content: dict[tuple[str, str], list[ClipRecord]] = defaultdict(list)
        self._by_style_all: dict[str, list[ClipRecord]] = defaultdict(list)
        self._by_content_all: dict[str, list[ClipRecord]] = defaultdict(list)
        self._records_by_split: dict[str, list[ClipRecord]] = defaultdict(list)
        for record in self.records:
            self._records_by_split[record.split].append(record)
            self._by_split_style[(record.split, record.style)].append(record)
            self._by_split_content[(record.split, record.content)].append(record)
            self._by_style_all[record.style].append(record)
            self._by_content_all[record.content].append(record)

    def styles_for_stage(self, stage: str) -> tuple[str, ...]:
        if stage == "train":
            return self.style_split.train_styles
        if stage == "val":
            return self.style_split.val_styles
        if stage == "test" or stage == "unseen":
            return self.style_split.test_unseen_styles
        raise ValueError(f"Unknown stage {stage!r}; expected train, val or test")

    def _stage_split(self, stage: str) -> str | None:
        if not self.use_data_splits:
            return None
        if stage in {"test", "unseen"}:
            return "test"
        if stage == "val":
            return "val"
        if stage == "train":
            return "train"
        return None

    def _bucket(
        self, target: ClipRecord, *, mode: str, split: str | None, allowed: set[str] | None
    ) -> list[ClipRecord]:
        """Records this mode may consider at all (before the leak rules)."""
        if mode == "same_style":
            return (
                self._by_split_style.get((split, target.style), [])
                if split is not None
                else self._by_style_all.get(target.style, [])
            )
        if mode == "same_content":
            return (
                self._by_split_content.get((split, target.content), [])
                if split is not None
                else self._by_content_all.get(target.content, [])
            )
        # different_style: every record of the split that is not the target's style.
        if split is not None:
            return [
                record
                for record in self._records_by_split.get(split, [])
                if record.style != target.style
            ]
        return self.records

    def _accepts(
        self,
        record: ClipRecord,
        target: ClipRecord,
        *,
        mode: str,
        split: str | None,
        allowed: set[str] | None,
    ) -> bool:
        if record.clip_id == target.clip_id:
            return False
        if not self.allow_same_take and record.source_group >= 0 and record.source_group == target.source_group:
            return False
        if allowed is not None and record.style not in allowed:
            return False
        if split == "train" and record.style in self.held_out_styles:
            return False
        # The evidence each mode is supposed to carry.
        if mode == "same_style":
            return record.style == target.style and record.content != target.content
        if mode == "same_content":
            return record.content == target.content and record.style != target.style
        return record.style != target.style

    def draw_candidate(
        self,
        target: ClipRecord,
        *,
        mode: str,
        stage: str = "train",
        rng: np.random.Generator,
        attempts: int = 64,
    ) -> ClipRecord | None:
        """One reference drawn by rejection sampling.

        Materializing the candidate list is O(bucket), and a majority style's
        bucket holds most of the catalogue (SEED: ~105k of 142k records), which
        cost ~7 minutes per epoch when every target scanned it.  Drawing with a
        bounded number of attempts keeps the cost independent of the bucket size.
        """
        split = self._stage_split(stage)
        allowed = set(self.styles_for_stage(stage)) if split is None and stage != "all" else None
        bucket = self._bucket(target, mode=mode, split=split, allowed=allowed)
        if not bucket:
            return None
        for _ in range(int(attempts)):
            record = bucket[int(rng.integers(len(bucket)))]
            if self._accepts(record, target, mode=mode, split=split, allowed=allowed):
                return record
        return None

    def candidates(
        self,
        target: ClipRecord,
        *,
        mode: str,
        stage: str = "train",
    ) -> list[ClipRecord]:
        """Reference candidates for one target, split-safe and leak-free.

        With ``use_data_splits`` the stage selects the record's own data split
        (so a held-out actor never appears as a training reference or target);
        otherwise the stage selects by style vocabulary, which is the behaviour
        for stores without an actor holdout.
        """
        if mode not in PAIR_MODES:
            raise ValueError(f"Unknown pair mode {mode!r}; expected {list(PAIR_MODES)}")
        split = self._stage_split(stage)
        allowed = set(self.styles_for_stage(stage)) if split is None and stage != "all" else None
        bucket = self._bucket(target, mode=mode, split=split, allowed=allowed)
        return [
            record
            for record in bucket
            if self._accepts(record, target, mode=mode, split=split, allowed=allowed)
        ]

    def pairs_for(
        self,
        target: ClipRecord,
        *,
        mode: str,
        count: int = 1,
        stage: str = "train",
        generator: np.random.Generator | None = None,
    ) -> list[StylePair]:
        if generator is not None and not isinstance(generator, np.random.Generator):
            raise TypeError(
                "StylePairSampler needs a numpy.random.Generator, got "
                f"{type(generator).__name__}"
            )
        rng = generator or np.random.default_rng(self.seed)
        pairs: list[StylePair] = []
        for _ in range(int(count)):
            reference = self.draw_candidate(target, mode=mode, stage=stage, rng=rng)
            if reference is None:
                break
            pairs.append(StylePair(target=target, reference=reference, mode=mode))
        return pairs

    def sample(
        self,
        *,
        count: int,
        mode: str = "same_style",
        stage: str = "train",
        targets: Sequence[ClipRecord] | None = None,
        generator: np.random.Generator | None = None,
    ) -> list[StylePair]:
        """Draws ``count`` pairs, preferring targets that have references.

        ``generator`` is a ``numpy.random.Generator``; the pairing logic is index
        arithmetic over clip records, not tensor sampling.
        """
        if generator is not None and not isinstance(generator, np.random.Generator):
            raise TypeError(
                "StylePairSampler needs a numpy.random.Generator, got "
                f"{type(generator).__name__}"
            )
        rng = generator or np.random.default_rng(self.seed)
        split = self._stage_split(stage)
        if targets is not None:
            pool = list(targets)
        elif split is not None:
            pool = self._records_by_split.get(split, [])
        else:
            allowed = set(self.styles_for_stage(stage))
            pool = [record for style in allowed for record in self._by_style_all.get(style, [])]
        if not pool:
            return []
        order = rng.permutation(len(pool))
        pairs: list[StylePair] = []
        for index in order:
            target = pool[int(index)]
            drawn = self.pairs_for(
                target, mode=mode, count=1, stage=stage, generator=rng
            )
            if drawn:
                pairs.append(drawn[0])
            if len(pairs) >= int(count):
                break
        return pairs

    def write_audit(self, path: str | Path, *, sample_pairs: int = 512) -> dict[str, Any]:
        audit = build_pair_audit(
            self.records, style_split=self.style_split, sample_pairs=sample_pairs, seed=self.seed
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return audit


#: Performer codes as they appear in 100STYLE clip names (``BR``, ``TR1``, ...).
_PERFORMER_CODE = re.compile(r"^[A-Z]{1,3}\d?$")


def split_style_name(name: str) -> tuple[str, str]:
    """``Flapping_TR1`` -> ``("Flapping", "TR1")`` for 100STYLE-style names."""
    head, separator, suffix = str(name).rpartition("_")
    if separator and head and _PERFORMER_CODE.match(suffix):
        return head, suffix
    return str(name), ""


def labels_from_name(name: str) -> tuple[str, str, str]:
    """``(style, content, performer)`` derived from one clip name.

    Covers the naming conventions in this repo:

    * ``100style/Flapping_TR1`` -> ``("Flapping", "100style/Flapping_TR1", "TR1")``;
    * ``lafan/aiming1_subject1`` -> ``("lafan", "lafan/aiming1_subject1", "subject1")``;
    * ``Aeroplane_BR`` -> ``("Aeroplane", "Aeroplane_BR", "BR")``.

    ``content`` is the clip identity, so two references of the same style with
    different performers count as different content — which is exactly the
    same-style / different-content evidence the operator needs.  Style labels
    that the dataset provides itself always override this derivation.
    """
    text = str(name)
    package, separator, tail = text.partition("/")
    if separator and tail:
        head, underscore, suffix = tail.rpartition("_")
        if underscore and head and suffix.startswith("subject"):
            return package, text, suffix
        style, performer = split_style_name(tail)
        if performer:
            return style, text, performer
        return package, text, ""
    style, performer = split_style_name(text)
    return style, text, performer


def clip_records_from_store(store: Any) -> list[ClipRecord]:
    """Builds clip records from a v3 feature store or a v4 packed store.

    Splits come from the store's own table (``split_ids`` / ``clip_split``), so a
    pair audit can never disagree with the data pipeline about what is held out.
    """
    records: list[ClipRecord] = []
    if hasattr(store, "clip_label"):
        for clip_idx in range(int(store.num_clips)):
            label = store.clip_label(clip_idx)
            split_value = label.get("split")
            split = (
                SPLIT_NAMES[int(split_value)]
                if isinstance(split_value, int) and 0 <= int(split_value) < len(SPLIT_NAMES)
                else str(split_value or "train")
            )
            style = str(label.get("style") or "")
            content = str(label.get("action") or label.get("package") or "")
            performer = _performer_from_store(store, clip_idx, int(label.get("source_group", -1)))
            records.append(
                ClipRecord(
                    clip_id=int(label["clip_id"]),
                    style=style or f"style_{label.get('source_id')}",
                    content=content or f"content_{label.get('source_id')}",
                    performer=performer,
                    source_group=int(label.get("source_group", -1)),
                    split=split,
                    variant=int(label.get("variant", 0)),
                    frames=int(store.clip_length[clip_idx]),
                )
            )
        return records
    style_names = tuple(getattr(store, "style_names", ()))
    action_names = tuple(getattr(store, "action_names", ()))
    for row, name in enumerate(tuple(store.range_names)):
        style, content, performer = labels_from_name(name)
        if style_names:
            style_id = int(np.asarray(store.style_ids)[row])
            if 0 <= style_id < len(style_names):
                style, named_performer = split_style_name(style_names[style_id])
                performer = performer or named_performer
        if action_names:
            action_id = int(np.asarray(store.action_ids)[row])
            if 0 <= action_id < len(action_names):
                content = str(action_names[action_id])
        split_id = int(np.asarray(store.split_ids)[row])
        start = int(np.asarray(store.range_starts)[row])
        stop = int(np.asarray(store.range_stops)[row])
        records.append(
            ClipRecord(
                clip_id=row,
                style=style,
                content=content,
                performer=performer,
                source_group=int(np.asarray(store.source_clip_ids)[row]),
                split=SPLIT_NAMES[split_id] if 0 <= split_id < len(SPLIT_NAMES) else "train",
                variant=int(bool(np.asarray(store.range_mirror)[row])),
                frames=stop - start,
            )
        )
    return records


def _performer_from_store(store: Any, row: int, group: int) -> str:
    """Actor label when the store exposes one, otherwise empty ("unknown").

    The packed store carries ``source_performer_names`` plus a per-clip id
    (``clip_performer_id``, written from the catalogue's take actor).  Older
    stores and datasets without actor metadata return "" and the audit reports
    the performer analysis as unavailable: inventing a performer from a group id
    would make the overlap analysis look measured while it only reported group
    identity.
    """
    names = getattr(store, "source_performer_names", None)
    ids = getattr(store, "clip_performer_id", None)
    if names and ids is not None:
        index = int(ids[int(row)])
        if 0 <= index < len(names):
            return str(names[index])
    for attribute in ("source_actor_names", "actor_names", "source_group_names"):
        labels = getattr(store, attribute, None)
        if not labels:
            continue
        index = int(group) if attribute == "source_group_names" else int(row)
        if 0 <= index < len(labels):
            return str(labels[index])
    return ""


__all__ = [
    "PAIR_MODES",
    "SPLIT_NAMES",
    "ClipRecord",
    "StylePair",
    "StylePairSampler",
    "StyleSplit",
    "build_pair_audit",
    "clip_records_from_store",
    "labels_from_name",
    "split_style_name",
    "split_styles_by_performer",
]
