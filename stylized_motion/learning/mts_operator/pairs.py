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
TARGET_SAMPLING_MODES = ("clip_uniform", "style_uniform")

#: The ten 100STYLE clip suffixes.  They are **actions** (``Flapping_FW`` is the
#: FW clip of the Flapping style), not performers: the dataset was recorded by one
#: actor, and the ten suffixes are what the audit must not turn into ten actors.
STYLE100_ACTIONS: tuple[str, ...] = (
    "BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3",
)
STYLE100_PACKAGE = "100style"
#: Dataset-level marker for the single 100STYLE actor, with its provenance.
STYLE100_ACTOR = "100style_actor_0"
STYLE100_ACTOR_SOURCE = "100style_single_actor_marker"


def style100_action_family(action: str) -> str:
    """``(BR, BW, FR, FW, ID, SR, SW, TR1, TR2, TR3)`` -> a family label.

    The three transition clips belong to one family, so a per-family count does
    not fragment into TR1/TR2/TR3.
    """
    text = str(action).strip().upper()
    return "TR" if text in {"TR1", "TR2", "TR3"} else text


def split_style100_action(name: str) -> tuple[str, str] | None:
    """``Flapping_FW`` -> ``("Flapping", "FW")`` when the suffix is a 100STYLE action."""
    head, separator, suffix = str(name).strip().rpartition("_")
    if separator and head and suffix.upper() in STYLE100_ACTIONS:
        return head, suffix.upper()
    return None


def is_style100_name(name: str) -> bool:
    """True for names that carry the 100STYLE package explicitly."""
    package, separator, tail = str(name).partition("/")
    return bool(separator and tail) and package.lower() == STYLE100_PACKAGE


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
    target_sampling: str = "clip_uniform",
    window_frames: int | None = None,
    dataset: str | None = None,
    label_provenance: dict[str, Any] | None = None,
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
    if target_sampling not in TARGET_SAMPLING_MODES:
        raise ValueError(
            f"Unknown target_sampling {target_sampling!r}; expected {list(TARGET_SAMPLING_MODES)}"
        )
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
    # The axes are computed before the warnings that read them: the earlier
    # version referenced ``performer_axis`` 30 lines before assigning it, so the
    # overlap warning raised UnboundLocalError instead of warning.
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
    if not performers_known:
        performer_axis["zero_shot_performer_supported"] = False
        performer_axis["note"] = (
            "no actor labels were available, so no performer split can be claimed"
        )
    style_axis = {
        "train_styles": sorted(split_styles["train"]),
        "styles_in_test_only": test_only_styles,
        "held_out_styles": sorted(held_out),
        "zero_shot_style_supported": bool(test_only_styles or held_out),
    }

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

    sampler = StylePairSampler(
        records,
        style_split=split,
        seed=seed,
        held_out_styles=sorted(held_out),
        window_frames=window_frames,
        target_sampling=target_sampling,
    )
    leakage = {"same_clip": 0, "same_take": 0, "pairs": 0, "verified": 0}
    for pair in sampler.sample(count=sample_pairs, mode="same_style", target_sampling=target_sampling):
        leakage["pairs"] += 1
        leakage["verified"] += 1 if pair.same_style and not pair.same_content else 0
        leakage["same_clip"] += int(pair.same_clip)
        leakage["same_take"] += int(pair.same_take)
    pair_report = sampler.pair_report(
        count=sample_pairs, mode="same_style", target_sampling=target_sampling
    )
    configured_styles = sorted({record.style for record in records})
    eligible_targets = sampler.eligible_targets(stage="train")
    eligible_styles = sorted({record.style for record in eligible_targets})
    sampled_styles = sorted(pair_report["pairs_per_style"])
    style_vocabulary = {
        # configured: what the run declared; eligible: what the stage could have
        # used; sampled: what the draws actually touched.  ``len(style_split)``
        # was previously reported as if it were the sampled vocabulary.
        "configured": configured_styles,
        "eligible": eligible_styles,
        "sampled": sampled_styles,
        "configured_count": len(configured_styles),
        "eligible_count": len(eligible_styles),
        "sampled_count": len(sampled_styles),
        "eligible_target_clips": len(eligible_targets),
        "excluded_by_window": int(pair_report["excluded_clips"].get("window_unavailable", 0)),
        "excluded_by_holdout": int(pair_report["excluded_clips"].get("heldout_style", 0)),
    }
    same_style_evidence = {
        "styles_with_evidence": sorted(
            style for style, contents in contents_per_style.items() if len(contents) > 1
        ),
        "styles_without_evidence": [
            {"style": style, "reason": "single_content_label"}
            for style, contents in sorted(contents_per_style.items())
            if len(contents) <= 1
        ],
    }

    styles_train = [
        style for style in split.train_styles if clips_per_style.get(style, 0) > 0
    ]
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
        "style_vocabulary": style_vocabulary,
        "same_style_evidence": same_style_evidence,
        "pair_report": pair_report,
        "label_provenance": dict(label_provenance or {}),
        "target_sampling": target_sampling,
        "window_frames": None if window_frames is None else int(window_frames),
        "dataset": None if dataset is None else str(dataset),
        "held_out_styles": sorted(held_out),
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
        window_frames: int | None = None,
        target_sampling: str = "clip_uniform",
    ) -> None:
        if not records:
            raise ValueError("StylePairSampler needs at least one clip record")
        if target_sampling not in TARGET_SAMPLING_MODES:
            raise ValueError(
                f"Unknown target_sampling {target_sampling!r}; expected {list(TARGET_SAMPLING_MODES)}"
            )
        if window_frames is not None and int(window_frames) <= 0:
            raise ValueError("window_frames must be positive when given")
        self.records = list(records)
        self.style_split = style_split or split_styles_by_performer(self.records, seed=seed)
        self.seed = int(seed)
        self.allow_same_take = bool(allow_same_take)
        #: Styles the operator must never train on (the unseen-style axis).
        self.held_out_styles = {str(style) for style in held_out_styles}
        #: When records carry a real split (an actor holdout), stage follows it.
        self.use_data_splits = bool(use_data_splits)
        #: Clips shorter than this cannot provide a training window and are never
        #: paired; ``None`` means the caller already filtered them.
        self.window_frames = None if window_frames is None else int(window_frames)
        self.target_sampling = str(target_sampling)
        #: Why a candidate was rejected, counted over the sampler's lifetime.
        self.rejections: Counter[str] = Counter()
        self.targets_skipped: Counter[str] = Counter()
        self.draw_attempts = 0
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

    def has_window(self, record: ClipRecord) -> bool:
        """Whether the clip can supply a training window at all."""
        return self.window_frames is None or int(record.frames) >= self.window_frames

    def _reject_reason(
        self,
        record: ClipRecord,
        target: ClipRecord,
        *,
        mode: str,
        split: str | None,
        allowed: set[str] | None,
        stage: str,
    ) -> str | None:
        """Why ``record`` may not be ``target``'s reference, or ``None``."""
        if not self.has_window(record):
            return "window_unavailable"
        if record.clip_id == target.clip_id:
            return "same_clip"
        if not self.allow_same_take and record.source_group >= 0 and record.source_group == target.source_group:
            return "same_take"
        if record.split != target.split:
            # A held-out actor's clip must never be a training reference, even
            # when the stage selects by style vocabulary rather than by split.
            return "cross_split"
        if allowed is not None and record.style not in allowed:
            return "style_not_in_stage"
        if stage == "train" and record.style in self.held_out_styles:
            # The held-out filter used to apply only when the stage resolved to a
            # data split, so a style held out for the unseen-style axis was still
            # sampled in the vocabulary-only path.
            return "heldout_style"
        # The evidence each mode is supposed to carry.
        if mode == "same_style":
            if record.style != target.style:
                return "not_same_style"
            if record.content == target.content:
                return "same_content"
            return None
        if mode == "same_content":
            if record.content != target.content:
                return "not_same_content"
            if record.style == target.style:
                return "same_style"
            return None
        if record.style == target.style:
            return "not_different_style"
        return None

    def _accepts(
        self,
        record: ClipRecord,
        target: ClipRecord,
        *,
        mode: str,
        split: str | None,
        allowed: set[str] | None,
        stage: str = "train",
    ) -> bool:
        return (
            self._reject_reason(
                record, target, mode=mode, split=split, allowed=allowed, stage=stage
            )
            is None
        )

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
            self.rejections["empty_bucket"] += 1
            return None
        for _ in range(int(attempts)):
            self.draw_attempts += 1
            record = bucket[int(rng.integers(len(bucket)))]
            reason = self._reject_reason(
                record, target, mode=mode, split=split, allowed=allowed, stage=stage
            )
            if reason is None:
                return record
            self.rejections[reason] += 1
        self.rejections["attempts_exhausted"] += 1
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
            if self._reject_reason(
                record, target, mode=mode, split=split, allowed=allowed, stage=stage
            )
            is None
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

    def _target_reason(self, target: ClipRecord, *, stage: str) -> str | None:
        """Why ``target`` cannot be used as a target in this stage, or ``None``."""
        if not self.has_window(target):
            return "window_unavailable"
        split = self._stage_split(stage)
        if split is not None and target.split != split:
            return "cross_split"
        if stage == "train" and target.style in self.held_out_styles:
            return "heldout_style"
        if split is None and stage != "all":
            allowed = set(self.styles_for_stage(stage))
            if target.style not in allowed:
                return "style_not_in_stage"
        return None

    def eligible_targets(
        self, *, stage: str = "train", targets: Sequence[ClipRecord] | None = None
    ) -> list[ClipRecord]:
        """Targets the stage may use at all (window, split, holdout filters).

        Exposed so an audit can report the *eligible* vocabulary instead of
        quoting the configured one as if it had been sampled.
        """
        pool = list(self.records) if targets is None else list(targets)
        return [record for record in pool if self._target_reason(record, stage=stage) is None]

    def sample(
        self,
        *,
        count: int,
        mode: str = "same_style",
        stage: str = "train",
        targets: Sequence[ClipRecord] | None = None,
        generator: np.random.Generator | None = None,
        target_sampling: str | None = None,
    ) -> list[StylePair]:
        """Draws ``count`` pairs, preferring targets that have references.

        ``target_sampling`` is ``clip_uniform`` (every eligible clip is equally
        likely to be a target) or ``style_uniform`` (every eligible style is
        equally likely, then a clip of that style) — the latter is what keeps a
        majority style from dominating the objective.  Explicit ``targets`` are
        validated, not trusted: a target from another split or a held-out style is
        rejected with a reason instead of being paired.

        ``generator`` is a ``numpy.random.Generator``; the pairing logic is index
        arithmetic over clip records, not tensor sampling.
        """
        if generator is not None and not isinstance(generator, np.random.Generator):
            raise TypeError(
                "StylePairSampler needs a numpy.random.Generator, got "
                f"{type(generator).__name__}"
            )
        strategy = str(target_sampling or self.target_sampling)
        if strategy not in TARGET_SAMPLING_MODES:
            raise ValueError(
                f"Unknown target_sampling {strategy!r}; expected {list(TARGET_SAMPLING_MODES)}"
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
        eligible: list[ClipRecord] = []
        for record in pool:
            reason = self._target_reason(record, stage=stage)
            if reason is None:
                eligible.append(record)
            else:
                self.targets_skipped[reason] += 1
        if not eligible:
            return []
        order = self._target_order(eligible, strategy=strategy, rng=rng)
        pairs: list[StylePair] = []
        for target in order:
            drawn = self.pairs_for(target, mode=mode, count=1, stage=stage, generator=rng)
            if drawn:
                pairs.append(drawn[0])
            if len(pairs) >= int(count):
                break
        return pairs

    def _target_order(
        self, eligible: Sequence[ClipRecord], *, strategy: str, rng: np.random.Generator
    ) -> list[ClipRecord]:
        """The order targets are tried in, drawn without duplicating clips.

        ``style_uniform`` draws a style first and then a clip of that style, so a
        style with 10% of the clips is not 10x less likely to be trained on.  Rare
        clips are never copied to balance anything: the pool is the real catalogue.
        """
        if strategy == "clip_uniform":
            order = rng.permutation(len(eligible))
            return [eligible[int(index)] for index in order]
        by_style: dict[str, list[ClipRecord]] = defaultdict(list)
        for record in eligible:
            by_style[record.style].append(record)
        styles = sorted(by_style)
        style_order = [styles[int(index)] for index in rng.permutation(len(styles))]
        queues = {
            style: [by_style[style][int(position)] for position in rng.permutation(len(by_style[style]))]
            for style in styles
        }
        # Round robin: every round offers each style its next clip, so a style with
        # 100k clips does not take the whole draw.  Exhausting one style before
        # moving on looked like a style-uniform schedule but sampled a single style
        # for as long as that style had clips left (on SEED: 256/256 pairs).
        clips: list[ClipRecord] = []
        remaining = [style for style in style_order]
        while remaining:
            for style in list(remaining):
                queue = queues[style]
                if not queue:
                    remaining.remove(style)
                    continue
                clips.append(queue.pop())
        return clips

    def pair_report(
        self,
        *,
        count: int,
        mode: str = "same_style",
        stage: str = "train",
        target_sampling: str | None = None,
        targets: Sequence[ClipRecord] | None = None,
    ) -> dict[str, Any]:
        """Draws ``count`` pairs and reports what was used and what was rejected."""
        before = Counter(self.rejections)
        skipped_before = Counter(self.targets_skipped)
        pairs = self.sample(
            count=count,
            mode=mode,
            stage=stage,
            targets=targets,
            target_sampling=target_sampling,
        )
        per_style: Counter[str] = Counter(pair.target.style for pair in pairs)
        per_action: Counter[str] = Counter(pair.target.content for pair in pairs)
        per_action_family: Counter[str] = Counter(
            style100_action_family(pair.target.content) for pair in pairs
        )
        per_actor: Counter[str] = Counter(
            pair.target.performer or "unknown" for pair in pairs
        )
        return {
            "mode": mode,
            "stage": stage,
            "target_sampling": str(target_sampling or self.target_sampling),
            "pairs": len(pairs),
            "pairs_per_style": {style: int(per_style[style]) for style in sorted(per_style)},
            "pairs_per_action": {action: int(per_action[action]) for action in sorted(per_action)},
            "pairs_per_action_family": {
                family: int(per_action_family[family]) for family in sorted(per_action_family)
            },
            "pairs_per_actor": {actor: int(per_actor[actor]) for actor in sorted(per_actor)},
            "rejections": {
                reason: int(self.rejections[reason] - before[reason])
                for reason in sorted(self.rejections)
            },
            "targets_skipped": {
                reason: int(self.targets_skipped[reason] - skipped_before[reason])
                for reason in sorted(self.targets_skipped)
            },
            "excluded_clips": {
                "window_unavailable": sum(
                    1 for record in self.records if not self.has_window(record)
                ),
                "heldout_style": sum(
                    1 for record in self.records if record.style in self.held_out_styles
                ),
            },
            "unique_targets": len({pair.target.clip_id for pair in pairs}),
            "unique_references": len({pair.reference.clip_id for pair in pairs}),
        }

    def write_audit(
        self,
        path: str | Path,
        *,
        sample_pairs: int = 512,
        dataset: str | None = None,
        label_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        audit = build_pair_audit(
            self.records,
            style_split=self.style_split,
            sample_pairs=sample_pairs,
            seed=self.seed,
            held_out_styles=sorted(self.held_out_styles),
            target_sampling=self.target_sampling,
            window_frames=self.window_frames,
            dataset=dataset,
            label_provenance=label_provenance,
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


def labels_from_name(name: str, *, dataset: str | None = None) -> tuple[str, str, str]:
    """``(style, content, performer)`` derived from one clip name.

    The 100STYLE rule applies **only** to an explicitly 100STYLE source (the
    ``100style/`` package prefix, or ``dataset="100style"``): the suffix is the
    action and the performer is the dataset-level single-actor marker.

    * ``100style/Flapping_FW`` -> ``("Flapping", "FW", "100style_actor_0")``;
    * ``lafan/aiming1_subject1`` -> ``("lafan", "lafan/aiming1_subject1", "subject1")``;
    * ``Aeroplane_BR`` with no dataset -> ``("Aeroplane", "Aeroplane_BR", "")``:
      the suffix still groups the style, but it is never claimed as a performer
      (``BR``/``FW``/... are actions, and an unknown dataset must not borrow
      100STYLE's conventions).

    ``content`` is the clip identity, so two clips of the same style count as
    different content — the same-style / different-content evidence the operator
    needs.  Style labels the dataset provides itself always override this.
    """
    text = str(name)
    package, separator, tail = text.partition("/")
    style100 = str(dataset or "").lower() == STYLE100_PACKAGE or (
        bool(separator and tail) and package.lower() == STYLE100_PACKAGE
    )
    if style100:
        body = tail if separator and tail else text
        action = split_style100_action(body)
        if action is not None:
            style, action_name = action
            return style, action_name, STYLE100_ACTOR
        # Not a 100STYLE clip name: fall through to the generic reading instead of
        # flattening the label, so a mixed store keeps its packages and explicit
        # ``..._subjectN`` labels intact.
    if separator and tail:
        head, underscore, suffix = tail.rpartition("_")
        if underscore and head and suffix.lower().startswith("subject"):
            return package, text, suffix
        return package, text, ""
    style, _suffix = split_style_name(text)
    return style, text, ""


def clip_label_from_tables(
    store: Any, row: int, *, dataset: str | None = None
) -> dict[str, str]:
    """``(style, content, performer)`` of one range/clip row, table first.

    The single-row version of :func:`clip_records_from_store`, with the same
    conventions: explicit store columns win over name parsing, the 100STYLE suffix
    rule applies only to an explicitly 100STYLE source, and ``performer`` is empty
    ("unknown") rather than invented when the store has no actor information.  The
    data loaders attach this to a batch so a conditioned model reads the same
    labels the vocabulary was built from.
    """
    row = int(row)
    if hasattr(store, "clip_label"):
        label = store.clip_label(row)
        return {
            "style": str(label.get("style") or f"style_{label.get('source_id')}"),
            "action": str(
                label.get("action") or label.get("package") or f"content_{label.get('source_id')}"
            ),
            "performer": _performer_from_store(store, row, int(label.get("source_group", -1))),
        }
    style_names = tuple(getattr(store, "style_names", ()))
    action_names = tuple(getattr(store, "action_names", ()))
    style100 = str(dataset or "").lower() == STYLE100_PACKAGE
    style, content, performer = labels_from_name(store.range_names[row], dataset=dataset)
    if style_names:
        style_id = int(np.asarray(store.style_ids)[row])
        if 0 <= style_id < len(style_names):
            style = str(style_names[style_id])
            if style100:
                # Explicit style column, with a 100STYLE suffix still split off as
                # the action; no performer is read out of the style label.
                action = split_style100_action(style)
                if action is not None:
                    style, action_name = action
                    performer = performer or STYLE100_ACTOR
                    if not action_names:
                        content = action_name
    if action_names:
        action_id = int(np.asarray(store.action_ids)[row])
        if 0 <= action_id < len(action_names):
            content = str(action_names[action_id])
    return {"style": style, "action": content, "performer": performer}


def clip_records_from_store(store: Any, *, dataset: str | None = None) -> list[ClipRecord]:
    """Builds clip records from a v3 feature store or a v4 packed store.

    Splits come from the store's own table (``split_ids`` / ``clip_split``), so a
    pair audit can never disagree with the data pipeline about what is held out.

    Explicit store metadata wins over name parsing: the style/action columns and
    the actor table are used as they are.  The name fallback only reports what it
    can support — a style group, plus a performer when the name really carries an
    actor label (``..._subjectN``) — and it never re-parses a style label to
    invent an actor, which is how the old code turned ``Flapping_FW`` into an
    "actor FW".  Pass ``dataset="100style"`` to opt into the 100STYLE suffix rule.
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
        labels = clip_label_from_tables(store, row, dataset=dataset)
        style, content, performer = labels["style"], labels["action"], labels["performer"]
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
    # An actor column, not a group id: ``source_group_names`` is deliberately not
    # consulted, because a group is a take identity, and calling it a performer
    # would make the overlap analysis look measured when it only reported groups.
    for attribute in ("source_actor_names", "actor_names"):
        labels = getattr(store, attribute, None)
        if not labels:
            continue
        if 0 <= int(row) < len(labels):
            return str(labels[int(row)])
    return ""


__all__ = [
    "PAIR_MODES",
    "SPLIT_NAMES",
    "STYLE100_ACTIONS",
    "STYLE100_ACTOR",
    "STYLE100_ACTOR_SOURCE",
    "TARGET_SAMPLING_MODES",
    "ClipRecord",
    "StylePair",
    "StylePairSampler",
    "StyleSplit",
    "build_pair_audit",
    "clip_label_from_tables",
    "clip_records_from_store",
    "is_style100_name",
    "labels_from_name",
    "split_style100_action",
    "split_style_name",
    "style100_action_family",
    "split_styles_by_performer",
]
