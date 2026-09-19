#!/usr/bin/env python
"""N02: the content x style coverage table, the label baselines and the dev benchmark.

Three questions, one read-only pass over the store:

1. **Coverage** -- how many clips and takes every (canonical content, style) cell
   holds in each split.  A cell that is empty stays empty and is listed as
   unavailable with its reason; it is never filled by borrowing a label.
2. **How much style the content label gives away** -- the accuracy of predicting
   style from the content label alone (train-majority per content), under the raw
   labels and under the v1 canonical map.  The drop is the point of the merge: it
   is the size of the shortcut the round-1 arms could take.
3. **The dev benchmark** -- a small, balanced, take-grouped manifest with real
   positives and negatives for two subtasks: same-style cross-content reference
   consistency, and a real style edit at fixed source content.  Rows are locked
   (clips, starts, groups, seeds) so every later evaluation scores the same rows.

    python scripts/audit_mts_content_style.py \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --content-schema data/configs/mts_content_schema_v1.yaml \
      --output outputs/mts_next_round_20260918/N02/coverage.json \
      --manifest-output outputs/mts_next_round_20260918/N02/benchmark_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    split_table_identity,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.content_schema import (  # noqa: E402
    ContentSchema,
    identity_content_schema,
)
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.summary import write_summary  # noqa: E402
from stylized_motion.learning.mts_operator.windows import windows_by_clip  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")
#: The styles a *reference* arm can be trained to read.  They come from the pair
#: sampler's own eligibility rule (pairs.py), not from a list typed here.
ELIGIBLE_STYLE_SOURCE = "the style-pair sampler's train-eligible targets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Content x style coverage and dev benchmark (read-only).")
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument(
        "--content-schema",
        type=Path,
        default=REPO_ROOT / "data" / "configs" / "mts_content_schema_v1.yaml",
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407, help="Sampling seed for the dev benchmark.")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20260918,
        help="The benchmark's own evaluation seed (masks, draws); never a training seed.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "outputs/mts_revision2/operator_styleid_logit_s3407_20260918_1814/validation_protocol.json",
        help="A frozen protocol whose content x style table should be reported.",
    )
    parser.add_argument("--cases-per-cell", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    return parser


def clip_table(store: Any) -> dict[str, Any]:
    """The store's own clip columns, in one place."""
    return {
        "num_clips": int(store.num_clips),
        "style": [str(value) for value in store.source_style_names],
        "action": [str(value) for value in store.source_action_names],
        "clip_style_id": np.asarray(store.clip_style_id),
        "clip_action_id": np.asarray(store.clip_action_id),
        "clip_split": np.asarray(store.clip_split),
        "clip_source_group": np.asarray(store.clip_source_group),
        "clip_source_id": np.asarray(store.clip_source_id),
        "clip_mirror": np.asarray(store.clip_mirror),
        "clip_length": np.asarray(store.clip_length),
    }


def coverage(table: dict[str, Any], schema: ContentSchema) -> dict[str, Any]:
    """Clips and takes per (canonical content, style) cell, per split."""
    report: dict[str, Any] = {
        "schema": schema.as_dict(),
        "splits": {},
        "cells": {},
        "undeclared_labels": {},
    }
    for split_id, split_name in enumerate(SPLIT_NAMES):
        rows = np.flatnonzero(table["clip_split"] == split_id)
        cells: Counter = Counter()
        takes: defaultdict = defaultdict(set)
        for clip in rows.tolist():
            content = schema.canonical(table["action"][int(table["clip_action_id"][clip])])
            style = table["style"][int(table["clip_style_id"][clip])]
            cells[(content, style)] += 1
            takes[(content, style)].add(int(table["clip_source_group"][clip]))
        raw_labels = [
            table["action"][int(table["clip_action_id"][clip])] for clip in rows.tolist()
        ]
        report["splits"][split_name] = {
            "clips": int(len(rows)),
            "takes": int(len({int(table["clip_source_group"][clip]) for clip in rows.tolist()})),
            "styles": sorted({style for _content, style in cells}),
            "contents": sorted({content for content, _style in cells}),
            "undeclared_labels": schema.undeclared(raw_labels),
        }
        report["undeclared_labels"][split_name] = schema.undeclared(raw_labels)
        report["cells"][split_name] = [
            {
                "content": content,
                "style": style,
                "clips": int(count),
                "takes": int(len(takes[(content, style)])),
            }
            for (content, style), count in sorted(cells.items())
        ]
    return report


def style_from_content_baseline(
    table: dict[str, Any], schema: ContentSchema, *, styles: set[str] | None = None
) -> dict[str, Any]:
    """How well the content label alone predicts style (train-majority per content).

    This is the shortcut the round-1 protocol shipped: with the raw labels, the
    content column nearly tells the model whether the clip is neutral.  The same
    measurement under the canonical map is the evidence that the merge removes it.
    """
    def label_of(clip: int) -> tuple[str, str]:
        return (
            schema.canonical(table["action"][int(table["clip_action_id"][clip])]),
            table["style"][int(table["clip_style_id"][clip])],
        )

    train = np.flatnonzero(table["clip_split"] == 0).tolist()
    val = np.flatnonzero(table["clip_split"] == 1).tolist()
    majority: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for clip in train:
        content, style = label_of(clip)
        if styles is not None and style not in styles:
            continue
        majority[content][style] += 1
        global_counts[style] += 1
    if not majority:
        raise ValueError("The train split has no clips for the requested styles")
    best_per_content = {content: counts.most_common(1)[0][0] for content, counts in majority.items()}
    global_best = global_counts.most_common(1)[0][0]
    scored = [clip for clip in val if styles is None or label_of(clip)[1] in styles]
    correct = sum(1 for clip in scored if label_of(clip)[1] == best_per_content.get(label_of(clip)[0]))
    correct_global = sum(1 for clip in scored if label_of(clip)[1] == global_best)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for clip in scored:
        content, style = label_of(clip)
        confusion[best_per_content.get(content, "<unknown>")][style] += 1
    return {
        "rows": len(scored),
        "majority_style_per_content": dict(sorted(best_per_content.items())),
        "accuracy_from_content": None if not scored else correct / len(scored),
        "global_majority_style": global_best,
        "accuracy_from_the_global_majority": None if not scored else correct_global / len(scored),
        "chance_uniform_over_styles": None
        if not styles
        else 1.0 / len(styles),
        "predicted_vs_true": {
            predicted: dict(sorted(counts.items())) for predicted, counts in sorted(confusion.items())
        },
    }


def eligible_styles(store: Any, *, seed: int = 3407, frames: int = 64) -> dict[str, Any]:
    """The styles a reference arm can actually be *trained* on: the pairable ones.

    ``StyleSplit.train_styles`` is an intention; what the operator's style index
    ends up holding is the styles for which the sampler really produces a
    same-style pair at the training stage (``hurry to neutral`` is in the split but
    has no pairable partner, and the round-1 checkpoints hold three ids, not four).
    The distinction is measured here rather than assumed.
    """
    from stylized_motion.learning.mts_operator.pairs import StylePairSampler

    records = clip_records_from_store(store)
    style_split = split_styles_by_performer(records, seed=int(seed))
    sampler = StylePairSampler(
        records, style_split=style_split, seed=int(seed), window_frames=int(frames)
    )
    pairable = sorted(
        {
            str(target.style)
            for target in sampler.eligible_targets(stage="train")
            if sampler.pairs_for(target, mode="same_style", count=1, stage="train")
        }
    )
    return {
        "source": ELIGIBLE_STYLE_SOURCE,
        "eligible": pairable,
        "split_train_styles": sorted(str(style) for style in style_split.train_styles),
        "val_styles": sorted(str(style) for style in style_split.val_styles),
        "unseen_styles": sorted(str(style) for style in style_split.test_unseen_styles),
        "all_styles": sorted({str(record.style) for record in records}),
        "note": "'hurry', 'old' and the 'to neutral' transitions are separate labels; the "
        "zero-shot axis has no trained checkpoint and stays not_evaluated",
    }


def style_impurity(table: dict[str, Any], schema: ContentSchema) -> dict[str, Any]:
    """How much of the style the content label tells, per content class.

    ``majority_share`` is the accuracy a model gets by always predicting the
    content's most common style; ``non_neutral_share`` is what an arm trained to
    read "is this clip non-neutral?" could get from the label alone.  Both are
    reported per split so the round-1 shortcut is visible as a number, not a worry.
    """
    report: dict[str, Any] = {}
    for split_id, split_name in enumerate(SPLIT_NAMES):
        rows = np.flatnonzero(table["clip_split"] == split_id).tolist()
        counts: dict[str, Counter] = defaultdict(Counter)
        for clip in rows:
            content = schema.canonical(table["action"][int(table["clip_action_id"][clip])])
            style = table["style"][int(table["clip_style_id"][clip])]
            counts[content][style] += 1
        entries = {}
        for content, styles in sorted(counts.items()):
            total = sum(styles.values())
            majority_style, majority_count = styles.most_common(1)[0]
            non_neutral = sum(count for style, count in styles.items() if style != "neutral")
            entries[content] = {
                "clips": int(total),
                "styles": dict(sorted(styles.items())),
                "majority_style": majority_style,
                "majority_share": majority_count / total,
                "non_neutral_share": non_neutral / total,
                "style_entropy_bits": float(
                    -sum(
                        (count / total) * np.log2(count / total)
                        for count in styles.values()
                        if count > 0
                    )
                ),
            }
        report[split_name] = entries
    return report


def protocol_content_table(path: Path, schema: ContentSchema) -> dict[str, Any]:
    """The frozen protocol's own content x style table, raw and canonical.

    This is the table plan §5 is about: the 213 rows' content labels nearly
    determine their style, so a constant arm could read the style off the content.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["items"]
    raw: Counter = Counter()
    canonical: Counter = Counter()
    for item in items:
        content = str(item.get("content", ""))
        style = str(item.get("style", ""))
        raw[(content, style)] += 1
        canonical[(schema.canonical(content), style)] += 1

    def table_of(counts: Counter) -> dict[str, Any]:
        by_content: dict[str, Counter] = defaultdict(Counter)
        for (content, style), count in counts.items():
            by_content[content][style] += count
        entries = {}
        for content, styles in sorted(by_content.items()):
            total = sum(styles.values())
            non_neutral = sum(count for style, count in styles.items() if style != "neutral")
            entries[content] = {
                "rows": int(total),
                "styles": dict(sorted(styles.items())),
                "non_neutral_share": non_neutral / total,
            }
        return entries

    return {
        "protocol_id": payload.get("protocol_id"),
        "rows": len(items),
        "legacy_raw": table_of(raw),
        "canonical_v1": table_of(canonical),
        "note": "the canonical map cannot relabel these rows for the trained arms; it is what a "
        "retrained model would see",
    }


def _window_start(windows: dict[int, list[Any]], clip: int) -> int | None:
    entries = windows.get(int(clip))
    if not entries:
        return None
    return int(min(int(getattr(request, "target_start", 0)) for request in entries))


def build_benchmark(
    table: dict[str, Any],
    windows: dict[int, list[Any]],
    schema: ContentSchema,
    *,
    eligible: list[str],
    cases_per_cell: int,
    seed: int,
    eval_seed: int,
    primary_content: str = "Basic Locomotion",
) -> dict[str, Any]:
    """The locked dev benchmark: real positives, real negatives, take-isolated.

    Subtask ``same_style_cross_content``: the query and its reference share a
    style but not a content, so "the reference describes the style" is the only
    thing that can explain a consistent response.

    Subtask ``edit_to_target_style``: the source content is fixed and the target
    style changes, with a real reference clip of the target style at the same
    (canonical) content taken from a different take.

    Every row states its clips, starts, takes, raw labels and the legacy
    conditioning caveat.  A cell that cannot supply a legal row is listed as
    unavailable with the reason that stopped it -- never filled by re-labelling a
    clip or copying a take.
    """
    rng = np.random.default_rng(int(seed))
    val = np.flatnonzero(table["clip_split"] == 1).tolist()
    val = [clip for clip in val if _window_start(windows, clip) is not None]
    by_style_content: dict[tuple[str, str], list[int]] = defaultdict(list)
    for clip in val:
        style = table["style"][int(table["clip_style_id"][clip])]
        content = schema.canonical(table["action"][int(table["clip_action_id"][clip])])
        by_style_content[(style, content)].append(clip)
    for key in by_style_content:
        by_style_content[key].sort()

    def row(clip: int) -> dict[str, Any]:
        return {
            "clip": int(clip),
            "start": int(_window_start(windows, clip) or 0),
            "style": table["style"][int(table["clip_style_id"][clip])],
            "content_raw": table["action"][int(table["clip_action_id"][clip])],
            "content_canonical": schema.canonical(
                table["action"][int(table["clip_action_id"][clip])]
            ),
            "take": int(table["clip_source_group"][clip]),
            "source_id": int(table["clip_source_id"][clip]),
            "mirror": bool(table["clip_mirror"][clip]),
        }

    def legal(query: int, reference: int, *, same_style: bool, same_content: bool) -> bool:
        left, right = row(query), row(reference)
        if left["take"] == right["take"] or left["source_id"] == right["source_id"]:
            return False
        if bool(left["mirror"]) != bool(right["mirror"]):
            return False
        if same_style and left["style"] != right["style"]:
            return False
        if not same_style and left["style"] == right["style"]:
            return False
        if same_content and left["content_canonical"] != right["content_canonical"]:
            return False
        if not same_content and left["content_canonical"] == right["content_canonical"]:
            return False
        return True

    unavailable: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    declared = set(schema.declared)

    def conditionable(clip: int) -> bool:
        """The *query* carries the content condition, so its label must be trained."""
        return (not declared) or row(clip)["content_raw"] in declared

    # Subtask A: same style, different content.  The query is drawn from a declared
    # content (the arm has to be conditioned on it); the reference may come from an
    # undeclared one, because a reference is only ever tokens.
    for style in eligible:
        queries = [clip for clip in val if row(clip)["style"] == style and conditionable(clip)]
        rng.shuffle(queries)
        picked = 0
        for query in queries:
            if picked >= max(1, int(cases_per_cell)):
                break
            candidates = [
                clip
                for content_key in sorted(by_style_content)
                if content_key[0] == style
                for clip in by_style_content[content_key]
                if legal(query, clip, same_style=True, same_content=False)
            ]
            if not candidates:
                continue
            reference = int(candidates[int(rng.integers(len(candidates)))])
            rows.append(
                {
                    "row_id": len(rows),
                    "subtask": "same_style_cross_content",
                    "split": "val",
                    "query": row(query),
                    "reference": row(reference),
                    "source_style": style,
                    "target_style": style,
                    "transfer": False,
                    "legacy_conditioning": True,
                    "note": "the reference shares the style but not the canonical content",
                }
            )
            picked += 1
        if picked == 0:
            unavailable.append(
                {
                    "subtask": "same_style_cross_content",
                    "style": style,
                    "reason": "no val pair with the same style, a different canonical content, a "
                    "different take and a different source clip",
                }
            )

    # Subtask B: fixed source content, swap the target style, real target-style reference.
    sources = [
        clip
        for clip in val
        if schema.canonical(table["action"][int(table["clip_action_id"][clip])]) == primary_content
    ]
    for style in eligible:
        style_sources = [
            clip for clip in sources if table["style"][int(table["clip_style_id"][clip])] == style
        ]
        rng.shuffle(style_sources)
        picked = 0
        for source in style_sources:
            if picked >= max(1, int(cases_per_cell)):
                break
            targets = [other for other in eligible if other != style]
            made = 0
            for target in targets:
                candidates = [
                    clip
                    for clip in by_style_content.get((target, primary_content), [])
                    if legal(source, clip, same_style=False, same_content=True)
                ]
                if not candidates:
                    unavailable.append(
                        {
                            "subtask": "edit_to_target_style",
                            "content": primary_content,
                            "source_style": style,
                            "target_style": target,
                            "reason": "no val reference clip with the target style at this "
                            "canonical content on a different take and source",
                        }
                    )
                    continue
                reference = int(candidates[int(rng.integers(len(candidates)))])
                rows.append(
                    {
                        "row_id": len(rows),
                        "subtask": "edit_to_target_style",
                        "split": "val",
                        "query": row(source),
                        "reference": row(reference),
                        "source_style": style,
                        "target_style": target,
                        "transfer": True,
                        "legacy_conditioning": True,
                        "note": "source and reference share the canonical content; the style "
                        "changes, and the legacy raw condition is kept for the round-1 arms",
                    }
                )
                made += 1
            if made:
                picked += 1
        if picked == 0:
            unavailable.append(
                {
                    "subtask": "edit_to_target_style",
                    "content": primary_content,
                    "source_style": style,
                    "target_style": None,
                    "reason": "no val source clip with this style at the primary content",
                }
            )

    cells = []
    for (style, content), clips in sorted(by_style_content.items()):
        takes = len({int(table["clip_source_group"][clip]) for clip in clips})
        cells.append(
            {
                "style": style,
                "content": content,
                "val_clips": len(clips),
                "val_takes": takes,
                "eligible_style": style in set(eligible),
                "primary_content": content == primary_content,
            }
        )
    payload = {
        "kind": "mts_content_style_benchmark",
        "schema": schema.as_dict(),
        "primary_content": primary_content,
        "eligible_styles": list(eligible),
        "sampling_seed": int(seed),
        "eval_seed": int(eval_seed),
        "frames": int(64),
        "rows": rows,
        "unavailable": unavailable,
        "cells": cells,
        "notes": {
            "grouping": "a statistical group is (take, mirror); rows never pair two clips of one "
            "take, and the pairs sampler's leak rule is the same one",
            "legacy_conditioning": "the round-1 arms were trained on raw action ids; these rows keep "
            "the raw label so they can be scored under that conditioning, and the canonical content "
            "is what a retrained model would use",
            "test_split": "the test split is not sampled here: it is scored once, after the recipe "
            "is fixed",
        },
    }
    return payload


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = ContentSchema.read(args.content_schema)
    legacy = identity_content_schema(declared=schema.declared)
    store = open_any_token_store(args.token_store)
    try:
        table = clip_table(store)
        identity = {
            "store_path": str(args.token_store),
            **{key: str(value) for key, value in validate_store_binding(store, store_kind="token").items()},
            **{key: value for key, value in split_table_identity(store).items()},
        }
        styles = eligible_styles(store)
        report: dict[str, Any] = {
            "kind": "mts_content_style_audit",
            "schema": schema.as_dict(),
            "store_identity": identity,
            "eligible_styles": styles,
            "coverage": coverage(table, schema),
            "coverage_legacy": {
                "splits": coverage(table, legacy)["splits"],
            },
            "style_from_content": {
                "canonical_v1": style_from_content_baseline(table, schema),
                "legacy_raw": style_from_content_baseline(table, legacy),
                "canonical_v1_eligible_styles_only": style_from_content_baseline(
                    table, schema, styles=set(styles["eligible"])
                ),
                "legacy_raw_eligible_styles_only": style_from_content_baseline(
                    table, legacy, styles=set(styles["eligible"])
                ),
                "note": "accuracy of predicting style from the content label alone; the canonical "
                "map is the fix, and the difference is the size of the shortcut",
            },
            "style_impurity": {
                "canonical_v1": style_impurity(table, schema),
                "legacy_raw": style_impurity(table, legacy),
            },
        }
        if args.protocol and Path(args.protocol).exists():
            report["protocol_content_table"] = {
                "canonical_v1": protocol_content_table(args.protocol, schema),
                "legacy_raw": protocol_content_table(args.protocol, legacy),
            }
        windows = windows_by_clip(store, "val", frames=int(args.frames))
        benchmark = build_benchmark(
            table,
            windows,
            schema,
            eligible=list(styles["eligible"]),
            cases_per_cell=int(args.cases_per_cell),
            seed=int(args.seed),
            eval_seed=int(args.eval_seed),
        )
        benchmark["store_identity"] = identity
        benchmark["fingerprint_sha256"] = fingerprint(
            {key: value for key, value in benchmark.items() if key != "fingerprint_sha256"}
        )
        report["benchmark"] = {
            "rows": len(benchmark["rows"]),
            "by_subtask": dict(Counter(row["subtask"] for row in benchmark["rows"])),
            "unavailable": benchmark["unavailable"],
            "fingerprint_sha256": benchmark["fingerprint_sha256"],
        }
        write_summary(args.output, report)
        if args.manifest_output:
            write_summary(args.manifest_output, benchmark)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "benchmark": str(args.manifest_output),
                    "eligible_styles": styles["eligible"],
                    "style_from_content_accuracy": {
                        name: None
                        if entry["accuracy_from_content"] is None
                        else round(entry["accuracy_from_content"], 4)
                        for name, entry in report["style_from_content"].items()
                        if isinstance(entry, dict)
                    },
                    "benchmark_rows": len(benchmark["rows"]),
                    "benchmark_unavailable": len(benchmark["unavailable"]),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
