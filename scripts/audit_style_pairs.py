#!/usr/bin/env python
"""Audit style pairs before any operator training (MTS-FSQ plan, Phase 1).

    python scripts/audit_style_pairs.py \
      --feature-database data/processed/100style_pruned_90/fsq_window_index \
      --output outputs/mts_pairs/audit \
      --sample-pairs 512

Writes ``style_pair_audit.json``.  If the audit cannot find same-style /
different-content evidence, that is a blocking finding for the operator stages,
not something to work around in the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    TARGET_SAMPLING_MODES,
    StylePairSampler,
    clip_records_from_store,
    split_styles_by_performer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit style pairs and splits for MTS operator training.")
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Directory for style_pair_audit.json.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--unseen-fraction", type=float, default=0.2)
    parser.add_argument("--sample-pairs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-clips", type=int, default=0, help="0 = every clip.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Source dataset name; '100style' enables the suffix-is-an-action rule.",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=0,
        help="Training window length; clips shorter than this are excluded (0 = no filter).",
    )
    parser.add_argument(
        "--held-out-styles",
        nargs="*",
        default=[],
        help="Styles excluded from operator training (the unseen-style axis).",
    )
    parser.add_argument(
        "--target-sampling",
        choices=list(TARGET_SAMPLING_MODES),
        default="style_uniform",
        help="How target clips are drawn once a style is chosen.",
    )
    return parser


def audit_actor_source(records) -> str:
    """Where the performer labels came from, for the audit's provenance block."""
    performers = {record.performer for record in records if record.performer}
    if not performers:
        return "unknown"
    if len(performers) == 1 and "100style_actor" in next(iter(performers)):
        return "100style_marker"
    return "store_or_filename"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = open_any_feature_store(args.feature_database)
    try:
        records = clip_records_from_store(store, dataset=args.dataset)
    finally:
        store.close()
    if args.max_clips > 0:
        records = records[: args.max_clips]
    if not records:
        raise ValueError("The store produced no clip records to audit")
    if len({record.style for record in records}) < 3:
        raise ValueError(
            "Style pair auditing needs at least three distinct styles; "
            f"found {sorted({record.style for record in records})}"
        )
    split = split_styles_by_performer(
        records,
        val_fraction=float(args.val_fraction),
        unseen_fraction=float(args.unseen_fraction),
        seed=int(args.seed),
    )
    sampler = StylePairSampler(
        records,
        style_split=split,
        seed=int(args.seed),
        held_out_styles=list(args.held_out_styles),
        window_frames=int(args.window_frames) or None,
        target_sampling=str(args.target_sampling),
    )
    output_dir = args.output or Path("outputs/mts_pairs/audit")
    audit = sampler.write_audit(
        output_dir / "style_pair_audit.json",
        sample_pairs=int(args.sample_pairs),
        dataset=args.dataset,
        label_provenance={
            "dataset": args.dataset,
            "actor_source": audit_actor_source(records),
            "note": (
                "style/action/actor came from the store's own tables when present; "
                "otherwise the name fallback only reports what it can support"
            ),
        },
    )
    summary = {
        "clips": audit["clips"],
        "style_groups": audit["style_groups"],
        "same_clip_leakage": audit["same_clip_leakage"],
        "train_styles": len(audit["train_styles"]),
        "val_styles": len(audit["val_styles"]),
        "test_unseen_styles": len(audit["test_unseen_styles"]),
        "style_vocabulary": {
            key: audit["style_vocabulary"][key]
            for key in ("configured_count", "eligible_count", "sampled_count")
        },
        "pair_report": audit["pair_report"],
        "performer_analysis": audit["performer_analysis"],
        "warnings": audit["warnings"],
        "median_content_diversity": sorted(audit["content_diversity"].values())[
            len(audit["content_diversity"]) // 2
        ]
        if audit["content_diversity"]
        else 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir / 'style_pair_audit.json'}")


if __name__ == "__main__":
    main()
