#!/usr/bin/env python
"""Evaluate a trained MTS operator (MTS-FSQ plan, Phase 5 / §8).

    python scripts/evaluate_mts_operator.py \
      --checkpoint outputs/mts_operator/birth_death/seed3407/best.pt \
      --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
      --feature-database data/processed/100style_pruned_90/fsq_window_index \
      --split test \
      --output outputs/mts_eval/main

Writes ``operator_metrics.json`` (per-batch rows plus aggregates),
``strength_curve.csv`` and ``physics.csv``.  The reference comparison is
correct / wrong-style / random, which is the Phase 4 exit check; style retrieval
needs no external classifier because the reference that explains the target best
is the model's own ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    checkpoint_style_index,
    file_sha256,
    load_operator_bundle,
    require_token_store_binding,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.eval_protocol import (  # noqa: E402
    build_eval_rows,
    read_eval_manifest,
    retrieval_report,
    write_eval_manifest,
)
from stylized_motion.learning.mts_operator import (  # noqa: E402
    LayoutAdapter,
    MaskGenerator,
    OperatorBatch,
)
from stylized_motion.learning.mts_operator.masking import MaskConfig  # noqa: E402
from stylized_motion.learning.mts_operator.metrics import (  # noqa: E402
    aggregate,
    comparison_physics,
    physics_metrics,
    reference_sensitivity,
    strength_response,
    support_locality,
    style_retrieval,
    token_likelihood_diagnostics,
    token_likelihood_per_sample,
    unavailable_metrics,
)
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    StylePairSampler,
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    ContentVocabulary,
    PairedBatchSource,
    TokenSource,
    windows_by_clip,
)
from stylized_motion.learning.nef_probe import (  # noqa: E402
    KinematicContext,
    reference_positions_for_fk,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402
from stylized_motion.learning.runner import choose_device, set_seed  # noqa: E402
from stylized_motion.learning.mts_operator.sampling import (  # noqa: E402
    CommonRandomNumbers,
    paired_comparison,
)


def _require_finite(value: Any) -> None:
    """NaN is not a number: a metric that was not computed is written as null."""
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(
            f"A metric is not finite ({value!r}); uncomputed metrics must be null, not NaN"
        )


def build_row_batch(
    model: Any,
    row: int,
    batch: OperatorBatch,
    candidates: Sequence[torch.Tensor],
    *,
    candidate_valid: Sequence[torch.Tensor] | None = None,
) -> list[float]:
    """Target-token NLL of one row under each candidate reference.

    The row is taken from the *complete* batch (visible/hard/valid/condition all
    intact) and only the reference is swapped, so retrieval cannot silently score
    a different protocol than the rest of the run.
    """
    if not candidates:
        return []
    import dataclasses

    from stylized_motion.learning.mts_operator.metrics import _nll, _row

    row_batch = _row(batch, int(row))
    # Only the reference changes: the row keeps its visible/hard/valid masks and its
    # condition, so retrieval scores the same protocol as every other metric.  (It
    # used to blank the visible mask, which stopped being possible once a missing
    # mask became an error rather than "everything is visible".)
    scores: list[float] = []
    for position, candidate in enumerate(candidates):
        reference = candidate if candidate.ndim == 3 else candidate.unsqueeze(0)
        payload: dict[str, Any] = {"reference_tokens": reference}
        if candidate_valid is not None:
            valid = candidate_valid[position]
            payload["reference_valid_mask"] = valid if valid.ndim == 2 else valid.unsqueeze(0)
        scores.append(_nll(model, dataclasses.replace(row_batch, **payload)))
    return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a reference-conditioned MTS operator.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Operator bundle. Required unless --build-manifest-only is used.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        default=None,
        help="Frozen tokenizer. Required unless --build-manifest-only is used.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Read a frozen evaluation manifest instead of building one.",
    ),
    parser.add_argument(
        "--build-manifest-only",
        action="store_true",
        help="Build and write the manifest from the data identity, then exit without a model.",
    ),
    parser.add_argument(
        "--style-label",
        default=None,
        help="Reference label for a style-ID model (mapped through the checkpoint's style_to_id).",
    ),
    parser.add_argument("--dataset", default=None, help="Source dataset name (e.g. 100style)."),
    parser.add_argument(
        "--content-kind",
        choices=["none", "action_id"],
        default=None,
        help="Override the action condition; must agree with the checkpoint's own kind.",
    ),
    parser.add_argument(
        "--transport-checkpoint",
        type=Path,
        default=None,
        help="Defaults to the path recorded in the operator checkpoint metadata.",
    )
    parser.add_argument("--feature-database", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--mask-kind", default="full_generation")
    parser.add_argument("--strengths", nargs="*", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Strength of the style edit used for the source/base/styled physics comparison.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Monotonic filling steps for the styled draw (1 = one-shot sampling).",
    )
    parser.add_argument(
        "--support",
        nargs="*",
        default=None,
        help="Region names for the edit support (several regions = a disjoint multi-region "
        "edit). Omitted or empty means the whole body, never an empty edit.",
    )
    parser.add_argument("--graph-radius", type=int, default=0)
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "STOP"),
        help="Temporal support: only frames in [START, STOP) are editable.",
    )
    parser.add_argument("--label", default="", help="Row label for the experiment matrix.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--held-out-styles",
        nargs="*",
        default=None,
        help="Styles the operator was never trained on (the unseen-style axis).",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for operator_metrics.json, eval_manifest.json and eval_rows.jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(int(args.seed), deterministic=False)
    device = choose_device(args.device)

    # The manifest describes the *protocol*, not a model: building it must not need
    # a checkpoint and must not load the tokenizer (C01b), so the branch comes before
    # any model or tokenizer load.
    if args.build_manifest_only:
        if args.token_store is None and args.feature_database is None:
            raise SystemExit(
                "--build-manifest-only needs --token-store or --feature-database to read the "
                "clip table and window metadata"
            )
        store = (
            open_any_token_store(args.token_store)
            if args.token_store
            else open_any_feature_store(args.feature_database)
        )
        records = clip_records_from_store(
            store, dataset=None if args.dataset is None else str(args.dataset)
        )
        token_source = TokenSource(
            store=store,
            windows_by_clip=windows_by_clip(store, args.split, frames=int(args.frames)),
            # No encoder is loaded here: the manifest only needs clip identity and
            # window geometry, never motion.
            tokenizer=None,
            feature_stats=None,
            adapter=None,
            frames=int(args.frames),
            history=0,
            rng=np.random.default_rng(int(args.seed)),
        )
        rows = build_eval_rows(
            records,
            token_source,
            split=args.split,
            samples=int(args.batches) * int(args.batch_size),
            batch_size=int(args.batch_size),
            frames=int(args.frames),
            seed=int(args.seed),
            mask_kinds=[args.mask_kind],
            region=",".join(args.support) if args.support else "",
            graph_radius=int(args.graph_radius),
            frame_range=tuple(args.frame_range) if args.frame_range else None,
        )
        output_dir = args.output or Path("outputs/mts_eval")
        path = write_eval_manifest(
            output_dir / "eval_manifest.json",
            rows,
            identity={
                "representation_id": getattr(store, "representation_id", None),
                "feature_schema_hash": getattr(store, "feature_schema_hash", None),
                "normalization_hash": getattr(store, "normalization_hash", None),
                "split_manifest_hash": getattr(store, "split_manifest_hash", None),
            },
            protocol={
                "dataset": args.dataset,
                "split": args.split,
                "frames": int(args.frames),
                "mask_kind": args.mask_kind,
            },
        )
        print(
            f"built {len(rows)} evaluation rows at {path} (no model was loaded); "
            f"unavailable roles: "
            f"{sorted({reason for row in rows for reason in row.reasons})}",
            flush=True,
        )
        store.close()
        return

    if args.tokenizer_checkpoint is None:
        raise SystemExit("--tokenizer-checkpoint is required to evaluate a model")
    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(
        args.tokenizer_checkpoint, torch.device("cpu")
    )
    tokenizer = tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    adapter = LayoutAdapter(tokenizer.token_layout(), num_levels=int(tokenizer.num_levels))
    tokenizer_metadata = tokenizer.representation_metadata()

    checkpoint, model = load_operator_bundle(
        args.checkpoint,
        adapter=adapter,
        tokenizer_identity=tokenizer_metadata,
        # The tokenizer *file*, not only its metadata block: the recorded SHA is
        # what separates two tokenizers that share a layout (F05).
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        device=device,
    )
    model_config = dict(checkpoint.get("metadata", {}).get("model_config") or {})
    metrics_meta = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    operator_kind = str(model_config.get("operator", {}).get("name", "unknown"))
    encoder_kind = str(model_config.get("style_encoder", {}).get("kind", "reference"))
    if args.transport_checkpoint is not None:
        recorded = checkpoint.get("provenance", {}).get("upstream_transport_sha256")
        actual = file_sha256(args.transport_checkpoint)
        if recorded is not None and str(recorded) != actual:
            raise ValueError(
                f"--transport-checkpoint {args.transport_checkpoint} has SHA-256 {actual}, but "
                f"this operator was trained against {recorded}; pass the file it was trained on "
                "or drop the flag (the bundle already carries the weights)"
            )
    transport = model.transport
    transport_dim = int(transport.dim)

    store_path = args.token_store
    feature_path = args.feature_database
    if store_path:
        store = open_any_token_store(store_path)
        store_kind = "token"
    elif feature_path:
        store = open_any_feature_store(feature_path)
        store_kind = "feature"
    else:
        raise ValueError("Pass --feature-database or --token-store to read evaluation windows")
    if store_kind == "token":
        require_token_store_binding(
            store, tokenizer_checkpoint=args.tokenizer_checkpoint, checkpoint=checkpoint,
            where="eval token store",
        )
    validate_store_binding(store, store_kind=store_kind)
    records = clip_records_from_store(
        store, dataset=None if args.dataset is None else str(args.dataset)
    )
    style_split = split_styles_by_performer(records, seed=int(args.seed))
    sampler = StylePairSampler(
        records, style_split=style_split, seed=int(args.seed), held_out_styles=tuple(args.held_out_styles or ())
    )
    windows = windows_by_clip(store, args.split, frames=int(args.frames))
    history = int(tokenizer.history_frames)
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    pair_rng = np.random.default_rng(int(args.seed))
    # One shared window reader (windows.py) instead of a script-local copy.
    token_source = TokenSource(
        store=store,
        windows_by_clip=windows,
        tokenizer=None if args.token_store else tokenizer,
        feature_stats=None if args.token_store else feature_stats,
        adapter=adapter,
        frames=int(args.frames),
        history=history,
        rng=np.random.default_rng(int(args.seed)),
    )
    # The action condition comes from the frozen transport's own vocabulary, or
    # from `--content-kind none`; it is never faked with a row index.
    # The checkpoint is the authority on the condition kind; --content-kind is an
    # override that must agree with it, not a per-invocation reminder to type.
    content_vocabulary = getattr(transport, "content_vocabulary", None)
    if content_vocabulary is None:
        content_vocabulary = ContentVocabulary(kind="none")
    if args.content_kind is not None and str(args.content_kind) != content_vocabulary.kind:
        raise ValueError(
            f"--content-kind {args.content_kind!r} disagrees with the checkpoint's "
            f"{content_vocabulary.kind!r}; drop the flag or fix the checkpoint"
        )
    print(f"content condition: {content_vocabulary.kind}", flush=True)
    mask_generator = MaskGenerator(MaskConfig(mixture={args.mask_kind: 1.0}))
    # FK needs a skeleton, not the mirror-averaged ref_pos; a mirrored clip
    # (clip_mirror, the `_M` variants) additionally needs the mirrored skeleton,
    # otherwise its torso folds over and every world-space number is wrong.
    def _kinematic_for(mirror: bool):
        """The FK context for one mirror group.

        The plain group falls back to the stored ``ref_pos`` when no bind
        contract applies (synthetic fixtures), exactly as before; the mirrored
        group needs the mirrored skeleton, and without one it is skipped rather
        than posed on the wrong offsets.
        """
        if not (feature_stats and "ref_pos" in feature_stats):
            return None
        reference = reference_positions_for_fk(feature_stats, mirror=mirror)
        if mirror and reference is None:
            return None
        return KinematicContext.from_feature_stats(
            feature_stats, reference_positions=reference
        ).to(device)

    kinematic = _kinematic_for(False)
    kinematic_mirrored = _kinematic_for(True)

    def window_is_mirrored(clip_id: int) -> bool:
        flags = getattr(store, "clip_mirror", None)
        if flags is None or not 0 <= int(clip_id) < len(flags):
            return False
        return bool(flags[int(clip_id)])

    def tokens_for(clip_id: int) -> torch.Tensor | None:
        sample = token_source.window(int(clip_id))
        return None if sample is None else sample.tokens

    # The manifest is the protocol: built once (or read from disk), then only read.
    if args.eval_manifest is not None:
        manifest_meta, eval_rows = read_eval_manifest(args.eval_manifest)
        manifest_meta = dict(manifest_meta)
        print(f"eval manifest: {manifest_meta['rows']} rows from {args.eval_manifest}", flush=True)
    else:
        eval_rows = build_eval_rows(
            records,
            token_source,
            split=args.split,
            samples=int(args.batches) * int(args.batch_size),
            batch_size=int(args.batch_size),
            frames=int(args.frames),
            seed=int(args.seed),
            mask_kinds=[args.mask_kind],
            region=",".join(args.support) if args.support else "",
            graph_radius=int(args.graph_radius),
            frame_range=tuple(args.frame_range) if args.frame_range else None,
            condition=content_vocabulary.kind,
        )
        manifest_path = args.output / "eval_manifest.json" if args.output else Path("outputs/mts_eval/eval_manifest.json")
        write_eval_manifest(
            manifest_path,
            eval_rows,
            identity={
                "representation_id": getattr(store, "representation_id", None),
                "feature_schema_hash": getattr(store, "feature_schema_hash", None),
                "normalization_hash": getattr(store, "normalization_hash", None),
                "split_manifest_hash": getattr(store, "split_manifest_hash", None),
                "checkpoint": str(args.checkpoint),
            },
            protocol={
                "dataset": args.dataset,
                "split": args.split,
                "frames": int(args.frames),
                "mask_kind": args.mask_kind,
                "region": list(args.support or []),
                "graph_radius": int(args.graph_radius),
                "frame_range": list(args.frame_range) if args.frame_range else None,
                "condition": content_vocabulary.kind,
                "held_out_styles": list(args.held_out_styles or ()),
            },
        )
        print(f"eval manifest: built {len(eval_rows)} rows at {manifest_path}", flush=True)
    if args.held_out_styles:
        exposure = (checkpoint.get("provenance") or {}).get("training_exposure") or {}
        recorded = exposure.get("held_out_styles")
        if recorded is None:
            raise ValueError(
                "--held-out-styles was given but the checkpoint records no training exposure, "
                "so the claim cannot be checked; it is not evidence on its own"
            )
        claimed = sorted(str(style) for style in args.held_out_styles)
        if claimed != sorted(str(style) for style in recorded):
            raise ValueError(
                f"--held-out-styles {claimed} disagrees with the checkpoint's recorded held-out "
                f"styles {sorted(str(style) for style in recorded)}"
            )
    style_label_id = None
    if args.style_label is None and encoder_kind == "style_id":
        # Say what is missing before the model raises a generic "needs style ids":
        # a style-ID checkpoint has no reference input, so the evaluation has to
        # name the style it is scoring.
        raise ValueError(
            "This checkpoint uses a style-ID encoder: pass --style-label <label> (one of "
            f"{sorted(checkpoint_style_index(checkpoint) or {})}) to state which style the "
            "evaluation conditions on; a style-ID model has no reference to read it from."
        )
    if args.style_label is not None:
        stored_style_index = checkpoint_style_index(checkpoint)
        if stored_style_index is None:
            raise ValueError(
                "--style-label needs a style-ID checkpoint with a stored style_to_id map; "
                "this checkpoint has none"
            )
        if str(args.style_label) not in stored_style_index:
            raise ValueError(
                f"Unknown style label {args.style_label!r}: the checkpoint was trained on "
                f"{sorted(stored_style_index)}; an untrained id must not stand in for an unseen style"
            )
        style_label_id = int(stored_style_index[str(args.style_label)])

    rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    strength_rows: list[dict[str, float]] = []
    physics_rows: list[dict[str, float]] = []
    retrieval_scores: list[list[float]] = []
    retrieval_positives: list[list[int]] = []
    batch_size = int(args.batch_size)
    for batch_index, start in enumerate(range(0, len(eval_rows), batch_size)):
        chunk = eval_rows[start : start + batch_size]
        if not chunk:
            continue
        targets = [token_source.read(token_source.request(row.target.clip_id, row.target.start)) for row in chunk]
        references = []
        for row in chunk:
            if row.correct is None:
                # A style-ID model needs no reference; a reference model cannot score
                # this row at all, and says so instead of inventing one.
                if style_label_id is not None:
                    references.append(token_source.read(token_source.request(row.target_clip, row.target_start)))
                    continue
                raise ValueError(
                    f"Row {row.sample_id} has no legal same-style reference "
                    f"({row.reasons.get('candidates')}); an evaluation cannot score it"
                )
            references.append(token_source.read(token_source.request(row.correct.clip_id, row.correct.start)))
        target_tokens = torch.stack(targets).to(device)
        mask_generator = MaskGenerator(MaskConfig(mixture={chunk[0].mask_kind: 1.0}))
        mask = mask_generator.sample_kind(
            chunk[0].mask_kind, len(chunk), int(target_tokens.shape[1]),
            adapter=adapter,
            generator=torch.Generator(device="cpu").manual_seed(int(chunk[0].mask_seed)),
            device=torch.device("cpu"),
        )
        content_condition = None
        if not content_vocabulary.unconditional:
            # Always the *target's* action: swapping the style label must not touch
            # the content condition (feeding the style id here silently conditioned
            # the transport on a different action, or on an id it never learned).
            content_condition = content_vocabulary.vector([row.target.action for row in chunk]).to(device)
        # The manifest states the edit region; it is applied here, so every metric
        # of this batch (likelihood, sensitivity, strength, locality, physics) is
        # measured under the same condition instead of one of them quietly using
        # the whole body.
        hard_mask = None
        regions = {row.region for row in chunk if row.region}
        if len(regions) > 1:
            raise ValueError(f"A batch mixes regions {sorted(regions)}; split the manifest")
        if regions:
            region = next(iter(regions))
            radius = int(chunk[0].graph_radius)
            frame_range = chunk[0].frame_range
            hard_mask = adapter.hard_mask(
                [name for name in str(region).split(",") if name],
                graph_radius=radius,
                frame_range=tuple(frame_range) if frame_range else None,
                length=int(target_tokens.shape[1]),
                device=device,
            ).to(device)
        visible = mask.visible_mask.to(device)
        if hard_mask is not None:
            visible = visible & ~hard_mask.unsqueeze(0).expand_as(visible)
        batch = OperatorBatch(
            target_tokens=target_tokens,
            reference_tokens=torch.stack(references).to(device),
            # Built on the compute device: a style-ID model is on CUDA in a
            # real run, and a CPU index tensor aborts the embedding there.
            style_ids=torch.full((len(chunk),), style_label_id, dtype=torch.long, device=device)
            if style_label_id is not None
            else None,
            visible_mask=visible,
            hard_mask=hard_mask,
            content_condition=content_condition,
            sample_metadata=[row.as_dict() for row in chunk],
            strength=1.0,
        )
        row_report: dict[str, Any] = {
            "batch": float(batch_index),
            "samples": int(len(chunk)),
            "mask_kind": chunk[0].mask_kind,
            "mask_seed": int(chunk[0].mask_seed),
            "sample_ids": [int(item.sample_id) for item in chunk],
        }
        # Every NLL/TV number is measured on the same complete batch, after the
        # condition is set, so the three references are comparable.
        wrong_tokens = [
            None
            if item.wrong is None
            else token_source.read(token_source.request(item.wrong.clip_id, item.wrong.start))
            for item in chunk
        ]
        random_tokens = [
            None
            if item.random is None
            else token_source.read(token_source.request(item.random.clip_id, item.random.start))
            for item in chunk
        ]
        if any(item is not None for item in wrong_tokens):
            row_report.update(
                reference_sensitivity(
                    model,
                    batch,
                    wrong_reference=None
                    if any(item is None for item in wrong_tokens)
                    else torch.stack(wrong_tokens).to(device),
                    random_reference=None
                    if any(item is None for item in random_tokens)
                    else torch.stack(random_tokens).to(device),
                )
            )
        else:
            row_report["wrong_reference"] = None
            row_report["wrong_reference_reason"] = "no_legal_different_style_clip"
        row_report.update(token_likelihood_diagnostics(model, batch))
        per_sample_likelihood = token_likelihood_per_sample(model, batch)
        if args.support:
            support = adapter.hard_mask(
                list(args.support), graph_radius=int(args.graph_radius), length=target_tokens.shape[1],
                frame_range=tuple(args.frame_range) if args.frame_range else None,
                device=device,
            )
            support_batch = OperatorBatch(
                target_tokens=target_tokens,
                reference_tokens=batch.reference_tokens,
                visible_mask=mask.visible_mask.to(device),
                hard_mask=support,
                content_condition=batch.content_condition,
                strength=1.0,
            )
            row_report.update(support_locality(model, support_batch))
        rows.append(row_report)

        for point in strength_response(model, batch, strengths=list(args.strengths)):
            strength_rows.append({"batch": float(batch_index), **point})

        if kinematic is not None:
            # Three motions, three comparisons, never averaged together:
            #   source -- the recorded target tokens,
            #   base   -- a draw from the *frozen base transport* (not strength=0,
            #             whose distribution is the identity for operator families
            #             with an anchor),
            #   styled -- a draw from the operator at the configured strength.
            # The base and styled draws share one CRN cell, so the pair is coupled
            # and a difference is attributable to the operator.
            crn = CommonRandomNumbers(seed=int(args.seed))
            styled_batch = OperatorBatch(
                target_tokens=target_tokens,
                reference_tokens=batch.reference_tokens,
                style_ids=batch.style_ids,
                visible_mask=mask.visible_mask.to(device),
                hard_mask=hard_mask,
                content_condition=batch.content_condition,
                strength=float(args.strength),
            )
            base_tokens = model.generate_edit(
                styled_batch, crn=crn, sample_id=int(batch_index), step_id=0, use_base=True
            )
            styled_tokens = model.generate_edit(
                styled_batch,
                crn=crn,
                sample_id=int(batch_index),
                step_id=0,
                use_base=False,
                steps=int(args.steps),
            )
            with torch.no_grad():
                source_motion = tokenizer.decode_indices(target_tokens)
                base_motion = tokenizer.decode_indices(base_tokens)
                styled_motion = tokenizer.decode_indices(styled_tokens)
            interval = (
                (int(args.frame_range[0]), int(args.frame_range[1]))
                if args.frame_range
                else (0, int(target_tokens.shape[1]))
            )
            # One FK context per mirror group: a batch may mix plain and mirrored
            # clips, and their reference skeletons differ.
            batch_flags = [window_is_mirrored(row.target.clip_id) for row in chunk]
            groups = sorted(set(batch_flags))
            for group_mirror in groups:
                indices = [index for index, flag in enumerate(batch_flags) if flag == group_mirror]
                context = kinematic_mirrored if group_mirror else kinematic
                if context is None:
                    continue
                selector = torch.tensor(indices, device=source_motion.device)
                comparison = comparison_physics(
                    source_motion=source_motion[selector],
                    base_motion=base_motion[selector],
                    styled_motion=styled_motion[selector],
                    kinematic=context,
                    edit_interval=interval,
                )
                for name, values in comparison.items():
                    physics_rows.append(
                        {
                            "batch": float(batch_index),
                            "samples": float(len(indices)),
                            "mirrored": bool(group_mirror),
                            "strength": float(args.strength),
                            "steps": int(args.steps),
                            **values,
                        }
                    )
            # The coupled base/styled pair also reports how many tokens the edit
            # actually moved, which the decoded physics cannot show by itself.
            paired = paired_comparison(
                model(styled_batch).base_probabilities,
                model(styled_batch).probabilities,
                crn=crn,
                hard_mask=hard_mask,
                sample_id=int(batch_index),
            )
            row_report["base_vs_styled_sampled"] = paired["changed_token_ratio"]
            row_report["base_vs_styled_tv"] = paired["total_variation"]
            row_report["edited_token_count"] = int((styled_tokens != base_tokens).sum())

        # Multi-positive retrieval: every same-style candidate is a correct answer.
        # A style-ID model has no reference encoder, so it simply does not enter this
        # loop instead of running it and relabelling the result afterwards.
        for row_index, item in enumerate(chunk):
            retrieval_scores.append([])
            retrieval_positives.append([int(index) for index in item.positive_indices])
            retrieval_rank: int | None = None
            if encoder_kind == "reference":
                candidates = [
                    token_source.read(token_source.request(candidate.clip_id, candidate.start))
                    for candidate in item.candidates
                ]
                retrieval_scores[-1] = build_row_batch(
                    model, row_index, batch, candidates
                )
                # Multi-positive rank: the best positive candidate's position, with
                # ties broken by the manifest's fixed candidate order (the same rule
                # retrieval_report aggregates).
                scores = retrieval_scores[-1]
                positives = {int(index) for index in item.positive_indices}
                if scores and positives:
                    best = min(positives, key=lambda index: (float(scores[index]), index))
                    retrieval_rank = 1 + sum(
                        1 for value in scores if float(value) < float(scores[best])
                    )
            else:
                retrieval_scores[-1] = []
            likelihood = per_sample_likelihood[row_index]
            per_sample_rows.append(
                {
                    "sample_id": int(item.sample_id),
                    "batch": int(batch_index),
                    "split": item.split,
                    "target": item.target.as_dict(),
                    "correct_reference": None if item.correct is None else item.correct.as_dict(),
                    "wrong_reference": None if item.wrong is None else item.wrong.as_dict(),
                    "random_real_reference": None if item.random is None else item.random.as_dict(),
                    "candidates": len(item.candidates),
                    "positives": len(item.positive_indices),
                    "mask_kind": item.mask_kind,
                    "mask_seed": int(item.mask_seed),
                    "sample_seed": int(item.sample_seed),
                    "condition": item.condition,
                    "style_label": args.style_label,
                    "reasons": dict(item.reasons),
                    "supervised_tokens": likelihood["supervised_tokens"],
                    "target_token_nll_base": likelihood["target_token_nll_base"],
                    "target_token_nll_styled": likelihood["target_token_nll_styled"],
                    "target_token_nll_delta": likelihood["target_token_nll_delta"],
                    "retrieval_rank": retrieval_rank,
                    "retrieval_top1_hit": None if retrieval_rank is None else retrieval_rank == 1,
                }
            )

    retrieval = retrieval_report(retrieval_scores, retrieval_positives)
    if encoder_kind == "style_id":
        retrieval = {**retrieval, "not_applicable": "style-ID model: reference retrieval is N/A"}
    physics_summary: dict[str, Any] = {}
    for comparison_name in ("source_to_base", "base_to_styled", "source_to_styled"):
        subset = [row for row in physics_rows if row.get("comparison") == comparison_name]
        physics_summary[comparison_name] = aggregate(subset) if subset else None

    summary = {
        "checkpoint": str(args.checkpoint),
        "label": args.label or f"{operator_kind}:{encoder_kind}",
        "split": args.split,
        "regions": list(args.support) if args.support else [],
        "graph_radius": int(args.graph_radius),
        "frame_range": list(args.frame_range) if args.frame_range else None,
        "operator": operator_kind,
        "style_encoder": encoder_kind,
        "batches": len(rows),
        "mask_kind": args.mask_kind,
        "strengths": list(args.strengths),
        "strength": float(args.strength),
        "steps": int(args.steps),
        "aggregate": aggregate(rows),
        "style_retrieval_top1": retrieval.get("top1_accuracy"),
        "style_retrieval": retrieval,
        "physics": aggregate(physics_rows),
        "physics_per_comparison": physics_summary,
        "strength_curve": strength_rows,
        # The two protocols are named separately: the masked NLL is teacher-forced
        # under a stated mask, while the physics/strength numbers come from iterative
        # generation that never observes a target token inside the edit region.
        "protocols": {
            "masked_nll": "teacher-forced: the manifest's per-row mask hides tokens, "
            "targets are the real tokens",
            "generated": f"iterative: generate_edit(schedule=monotonic_frame_coordinate, "
            f"steps={int(args.steps)}) with no target token observed inside the region",
            "physics": "decoded source/base/styled motions, three comparisons kept apart",
        },
        "not_computed": unavailable_metrics(),
    }
    output = args.output or Path("outputs/mts_eval/run")
    output.mkdir(parents=True, exist_ok=True)
    _require_finite(summary)
    (output / "operator_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if per_sample_rows:
        with (output / "eval_rows.jsonl").open("w", encoding="utf-8") as handle:
            for item in per_sample_rows:
                _require_finite(item)
                handle.write(json.dumps(item, sort_keys=True, default=str) + "\n")
    if strength_rows:
        with (output / "strength_curve.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(strength_rows[0]))
            writer.writeheader()
            writer.writerows(strength_rows)
    if physics_rows:
        with (output / "physics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(physics_rows[0]))
            writer.writeheader()
            writer.writerows(physics_rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "batches": len(rows),
                "style_retrieval_top1": summary["style_retrieval_top1"],
                "nll": summary["aggregate"].get("nll_correct"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
