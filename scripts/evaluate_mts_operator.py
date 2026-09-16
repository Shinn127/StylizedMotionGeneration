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
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store, open_token_store  # noqa: E402
from stylized_motion.learning.mts_operator import (  # noqa: E402
    LayoutAdapter,
    MaskGenerator,
    MotionTransportTransformer,
    OperatorBatch,
    load_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.masking import MaskConfig  # noqa: E402
from stylized_motion.learning.mts_operator.metrics import (  # noqa: E402
    aggregate,
    content_preservation,
    physics_metrics,
    reference_sensitivity,
    strength_response,
    support_locality,
    style_retrieval,
    unavailable_metrics,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator  # noqa: E402
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    StylePairSampler,
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.style_encoder import (  # noqa: E402
    GlobalStyleEncoder,
    StyleIDEncoder,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    read_window_tokens,
    windows_by_clip,
)
from stylized_motion.learning.nef_probe import KinematicContext  # noqa: E402
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402
from stylized_motion.learning.runner import choose_device, set_seed  # noqa: E402


_OPERATOR_CTOR_KEYS = frozenset(
    {
        "hidden_dim",
        "coordinate_dim",
        "identity_mix",
        "max_rate",
        "uniformization_tolerance",
        "max_terms",
        "style_dim",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a reference-conditioned MTS operator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
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
    parser.add_argument("--support", nargs="*", default=None, help="Region names for locality metrics.")
    parser.add_argument("--graph-radius", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(int(args.seed), deterministic=False)
    device = choose_device(args.device)

    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(
        args.tokenizer_checkpoint, torch.device("cpu")
    )
    tokenizer = tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    adapter = LayoutAdapter(tokenizer.token_layout(), num_levels=int(tokenizer.num_levels))
    token_spec = adapter.token_spec(representation_id=tokenizer.representation_id)
    tokenizer_metadata = tokenizer.representation_metadata()

    style_config: dict[str, object] = {}
    operator_kind = "unknown"
    encoder_kind = "unknown"
    transport_dim = 256
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    extra = checkpoint.get("metadata", {})
    model_config = extra.get("model_config", {}) if isinstance(extra, dict) else {}
    metrics_meta = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    operator_kind = str(metrics_meta.get("operator", "unknown"))
    encoder_kind = str(metrics_meta.get("style_encoder_kind", "reference"))

    def build_transport(stored):
        nonlocal transport_dim
        model = MotionTransportTransformer(adapter, **stored)
        transport_dim = int(model.dim)
        return model

    transport_path = args.transport_checkpoint or metrics_meta.get("transport_checkpoint")
    if transport_path is None:
        raise ValueError(
            "The operator checkpoint records no transport path; pass --transport-checkpoint"
        )
    _, transport = load_mts_checkpoint(
        Path(str(transport_path)),
        kind="transport",
        build_model=build_transport,
        device=device,
        token_spec=token_spec,
        tokenizer_metadata=tokenizer_metadata,
    )
    transport.eval()

    if encoder_kind == "style_id":
        style_encoder = StyleIDEncoder(
            num_styles=int(model_config.get("style_encoder", {}).get("num_styles", 1)),
            output_dim=int(model_config.get("style_encoder", {}).get("output_dim", transport_dim)),
        )
    else:
        encoder_config = dict(model_config.get("style_encoder") or {})
        encoder_config.pop("kind", None)
        style_encoder = GlobalStyleEncoder(adapter, **encoder_config)

    from stylized_motion.learning.mts_operator import build_operator  # local import keeps the CLI light

    # The checkpoint stores the operator's *description*; only its constructor
    # arguments may be replayed, and the level count / stream width come from the
    # live tokenizer and transport.
    operator_description = dict(model_config.get("operator") or {})
    operator_name = str(operator_description.get("name", operator_kind))
    operator_config = {
        key: value
        for key, value in dict(operator_description.get("config") or {}).items()
        if key in _OPERATOR_CTOR_KEYS
    }
    operator = build_operator(
        operator_name,
        num_levels=adapter.num_levels,
        stream_dim=transport_dim,
        **operator_config,
    )
    model = MtsStyleOperator(
        adapter, transport=transport, style_encoder=style_encoder, operator=operator, freeze_transport=True
    ).to(device)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Operator checkpoint is missing its model state dict")
    model.load_state_dict(state)
    model.eval()

    store_path = args.token_store
    feature_path = args.feature_database
    if store_path:
        store = open_token_store(store_path)
    elif feature_path:
        store = open_any_feature_store(feature_path)
    else:
        raise ValueError("Pass --feature-database or --token-store to read evaluation windows")
    records = clip_records_from_store(store)
    style_split = split_styles_by_performer(records, seed=int(args.seed))
    sampler = StylePairSampler(records, style_split=style_split, seed=int(args.seed))
    windows = windows_by_clip(store, args.split, frames=int(args.frames))
    history = int(tokenizer.history_frames)
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    shards: dict[int, Any] = {}
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    pair_rng = np.random.default_rng(int(args.seed))
    mask_generator = MaskGenerator(MaskConfig(mixture={args.mask_kind: 1.0}))
    kinematic = (
        KinematicContext.from_feature_stats(feature_stats)
        if feature_stats and "ref_pos" in feature_stats
        else None
    )

    def tokens_for(clip_id: int) -> torch.Tensor | None:
        candidates = windows.get(int(clip_id))
        if not candidates:
            return None
        request = candidates[int(torch.randint(len(candidates), (1,), generator=generator).item())]
        return read_window_tokens(
            store,
            request,
            frames=int(args.frames),
            history=history,
            tokenizer=None if args.token_store else tokenizer,
            feature_stats=None if args.token_store else feature_stats,
            shards=shards,
        )

    rows: list[dict[str, float]] = []
    strength_rows: list[dict[str, float]] = []
    physics_rows: list[dict[str, float]] = []
    retrieval_hits: list[int] = []
    for _ in range(int(args.batches)):
        pairs = sampler.sample(
            count=int(args.batch_size), mode="same_style", stage=args.split, generator=pair_rng
        )
        if not pairs:
            break
        targets, references, wrong_references, random_references, content_ids = [], [], [], [], []
        for index, pair in enumerate(pairs):
            target = tokens_for(pair.target.clip_id)
            reference = tokens_for(pair.reference.clip_id)
            if target is None or reference is None:
                continue
            targets.append(target)
            references.append(reference)
            wrong_references.append(reference.roll(1, dims=0))
            random_references.append(
                torch.randint(0, adapter.num_levels, target.shape, generator=generator)
            )
            content_ids.append(index % 4)
        if not targets:
            continue
        target_tokens = torch.stack(targets).to(device)
        mask = mask_generator.sample_kind(
            args.mask_kind, target_tokens.shape[0], target_tokens.shape[1], adapter=adapter,
            generator=generator, device=device,
        )
        batch = OperatorBatch(
            target_tokens=target_tokens,
            reference_tokens=torch.stack(references).to(device),
            visible_mask=mask.visible_mask,
            content_condition=torch.tensor(content_ids, device=device, dtype=torch.long),
            strength=1.0,
        )
        row = reference_sensitivity(
            model,
            batch,
            wrong_reference=torch.stack(wrong_references).to(device),
            random_reference=torch.stack(random_references).to(device),
        )
        row.update(content_preservation(model, batch))
        if args.support:
            support = adapter.hard_mask(
                list(args.support), graph_radius=int(args.graph_radius), length=target_tokens.shape[1],
                device=device,
            )
            batch = OperatorBatch(
                target_tokens=target_tokens,
                reference_tokens=batch.reference_tokens,
                visible_mask=mask.visible_mask,
                hard_mask=support,
                content_condition=batch.content_condition,
            )
            row.update(support_locality(model, batch))
        rows.append(row)

        for point in strength_response(model, batch, strengths=list(args.strengths)):
            strength_rows.append({"batch": float(len(rows)), **point})

        if kinematic is not None:
            for strength in (0.0, 1.0):
                styled_batch = OperatorBatch(
                    target_tokens=target_tokens,
                    reference_tokens=batch.reference_tokens,
                    hard_mask=batch.hard_mask,
                    content_condition=batch.content_condition,
                    strength=strength,
                )
                edited_tokens = model.generate_edit(
                    styled_batch, generator=torch.Generator().manual_seed(int(args.seed))
                )
                with torch.no_grad():
                    base_motion = tokenizer.decode_indices(target_tokens)
                    edited_motion = tokenizer.decode_indices(edited_tokens)
                physics = physics_metrics(
                    baseline_motion=base_motion,
                    edited_motion=edited_motion,
                    kinematic=kinematic,
                    edit_interval=(0, target_tokens.shape[1]),
                )
                physics_rows.append({"strength": float(strength), **physics})

        # Style retrieval: every target must rank its own reference first.
        retrieval = style_retrieval(
            model,
            batch,
            candidate_sets=[
                [batch.reference_tokens[index] for index in range(target_tokens.shape[0])]
                for _ in range(target_tokens.shape[0])
            ],
            correct_index=list(range(target_tokens.shape[0])),
        )
        retrieval_hits.append(int(round(retrieval["top1_accuracy"])))

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "operator": operator_kind,
        "style_encoder": encoder_kind,
        "batches": len(rows),
        "mask_kind": args.mask_kind,
        "strengths": list(args.strengths),
        "aggregate": aggregate(rows),
        "style_retrieval_top1": float(np.mean(retrieval_hits)) if retrieval_hits else None,
        "physics": aggregate(physics_rows),
        "strength_curve": strength_rows,
        "not_computed": unavailable_metrics(),
    }
    output = args.output or Path("outputs/mts_eval/run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "operator_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
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
