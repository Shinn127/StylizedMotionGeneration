#!/usr/bin/env python
"""Generate styled token edits and decoded motion from a trained MTS operator.

    python scripts/generate_mts_operator.py \
      --checkpoint outputs/mts_operator/birth_death/seed3407/best.pt \
      --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
      --transport-checkpoint outputs/mts_transport/seed3407/best.pt \
      --feature-database ... --split test \
      --content-clip 0 --style-clip 3 \
      --regions left_arm --graph-radius 1 --frame-range 16 48 \
      --strength 1.0 --seed 1234 --locked-edit \
      --output outputs/mts_samples/example

Every draw is a locked edit over an explicit region: the observed set is stated
outright (``visible = ~support``), ``--no-locked-edit`` is refused unless the
region is the whole body (whole-body generation is written as
``--regions whole_body``), and each of ``--samples`` draws uses its own
``sample_id`` so the draws are independent yet reproducible from ``--seed``.

Writes ``tokens.npy`` (styled draws), ``source_tokens.npy``,
``base_tokens.npy`` (argmax of the frozen transport, the "no style" reference),
``motion.npy`` / ``baseline_motion.npy`` / ``reference_motion.npy`` (decoded),
``support.npy`` / ``edit_mask.npy`` / ``visible_mask.npy`` and
``generation.json`` with the masks, per-draw sample ids and the
source / base / styled comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator import (  # noqa: E402
    CommonRandomNumbers,
    LayoutAdapter,
    OperatorBatch,
)
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    checkpoint_action_vocabulary,
    checkpoint_style_index,
    file_sha256,
    load_operator_bundle,
    require_token_store_binding,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.pairs import clip_records_from_store  # noqa: E402
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    ContentVocabulary,
    TokenSource,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402
from stylized_motion.learning.runner import choose_device, set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a locally supported style edit.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-database", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--dataset", default=None, help="Source dataset name (e.g. 100style).")
    parser.add_argument("--content-clip", type=int, required=True, help="Target clip row index.")
    parser.add_argument(
        "--style-clip",
        type=int,
        default=None,
        help="Reference clip row index (reference encoders only).",
    )
    parser.add_argument(
        "--style-label",
        default=None,
        help="Style label for a style-ID model, mapped through the checkpoint's style_to_id.",
    )
    parser.add_argument("--regions", nargs="*", default=["left_arm"], help="NEF region names, or whole_body.")
    parser.add_argument("--graph-radius", type=int, default=1)
    parser.add_argument("--frame-range", nargs=2, type=int, default=None, help="Half-open [start, stop).")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Iterative generation steps: the region is filled monotonically, each "
        "position committed exactly once.",
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4, help="Independent draws per condition.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--locked-edit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def resolve_generation_region(
    support: torch.Tensor, *, locked_edit: bool, frames: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """The explicit region/visible pair a revision-2 generation run uses.

    Returns ``(region, visible_mask, whole_body)``.  ``--no-locked-edit`` is only
    meaningful for a whole-body region: with a partial support it would sample
    positions the caller explicitly excluded, so it is refused instead.
    """
    region = support.to(torch.bool).unsqueeze(0).expand(1, frames, support.shape[-1])
    whole_body = bool(region.all())
    if not locked_edit and not whole_body:
        raise ValueError(
            "--no-locked-edit with a partial support would sample outside the requested "
            "region; revision 2 expresses whole-body generation as --regions whole_body"
        )
    return region, ~region, whole_body


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

    # One bundle: the operator file carries its transport weights and its own
    # configuration, so generation needs nothing but the tokenizer.
    checkpoint, model = load_operator_bundle(
        args.checkpoint,
        adapter=adapter,
        tokenizer_identity=tokenizer.representation_metadata(),
        # The tokenizer *file*, not only its metadata: the recorded SHA is what
        # separates two tokenizers that share a layout (F05).
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        device=device,
    )
    model_config = dict(checkpoint.get("metadata", {}).get("model_config") or {})
    metrics_meta = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    operator_name = str(model_config.get("operator", {}).get("name", "operator"))
    encoder_kind = str(model_config.get("style_encoder", {}).get("kind", "reference"))
    if args.transport_checkpoint is not None:
        recorded = checkpoint.get("provenance", {}).get("upstream_transport_sha256")
        actual = file_sha256(args.transport_checkpoint)
        if recorded is not None and str(recorded) != actual:
            raise ValueError(
                f"--transport-checkpoint {args.transport_checkpoint} has SHA-256 {actual}, but "
                f"this operator was trained against {recorded}"
            )
    transport = model.transport
    content_vocabulary = getattr(transport, "content_vocabulary", None)
    if content_vocabulary is None:
        stored_actions = checkpoint_action_vocabulary(checkpoint)
        content_vocabulary = (
            ContentVocabulary.from_dict(stored_actions) if stored_actions else ContentVocabulary()
        )

    if args.token_store:
        store = open_any_token_store(args.token_store)
        store_kind = "token"
    elif args.feature_database:
        store = open_any_feature_store(args.feature_database)
        store_kind = "feature"
    else:
        raise ValueError("Pass --feature-database or --token-store to read clips")
    if store_kind == "token":
        require_token_store_binding(
            store, tokenizer_checkpoint=args.tokenizer_checkpoint, checkpoint=checkpoint,
            where="generation token store",
        )
    validate_store_binding(store, store_kind=store_kind)
    windows = windows_by_clip(store, args.split, frames=int(args.frames))
    history = int(tokenizer.history_frames)
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    # Shared window reader; the action condition comes from the frozen transport's
    # own vocabulary rather than from a positional guess.
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
    content_vocabulary = getattr(transport, "content_vocabulary", None) or ContentVocabulary()

    def tokens_for(clip_id: int) -> torch.Tensor:
        sample = token_source.window(int(clip_id))
        if sample is None:
            raise ValueError(f"Clip {clip_id} has no {args.frames}-frame window in split {args.split!r}")
        return sample.tokens

    target_tokens = tokens_for(int(args.content_clip))
    style_label_id = None
    if encoder_kind == "style_id":
        stored_style_index = checkpoint_style_index(checkpoint)
        if args.style_label is None:
            raise ValueError(
                "A style-ID model needs --style-label; the reference clip is not its input"
            )
        if stored_style_index is None:
            raise ValueError("The checkpoint records no style_to_id map")
        if str(args.style_label) not in stored_style_index:
            raise ValueError(
                f"Unknown style label {args.style_label!r}; the checkpoint knows "
                f"{sorted(stored_style_index)}"
            )
        style_label_id = int(stored_style_index[str(args.style_label)])
        if args.style_clip is not None:
            # Accepted-but-ignored is how "the run used the reference" gets claimed
            # for a model that cannot read one.
            raise SystemExit(
                "This is a style-ID checkpoint: --style-label is its style input and "
                "--style-clip is not read. Drop --style-clip."
            )
        reference_tokens = target_tokens
    else:
        if args.style_clip is None:
            raise ValueError("A reference model needs --style-clip")
        reference_tokens = tokens_for(int(args.style_clip))
    content_condition = None
    if not content_vocabulary.unconditional:
        action = next(
            (
                record.content
                for record in clip_records_from_store(
                    store, dataset=None if args.dataset is None else str(args.dataset)
                )
                if int(record.clip_id) == int(args.content_clip)
            ),
            None,
        )
        if action is None:
            raise ValueError(f"No action label for clip {int(args.content_clip)}")
        content_condition = content_vocabulary.vector([action]).to(device)
        print(f"content condition: action {action!r} -> id {int(content_condition.item())}", flush=True)
    support = adapter.hard_mask(
        list(args.regions),
        graph_radius=int(args.graph_radius),
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        length=target_tokens.shape[0],
        device=device,
    )

    # Revision 2 has one generation path: the edit region is the support, and
    # everything else is copied.  "Unlocked" whole-body sampling must be written
    # as a whole-body region instead of quietly leaving the support.
    region, visible_mask, whole_body = resolve_generation_region(
        support, locked_edit=bool(args.locked_edit), frames=int(target_tokens.shape[0])
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    crn = CommonRandomNumbers(seed=int(args.seed))
    edited_rows, styled_rows, base_rows = [], [], []
    for sample in range(int(args.samples)):
        batch = OperatorBatch(
            target_tokens=target_tokens[None].to(device),
            reference_tokens=reference_tokens[None].to(device),
            style_ids=None if style_label_id is None else torch.tensor([style_label_id], device=device),
            # Stated outright: the region is authored, the rest is evidence.
            visible_mask=visible_mask.to(device),
            hard_mask=support,
            content_condition=content_condition,
            strength=float(args.strength),
        )
        with torch.no_grad():
            result = model(batch)
            # The base is the frozen transport's own distribution.  A strength=0
            # forward is NOT the base for the arbitrary kernel (its lambda=0 is a
            # uniform kernel), so it must not be used as the reference.
            base_rows.append(result.base_probabilities.argmax(dim=-1)[0].cpu())
            styled_rows.append(result.probabilities[0].cpu())
            drawn, commits = model.generate_edit(
                batch, crn=crn, sample_id=sample, step_id=0,
                steps=int(args.steps), return_trace=True,
            )
            if sample == 0:
                np.save(output / "commit_trace.npy",
                        torch.stack(commits).cpu().numpy() if commits else np.zeros(0))
        edited_rows.append(drawn[0].cpu())
        edit_mask = batch.effective_edit_mask(model.spec).cpu()

    edited = torch.stack(edited_rows)
    probabilities = torch.stack(styled_rows)
    base_tokens = torch.stack(base_rows)
    with torch.no_grad():
        baseline_motion = tokenizer.decode_indices(target_tokens[None].to(device))[0].cpu()
        reference_motion = tokenizer.decode_indices(reference_tokens[None].to(device))[0].cpu()
        edited_motion = tokenizer.decode_indices(edited.to(device)).cpu()

    output = args.output
    np.save(output / "tokens.npy", edited.numpy())
    np.save(output / "source_tokens.npy", target_tokens.numpy())
    np.save(output / "base_tokens.npy", base_tokens.numpy())
    np.save(output / "motion.npy", edited_motion.numpy())
    np.save(output / "baseline_motion.npy", baseline_motion.numpy())
    np.save(output / "reference_motion.npy", reference_motion.numpy())
    np.save(output / "support.npy", support.cpu().numpy())
    np.save(output / "edit_mask.npy", edit_mask.cpu().numpy())
    np.save(output / "visible_mask.npy", visible_mask.cpu().numpy())
    # [1, T, K] masks expanded over the sample axis: a boolean mask must match the
    # leading dimensions of what it indexes, and the draws are [S, T, K].
    region_cpu = region.cpu()[0]
    support_cpu = support.cpu()
    source = target_tokens[None]
    source_stack = source.expand_as(edited)
    edited_region = region_cpu.unsqueeze(0).expand_as(edited)
    source_region = region_cpu.unsqueeze(0).expand_as(source_stack)
    outside_region = (~region_cpu).unsqueeze(0).expand_as(edited)
    summary = {
        "checkpoint": str(args.checkpoint),
        "operator": operator_name,
        "style_encoder": encoder_kind,
        "content_clip": int(args.content_clip),
        "style_clip": None if args.style_clip is None else int(args.style_clip),
        "regions": list(args.regions),
        "graph_radius": int(args.graph_radius),
        "frame_range": list(args.frame_range) if args.frame_range else None,
        "strength": float(args.strength),
        "steps": int(args.steps),
        "locked_edit": bool(args.locked_edit),
        "whole_body": whole_body,
        "samples": int(args.samples),
        "sample_ids": list(range(int(args.samples))),
        "seed": int(args.seed),
        "crn_seed": int(crn.seed),
        "crn_keys": sorted(str(key) for key in crn.uniforms_cache),
        "support_coordinates": int(support_cpu.sum()),
        "support_fraction": float(support_cpu.float().mean()),
        # source = the input clip, base = the frozen transport without style,
        # styled = the edited draw.  Kept apart so "changed" cannot be confused
        # with "the operator did something".
        "outside_region_unchanged": bool((edited[outside_region] == source_stack[outside_region]).all()),
        "source_changed_ratio": float(
            (edited[edited_region] != source_stack[source_region]).float().mean()
        ),
        "base_changed_ratio": float(
            (edited[edited_region] != base_tokens[edited_region]).float().mean()
        ),
        "base_vs_source_changed_ratio": float(
            (base_tokens[edited_region] != source_stack[source_region]).float().mean()
        ),
        "mean_max_probability": float(probabilities.max(dim=-1).values.mean()),
    }
    (output / "generation.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(
        f"wrote tokens.npy / source_tokens.npy / base_tokens.npy / motion.npy / "
        f"support.npy / edit_mask.npy / visible_mask.npy / generation.json into {output}"
    )


if __name__ == "__main__":
    main()
