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

Writes ``tokens.npy`` (target / reference / edited), ``motion.npy`` (decoded
motion features for the same three rows) and ``generation.json`` with the
support, strength and seeds.  With ``--locked-edit`` (the default) tokens outside
the support are copied from the target, so an off-region comparison is exact.
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
    MotionTransportTransformer,
    OperatorBatch,
    build_operator,
    load_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator  # noqa: E402
from stylized_motion.learning.mts_operator.style_encoder import (  # noqa: E402
    GlobalStyleEncoder,
    StyleIDEncoder,
)
from stylized_motion.learning.mts_operator.windows import read_window_tokens, windows_by_clip  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Generate a locally supported style edit.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-database", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--content-clip", type=int, required=True, help="Target clip row index.")
    parser.add_argument("--style-clip", type=int, required=True, help="Reference clip row index.")
    parser.add_argument("--regions", nargs="*", default=["left_arm"], help="NEF region names, or whole_body.")
    parser.add_argument("--graph-radius", type=int, default=1)
    parser.add_argument("--frame-range", nargs=2, type=int, default=None, help="Half-open [start, stop).")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4, help="Independent draws per condition.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--locked-edit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics_meta = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    model_config = checkpoint.get("metadata", {}).get("model_config", {}) or {}
    encoder_kind = str(metrics_meta.get("style_encoder_kind", "reference"))
    operator_name = str(metrics_meta.get("operator", "birth_death"))
    transport_path = args.transport_checkpoint or metrics_meta.get("transport_checkpoint")
    if transport_path is None:
        raise ValueError(
            "The operator checkpoint records no transport path; pass --transport-checkpoint"
        )
    transport_dim = 256

    def build_transport(stored):
        nonlocal transport_dim
        model = MotionTransportTransformer(adapter, **stored)
        transport_dim = int(model.dim)
        return model

    _, transport = load_mts_checkpoint(
        Path(str(transport_path)),
        kind="transport",
        build_model=build_transport,
        device=device,
        token_spec=token_spec,
        tokenizer_metadata=tokenizer.representation_metadata(),
    )
    transport.eval()

    if encoder_kind == "style_id":
        encoder_config = dict(model_config.get("style_encoder") or {})
        style_encoder = StyleIDEncoder(
            num_styles=int(encoder_config.get("num_styles", 1)),
            output_dim=int(encoder_config.get("output_dim", transport_dim)),
        )
    else:
        encoder_config = dict(model_config.get("style_encoder") or {})
        encoder_config.pop("kind", None)
        style_encoder = GlobalStyleEncoder(adapter, **encoder_config)
    operator_description = dict(model_config.get("operator") or {})
    operator_name = str(operator_description.get("name", operator_name))
    operator_config = {
        key: value
        for key, value in dict(operator_description.get("config") or {}).items()
        if key in _OPERATOR_CTOR_KEYS
    }
    operator = build_operator(
        operator_name, num_levels=adapter.num_levels, stream_dim=transport_dim, **operator_config
    )
    model = MtsStyleOperator(
        adapter, transport=transport, style_encoder=style_encoder, operator=operator, freeze_transport=True
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.token_store:
        store = open_any_token_store(args.token_store)
    elif args.feature_database:
        store = open_any_feature_store(args.feature_database)
    else:
        raise ValueError("Pass --feature-database or --token-store to read clips")
    windows = windows_by_clip(store, args.split, frames=int(args.frames))
    history = int(tokenizer.history_frames)
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    shards: dict[int, Any] = {}

    def tokens_for(clip_id: int) -> torch.Tensor:
        candidates = windows.get(int(clip_id))
        if not candidates:
            raise ValueError(f"Clip {clip_id} has no {args.frames}-frame window in split {args.split!r}")
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

    target_tokens = tokens_for(int(args.content_clip))
    reference_tokens = tokens_for(int(args.style_clip))
    support = adapter.hard_mask(
        list(args.regions),
        graph_radius=int(args.graph_radius),
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        length=target_tokens.shape[0],
        device=device,
    )

    crn = CommonRandomNumbers(seed=int(args.seed))
    edited_rows, styled_rows = [], []
    for sample in range(int(args.samples)):
        batch = OperatorBatch(
            target_tokens=target_tokens[None].to(device),
            reference_tokens=reference_tokens[None].to(device),
            hard_mask=support,
            strength=float(args.strength),
        )
        with torch.no_grad():
            if args.locked_edit:
                drawn = model.generate_edit(batch, crn=crn)
            else:
                result = model(batch)
                from stylized_motion.learning.mts_operator import sample_tokens

                drawn = sample_tokens(result.probabilities, generator=generator)
        edited_rows.append(drawn[0].cpu())
        with torch.no_grad():
            styled_rows.append(model(batch).probabilities[0].cpu())

    edited = torch.stack(edited_rows)
    probabilities = torch.stack(styled_rows)
    with torch.no_grad():
        baseline_motion = tokenizer.decode_indices(target_tokens[None].to(device))[0].cpu()
        reference_motion = tokenizer.decode_indices(reference_tokens[None].to(device))[0].cpu()
        edited_motion = tokenizer.decode_indices(edited.to(device)).cpu()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "tokens.npy", edited.numpy())
    np.save(output / "motion.npy", edited_motion.numpy())
    np.save(output / "baseline_motion.npy", baseline_motion.numpy())
    np.save(output / "reference_motion.npy", reference_motion.numpy())
    np.save(output / "support.npy", support.cpu().numpy())
    summary = {
        "checkpoint": str(args.checkpoint),
        "operator": operator_name,
        "style_encoder": encoder_kind,
        "content_clip": int(args.content_clip),
        "style_clip": int(args.style_clip),
        "regions": list(args.regions),
        "graph_radius": int(args.graph_radius),
        "frame_range": list(args.frame_range) if args.frame_range else None,
        "strength": float(args.strength),
        "locked_edit": bool(args.locked_edit),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "support_coordinates": int(support.sum()),
        "support_fraction": float(support.float().mean()),
        "outside_support_unchanged": bool(
            (edited[:, ~support.cpu()] == target_tokens[None][:, ~support.cpu()]).all()
        )
        if args.locked_edit
        else None,
        "mean_max_probability": float(probabilities.max(dim=-1).values.mean()),
        "changed_support_ratio": float(
            (edited[:, support.cpu()] != target_tokens[None][:, support.cpu()]).float().mean()
        ),
    }
    (output / "generation.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote tokens.npy / motion.npy / generation.json into {output}")


if __name__ == "__main__":
    main()
