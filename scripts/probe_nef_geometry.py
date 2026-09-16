#!/usr/bin/env python
"""Probe NEF-FSQ level geometry and temporal influence on real token windows.

Phase 0 of ``docs/MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md``: before any style
operator is built, measure whether adjacent FSQ levels are neighbouring decoded
motion states, and how far one token edit propagates through the causal decoder.

    python scripts/probe_nef_geometry.py \
      --checkpoint outputs/nef_fsq_40x9/best.pt \
      --feature-database data/processed/100style_pruned_90/fsq_window_index \
      --split test --max-clips 256 --output outputs/nef_probe/v1

Writes ``probe_geometry.json`` (full per-coordinate report) and
``probe_geometry.csv`` (one row per coordinate) into the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.sampling import FixedWindowSampler  # noqa: E402
from stylized_motion.learning.nef_eval import validate_checkpoint_store  # noqa: E402
from stylized_motion.learning.nef_layout import NEF_STREAM_NAMES  # noqa: E402
from stylized_motion.learning.nef_probe import (  # noqa: E402
    KinematicContext,
    LevelGeometryProbe,
    json_dumps,
    model_space_window,
    read_probe_window,
    temporal_influence_width,
    write_probe_csv,
)
from stylized_motion.learning.representation import (  # noqa: E402
    NEF_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import choose_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe NEF-FSQ level geometry, decoded effects and decoder influence width."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-clips", type=int, default=256, help="Windows to probe (0 = all).")
    parser.add_argument("--far-samples", type=int, default=2, help="Far-level jumps per coordinate.")
    parser.add_argument("--far-min-distance", type=int, default=None)
    parser.add_argument("--decode-rows", type=int, default=1024, help="Rows per decoder call.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--root-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--no-kinematics", action="store_true", help="Skip FK/root/contact metrics.")
    parser.add_argument(
        "--influence-streams",
        nargs="*",
        default=["left_arm_node", "global"],
        help="Streams measured by the temporal-influence probe.",
    )
    parser.add_argument("--influence-frame", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = choose_device(args.device)
    store = open_any_feature_store(args.feature_database)
    try:
        checkpoint, model = load_representation_checkpoint(args.checkpoint, device)
        if model.family != NEF_FSQ_FAMILY:
            raise ValueError(
                f"--checkpoint must hold a {NEF_FSQ_FAMILY!r} model, got {model.family!r}"
            )
        validate_checkpoint_store(checkpoint, model, store)
        model = model.to(device).eval()
        module = model.module
        history = int(model.history_frames)
        windows = list(
            FixedWindowSampler(store, args.split, target_frames=64, stride=64, include_tail=True)
        )
        if args.max_clips > 0:
            windows = windows[: args.max_clips]
        if not windows:
            raise ValueError(f"Split {args.split!r} has no 64-frame windows to probe")

        feature_stats = checkpoint["feature_stats"]
        shards: dict[int, object] = {}
        token_windows = []
        with torch.no_grad():
            for request in windows:
                window = read_probe_window(store, request, history=history, shards=shards)
                motion = model_space_window(window, store, feature_stats).to(device)
                tokens = model.encode_indices(motion[None])
                # Only the frames with a full 63-frame history are probed; the
                # leading frames exist to give the encoder its context.
                token_windows.append(tokens[0, history:])
        tokens = torch.stack(token_windows)
        if tokens.shape[1:] != (64, module.layout.num_coordinates):
            raise RuntimeError(f"Unexpected token window shape {tuple(tokens.shape)}")

        kinematic = None
        if not args.no_kinematics:
            kinematic = KinematicContext.from_feature_stats(
                feature_stats, dt=args.root_dt
            )
        probe = LevelGeometryProbe(
            model,
            kinematic=kinematic,
            far_samples=args.far_samples,
            far_min_distance=args.far_min_distance,
            decode_rows=args.decode_rows,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        report = probe.run(tokens, generator=generator)

        influences = []
        for stream in args.influence_streams:
            if stream not in NEF_STREAM_NAMES:
                raise ValueError(f"Unknown NEF stream {stream!r}")
            influences.append(
                temporal_influence_width(
                    model, tokens, stream=stream, frame=int(args.influence_frame)
                )
            )

        payload = {
            "kind": "nef_probe",
            "checkpoint": str(args.checkpoint),
            "feature_database": str(args.feature_database),
            "split": args.split,
            "windows": len(windows),
            "history_frames": history,
            "representation_id": model.representation_id,
            "layout_hash": module.layout.layout_hash(),
            "seed": int(args.seed),
            "geometry": report,
            "temporal_influence": influences,
        }
        output = args.output
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "probe_geometry.json").write_text(
                json_dumps(payload) + "\n", encoding="utf-8"
            )
            write_probe_csv(output / "probe_geometry.csv", report)
        summary = {
            "windows": len(windows),
            **report["summary"],
            "temporal_influence": [
                {
                    "stream": item["stream"],
                    "frames_after": item["frames_after"],
                    "within_contract": item["within_contract"],
                }
                for item in influences
            ],
        }
        print(json.dumps(summary, indent=2, default=str))
        if output is not None:
            print(f"wrote {output / 'probe_geometry.json'} and {output / 'probe_geometry.csv'}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
