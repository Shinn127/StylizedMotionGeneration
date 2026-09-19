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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.sampling import FixedWindowSampler  # noqa: E402
from stylized_motion.learning.nef_eval import validate_checkpoint_store  # noqa: E402
from stylized_motion.learning.nef_layout import NEF_STREAM_NAMES  # noqa: E402
from stylized_motion.learning.nef_probe import (  # noqa: E402
    PROBE_PROTOCOL_REVISION,
    KinematicContext,
    reference_positions_for_fk,
    LevelGeometryProbe,
    json_dumps,
    model_space_window,
    store_normalized_window,
    read_probe_window,
    stratified_span_probe,
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
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Also run the protocol-v2 stratified span probe (single pulse and fixed "
        "signed spans over one shared legal support) and write probe_stratified.*",
    )
    parser.add_argument(
        "--stratified-frames",
        type=int,
        default=128,
        help="Window length of the stratified probe: a complete tail needs "
        "frames >= frame + span + decoder RF - 1.",
    )
    parser.add_argument("--stratified-frame", type=int, default=32)
    parser.add_argument("--stratified-span", type=int, default=4)
    parser.add_argument("--stratified-magnitudes", nargs="*", type=int, default=[1, 3])
    parser.add_argument("--stratified-signs", nargs="*", type=int, default=[1, -1])
    parser.add_argument(
        "--stratified-coordinates",
        nargs="*",
        type=int,
        default=None,
        help="Coordinates of the stratified probe (default: one per stream).",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser




def _default_stratified_coordinates(module: Any) -> list[int]:
    """One coordinate per NEF stream: the support is spread over the whole body."""
    layout = module.layout
    chosen: list[int] = []
    for stream in layout.coordinate_order:
        span = layout.stream_slices[stream]
        chosen.append(int(span.start))
    return chosen


def run_stratified(
    args: argparse.Namespace,
    *,
    store: Any,
    model: Any,
    module: Any,
    history: int,
    feature_stats: Mapping[str, Any],
    kinematic: KinematicContext | None,
    device: torch.device,
) -> dict[str, Any]:
    """Protocol v2: single pulses and fixed signed spans with a shared support.

    The window is chosen so the decoder's tail fits: the probe reads
    ``stratified_frames`` tokens and states ``temporal_probe_complete`` per row,
    because a window that ends inside the influence cannot say where it stopped.
    """
    frames = int(args.stratified_frames)
    receptive_field = int(getattr(module, "receptive_field", 0) or 0)
    needs = int(args.stratified_frame) + int(args.stratified_span) + max(0, receptive_field - 1)
    coordinates = (
        list(args.stratified_coordinates)
        if args.stratified_coordinates
        else _default_stratified_coordinates(module)
    )
    windows = list(
        FixedWindowSampler(
            store, args.split, target_frames=frames, stride=frames, include_tail=True
        )
    )
    if args.max_clips > 0:
        windows = windows[: args.max_clips]
    if not windows:
        raise ValueError(
            f"Split {args.split!r} has no {frames}-frame window; lower --stratified-frames"
        )
    shards: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    incomplete = 0
    with torch.no_grad():
        for request in windows:
            window = read_probe_window(store, request, history=history, shards=shards)
            motion = model_space_window(
                store_normalized_window(store, window), store, feature_stats
            ).to(device)
            tokens = model.encode_indices(motion[None])[0, history:]
            if int(tokens.shape[0]) != frames:
                raise RuntimeError(
                    f"Stratified probe expected {frames} frames, got {int(tokens.shape[0])}"
                )
            for coordinate in coordinates:
                report = stratified_span_probe(
                    module,
                    tokens[None],
                    coordinate=int(coordinate),
                    frame=int(args.stratified_frame),
                    span=int(args.stratified_span),
                    magnitudes=tuple(int(value) for value in args.stratified_magnitudes),
                    signs=tuple(int(value) for value in args.stratified_signs),
                    kinematic=kinematic,
                )
                incomplete += 0 if report["temporal_probe_complete"] else 1
                for row in report["rows"]:
                    rows.append(
                        {
                            "clip_id": int(getattr(request, "variant_idx", -1)),
                            "target_start": int(getattr(request, "target_start", -1)),
                            "window_frames": frames,
                            **{key: value for key, value in report.items() if key != "rows"},
                            **row,
                        }
                    )
    return {
        "kind": "stratified_probe",
        "protocol_revision": int(PROBE_PROTOCOL_REVISION),
        "checkpoint": str(args.checkpoint),
        "feature_database": str(args.feature_database),
        "split": args.split,
        "windows": len(windows),
        "window_frames": frames,
        "history_frames": int(history),
        "coordinates": [int(value) for value in coordinates],
        "frame": int(args.stratified_frame),
        "span": int(args.stratified_span),
        "magnitudes": [int(value) for value in args.stratified_magnitudes],
        "signs": [int(value) for value in args.stratified_signs],
        "decoder_receptive_field": receptive_field,
        "tail_frames_needed_for_complete_probe": int(needs),
        "rows_with_incomplete_tail": int(incomplete),
        "seed": int(args.seed),
        "units": {"feature": "mean |decoded difference|", "world": "metres"},
        "rows": rows,
    }


def write_stratified_csv(path: str | Path, report: Mapping[str, Any]) -> Path:
    """One row per (clip, coordinate, sign, magnitude) of the stratified probe."""
    import csv

    rows = list(report.get("rows", []))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "clip_id",
        "target_start",
        "coordinate",
        "sign",
        "magnitude",
        "kind",
        "frame_range",
        "spec_visible",
        "legal_frames_in_span",
        "excluded_by_level_range",
        "feature_l1_inside_support",
        "feature_l1_mean",
        "feature_last_changed_frame",
        "world_fk_max",
        "world_fk_tail_max",
        "world_root_tail_max",
        "temporal_probe_complete",
        "decoder_receptive_field",
        "protocol_revision",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path




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
                motion = model_space_window(
                    store_normalized_window(store, window), store, feature_stats
                ).to(device)
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
                feature_stats,
                dt=args.root_dt,
                reference_positions=reference_positions_for_fk(feature_stats),
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

        stratified = None
        if args.stratified:
            stratified = run_stratified(
                args,
                store=store,
                model=model,
                module=module,
                history=history,
                feature_stats=feature_stats,
                kinematic=kinematic,
                device=device,
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
            if stratified is not None:
                # A separate file: the historical geometry results are never rewritten
                # by a run with a different protocol revision.
                (output / "probe_stratified.json").write_text(
                    json_dumps(stratified) + "\n", encoding="utf-8"
                )
                write_stratified_csv(output / "probe_stratified.csv", stratified)
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
            if stratified is not None:
                print(f"wrote {output / 'probe_stratified.json'} and {output / 'probe_stratified.csv'}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
