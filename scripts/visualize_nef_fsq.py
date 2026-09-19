#!/usr/bin/env python
"""Visualize what an NEF-FSQ tokenizer actually does to motion.

Three views, all from one real window of a real store:

* character stills rendered through the production SOMA pipeline, for the source
  motion and for the motion decoded from its tokens, at a few frames;
* a side-by-side montage of those stills;
* a token map: the 40 coordinates x T token grid the encoder produced, with the
  13 stream boundaries drawn on it.

The reconstruction is ``decode(encode(x))`` -- exactly the path inference uses --
so the stills show the tokenizer's own round trip, not a model's fit.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/visualize_nef_fsq.py \
      --checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --feature-database data/processed/seed_soma_pruned_v4_ah \
      --split test --action "Basic Locomotion Neutral" \
      --output outputs/nef_fsq_viz/walk
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.anim.features import (  # noqa: E402
    bind_reference_positions,
    denormalize_motion_features,
    deserialize_motion_feature_stats,
    reconstruct_motion_state_from_features,
    stats_with_reference_skeleton,
)
from stylized_motion.data.packed_store import open_any_feature_store  # noqa: E402
from stylized_motion.data.sampling import SampleRequest  # noqa: E402
from stylized_motion.learning.nef_data import (  # noqa: E402
    model_space_window,
    store_normalized_window,
)
from stylized_motion.learning.nef_layout import NEF_STREAM_NAMES  # noqa: E402
from stylized_motion.learning.nef_probe import read_probe_window  # noqa: E402
from stylized_motion.learning.representation import (  # noqa: E402
    NEF_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import choose_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render source vs NEF-FSQ-decoded motion plus the token map."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--clip", type=int, default=None, help="Explicit clip index (default: first match).")
    parser.add_argument("--start", type=int, default=None, help="Window start (default: the clip's first full window).")
    parser.add_argument("--style", type=str, default=None, help="Only consider clips with this style label.")
    parser.add_argument("--action", type=str, default=None, help="Only consider clips with this action label.")
    parser.add_argument("--frames", type=int, default=64, help="Window length to visualize.")
    parser.add_argument(
        "--still-frames",
        type=int,
        nargs="*",
        default=None,
        help="Frame indices inside the window to render (default: four evenly spaced).",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=32,
        help="Frames per motion rendered into the side-by-side GIF (0 = no GIF).",
    )
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument(
        "--difference-gain",
        type=float,
        default=4.0,
        help="Amplification of the source/reconstruction pixel difference in the montage.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--pipeline", choices=["somaview", "genoview"], default="somaview")
    parser.add_argument(
        "--resources-root",
        type=Path,
        default=None,
        help="Viewer resources. Default: data/assets/somaview_quinn (the Quinn mesh bound to the "
        "SOMA skeleton); the older data/assets/somaview character is not the one the paper figures use.",
    )
    parser.add_argument(
        "--base-color-map",
        type=Path,
        default=None,
        help="Quinn base-color texture (default: <resources-root>/quinn_base_color.jpg if present).",
    )
    parser.add_argument(
        "--normal-map",
        type=Path,
        default=None,
        help="Quinn normal map (default: <resources-root>/quinn_normal.jpg if present).",
    )
    parser.add_argument("--skeleton", action="store_true", help="Draw the joint overlay into the stills.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--white-background", action="store_true", help="Paper-figure mode.")
    parser.add_argument("--camera-distance", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-render", action="store_true", help="Only the token map and the numbers.")
    parser.add_argument(
        "--sweep",
        type=int,
        default=0,
        help="Instead of one clip, score this many test clips stratified over actions and write "
        "sweep.csv/sweep.png (no renders).",
    )
    return parser


def sweep_clips(store, model, module, checkpoint, args, *, count: int) -> None:
    """Score ``count`` test clips spread over the action table, then plot the spread.

    Two rendered clips are an anecdote; this is the distribution a claim would rest on.
    """
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    feature_stats = dict(checkpoint["feature_stats"])
    model_stats, metadata = deserialize_motion_feature_stats(feature_stats)
    plain_stats = stats_with_reference_skeleton(model_stats, metadata["names"])
    mirrored_stats = stats_with_reference_skeleton(model_stats, metadata["names"], mirror=True)
    parents = metadata["parents"]
    names = metadata["names"]
    frames = int(args.frames)
    history = int(model.history_frames)
    device = next(model.parameters()).device

    action_ids = np.asarray(store.clip_action_id)
    action_names = list(store.source_action_names)
    splits = np.asarray(store.clip_split)
    per_action = max(1, int(np.ceil(count / max(len(action_names), 1))))
    chosen: list[int] = []
    for index in range(len(action_names)):
        matches = np.flatnonzero((action_ids == index) & (splits == 2))
        picked = 0
        for clip in matches:
            clip = int(clip)
            if int(store.clip_length[clip]) < frames:
                continue
            chosen.append(clip)
            picked += 1
            if picked >= per_action:
                break
        if len(chosen) >= count:
            break
    chosen = chosen[:count]

    rows: list[dict] = []
    shards: dict = {}
    for clip in chosen:
        length = int(store.clip_length[clip])
        offset = int(store.clip_offset[clip])
        start = offset + max(0, (length - frames) // 2)
        request = SampleRequest(
            shard_idx=int(store.clip_shard[clip]),
            target_start=int(start),
            target_frames=frames,
            variant_idx=int(clip),
        )
        with torch.no_grad():
            window = read_probe_window(store, request, history=history, shards=shards)
            motion = model_space_window(
                store_normalized_window(store, window), store, feature_stats
            ).to(device)
            tokens = model.encode_indices(motion[None])
            decoded = module.decode_from_indices(tokens)
            reencoded = model.encode_indices(decoded)
        fixed_rate = float((tokens[0, -frames:] == reencoded[0, -frames:]).float().mean())
        source_raw = np.asarray(store.read_window(clip, int(start), frames), dtype=np.float32)
        recon_raw = denormalize_motion_features(
            decoded[0, -frames:].detach().cpu().numpy(), model_stats
        ).astype(np.float32)
        mirrored = bool(getattr(store, "clip_mirror", np.zeros(1, dtype=bool))[clip])
        clip_stats = mirrored_stats if mirrored else plain_stats
        error = joint_error_stats(source_raw, recon_raw, clip_stats, parents, names)
        label = store.clip_label(clip)
        rows.append({
            "clip": int(clip),
            "clip_name": str(store.range_names[clip]),
            "mirror": mirrored,
            "action": str(label.get("action") or ""),
            "style": str(label.get("style") or ""),
            "token_fixed_rate": fixed_rate,
            "feature_abs_error_mean": float(np.abs(recon_raw - source_raw).mean()),
            "joint_error_mean_m": error["mean"],
            "joint_error_max_m": error["max"],
        })
        print(
            f"{len(rows)}/{len(chosen)} clip {clip} ({rows[-1]['action']}): "
            f"fixed={fixed_rate:.3f} joint mean={error['mean']:.4f} m",
            flush=True,
        )

    csv_path = output / "sweep.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in rows[0]) + "\n")
    plot_sweep(output / "sweep.png", rows, frames=frames)
    summary = {
        "checkpoint": str(args.checkpoint),
        "representation_id": model.representation_id,
        "clips": len(rows),
        "frames_per_clip": frames,
        "split": "test",
        "selection": f"first {per_action} clips per action with a full window, "
        f"{len({row['action'] for row in rows})} actions",
        "mirrored_clips": int(sum(1 for row in rows if row["mirror"])),
        "token_fixed_rate_mean": float(np.mean([row["token_fixed_rate"] for row in rows])),
        "joint_error_mean_m": float(np.mean([row["joint_error_mean_m"] for row in rows])),
        "joint_error_median_m": float(np.median([row["joint_error_mean_m"] for row in rows])),
        "joint_error_p90_m": float(np.percentile([row["joint_error_mean_m"] for row in rows], 90)),
        "joint_error_max_m": float(np.max([row["joint_error_max_m"] for row in rows])),
    }
    (output / "sweep.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def plot_sweep(path: Path, rows: list[dict], *, frames: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_action: dict[str, list[float]] = {}
    for row in rows:
        by_action.setdefault(str(row["action"]), []).append(float(row["joint_error_mean_m"]))
    actions = sorted(by_action, key=lambda name: -float(np.mean(by_action[name])))
    height = max(4.8, 0.34 * len(actions) + 2.2)
    figure, axes = plt.subplots(1, 2, figsize=(13, height), dpi=140)
    wrapped = [name if len(name) <= 14 else name.replace(" ", "\n", 1) for name in actions]
    axes[0].barh(
        wrapped[::-1],
        [float(np.mean(by_action[name])) for name in actions][::-1],
        color="#c8641e",
    )
    axes[0].set_xlabel("mean joint error (m)")
    axes[0].set_title(
        f"NEF-FSQ round trip, {frames}-frame test windows\naveraged per action", fontsize=10
    )
    axes[0].tick_params(labelsize=7)
    axes[1].scatter(
        [float(row["token_fixed_rate"]) for row in rows],
        [float(row["joint_error_mean_m"]) for row in rows],
        s=22, color="#20507a",
    )
    axes[1].set_xlabel("token fixed-point rate (re-encode of the decode)")
    axes[1].set_ylabel("mean joint error (m)")
    axes[1].set_title(f"{len(rows)} clips", fontsize=10)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def unmirrored_sibling(store: Any, clip: int) -> int | None:
    """The same take's unmirrored variant, when the store has one.

    A stored mirror (``clip_mirror``/``_M``) is a *reflection* of the take.  Its
    features need the mirrored reference skeleton to reconstruct, but the SOMA
    mesh's bind pose and skin weights are authored for the unmirrored chain --
    and the chain bones lie along the mirror axis, so posing a mirror on this
    mesh rolls them 180 degrees about their own axis and pinches the skin.
    Rendering the unmirrored variant of the same take shows the same motion
    without that artefact; the joint metrics keep the mirror handling (see
    ``stats_with_reference_skeleton(..., mirror=...)``).
    """
    if not bool(np.asarray(getattr(store, "clip_mirror", [False]))[int(clip)]):
        return None
    group = int(np.asarray(store.clip_source_group)[int(clip)])
    for row in np.flatnonzero(np.asarray(store.clip_source_group) == group):
        row = int(row)
        if row == int(clip):
            continue
        if bool(np.asarray(store.clip_mirror)[row]):
            continue
        if int(np.asarray(store.clip_length)[row]) != int(np.asarray(store.clip_length)[int(clip)]):
            continue
        return row
    return None


def pick_window(store, args) -> tuple[int, int]:
    """One clip and the start of its first full window, filtered by labels."""
    total = int(store.num_clips) if hasattr(store, "num_clips") else len(store.range_names)
    split_id = {"train": 0, "val": 1, "test": 2}[args.split]
    splits = np.asarray(store.clip_split if hasattr(store, "clip_split") else store.split_ids)
    for clip in range(total):
        if args.clip is not None and clip != int(args.clip):
            continue
        if int(splits[clip]) != split_id:
            continue
        label = store.clip_label(clip) if hasattr(store, "clip_label") else {}
        style = str(label.get("style") or "")
        action = str(label.get("action") or "")
        if args.style is not None and style != args.style:
            continue
        if args.action is not None and action != args.action:
            continue
        length = int(store.clip_length[clip]) if hasattr(store, "clip_length") else None
        if length is None or length < int(args.frames):
            continue
        offset = int(store.clip_offset[clip]) if hasattr(store, "clip_offset") else 0
        starts = np.arange(offset, offset + length - int(args.frames) + 1, int(args.frames))
        start = int(starts[0]) if len(starts) else offset
        if args.start is not None:
            start = int(args.start)
        return int(clip), start
    raise SystemExit(
        f"no {args.split!r} clip with a full {int(args.frames)}-frame window matched "
        f"style={args.style!r} action={args.action!r} clip={args.clip}"
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    checkpoint, model = load_representation_checkpoint(args.checkpoint, torch.device("cpu"))
    if model.family != NEF_FSQ_FAMILY:
        raise ValueError(f"--checkpoint must hold a {NEF_FSQ_FAMILY} model, got {model.family!r}")
    device = choose_device(args.device)
    model = model.to(device).eval()
    module = model.module
    history = int(model.history_frames)
    store = open_any_feature_store(args.feature_database)
    try:
        if int(args.sweep) > 0:
            sweep_clips(store, model, module, checkpoint, args, count=int(args.sweep))
            return
        clip, start = pick_window(store, args)
        requested_clip = int(clip)
        relative_start = int(start) - int(store.clip_offset[clip])
        sibling = unmirrored_sibling(store, clip)
        if sibling is not None:
            print(
                f"clip {clip} is a stored mirror ({store.range_names[clip]}); rendering its "
                f"unmirrored variant {sibling} ({store.range_names[sibling]}) -- the same take, "
                "because posing a mirror on this mesh twists the skin (bind pose and bone axes "
                "are authored for the unmirrored chain)",
                flush=True,
            )
            clip = int(sibling)
            # The sibling's frames are at a different shard offset; keep the same
            # position inside the take.
            start = int(store.clip_offset[clip]) + max(
                0, min(relative_start, int(store.clip_length[clip]) - int(args.frames))
            )
        label = store.clip_label(clip) if hasattr(store, "clip_label") else {}
        length = int(store.clip_length[clip])
        request = SampleRequest(
            shard_idx=int(store.clip_shard[clip]),
            target_start=int(start),
            target_frames=int(args.frames),
            variant_idx=int(clip),
        )
        feature_stats = dict(checkpoint["feature_stats"])
        model_stats, metadata = deserialize_motion_feature_stats(feature_stats)
        # The store's ref_pos is a mirror-averaged mean, not a skeleton; FK and the
        # renderer need the bind contract or the spine collapses.  Mirrored clips
        # (`clip_mirror`, the `_M` variants) additionally need the lateral offsets
        # negated, otherwise their torso folds over.
        clip_mirror = bool(getattr(store, "clip_mirror", np.zeros(1, dtype=bool))[clip])
        model_stats = stats_with_reference_skeleton(
            model_stats, metadata["names"], mirror=clip_mirror
        )
        reference_skeleton = (
            "bind_bvh_mirrored" if clip_mirror else "bind_bvh"
        ) if bind_reference_positions(metadata["names"]) is not None else "store_ref_pos"

        # Read the window in the encoder's own input space, then run the round trip.
        with torch.no_grad():
            window = read_probe_window(store, request, history=history)
            motion = model_space_window(
                store_normalized_window(store, window), store, feature_stats
            ).to(device)
            tokens = model.encode_indices(motion[None])
            decoded = module.decode_from_indices(tokens)
        tokens = tokens[0].detach().cpu()
        window_tokens = tokens[history:] if tokens.shape[0] > int(args.frames) else tokens
        window_tokens = window_tokens[: int(args.frames)]
        decoded_window = decoded[0, history:] if decoded.shape[1] > int(args.frames) else decoded[0]
        decoded_window = decoded_window[: int(args.frames)]

        # Both sides back to raw feature space for rendering.
        source_raw = np.asarray(
            store.read_window(clip, int(start), int(args.frames)), dtype=np.float32
        )
        recon_raw = denormalize_motion_features(
            decoded_window.detach().cpu().numpy(), model_stats
        ).astype(np.float32)

        round_trip = int((model.encode_indices(decoded)[0, history : history + int(args.frames)] == window_tokens).sum())
        total_tokens = int(window_tokens.numel())
        per_frame_error = np.abs(recon_raw[: int(args.frames)] - source_raw[: int(args.frames)])

        still_frames = args.still_frames
        if not still_frames:
            span = int(args.frames) - 1
            still_frames = [int(round(span * fraction)) for fraction in (0.0, 0.33, 0.66, 1.0)]
        still_frames = [int(value) for value in still_frames]

        summary = {
            "checkpoint": str(args.checkpoint),
            "representation_id": model.representation_id,
            "feature_database": str(args.feature_database),
            "reference_skeleton": reference_skeleton,
            "clip_mirror": clip_mirror,
            "clip": int(clip),
            "requested_clip": requested_clip,
            "unmirrored_variant_of": requested_clip if int(clip) != requested_clip else None,
            "clip_name": str(getattr(store, "range_names", [""])[clip]) if hasattr(store, "range_names") else "",
            "clip_length": length,
            "split": args.split,
            "style": str(label.get("style") or ""),
            "action": str(label.get("action") or ""),
            "window_start": int(start),
            "frames": int(args.frames),
            "history_frames": history,
            "num_coordinates": int(module.layout.num_coordinates),
            "num_levels": int(module.num_levels),
            "token_round_trip_accuracy": round_trip / max(total_tokens, 1),
            "token_round_trip_count": [round_trip, total_tokens],
            "feature_abs_error_mean": float(per_frame_error.mean()),
            "feature_abs_error_p95": float(np.percentile(per_frame_error, 95)),
            "still_frames": still_frames,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: summary[key] for key in (
            "clip", "clip_name", "style", "action", "window_start", "frames",
            "token_round_trip_accuracy", "feature_abs_error_mean", "feature_abs_error_p95",
        )}, indent=2))

        # Token map: 40 coordinates over time, colored by FSQ level.
        plot_token_map(
            output / "token_map.png",
            window_tokens.numpy(),
            module.layout,
            int(module.num_levels),
            title=(
                f"NEF-FSQ tokens  |  clip {clip} ({summary['style']} / {summary['action']})  "
                f"frames {start}..{start + int(args.frames)}"
            ),
        )
        # Reconstruction error per coordinate, next to the map: where the levels are off.
        plot_coordinate_error(output / "coordinate_error.png", per_frame_error)

        # And the same error in metres, through the same FK the renderer uses: a
        # number a motion reader can judge, not just "raw feature units".
        joint_error = joint_error_stats(
            source_raw, recon_raw, model_stats, metadata["parents"], metadata["names"]
        )
        plot_joint_error(
            output / "joint_error.png", joint_error["per_frame"], joint_error["per_joint"],
            joint_error["names"],
            root_per_frame=joint_error["root_per_frame"],
            local_per_frame=joint_error["local_per_frame"],
        )
        summary["joint_error_m"] = {
            "mean": joint_error["mean"],
            "p95": joint_error["p95"],
            "max": joint_error["max"],
            "per_joint_mean": {
                name: float(value)
                for name, value in zip(joint_error["names"], joint_error["per_joint"])
            },
        }

        np.save(output / "source_raw.npy", source_raw)
        np.save(output / "recon_raw.npy", recon_raw)
        np.save(output / "tokens.npy", window_tokens.numpy())

        if args.skip_render:
            print(f"wrote {output} (renders skipped)")
            return

        source_db = output / "database_source.npz"
        recon_db = output / "database_recon.npz"
        for path, features, name in (
            (source_db, source_raw, "source"),
            (recon_db, recon_raw, "recon"),
        ):
            database = database_from_features(
                features, model_stats, metadata, f"{summary['clip_name'] or clip}_{name}"
            )
            np.savez(path, **database)

        camera_target = fixed_camera_target(
            model_stats, source_raw, metadata["parents"], int(args.frames)
        )
        stills = render_stills(
            args, output, source_db, recon_db, still_frames, camera_target=camera_target
        )
        video_frames = sample_frames(int(args.frames), int(args.video_frames))
        video_stills = (
            render_stills(
                args, output, source_db, recon_db, video_frames, camera_target=camera_target
            )
            if video_frames and video_frames != still_frames
            else stills
        )
        montage(
            output / "compare_frames.png",
            stills,
            still_frames,
            gain=float(args.difference_gain),
            mirrored_clip=clip_mirror,
        )
        if video_frames:
            side_by_side_gif(
                output / "compare.gif",
                video_stills,
                video_frames,
                fps=int(args.video_fps),
            )
        summary["renders"] = stills
        summary["video_frames"] = video_frames
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    finally:
        store.close()


def sample_frames(frames: int, count: int) -> list[int]:
    """``count`` frame indices spread over the window, without duplicates."""
    if int(count) <= 0:
        return []
    if int(count) >= int(frames):
        return list(range(int(frames)))
    return sorted({int(round(value)) for value in np.linspace(0, int(frames) - 1, int(count))})


def joint_error_stats(
    source_raw: np.ndarray, recon_raw: np.ndarray, model_stats, parents: np.ndarray, names
) -> dict:
    """Per-joint world-space error in metres, from the same reconstruction the renderer uses."""
    source = reconstruct_motion_state_from_features(
        x=source_raw, stats=model_stats, parents=parents, normalized=False
    )
    recon = reconstruct_motion_state_from_features(
        x=recon_raw, stats=model_stats, parents=parents, normalized=False
    )
    source_global = np.asarray(source.global_positions)
    recon_global = np.asarray(recon.global_positions)
    delta = source_global - recon_global
    error = np.linalg.norm(delta, axis=-1)
    # Split the error, because they mean different things: a wrong root trajectory
    # is a drift a longer window would amplify, a wrong body-local error is a pose
    # the 40x9 alphabet could not represent on this clip.
    root = np.linalg.norm(delta[:, 0], axis=-1)
    local = np.linalg.norm(delta - delta[:, :1], axis=-1)
    return {
        "per_frame": error.mean(axis=1),
        "per_frame_max": error.max(axis=1),
        "per_joint": error.mean(axis=0),
        "root_per_frame": root,
        "local_per_frame": local.mean(axis=1),
        "mean": float(error.mean()),
        "p95": float(np.percentile(error, 95)),
        "max": float(error.max()),
        "root_mean": float(root.mean()),
        "root_end": float(root[-1]),
        "local_mean": float(local.mean()),
        "local_p95": float(np.percentile(local, 95)),
        "names": [str(name) for name in names],
    }


def plot_joint_error(
    path: Path,
    per_frame: np.ndarray,
    per_joint: np.ndarray,
    names,
    *,
    root_per_frame: np.ndarray | None = None,
    local_per_frame: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.0), dpi=140)
    axes[0].plot(per_frame, label="mean over joints", color="#20507a")
    if root_per_frame is not None:
        axes[0].plot(root_per_frame, label="root position only", color="#2f8f4e", linewidth=1.2)
    if local_per_frame is not None:
        axes[0].plot(local_per_frame, label="body-local (pose)", color="#b03a2e", linewidth=1.2)
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("world error (m)")
    axes[0].set_title("round-trip joint error per frame", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    order = np.argsort(per_joint)[::-1]
    axes[1].barh(
        [str(names[index]) for index in order][:14][::-1],
        [float(per_joint[index]) for index in order][:14][::-1],
        color="#c8641e",
    )
    axes[1].set_xlabel("mean world error (m)")
    axes[1].set_title("worst 14 joints", fontsize=10)
    axes[1].tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_token_map(path: Path, tokens: np.ndarray, layout, levels: int, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    coordinates = tokens.shape[1]
    figure, axes = plt.subplots(figsize=(length_figure(tokens.shape[0]), 7.5), dpi=140)
    cmap = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.05, 0.95, levels)))
    norm = BoundaryNorm(np.arange(-0.5, levels + 0.5, 1.0), cmap.N)
    image = axes.imshow(
        tokens.T, aspect="auto", origin="lower", cmap=cmap, norm=norm,
        interpolation="nearest",
    )
    stream_of = np.empty(coordinates, dtype=int)
    names = []
    for index, name in enumerate(layout.stream_order if hasattr(layout, "stream_order") else layout.stream_slices):
        sl = layout.stream_slices[name]
        stream_of[sl] = index
        names.append(name)
    boundaries = np.flatnonzero(np.diff(stream_of)) + 0.5
    for boundary in boundaries:
        axes.axhline(float(boundary), color="white", linewidth=0.8, alpha=0.7)
    centers = [float(np.mean(np.flatnonzero(stream_of == index))) for index in range(len(names))]
    axes.set_yticks(centers, labels=[name.replace("_", " ") for name in names], fontsize=7)
    axes.set_xlabel("frame")
    axes.set_title(title, fontsize=9)
    bar = figure.colorbar(image, ax=axes, ticks=range(levels), pad=0.01, fraction=0.03)
    bar.set_label("FSQ level")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def length_figure(frames: int) -> float:
    return float(min(max(frames / 12.0, 6.0), 16.0))


def plot_coordinate_error(path: Path, error: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(12, 3.2), dpi=140)
    axes.imshow(error.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    axes.set_xlabel("frame")
    axes.set_ylabel("coordinate")
    axes.set_title("per-coordinate round-trip error (raw feature units)", fontsize=9)
    figure.colorbar(axes.images[0], ax=axes, pad=0.01, fraction=0.03)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def render_stills(
    args, output: Path, source_db: Path, recon_db: Path, frames: list[int], *, camera_target=None
) -> dict:
    """One render per (motion, frame) through the production stills path.

    Each still is its own process: raylib windows do not survive being opened and
    closed inside a process that already holds a CUDA context, and a per-frame
    process also makes a crashing frame reportable instead of taking the run down.

    ``camera_target`` pins the camera in the world.  Without it the character scene
    follows the root, so a sub-millimetre reconstruction difference moves the whole
    view and the difference image shows the floor grid rather than the body.
    """
    import os

    resources_root, base_color, normal_map = resolve_viewer_assets(args)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    rendered: dict[str, list[str]] = {"source": [], "recon": []}
    for name, database in (("source", source_db), ("recon", recon_db)):
        for frame in frames:
            target = output / f"{name}_frame_{frame:03d}.png"
            command = [
                sys.executable, "-m", "stylized_motion.anim.render_stills",
                "--database", str(database),
                "--pipeline", args.pipeline,
                "--resources-root", str(resources_root),
                "--output", str(target),
                "--frame", str(frame),
                "--width", str(args.width),
                "--height", str(args.height),
                "--camera-distance", str(args.camera_distance),
            ]
            if base_color is not None:
                command += ["--base-color-map", str(base_color)]
            if normal_map is not None:
                command += ["--normal-map", str(normal_map)]
            if args.skeleton:
                command.append("--skeleton")
            if camera_target is not None:
                command += ["--camera-target", *[f"{value:.4f}" for value in camera_target]]
            if args.white_background:
                command.append("--white-background")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=600, check=False,
                cwd=str(REPO_ROOT), env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"render failed for {name} frame {frame}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
                )
            rendered[name].append(str(target))
    return rendered


def database_from_features(features, model_stats, metadata, range_name: str) -> dict:
    """The viewer database, built with the *bind* skeleton.

    ``genoview.build_database_from_feature_array`` would read ``ref_pos`` back out
    of a stats source file, which for this store is the mirror-averaged mean and
    collapses the spine; passing the corrected in-memory stats keeps the joint
    centres where the mesh's bind pose has them.
    """
    state = reconstruct_motion_state_from_features(
        x=np.asarray(features, dtype=np.float32),
        stats=model_stats,
        parents=np.asarray(metadata["parents"], dtype=np.int32),
        normalized=False,
    )
    frames = int(len(state.local_positions))
    return {
        "positions": state.local_positions.astype(np.float32),
        "rotations": state.local_rotations.astype(np.float32),
        "velocities": state.local_velocities.astype(np.float32),
        "angular_velocities": state.local_angular_velocities.astype(np.float32),
        "contacts": np.asarray(state.contacts > 0.5, dtype=np.uint8),
        "parents": np.asarray(metadata["parents"], dtype=np.int32),
        "names": np.asarray(metadata["names"], dtype=object),
        "range_starts": np.asarray([0], dtype=np.int32),
        "range_stops": np.asarray([frames], dtype=np.int32),
        "range_names": np.asarray([range_name], dtype=object),
        "range_mirror": np.asarray([False], dtype=bool),
        "joint_subset": np.asarray(str(metadata["joint_subset"]), dtype=object),
    }


def resolve_viewer_assets(args) -> tuple[Path, Path | None, Path | None]:
    """The viewer resources the figures use: the Quinn mesh bound to SOMA.

    The bare ``data/assets/somaview`` character is a different mesh with the
    viewer's default tint, and rendering it (as this script first did) shows the
    wrong body and an orange, untextured surface.
    """
    root = args.resources_root
    if root is None:
        root = (
            REPO_ROOT / "data" / "assets" / "somaview_quinn"
            if args.pipeline == "somaview"
            else REPO_ROOT / "data" / "assets" / "genoview"
        )
    root = Path(root)
    if not (root / ("SOMA.bin" if args.pipeline == "somaview" else "Geno.bin")).exists():
        raise SystemExit(f"{root} does not look like a {args.pipeline} resource directory")
    base_color = args.base_color_map or (root / "quinn_base_color.jpg")
    normal_map = args.normal_map or (root / "quinn_normal.jpg")
    return (
        root,
        base_color if Path(base_color).exists() else None,
        normal_map if Path(normal_map).exists() else None,
    )


def fixed_camera_target(
    model_stats, source_features: np.ndarray, parents: np.ndarray, frames: int
) -> tuple:
    """A world-fixed camera target for the clip: mean root x/z, mean body height.

    Computed from *global* (post-FK) positions: the SOMA rig puts the spine chain
    along the local x axis, so a "height" read off the local positions would aim
    the camera at the knees.  Fixing the target keeps the difference image about
    the body instead of about camera parallax.
    """
    state = reconstruct_motion_state_from_features(
        x=source_features, stats=model_stats, parents=parents, normalized=False
    )
    global_positions = np.asarray(state.global_positions)
    roots = global_positions[:, 0]
    height = float(np.mean(global_positions[..., 1]))
    return (float(roots[:, 0].mean()), height, float(roots[:, 2].mean()))


def _label_font(size: int = 18):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _character_crop(images) -> tuple[int, int, int, int]:
    """Bounding box around everything that is not the flat background.

    The production camera leaves a large empty band above the character; keeping
    it would make the comparison panels mostly air.
    """
    boxes = []
    for image in images:
        array = np.asarray(image.convert("RGB"), dtype=np.int16)
        background = array[2, 2]  # top-left corner is sky/floor in every scene mode
        mask = (np.abs(array - background).sum(axis=2) > 24)
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        if len(rows) and len(columns):
            boxes.append((int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1))
    if not boxes:
        return (0, 0, images[0].width, images[0].height)
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    margin = 12
    return (
        max(left - margin, 0),
        max(top - margin, 0),
        min(right + margin, images[0].width),
        min(bottom + margin, images[0].height),
    )


def montage(
    path: Path, stills: dict, frames: list[int], *, gain: float = 8.0, mirrored_clip: bool = False
) -> None:
    """Source row, reconstruction row and an amplified difference row."""
    from PIL import Image, ImageChops, ImageDraw

    source = [Image.open(item).convert("RGB") for item in stills["source"]]
    recon = [Image.open(item).convert("RGB") for item in stills["recon"]]
    box = _character_crop(source + recon)
    panels = {
        "source": [image.crop(box) for image in source],
        "NEF-FSQ round trip": [image.crop(box) for image in recon],
        "difference x{:g}".format(gain): [
            ImageChops.difference(a, b).point(lambda value: min(255, int(value * gain)))
            for a, b in zip(source, recon)
        ],
    }
    width, height = panels["source"][0].size
    label_height = 30
    rows = list(panels)
    canvas = Image.new("RGB", (width * len(frames), (height + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(18)
    for row, name in enumerate(rows):
        top = row * (height + label_height)
        for column, frame in enumerate(frames):
            x = column * width
            canvas.paste(panels[name][column], (x, top + label_height))
            suffix = "  [mirrored clip]" if mirrored_clip else ""
        draw.text((x + 10, top + 6), f"{name}  |  frame {frame}{suffix}", fill="black", font=font)
    canvas.save(path)


def side_by_side_gif(path: Path, stills: dict, frames: list[int], *, fps: int) -> None:
    """Source | reconstruction animated side by side (GIF: no ffmpeg needed)."""
    from PIL import Image, ImageDraw

    source = [Image.open(item).convert("RGB") for item in stills["source"]]
    recon = [Image.open(item).convert("RGB") for item in stills["recon"]]
    box = _character_crop(source + recon)
    pairs = [(a.crop(box), b.crop(box)) for a, b in zip(source, recon)]
    width, height = pairs[0][0].size
    label_height = 26
    font = _label_font(16)
    composites = []
    for (a, b), frame in zip(pairs, frames):
        panel = Image.new("RGB", (width * 2, height + label_height), "white")
        panel.paste(a, (0, label_height))
        panel.paste(b, (width, label_height))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 5), f"source  frame {frame}", fill="black", font=font)
        draw.text((width + 10, 5), f"NEF-FSQ round trip  frame {frame}", fill="black", font=font)
        composites.append(panel)
    composites[0].save(
        path,
        save_all=True,
        append_images=composites[1:],
        duration=int(round(1000 / max(int(fps), 1))),
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
