#!/usr/bin/env python
"""Render a ``generate_mts_operator.py`` output directory as character stills.

``generate_mts_operator.py`` writes the decoded feature windows of a generation
(``reference_motion.npy`` = the source clip's own decode, ``baseline_motion.npy``
= the frozen transport's argmax decode, ``motion.npy`` = the styled draws) but no
pictures.  This script turns those windows into the same Quinn stills the NEF-FSQ
tokenizer check uses: per-frame ``database_*`` npz -> the production
``stylized_motion.anim.render_stills`` path, plus one montage with the edit
difference and a ``summary.json`` with the numbers.

The motions are in the tokenizer's *decode* space (the normalized feature space
training uses), so they are denormalized with the tokenizer checkpoint's own
``feature_stats``, and the viewer database is built with the bind skeleton -- the
store's ``ref_pos`` is the mirror-averaged mean and collapses the spine (see
``docs/nef_fsq_visual_check.md``).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/render_mts_generation.py \
      --generation outputs/mts_generation_demo/styleid_injured_torso \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --label-styled "styled: injured torso" \
      --output outputs/mts_generation_demo/styleid_injured_torso_renders
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from stylized_motion.anim.features import (  # noqa: E402
    bind_reference_positions,
    denormalize_motion_features,
    deserialize_motion_feature_stats,
    stats_with_reference_skeleton,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402

from visualize_nef_fsq import (  # noqa: E402
    _character_crop,
    _label_font,
    database_from_features,
    fixed_camera_target,
    joint_error_stats,
    resolve_viewer_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a generation directory as character stills.")
    parser.add_argument("--generation", type=Path, default=None,
                        help="A generate_mts_operator.py output directory (motion.npy et al.).")
    parser.add_argument("--record-npz", type=Path, default=None,
                        help="Or a benchmark_mts_editing.py --save-motions record "
                        "(motions/caseNNN_<arm>_<region>.npz with source/base/styled).")
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--draw", type=int, default=0, help="Which styled sample to render (default 0).")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="The generated window is a stored mirror (`_M`): FK needs the mirrored bind "
        "skeleton. --record-npz reads this from the record's meta instead.",
    )
    parser.add_argument("--still-frames", type=int, nargs="*", default=None,
                        help="Frame indices to render (default: four evenly spaced).")
    parser.add_argument("--difference-gain", type=float, default=4.0)
    parser.add_argument("--label-source", default="source (reference clip)")
    parser.add_argument("--label-base", default="base (transport decode)")
    parser.add_argument("--label-styled", default="styled draw")
    parser.add_argument("--pipeline", choices=["somaview", "genoview"], default="somaview")
    parser.add_argument("--resources-root", type=Path, default=None)
    parser.add_argument("--base-color-map", type=Path, default=None)
    parser.add_argument("--normal-map", type=Path, default=None)
    parser.add_argument("--skeleton", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_record_npz(path: Path, draw: int) -> tuple[dict[str, np.ndarray], dict]:
    """A benchmark ``--save-motions`` record: one decoded window per role, plus its meta."""
    payload = np.load(path, allow_pickle=False)
    missing = [name for name in ("source", "base", "styled") if name not in payload]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")
    styled = np.atleast_2d(payload["styled"])
    meta: dict = {}
    if "meta" in payload:
        meta = json.loads(str(payload["meta"]))
    if styled.ndim == 3 and styled.shape[0] > 1:
        if not 0 <= int(draw) < int(styled.shape[0]):
            raise SystemExit(f"--draw {draw} out of range: {int(styled.shape[0])} samples")
        styled = styled[int(draw)]
    motions = {
        "source": np.asarray(payload["source"], dtype=np.float32),
        "base": np.asarray(payload["base"], dtype=np.float32),
        "styled": np.asarray(styled, dtype=np.float32),
    }
    return motions, meta


def load_generation_motions(directory: Path, draw: int) -> dict[str, np.ndarray]:
    """source / base / styled feature windows, as the generator wrote them."""
    reference = np.load(directory / "reference_motion.npy")
    baseline = np.load(directory / "baseline_motion.npy")
    styled = np.atleast_2d(np.load(directory / "motion.npy"))
    if styled.ndim != 3:
        raise SystemExit(f"{directory / 'motion.npy'} has shape {styled.shape}, expected [samples, T, D]")
    if not 0 <= int(draw) < int(styled.shape[0]):
        raise SystemExit(f"--draw {draw} out of range: {int(styled.shape[0])} samples")
    return {
        "source": np.asarray(reference, dtype=np.float32),
        "base": np.asarray(baseline, dtype=np.float32),
        "styled": np.asarray(styled[int(draw)], dtype=np.float32),
    }


def render_named_stills(
    named_databases: list[tuple[str, Path]], frames: list[int], args, output: Path, camera_target
) -> dict[str, list[str]]:
    """One subprocess render per (motion, frame), like the tokenizer check does."""
    resources_root, base_color, normal_map = resolve_viewer_assets(args)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    rendered: dict[str, list[str]] = {}
    for name, database in named_databases:
        rendered[name] = []
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
                    f"render failed for {name} frame {frame}:\n"
                    f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
                )
            rendered[name].append(str(target))
    return rendered


def montage_rows(path: Path, rows: list[tuple[str, list[Path]]], *, gain: float) -> None:
    """Named still rows plus an amplified difference row between consecutive rows."""
    images = {name: [Image.open(item).convert("RGB") for item in stills] for name, stills in rows}
    box = _character_crop([image for stills in images.values() for image in stills])
    width, height = images[rows[0][0]][0].crop(box).size
    label_height = 30
    panels: list[tuple[str, list[Image.Image]]] = [
        (name, [image.crop(box) for image in stills]) for name, stills in images.items()
    ]
    for (upper_name, _), (lower_name, _) in zip(rows, rows[1:]):
        panels.append(
            (
                f"difference {lower_name} - {upper_name}  x{gain:g}",
                [
                    ImageChops.difference(a, b).point(lambda value: min(255, int(value * gain)))
                    for a, b in zip(images[upper_name], images[lower_name])
                ],
            )
        )
    canvas = Image.new(
        "RGB", (width * len(rows[0][1]), (height + label_height) * len(panels)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = _label_font(18)
    for row_index, (name, stills) in enumerate(panels):
        top = row_index * (height + label_height)
        for column, image in enumerate(stills):
            canvas.paste(image, (column * width, top + label_height))
        draw.text((10, top + 6), name, fill="black", font=font)
    canvas.save(path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (args.generation is None) == (args.record_npz is None):
        raise SystemExit("pass exactly one of --generation or --record-npz")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    checkpoint, _model = load_representation_checkpoint(args.tokenizer_checkpoint, torch_device())
    model_stats, metadata = deserialize_motion_feature_stats(dict(checkpoint["feature_stats"]))
    parents = np.asarray(metadata["parents"], dtype=np.int32)
    names = [str(name) for name in metadata["names"]]

    if args.record_npz is not None:
        motions, record_meta = load_record_npz(args.record_npz, int(args.draw))
    else:
        motions = load_generation_motions(args.generation, int(args.draw))
        record_meta = {}
    # The stats' ref_pos is the *dataset mean* of local positions, not a skeleton:
    # the packed store mixes mirrored clips, whose lateral offsets cancel, so the
    # spine chain collapses onto the chest and the mesh skins through itself.  FK,
    # the camera target and the error numbers all need the bind skeleton (mirrored
    # for a stored `_M` window), exactly like the tokenizer check does.
    clip_mirror = bool(record_meta.get("mirror", False)) or bool(args.mirror)
    model_stats = stats_with_reference_skeleton(model_stats, names, mirror=clip_mirror)
    reference_skeleton = (
        "bind_bvh_mirrored" if clip_mirror else "bind_bvh"
    ) if bind_reference_positions(names) is not None else "store_ref_pos"
    raw = {name: denormalize_motion_features(window, model_stats).astype(np.float32)
           for name, window in motions.items()}
    frames = int(raw["source"].shape[0])

    still_frames = args.still_frames
    if not still_frames:
        span = frames - 1
        still_frames = [int(round(span * fraction)) for fraction in (0.0, 0.33, 0.66, 1.0)]
    still_frames = [int(value) for value in still_frames]

    named = {}
    run_name = args.generation.name if args.generation is not None else args.record_npz.stem
    for name, window in raw.items():
        database = output / f"database_{name}.npz"
        np.savez(
            database,
            **database_from_features(window, model_stats, metadata, f"{run_name}_{name}"),
        )
        named[name] = database

    camera_target = fixed_camera_target(model_stats, raw["source"], parents, frames)
    stills = render_named_stills(
        [(name, named[name]) for name in ("source", "base", "styled")],
        still_frames, args, output, camera_target,
    )
    still_paths = {name: [Path(item) for item in paths] for name, paths in stills.items()}

    montage_rows(
        output / "compare_generation.png",
        [
            (args.label_source, still_paths["source"]),
            (args.label_base, still_paths["base"]),
            (args.label_styled, still_paths["styled"]),
        ],
        gain=float(args.difference_gain),
    )

    def error_stats(upper: str, lower: str) -> dict:
        stats = joint_error_stats(raw[upper], raw[lower], model_stats, parents, names)
        return {
            "mean_m": stats["mean"], "p95_m": stats["p95"], "max_m": stats["max"],
            "root_mean_m": stats["root_mean"], "local_mean_m": stats["local_mean"],
        }

    def side_stats(upper: str, lower: str) -> dict:
        """Which *anatomical side* the difference lives on, by joint name.

        The camera views the character's front, so on screen the character's left
        arm appears on the image's right -- laterality is easy to misread from the
        picture; these numbers are the ground truth.
        """
        stats = joint_error_stats(raw[upper], raw[lower], model_stats, parents, names)
        per_joint = dict(zip(stats["names"], stats["per_joint"], strict=True))
        sides = {"left_m": [], "right_m": [], "mid_m": []}
        for name, value in per_joint.items():
            key = "left_m" if name.startswith("Left") else (
                "right_m" if name.startswith("Right") else "mid_m"
            )
            sides[key].append(float(value))
        return {
            "left_m": float(np.mean(sides["left_m"])),
            "right_m": float(np.mean(sides["right_m"])),
            "mid_m": float(np.mean(sides["mid_m"])),
            "top_movers": {
                name: round(float(value), 4)
                for name, value in sorted(per_joint.items(), key=lambda item: -item[1])[:4]
                if value > 0
            },
        }

    if args.generation is not None:
        generation = json.loads((args.generation / "generation.json").read_text(encoding="utf-8"))
    else:
        with np.load(args.record_npz, allow_pickle=False) as payload:
            generation = {"record_npz": str(args.record_npz), "meta": json.loads(str(payload["meta"]))}
    movers = side_stats("base", "styled")["top_movers"]
    summary = {
        "run": run_name,
        "reference_skeleton": reference_skeleton,
        "clip_mirror": clip_mirror,
        "generation_directory": None if args.generation is None else str(args.generation),
        "tokenizer_checkpoint": str(args.tokenizer_checkpoint),
        "draw": int(args.draw),
        "still_frames": still_frames,
        "base_to_styled_joint_error_m": error_stats("base", "styled"),
        "base_to_styled_by_side": side_stats("base", "styled"),
        "source_to_base_joint_error_m": error_stats("source", "base"),
        "source_to_base_by_side": side_stats("source", "base"),
        "generation": generation,
        "renders": stills,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Caption the montage with the joints that actually moved, so "which hand" can
    # be read off the figure instead of guessed from the camera angle.
    legend = "base→styled movers: " + ", ".join(
        f"{name} {value:.3f} m" for name, value in movers.items()
    ) if movers else "base→styled: no joint moved"
    annotate_bottom(output / "compare_generation.png", legend)
    print(f"wrote {output} (montage {output / 'compare_generation.png'}, frames {still_frames})")
    print(f"   {legend}")


def annotate_bottom(path: Path, text: str) -> None:
    """Append one white caption strip under the montage without rebuilding it."""
    image = Image.open(path)
    canvas = Image.new("RGB", (image.width, image.height + 36), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, image.height + 8), text, fill="black", font=_label_font(18))
    canvas.save(path)


def torch_device():
    import torch

    return torch.device("cpu")


if __name__ == "__main__":
    main()
