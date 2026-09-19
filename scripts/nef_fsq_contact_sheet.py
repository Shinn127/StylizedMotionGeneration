#!/usr/bin/env python
"""One figure that puts several ``visualize_nef_fsq.py`` runs next to each other.

Each row is one sequence: the source still, the NEF-FSQ round trip at the same
frame, and the amplified pixel difference.  The caption carries the numbers from
that run's ``summary.json`` so the figure and the metrics cannot drift apart.

Every row is cropped to its own character and scaled to the same panel size, so a
lying-down clip reads as well as a standing one.  Row-to-row size is therefore
not a scale comparison; the per-run ``compare_frames.png`` keeps the raw framing.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/nef_fsq_contact_sheet.py \
      --root outputs/nef_fsq_viz --output outputs/nef_fsq_viz/contact_sheet.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_nef_fsq import _character_crop, _label_font  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contact sheet over several visualization runs.")
    parser.add_argument("--root", type=Path, default=Path("outputs/nef_fsq_viz"))
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directory names under --root (default: every one with a summary.json).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=2,
        help="Which of the run's still_frames to show (default: the third one).",
    )
    parser.add_argument("--gain", type=float, default=4.0, help="Amplification of the difference panel.")
    parser.add_argument(
        "--panel-size", type=int, default=420, help="Edge length of one panel after cropping and scaling."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_row(directory: Path, frame_index: int) -> dict | None:
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = list(summary.get("still_frames") or [])
    if not frames:
        return None
    frame = int(frames[min(frame_index, len(frames) - 1)])
    source = directory / f"source_frame_{frame:03d}.png"
    recon = directory / f"recon_frame_{frame:03d}.png"
    if not source.exists() or not recon.exists():
        return None
    return {
        "name": directory.name,
        "summary": summary,
        "frame": frame,
        "source": Image.open(source).convert("RGB"),
        "recon": Image.open(recon).convert("RGB"),
    }


def caption(row: dict) -> str:
    summary = row["summary"]
    error = summary.get("joint_error_m") or {}
    name = summary.get("clip_name") or summary.get("clip")
    mirror = ""
    if summary.get("unmirrored_variant_of") is not None:
        mirror = f"  [shown unmirrored: {summary['requested_clip']} -> {summary['clip']}]"
    return (
        f"{row['name']}  |  {name}  |  f{row['frame']}"
        f"  |  token fixed {summary.get('token_round_trip_accuracy', float('nan')):.3f}"
        f"  |  joint err {error.get('mean', float('nan')):.3f} m"
        f" (p95 {error.get('p95', float('nan')):.3f}, max {error.get('max', float('nan')):.3f})"
        f"{mirror}"
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root: Path = args.root
    names = args.runs if args.runs is not None else sorted(p.name for p in root.iterdir() if p.is_dir())

    rows = [row for name in names if (row := load_row(root / name, int(args.frame_index))) is not None]
    if not rows:
        raise SystemExit(f"no visualization runs with stills under {root}")

    # One crop per row (that sequence's own character box), then a fixed panel
    # size, so a lying-down clip is as readable as a standing one.
    panel_size = int(args.panel_size)
    panels = []
    for row in rows:
        difference = ImageChops.difference(row["source"], row["recon"]).point(
            lambda value: min(255, int(value * float(args.gain)))
        )
        box = _character_crop([row["source"], row["recon"]])
        panels.append(
            [
                image.crop(box).resize((panel_size, panel_size), Image.LANCZOS)
                for image in (row["source"], row["recon"], difference)
            ]
        )

    grid_width = panel_size * 3
    header_height = 30
    label_height = 30
    column_titles = ("source", "NEF-FSQ round trip", f"difference x{float(args.gain):g}")
    font = _label_font(18)
    captions = [caption(row) for row in rows]

    # The panels are square; a caption may be wider than the three of them, so the
    # canvas is the wider of the two and the grid stays centred.
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    caption_width = max(int(scratch.textlength(text, font=font)) for text in captions)
    canvas_width = max(grid_width, caption_width + 20)
    left = (canvas_width - grid_width) // 2
    canvas = Image.new(
        "RGB", (canvas_width, header_height + (panel_size + label_height) * len(rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(column_titles):
        text_width = draw.textlength(title, font=font)
        draw.text(
            (left + column * panel_size + (panel_size - text_width) / 2, 6),
            title,
            fill="black",
            font=font,
        )
    for row_index, text in enumerate(captions):
        top = header_height + row_index * (panel_size + label_height)
        for column, panel in enumerate(panels[row_index]):
            canvas.paste(panel, (left + column * panel_size, top + label_height))
        draw.text((10, top + 6), text, fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output} ({len(rows)} sequences, frame index {int(args.frame_index)})")


if __name__ == "__main__":
    main()
