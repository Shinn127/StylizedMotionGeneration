#!/usr/bin/env python3
"""Temporal decimation for BVH files.

The hierarchy and frame values are kept verbatim. For a factor of ``2`` this
turns a 120 FPS BVH into a 60 FPS BVH by keeping frames 0, 2, 4, ... and
doubling ``Frame Time``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_FRAMES_RE = re.compile(r"^(\s*Frames:\s*)(\d+)(\s*)$")
_FRAME_TIME_RE = re.compile(r"^(\s*Frame Time:\s*)([-+0-9.eE]+)(\s*)$")


def _replace_line_value(line: str, pattern: re.Pattern[str], value: str) -> str | None:
    """Replace a header value while preserving indentation, spacing, and EOL."""
    body = line.rstrip("\r\n")
    eol = line[len(body) :]
    match = pattern.match(body)
    if match is None:
        return None
    return f"{match.group(1)}{value}{match.group(3)}{eol}"


def downsample_bvh(input_path: Path, output_path: Path, factor: int = 2) -> tuple[int, int]:
    """Downsample one BVH and return ``(input_frames, output_frames)``."""
    if factor < 1:
        raise ValueError("factor must be a positive integer")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    frames_index: int | None = None
    frame_time_index: int | None = None
    input_frames: int | None = None
    frame_time: float | None = None

    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        frame_match = _FRAMES_RE.match(body)
        if frame_match:
            frames_index = index
            input_frames = int(body.split(":", 1)[1].strip())
            continue
        frame_time_match = _FRAME_TIME_RE.match(body)
        if frame_time_match:
            frame_time_index = index
            frame_time = float(frame_time_match.group(2))

    if frames_index is None or input_frames is None:
        raise ValueError(f"Missing Frames header in {input_path}")
    if frame_time_index is None or frame_time is None:
        raise ValueError(f"Missing Frame Time header in {input_path}")
    if frame_time_index <= frames_index:
        raise ValueError(f"Invalid MOTION header order in {input_path}")

    data_lines = [line for line in lines[frame_time_index + 1 :] if line.strip()]
    if len(data_lines) != input_frames:
        raise ValueError(
            f"{input_path}: Frames header says {input_frames}, found {len(data_lines)} frame rows"
        )

    selected = data_lines[::factor]
    output_frames = len(selected)
    output_lines = list(lines[: frame_time_index + 1])
    output_lines[frames_index] = _replace_line_value(
        output_lines[frames_index], _FRAMES_RE, str(output_frames)
    ) or output_lines[frames_index]
    output_lines[frame_time_index] = _replace_line_value(
        output_lines[frame_time_index], _FRAME_TIME_RE, f"{frame_time * factor:.15g}"
    ) or output_lines[frame_time_index]
    output_lines.extend(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(output_lines), encoding="utf-8")
    return input_frames, output_frames


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".bvh")
    raise FileNotFoundError(input_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A BVH file or a directory containing BVH files")
    parser.add_argument("output", type=Path, help="Output BVH file or directory")
    parser.add_argument("--factor", type=int, default=2, help="Keep every Nth frame (default: 2)")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing output files to be replaced")
    args = parser.parse_args()

    if args.factor < 1:
        parser.error("--factor must be a positive integer")

    files = _input_files(args.input)
    if not files:
        parser.error(f"No .bvh files found under {args.input}")
    single_file = args.input.is_file()
    if single_file and len(files) == 1:
        destinations = [(files[0], args.output)]
    else:
        destinations = [(source, args.output / source.relative_to(args.input)) for source in files]

    for source, destination in destinations:
        if destination.exists() and not args.overwrite:
            parser.error(f"Output exists (use --overwrite): {destination}")
        before, after = downsample_bvh(source, destination, args.factor)
        print(f"{source} -> {destination} ({before} frames -> {after} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
