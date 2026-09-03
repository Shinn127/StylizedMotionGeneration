"""Pixel comparison for baseline stills.

Compares a current render against a committed baseline PNG and fails (exit
code 1) when the mean per-channel difference exceeds the threshold. Used to
keep the ``docs/assets/pbr_baseline`` set honest across rendering changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def compare_images(reference: Path, current: Path, threshold: float) -> tuple[float, float, bool]:
    from PIL import Image

    a = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(current).convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ValueError(f"Image sizes differ: {reference} {a.shape} vs {current} {b.shape}")
    diff = np.abs(a - b)
    mean = float(diff.mean())
    return mean, float(diff.max()), mean > threshold


def main():
    parser = argparse.ArgumentParser(description="Compare a still against a baseline PNG.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.02, help="Mean abs diff (0-1) above which comparison fails.")
    args = parser.parse_args()

    mean, peak, exceeded = compare_images(args.reference, args.current, args.threshold)
    status = "FAIL" if exceeded else "OK"
    print(f"{status}: mean diff {mean:.5f} (threshold {args.threshold}), peak {peak:.3f}")
    if exceeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
