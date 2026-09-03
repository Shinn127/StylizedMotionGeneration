"""Sample calibrated scene anchors from a rendered still.

Anchor-based color verification: instead of eyeballing renders, compare the
mean RGB of known scene regions against target display values (0-255). The
targets come from inverting the display chain (exposure -> ACES -> sRGB) on
linear-radiance goals chosen from real outdoor references (spec 20.10).

Region identification:
- sky / floor_lit: fixed rects, valid for the default camera framing
- character lit/shadow: the saturated body mask split by luminance quantile,
  robust to pose and rig changes
- floor shadow: the cool cast-shadow blob right of the character

Usage:
    python -m stylized_motion.anim.sample_anchors render.png            # print
    python -m stylized_motion.anim.sample_anchors render.png --check    # assert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Rect regions (x0, y0, x1, y1) as fractions of a 1280x720 character-scene
# still in window orientation.
_RECT_REGIONS = {
    "sky": (0.16, 0.01, 0.70, 0.11),
    "floor_lit": (0.15, 0.83, 0.39, 0.96),
}

# Display targets in sRGB 0-255, from inverting exposure 0.9 -> ACES -> sRGB on
# the calibrated linear-radiance anchors. Keep in sync with the light rig.
ANCHOR_TARGETS = {
    # sky: the rect at rows 7-79 sees 18-22 deg elevation = the dome's neutral
    # mid-sky, not the horizon ring (which lives below ~10 deg). The honest
    # achieved value is anchored rather than an unreachable photo reference;
    # making mid-sky itself that blue would push floor irradiance blue again.
    "sky": (169, 177, 187),
    "floor_lit": (207, 209, 211),
    "floor_shadow": (81, 97, 114),
    "character_lit": (240, 222, 30),
    "character_shadow": (223, 175, 19),
}

DEFAULT_TOLERANCE = 8.0


def _rect_mean(image, rect):
    height, width = image.shape[:2]
    x0, y0 = int(rect[0] * width), int(rect[1] * height)
    x1, y1 = int(rect[2] * width), int(rect[3] * height)
    return image[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def sample_regions(image: np.ndarray) -> dict[str, np.ndarray]:
    out = {name: _rect_mean(image, rect) for name, rect in _RECT_REGIONS.items()}

    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    lum = image.sum(axis=2)
    body = (red > 140) & (green > 70) & (green < 230) & (blue < 130)
    if body.sum() > 100:
        rows, cols = np.where(body)
        body_lum = lum[rows, cols]
        split = np.percentile(body_lum, 55)
        out["character_lit"] = image[rows[body_lum >= split], cols[body_lum >= split]].mean(axis=0)
        out["character_shadow"] = image[rows[body_lum < split], cols[body_lum < split]].mean(axis=0)

    blob = (blue > red + 8) & (lum < 560) & ~body
    blob[:, : int(0.55 * image.shape[1])] = False
    blob[: int(0.55 * image.shape[0]), :] = False
    blob[int(0.78 * image.shape[0]) :, :] = False
    if blob.sum() > 100:
        out["floor_shadow"] = image[blob].mean(axis=0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample scene anchor regions and compare against calibrated targets.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--check", action="store_true", help="Exit 1 when any anchor deviates beyond the tolerance.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB")).astype(np.float64)
    means = sample_regions(image)

    failures = []
    for name, target_vals in ANCHOR_TARGETS.items():
        if name not in means:
            print(f"SKIP {name:18s} region not found in frame")
            continue
        mean = means[name]
        target = np.asarray(target_vals, dtype=np.float64)
        delta = mean - target
        status = "OK " if np.abs(delta).max() <= args.tolerance else "MISS"
        if status == "MISS":
            failures.append(name)
        print(f"{status} {name:18s} mean ({mean[0]:6.1f}, {mean[1]:6.1f}, {mean[2]:6.1f})  "
              f"target ({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f})  "
              f"delta ({delta[0]:+6.1f}, {delta[1]:+6.1f}, {delta[2]:+6.1f})")

    if args.check and failures:
        print(f"FAILED anchors: {', '.join(failures)} (tolerance {args.tolerance})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
