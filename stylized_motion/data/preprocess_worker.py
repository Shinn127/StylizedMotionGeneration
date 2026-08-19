"""Importable process-pool workers for data preprocessing."""

from __future__ import annotations

from pathlib import Path

from stylized_motion.anim import bvh


def process_motion_pair(task):
    """Load one BVH and return its original and mirrored processed motions."""
    # Import lazily to avoid a preprocess <-> worker import cycle. The worker
    # function itself remains defined in this importable module for pickle.
    from .preprocess import _process_motion_data

    path, prune_ends_and_fingers = task
    path = Path(path)
    bvh_data = bvh.load(path.as_posix())
    motions = []
    for mirror in (False, True):
        motion = _process_motion_data(
            bvh_data,
            mirror,
            prune_ends_and_fingers=prune_ends_and_fingers,
        )
        motions.append((mirror, motion))
    return path, motions


__all__ = ["process_motion_pair"]
