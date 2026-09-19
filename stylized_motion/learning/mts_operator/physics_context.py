"""One physical measurement context: bind asset, units, joint order, mirror rule.

Physical numbers (root speed, foot contact, foot slide, FK joint error) are only
comparable when the skeleton, its units, the joint order and the mirror handling
are the same everywhere.  The first round measured FK against the store's
``ref_pos`` -- the *dataset mean* of the local positions -- and mirroring cancels
every constant axis-aligned offset, so the spine collapsed and the mirrored clips
folded over their own torso.  Every evaluator now takes its context from here, and
the context records what it used:

* the bind asset's path and SHA-256 (``data/assets/somaview/SOMA_bind.bvh`` by
  default) and its unit scale (the SOMA bind file is authored in centimetres);
* the joint order as a digest, so two "same" contexts can be compared;
* the mirror rule, applied by :func:`stylized_motion.anim.features.mirror_partner`
  (swap the side *and* negate the lateral offset -- a blanket x-flip is a
  different skeleton);
* ``PHYSICAL_METRIC_VERSION``, bumped when any of the above changes meaning.

Features are denormalized with the *store's* offset/scale before anything metric
is derived: a normalized channel is not a metre, and a normalized "contact" is not
a contact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stylized_motion.anim.features import (
    SOMA_BIND_BVH,
    SOMA_BIND_UNIT_SCALE,
    MotionFeatureStats,
    deserialize_motion_feature_stats,
    stats_with_reference_skeleton,
)

from .checkpoint import file_sha256

#: Bump when the skeleton source, the unit scale, the joint order or the mirror
#: rule changes meaning: results measured under two versions are not comparable.
PHYSICAL_METRIC_VERSION = 1


def joint_order_digest(names: Any, parents: Any) -> str:
    """A digest of the joint names *and* their parent indices, in order."""
    digest = hashlib.sha256()
    for index, (name, parent) in enumerate(zip(names, parents)):
        digest.update(f"{index}:{name}:{int(parent)};".encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicsContext:
    """Denormalization plus FK, for one skeleton and both mirror groups."""

    stats: MotionFeatureStats
    stats_mirrored: MotionFeatureStats
    names: tuple[str, ...]
    parents: tuple[int, ...]
    dt: float = 1.0 / 60.0
    contact_threshold: float = 0.15
    bind_asset: str | None = None
    bind_asset_sha256: str | None = None
    unit_scale: float = SOMA_BIND_UNIT_SCALE
    joint_order_sha256: str = ""
    skeleton_source: str = "stored_ref_pos"
    #: Free-form provenance from the checkpoint (window frame count, feature
    #: schema hash, ...), carried so a reader can name the exact measurement.
    provenance: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if int(self.stats.ref_pos.shape[0]) != len(self.names):
            raise ValueError("The reference skeleton and the joint names disagree on length")
        if len(self.parents) != len(self.names):
            raise ValueError("parents and names must have the same length")
        if not self.joint_order_sha256:
            object.__setattr__(self, "joint_order_sha256", joint_order_digest(self.names, self.parents))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_feature_stats(
        cls,
        feature_stats: Mapping[str, Any],
        *,
        dt: float = 1.0 / 60.0,
        contact_threshold: float = 0.15,
        bind_asset: str | Path | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "PhysicsContext":
        """The context for one checkpoint's feature statistics.

        A payload the bind contract cannot place (synthetic fixtures) keeps the
        stored ``ref_pos`` and says so in ``skeleton_source``; it never silently
        pretends to be the bind skeleton.
        """
        payload = dict(feature_stats)
        stats, metadata = deserialize_motion_feature_stats(payload)
        names = tuple(str(name) for name in metadata["names"])
        parents = tuple(int(value) for value in np.asarray(metadata["parents"]).tolist())
        mirror_free: MotionFeatureStats | None = None
        try:
            candidate = stats_with_reference_skeleton(stats, names)
        except (KeyError, ValueError, TypeError):
            candidate = stats
        if candidate is not stats:
            mirror_free = candidate
        if mirror_free is None:
            resolved = stats
            source = "stored_ref_pos"
            asset_path = None
            asset_sha = None
            unit_scale = float(SOMA_BIND_UNIT_SCALE)
        else:
            resolved = mirror_free
            source = "bind_asset"
            path = Path(bind_asset) if bind_asset is not None else SOMA_BIND_BVH
            asset_path = str(path)
            asset_sha = file_sha256(path) if path.exists() else None
            unit_scale = float(SOMA_BIND_UNIT_SCALE)
        mirrored = _mirrored_stats(resolved, names)
        return cls(
            stats=resolved,
            stats_mirrored=mirrored,
            names=names,
            parents=parents,
            dt=float(dt),
            contact_threshold=float(contact_threshold),
            bind_asset=asset_path,
            bind_asset_sha256=asset_sha,
            unit_scale=unit_scale,
            skeleton_source=source,
            provenance=provenance,
        )

    # -- use ---------------------------------------------------------------
    def stats_for(self, *, mirror: bool) -> MotionFeatureStats:
        return self.stats_mirrored if mirror else self.stats

    def to(self, device: torch.device) -> "PhysicsContext":
        return PhysicsContext(
            stats=_torch_stats(self.stats, device),
            stats_mirrored=_torch_stats(self.stats_mirrored, device),
            names=self.names,
            parents=self.parents,
            dt=self.dt,
            contact_threshold=self.contact_threshold,
            bind_asset=self.bind_asset,
            bind_asset_sha256=self.bind_asset_sha256,
            unit_scale=self.unit_scale,
            joint_order_sha256=self.joint_order_sha256,
            skeleton_source=self.skeleton_source,
            provenance=self.provenance,
        )

    def world_state(
        self,
        features_normalized: np.ndarray | torch.Tensor,
        *,
        mirror: bool = False,
        normalized: bool = True,
        contact_threshold: float | None = None,
        root_position0: Any | None = None,
        root_rotation0: Any | None = None,
        stats: MotionFeatureStats | None = None,
    ) -> Any:
        """Denormalized positions/rotations/root for one motion feature window.

        ``normalized=True`` (the default) means the input is the store's normalized
        feature tensor and is denormalized here with the store's own offset/scale;
        nothing downstream ever sees a normalized channel again.

        The feature vector carries the root's *velocity*, not its absolute pose, so
        the reconstructed root starts at the origin unless ``root_position0`` /
        ``root_rotation0`` state the frame it came from.  Measurements that compare
        against a ground truth pass the truth's own first frame; measurements of an
        edit keep the source's, which is what makes "the edit moved the root" a
        statement about the edit.

        ``stats`` overrides the skeleton (the oracle compares the bind skeleton
        against the store's own ``ref_pos`` with everything else held fixed).
        """
        from stylized_motion.anim.features import reconstruct_motion_state_from_features

        values = (
            features_normalized.detach().cpu().numpy()
            if isinstance(features_normalized, torch.Tensor)
            else np.asarray(features_normalized)
        )
        resolved = self.stats_for(mirror=mirror) if stats is None else stats
        return reconstruct_motion_state_from_features(
            values,
            _numpy_stats(resolved),
            parents=np.asarray(self.parents, dtype=np.int32),
            dt=float(self.dt),
            root_position0=None
            if root_position0 is None
            else np.asarray(_as_numpy(root_position0), dtype=np.float32),
            root_rotation0=None
            if root_rotation0 is None
            else np.asarray(_as_numpy(root_rotation0), dtype=np.float32),
            normalized=bool(normalized),
            contact_threshold=self.contact_threshold if contact_threshold is None else contact_threshold,
        )

    def kinematic(self, *, mirror: bool = False) -> Any:
        """The ``KinematicContext`` the operator metrics use, for one mirror group."""
        from stylized_motion.learning.nef_probe import KinematicContext

        stats = self.stats_for(mirror=mirror)
        return KinematicContext(
            feature_offset=torch.as_tensor(np.asarray(stats.offset, dtype=np.float32)),
            feature_scale=torch.as_tensor(np.asarray(stats.scale, dtype=np.float32)),
            ref_pos=torch.as_tensor(np.asarray(stats.ref_pos, dtype=np.float32)),
            parents=self.parents,
            names=self.names,
            dt=float(self.dt),
            contact_threshold=float(self.contact_threshold),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "physical_metric_version": PHYSICAL_METRIC_VERSION,
            "skeleton_source": self.skeleton_source,
            "bind_asset": self.bind_asset,
            "bind_asset_sha256": self.bind_asset_sha256,
            "unit_scale": float(self.unit_scale),
            "unit": "metre (features) / centimetre (SOMA bind file, scaled by unit_scale)",
            "joint_order_sha256": self.joint_order_sha256,
            "joints": len(self.names),
            "parents_sha256": hashlib.sha256(
                json.dumps(list(self.parents)).encode("utf-8")
            ).hexdigest(),
            "mirror_rule": "mirror_partner + lateral negate (features.mirror_partner)",
            "dt": float(self.dt),
            "contact_threshold": float(self.contact_threshold),
            "provenance": dict(self.provenance),
        }


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _mirrored_stats(stats: MotionFeatureStats, names: tuple[str, ...]) -> MotionFeatureStats:
    """The same stats with the bind skeleton mirrored (x negated, sides swapped)."""
    from stylized_motion.anim.features import bind_reference_positions

    reference = bind_reference_positions(list(names), mirror=True)
    if reference is None:
        reference = np.asarray(stats.ref_pos, dtype=np.float32).copy()
        reference[:, 0] *= -1.0
    resolved = np.asarray(reference, dtype=np.float32).copy()
    stored = np.asarray(stats.ref_pos, dtype=np.float32)
    for index, name in enumerate(names):
        if str(name) in {"Simulation", "Hips"}:
            resolved[index] = stored[index]
    return MotionFeatureStats(
        offset=np.asarray(stats.offset, dtype=np.float32),
        scale=np.asarray(stats.scale, dtype=np.float32),
        dist=np.asarray(stats.dist, dtype=np.float32),
        weights=np.asarray(stats.weights, dtype=np.float32),
        ref_pos=resolved,
    )


def _numpy_stats(stats: MotionFeatureStats) -> MotionFeatureStats:
    return MotionFeatureStats(
        offset=np.asarray(stats.offset, dtype=np.float32),
        scale=np.asarray(stats.scale, dtype=np.float32),
        dist=np.asarray(stats.dist, dtype=np.float32),
        weights=np.asarray(stats.weights, dtype=np.float32),
        ref_pos=np.asarray(stats.ref_pos, dtype=np.float32),
    )


def _torch_stats(stats: MotionFeatureStats, device: torch.device) -> MotionFeatureStats:
    return MotionFeatureStats(
        offset=torch.as_tensor(np.asarray(stats.offset, dtype=np.float32), device=device),
        scale=torch.as_tensor(np.asarray(stats.scale, dtype=np.float32), device=device),
        dist=torch.as_tensor(np.asarray(stats.dist, dtype=np.float32), device=device),
        weights=torch.as_tensor(np.asarray(stats.weights, dtype=np.float32), device=device),
        ref_pos=torch.as_tensor(np.asarray(stats.ref_pos, dtype=np.float32), device=device),
    )


__all__ = [
    "PHYSICAL_METRIC_VERSION",
    "PhysicsContext",
    "joint_order_digest",
]
