"""NEF-FSQ research probes: FSQ level geometry, decoded locality and temporal influence.

Phase 0 of the MTS-FSQ plan has to answer two falsifiable questions before any
style operator is trained:

1. does moving one FSQ coordinate by one level change the decoded motion less
   than moving it to a far level (:class:`LevelGeometryProbe`)?  The
   birth-death CTMC operator assumes adjacent levels are neighbouring motion
   states; if the decoded geometry says otherwise, that assumption is dead.
2. does a token edit with a local support stay local after decoding, and how
   far does the causal decoder carry it (:func:`locality_report`,
   :func:`temporal_influence_width`)?

Every probe measures *decoded* motion as well as token space, so the reported
numbers are physical (feature, FK, root, contact, velocity, jerk) rather than
index distances.  Nothing here mutates a model or a checkpoint.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stylized_motion.anim import quat
from stylized_motion.learning.losses import (
    _masked_weighted_mean,
    integrate_root_trajectory,
    reconstruct_joint_positions,
)
from stylized_motion.learning.nef_eval import (
    DECODER_INFLUENCE_FRAMES,
    contacts_from_toe_motion,
)
from stylized_motion.learning.nef_layout import NEFLayout

CSV_COLUMNS = (
    "coordinate",
    "stream",
    "family",
    "adjacent_distance",
    "far_distance",
    "adjacent_to_far_ratio",
    "direction_consistency",
    "feature_l1_adjacent",
    "feature_l1_far",
    "fk_owned_adjacent",
    "fk_owned_far",
    "fk_offtarget_max_adjacent",
    "fk_offtarget_max_far",
    "valid_adjacent_frames",
    "valid_far_frames",
)


# ---------------------------------------------------------------------------
# token perturbations


@dataclass(frozen=True)
class TokenPerturbation:
    """One coordinate-level token edit and the frames where it stayed legal."""

    coordinate: int
    offsets: torch.Tensor  # [B, T] signed level offsets that were applied
    tokens: torch.Tensor  # [B, T, C] edited copy of the token tensor
    valid: torch.Tensor  # [B, T] bool; False where the shift left [0, num_levels)

    @property
    def dtype_valid_ratio(self) -> float:
        return float(self.valid.float().mean())


def _validate_levels(indices: torch.Tensor, num_levels: int, coordinate: int) -> None:
    if num_levels <= 1:
        raise ValueError(f"num_levels must be > 1, got {num_levels}")
    values = indices[..., coordinate]
    if bool((values < 0).any()) or bool((values >= num_levels).any()):
        raise ValueError(
            f"Coordinate {coordinate} holds values outside [0, {num_levels - 1}]; "
            "the probe only edits legal FSQ levels"
        )


def _apply_offsets(
    indices: torch.Tensor,
    coordinate: int,
    offsets: torch.Tensor,
    num_levels: int,
) -> TokenPerturbation:
    values = indices[..., coordinate]
    shifted = values + offsets
    valid = (shifted >= 0) & (shifted <= num_levels - 1)
    tokens = indices.clone()
    tokens[..., coordinate] = shifted.clamp_(0, num_levels - 1)
    return TokenPerturbation(
        coordinate=int(coordinate),
        offsets=offsets.clone(),
        tokens=tokens,
        valid=valid,
    )


def adjacent_perturbations(
    indices: torch.Tensor, coordinate: int, *, num_levels: int = 9
) -> tuple[TokenPerturbation, TokenPerturbation]:
    """The ``+1`` and ``-1`` level moves of one coordinate."""
    _validate_levels(indices, num_levels, coordinate)
    step = torch.ones_like(indices[..., coordinate])
    return (
        _apply_offsets(indices, coordinate, step, num_levels),
        _apply_offsets(indices, coordinate, -step, num_levels),
    )


def far_perturbations(
    indices: torch.Tensor,
    coordinate: int,
    *,
    num_levels: int = 9,
    min_distance: int | None = None,
    samples: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[TokenPerturbation, ...]:
    """Sampled jumps of at least ``min_distance`` levels in either direction."""
    _validate_levels(indices, num_levels, coordinate)
    if min_distance is None:
        min_distance = max(2, num_levels // 2)
    min_distance = int(min_distance)
    if not 1 <= min_distance < num_levels:
        raise ValueError(f"min_distance must be in [1, {num_levels - 1}], got {min_distance}")
    if int(samples) < 1:
        raise ValueError(f"samples must be positive, got {samples}")
    magnitudes = torch.arange(min_distance, num_levels, dtype=torch.long)
    values = indices[..., coordinate]
    result = []
    for _ in range(int(samples)):
        # Draws happen on the generator's own device and then move, so a CPU
        # generator works with CUDA tokens and vice versa.
        sign = torch.where(
            torch.rand(values.shape, generator=generator) < 0.5, -1, 1
        ).to(values.device)
        magnitude = magnitudes[
            torch.randint(magnitudes.numel(), values.shape, generator=generator)
        ].to(values.device)
        result.append(_apply_offsets(indices, coordinate, sign * magnitude, num_levels))
    return tuple(result)


# ---------------------------------------------------------------------------
# kinematics context and measurement


def reference_positions_for_fk(
    feature_stats: Mapping[str, object], *, mirror: bool = False
) -> np.ndarray | None:
    """Bind-skeleton local offsets (metres) for a stats payload, or ``None``.

    A store's ``ref_pos`` is the dataset mean of the local positions; the mirrored
    clips cancel every constant axis-aligned offset, so FK on it collapses the
    spine.  This resolves the real skeleton from the SOMA bind contract, and
    returns ``None`` when the payload already carries a usable one (synthetic
    fixtures) so callers can fall back to the stored values.
    """
    from stylized_motion.anim.features import (
        deserialize_motion_feature_stats,
        stats_with_reference_skeleton,
    )

    payload = dict(feature_stats)
    if "dist" not in payload and "offset" in payload:
        # A partial stats payload is still enough to place a skeleton; the missing
        # dispersion is only used by the normalisation.
        payload["dist"] = np.ones_like(np.asarray(payload["offset"], dtype=np.float32))
    try:
        stats, metadata = deserialize_motion_feature_stats(payload)
    except (KeyError, ValueError, TypeError):
        return None  # too incomplete to contain a skeleton: keep the stored value
    corrected = stats_with_reference_skeleton(stats, metadata["names"], mirror=mirror)
    if corrected is stats:  # the bind skeleton was unavailable; keep the stored value
        return None
    return np.asarray(corrected.ref_pos, dtype=np.float32)


@dataclass(frozen=True)
class KinematicContext:
    """Denormalisation and FK context needed to measure world-space effects."""

    feature_offset: torch.Tensor
    feature_scale: torch.Tensor
    ref_pos: torch.Tensor
    parents: tuple[int, ...]
    names: tuple[str, ...]
    dt: float = 1.0 / 60.0
    contact_threshold: float = 0.15

    @classmethod
    def from_feature_stats(
        cls,
        feature_stats: Mapping[str, object],
        *,
        dt: float = 1.0 / 60.0,
        contact_threshold: float = 0.15,
        reference_positions: np.ndarray | None = None,
    ) -> KinematicContext:
        """Build the FK context, optionally with an explicit reference skeleton.

        ``reference_positions`` exists because a store's ``ref_pos`` is the
        *dataset mean* of the local positions: mirror augmentation cancels every
        constant axis-aligned offset, so FK on the stored value collapses the
        spine and hides joint motion.  Callers that report world-space numbers
        pass the bind skeleton's offsets (``features.stats_with_reference_skeleton``);
        the default keeps the stored value so synthetic fixtures are unaffected.
        """
        ref_values = (
            np.asarray(feature_stats["ref_pos"], dtype=np.float32)
            if reference_positions is None
            else np.asarray(reference_positions, dtype=np.float32)
        )
        return cls(
            feature_offset=torch.as_tensor(
                np.asarray(feature_stats["offset"], dtype=np.float32)
            ),
            feature_scale=torch.as_tensor(
                np.asarray(feature_stats["scale"], dtype=np.float32)
            ),
            ref_pos=torch.as_tensor(ref_values),
            parents=tuple(int(value) for value in np.asarray(feature_stats["parents"]).tolist()),
            names=tuple(str(name) for name in feature_stats["names"]),
            dt=float(dt),
            contact_threshold=float(contact_threshold),
        )

    @property
    def toe_indices(self) -> tuple[int, int] | None:
        try:
            return self.names.index("LeftToeBase"), self.names.index("RightToeBase")
        except ValueError:
            return None

    def to(self, device: torch.device) -> KinematicContext:
        return KinematicContext(
            feature_offset=self.feature_offset.to(device),
            feature_scale=self.feature_scale.to(device),
            ref_pos=self.ref_pos.to(device),
            parents=self.parents,
            names=self.names,
            dt=self.dt,
            contact_threshold=self.contact_threshold,
        )


@dataclass(frozen=True)
class _Baseline:
    """Decoder-side reference quantities shared by every perturbation."""

    decoded: torch.Tensor  # [B, T, motion_dim]
    valid: torch.Tensor  # [B, T] bool
    positions: torch.Tensor | None  # [B, T, J, 3] world FK
    root_positions: torch.Tensor | None  # [B, T, 3]
    root_rotations: torch.Tensor | None  # [B, T, 4]
    contacts: torch.Tensor | None  # [B, T, 2] bool


def _world_kinematics(
    decoded: torch.Tensor, kinematic: KinematicContext
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = reconstruct_joint_positions(
        decoded,
        kinematic.feature_offset,
        kinematic.feature_scale,
        kinematic.ref_pos,
        kinematic.parents,
        kinematic.dt,
        world_space=True,
    )
    root_positions, root_rotations = integrate_root_trajectory(
        decoded,
        kinematic.feature_offset,
        kinematic.feature_scale,
        kinematic.dt,
        return_positions=True,
        return_rotations=True,
    )
    assert root_positions is not None and root_rotations is not None
    return positions, root_positions, root_rotations


def _inferred_contacts(
    positions: torch.Tensor, kinematic: KinematicContext
) -> torch.Tensor | None:
    toe_indices = kinematic.toe_indices
    if toe_indices is None or positions.shape[1] < 2:
        return None
    return contacts_from_toe_motion(
        positions, toe_indices, kinematic.dt, threshold=kinematic.contact_threshold
    )


class _Measurer:
    """Compares one decoded perturbation against a shared baseline."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        feature_indices: Mapping[str, torch.Tensor],
        kinematic: KinematicContext | None,
        stream_joints: Mapping[str, Sequence[int]] | None = None,
        parents: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.feature_indices = feature_indices
        self.device = module_device(model)
        # FK/root/contact metrics must run where the decode runs; keeping the
        # context on CPU silently worked only because every earlier probe ran on
        # CPU.
        self.kinematic = None if kinematic is None else kinematic.to(self.device)
        self._stream_joints_by_name = dict(stream_joints or {})
        self._parents = tuple(int(value) for value in (parents or ()))

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.decode_from_indices(tokens.to(self.device))

    def baseline(self, indices: torch.Tensor, valid: torch.Tensor) -> _Baseline:
        decoded = self.decode(indices)
        positions = root_positions = root_rotations = contacts = None
        if self.kinematic is not None:
            positions, root_positions, root_rotations = _world_kinematics(decoded, self.kinematic)
            contacts = _inferred_contacts(positions, self.kinematic)
        return _Baseline(
            decoded=decoded,
            valid=valid,
            positions=positions,
            root_positions=root_positions,
            root_rotations=root_rotations,
            contacts=contacts,
        )

    def measure(
        self,
        decoded: torch.Tensor,
        base: _Baseline,
        *,
        stream: str,
        valid: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Returns scalar metrics and the mean owned-feature delta vector."""
        local_valid = valid.to(decoded.device)
        delta = decoded - base.decoded
        owned_index = self.feature_indices[stream].to(decoded.device)
        feature_mask = torch.zeros(decoded.shape[-1], dtype=torch.bool, device=decoded.device)
        feature_mask[owned_index] = True
        owned = delta[..., feature_mask]
        off_target = delta[..., ~feature_mask]

        metrics: dict[str, float] = {
            "feature_l1": float(_masked_weighted_mean(delta.abs(), local_valid)),
            "feature_l2": float(
                _masked_weighted_mean(delta.square().sum(-1).sqrt(), local_valid)
            ),
            "stream_feature_l1": float(_masked_weighted_mean(owned.abs(), local_valid)),
            "offtarget_feature_max": float(off_target.abs().max()) if off_target.numel() else 0.0,
        }

        pair_valid = local_valid[:, 1:] & local_valid[:, :-1]
        velocity_delta = (decoded[:, 1:] - decoded[:, :-1]) - (
            base.decoded[:, 1:] - base.decoded[:, :-1]
        )
        metrics["velocity_change"] = float(
            _masked_weighted_mean(velocity_delta[..., feature_mask].abs(), pair_valid)
        )
        triple_valid = local_valid[:, 2:] & local_valid[:, :-2]
        jerk_delta = (velocity_delta[:, 1:] - velocity_delta[:, :-1])
        metrics["jerk_change"] = float(
            _masked_weighted_mean(jerk_delta[..., feature_mask].abs(), triple_valid)
        )

        if self.kinematic is not None and base.positions is not None:
            assert base.root_positions is not None and base.root_rotations is not None
            positions, root_positions, root_rotations = _world_kinematics(
                decoded, self.kinematic
            )
            joint_change = (positions - base.positions).norm(dim=-1)  # [B, T, J]
            owned_joints = self._stream_joints(stream)
            joints = torch.as_tensor(owned_joints, dtype=torch.long, device=decoded.device)
            joint_mask = torch.zeros(
                joint_change.shape[-1], dtype=torch.bool, device=decoded.device
            )
            joint_mask[joints] = True
            # Rotating a joint moves its children, not itself: a stream that owns
            # one joint's rotation changes FK exactly on that joint's descendants.
            # Leakage is therefore measured on joints that are neither owned nor
            # descendants, otherwise every Edge stream would look like a 2 m leak.
            descendants = kinematic_descendants(self._kinematic_parents(), owned_joints) - set(
                owned_joints
            )
            descendant_mask = torch.zeros_like(joint_mask)
            if descendants:
                descendant_mask[
                    torch.as_tensor(sorted(descendants), dtype=torch.long, device=decoded.device)
                ] = True
            influenced = joint_mask | descendant_mask
            off_target_joints = joint_change[..., ~influenced]
            metrics["owns_joints"] = 1.0 if owned_joints else 0.0
            metrics["fk_owned_mean"] = (
                float(_masked_weighted_mean(joint_change[..., joint_mask], local_valid))
                if bool(joint_mask.any())
                else 0.0
            )
            metrics["fk_descendant_mean"] = (
                float(_masked_weighted_mean(joint_change[..., descendant_mask], local_valid))
                if bool(descendant_mask.any())
                else 0.0
            )
            if owned_joints:
                metrics["fk_influence_mean"] = float(
                    _masked_weighted_mean(joint_change[..., influenced], local_valid)
                )
                metrics["fk_offtarget_mean"] = float(
                    _masked_weighted_mean(off_target_joints, local_valid)
                )
                metrics["fk_offtarget_max"] = (
                    float(off_target_joints.max()) if off_target_joints.numel() else 0.0
                )
            else:
                # The global stream owns the root orientation/velocity and the
                # contacts, so an edit there moves the whole body by construction:
                # "off-target leakage" is undefined, and only the influence
                # magnitude is meaningful.
                metrics["fk_influence_mean"] = float(
                    _masked_weighted_mean(joint_change, local_valid)
                )
                metrics["fk_offtarget_mean"] = 0.0
                metrics["fk_offtarget_max"] = 0.0
            metrics["root_pos_change"] = float(
                _masked_weighted_mean(
                    (root_positions[:, 1:] - base.root_positions[:, 1:]).abs(),
                    local_valid[:, 1:],
                )
            )
            metrics["root_rot_change"] = float(
                _masked_weighted_mean(
                    quat.torch_quat_angle(root_rotations[:, 1:], base.root_rotations[:, 1:]),
                    local_valid[:, 1:],
                )
            )
            metrics["contact_feature_change"] = float(
                _masked_weighted_mean(delta[..., -2:].abs(), local_valid)
            )
            contacts = _inferred_contacts(positions, self.kinematic)
            if contacts is not None and base.contacts is not None:
                metrics["contact_flip_rate"] = float(
                    _masked_weighted_mean(
                        (contacts != base.contacts).any(dim=-1).to(delta.dtype), local_valid
                    )
                )
        direction_vector = _masked_mean_vector(owned, local_valid)
        return metrics, direction_vector

    def _stream_joints(self, stream: str) -> list[int]:
        if self._stream_joints_by_name:
            if stream not in self._stream_joints_by_name:
                raise ValueError(f"Unknown stream {stream!r} in the supplied ownership table")
            return list(self._stream_joints_by_name[stream])
        layout = getattr(self.model, "layout", None)
        if layout is None:
            raise ValueError("Kinematic probes require an NEF layout")
        return list(layout.stream_joints(stream))

    def _kinematic_parents(self) -> tuple[int, ...]:
        if self._parents:
            return self._parents
        if self.kinematic is not None:
            return tuple(int(value) for value in self.kinematic.parents)
        layout = getattr(self.model, "layout", None)
        if layout is None:
            raise ValueError("Kinematic probes require an NEF layout")
        return tuple(int(value) for value in layout.parents)


def _masked_mean_vector(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` [B, T, F] over the frames selected by ``mask`` [B, T]."""
    weights = mask.to(values.device, values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=(0, 1))
    count = weights.sum().clamp_min(1.0)
    return total / count


def _mean_of_dicts(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    result: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item]
        if values:
            result[key] = float(np.mean(values))
    return result


def _as_feature_mask(
    support: Sequence[int] | torch.Tensor | Sequence[bool], width: int, device: torch.device
) -> torch.Tensor:
    """Normalizes a feature support into a boolean ``[width]`` mask."""
    values = torch.as_tensor(support, device=device)
    if values.dtype == torch.bool:
        if values.numel() != int(width):
            raise ValueError(f"feature_support has {values.numel()} entries, expected {width}")
        return values
    mask = torch.zeros(int(width), dtype=torch.bool, device=device)
    mask[values.long()] = True
    return mask


# ---------------------------------------------------------------------------
# level geometry probe


class LevelGeometryProbe:
    """Tests whether adjacent FSQ levels are neighbouring decoded motion states."""

    #: Metrics compared between adjacent and far perturbations.
    PRIMARY_METRIC = "feature_l1"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        kinematic: KinematicContext | None = None,
        far_samples: int = 2,
        far_min_distance: int | None = None,
        decode_rows: int = 1024,
    ) -> None:
        module = getattr(model, "module", model)
        layout = getattr(module, "get_token_layout", lambda: getattr(module, "layout", None))()
        if not isinstance(layout, NEFLayout):
            raise TypeError("LevelGeometryProbe requires an NEF-FSQ model with a NEFLayout")
        self.model = model
        self.layout = layout
        self.num_levels = int(getattr(module, "num_levels"))
        self.motion_dim = int(getattr(module, "motion_dim"))
        self.feature_indices = layout.feature_indices(self.motion_dim)
        self.kinematic = kinematic
        self.far_samples = int(far_samples)
        self.far_min_distance = far_min_distance
        if int(decode_rows) < 1:
            raise ValueError("decode_rows must be positive")
        self.decode_rows = int(decode_rows)
        self._measurer = _Measurer(
            module,
            feature_indices=self.feature_indices,
            kinematic=kinematic,
            stream_joints={stream: layout.stream_joints(stream) for stream in layout.coordinate_order},
            parents=layout.parents,
        )
        self.device = self._measurer.device

    def run(
        self,
        indices: torch.Tensor,
        *,
        coordinates: Sequence[int] | None = None,
        valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        indices = indices.detach().to(self.device).long()
        if indices.ndim != 3 or indices.shape[-1] != self.layout.num_coordinates:
            raise ValueError(
                f"Expected tokens [B, T, {self.layout.num_coordinates}], got {tuple(indices.shape)}"
            )
        batch, frames, _ = indices.shape
        if valid_mask is None:
            valid = torch.ones(batch, frames, dtype=torch.bool, device=self.device)
        else:
            valid = valid_mask.to(self.device).bool()
            if valid.shape != (batch, frames):
                raise ValueError(f"valid_mask must have shape {(batch, frames)}, got {tuple(valid.shape)}")
        coordinates = (
            list(range(indices.shape[-1])) if coordinates is None else [int(c) for c in coordinates]
        )
        for coordinate in coordinates:
            if not 0 <= coordinate < self.layout.num_coordinates:
                raise ValueError(f"coordinate {coordinate} is outside the token width")

        plan: list[tuple[int, str, TokenPerturbation]] = []
        for coordinate in coordinates:
            for perturbation in adjacent_perturbations(
                indices, coordinate, num_levels=self.num_levels
            ):
                plan.append((coordinate, "adjacent", perturbation))
            for perturbation in far_perturbations(
                indices,
                coordinate,
                num_levels=self.num_levels,
                min_distance=self.far_min_distance,
                samples=self.far_samples,
                generator=generator,
            ):
                plan.append((coordinate, "far", perturbation))

        base = self._measurer.baseline(indices, valid)
        collected: dict[tuple[int, str], list[dict[str, float]]] = {}
        directions: dict[tuple[int, int], torch.Tensor] = {}
        valid_counts: dict[tuple[int, str], list[int]] = {}
        metadata = self.layout.coordinate_metadata()

        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(0)
        chunk_size = max(1, self.decode_rows // max(batch, 1))
        for start in range(0, len(plan), chunk_size):
            chunk = plan[start : start + chunk_size]
            stacked = torch.cat([perturbation.tokens for _, _, perturbation in chunk], dim=0)
            decoded = self._measurer.decode(stacked)
            for offset, (coordinate, kind, perturbation) in enumerate(chunk):
                values = decoded[offset * batch : (offset + 1) * batch]
                stream = self._stream_of(coordinate, metadata)
                metrics, direction = self._measurer.measure(
                    values, base, stream=stream, valid=perturbation.valid
                )
                collected.setdefault((coordinate, kind), []).append(metrics)
                valid_counts.setdefault((coordinate, kind), []).append(
                    int(perturbation.valid.sum())
                )
                if kind == "adjacent":
                    sign = int(perturbation.offsets.flatten()[0].item())
                    directions[(coordinate, sign)] = direction

        records: list[dict[str, Any]] = []
        for coordinate in coordinates:
            stream = self._stream_of(coordinate, metadata)
            adjacent = _mean_of_dicts(collected.get((coordinate, "adjacent"), []))
            far = _mean_of_dicts(collected.get((coordinate, "far"), []))
            # Named so a reader cannot mistake this arm for the support-matched far
            # move of protocol revision 2.
            far = {"far_kind": "random_per_frame_stress", **far}
            adjacent_distance = float(adjacent.get(self.PRIMARY_METRIC, 0.0))
            far_distance = float(far.get(self.PRIMARY_METRIC, 0.0))
            ratio = adjacent_distance / far_distance if far_distance > 0.0 else float("inf")
            record: dict[str, Any] = {
                "coordinate": int(coordinate),
                "stream": stream,
                "family": metadata[coordinate]["family"],
                "joints": list(metadata[coordinate]["joints"]),
                "adjacent_distance": adjacent_distance,
                "far_distance": far_distance,
                "adjacent_to_far_ratio": ratio,
                "direction_consistency": self._direction_consistency(directions, coordinate),
                "valid_adjacent_frames": int(sum(valid_counts.get((coordinate, "adjacent"), [0]))),
                "valid_far_frames": int(sum(valid_counts.get((coordinate, "far"), [0]))),
                "adjacent": adjacent,
                "far": far,
            }
            records.append(record)

        ratios = [record["adjacent_to_far_ratio"] for record in records]
        finite_ratios = [value for value in ratios if np.isfinite(value)]
        consistencies = [record["direction_consistency"] for record in records]
        summary = {
            "kind": "level_geometry",
            "protocol_revision": 1,
            # The legacy geometry screen compares adjacent steps against *random
            # per-frame* far jumps.  That is a stress test, not an ordinal-geometry
            # measurement: the far arm changes many positions by different amounts at
            # once.  The fair comparison (same source/coordinate/time support, ±1 vs
            # ±d) is `stratified_span_probe` in protocol revision 2, which writes its
            # own artifact and never overwrites this one.
            "far_perturbation": "random_per_frame_stress",
            "ordinal_geometry_note": (
                "adjacent_to_far_ratio is a screen over random per-frame far jumps; "
                "use the protocol-v2 stratified probe for a support-matched comparison"
            ),
            "coordinates": len(records),
            "num_levels": self.num_levels,
            "frames": int(frames),
            "batch": int(batch),
            "skeleton": self.layout.skeleton,
            "primary_metric": self.PRIMARY_METRIC,
            "far_min_distance": int(
                self.far_min_distance
                if self.far_min_distance is not None
                else max(2, self.num_levels // 2)
            ),
            "far_samples": self.far_samples,
            "kinematics": self.kinematic is not None,
            "mean_adjacent_to_far_ratio": float(np.mean(finite_ratios))
            if finite_ratios
            else float("inf"),
            "median_adjacent_to_far_ratio": float(np.median(finite_ratios))
            if finite_ratios
            else float("inf"),
            "coordinates_with_adjacent_smoother": int(
                sum(1 for value in finite_ratios if value < 1.0)
            ),
            "mean_direction_consistency": float(np.mean(consistencies)),
        }
        # Heuristic screen only: it decides whether birth-death is worth
        # implementing, never whether the tokenizer is "correct".
        summary["ordinal_geometry_supported"] = bool(
            summary["median_adjacent_to_far_ratio"] < 1.0
            and summary["mean_direction_consistency"] > 0.0
        )
        return {
            "kind": "level_geometry",
            "skeleton": self.layout.skeleton,
            "layout_hash": self.layout.layout_hash(),
            "summary": summary,
            "per_coordinate": records,
        }

    @staticmethod
    def _stream_of(coordinate: int, metadata: Sequence[Mapping[str, Any]]) -> str:
        return str(metadata[coordinate]["stream"])

    @staticmethod
    def _direction_consistency(
        directions: Mapping[tuple[int, int], torch.Tensor], coordinate: int
    ) -> float:
        plus = directions.get((coordinate, 1))
        minus = directions.get((coordinate, -1))
        if plus is None or minus is None:
            return 0.0
        denominator = plus.norm() * minus.norm()
        if float(denominator) <= 1e-12:
            return 0.0
        # +1 and -1 should move the decode along opposite directions of one
        # ordinal axis, so the negation of their cosine is the consistency.
        cosine = float(torch.dot(plus, minus) / denominator)
        return float(np.clip(-cosine, -1.0, 1.0))


# ---------------------------------------------------------------------------
# protocol revision 2: signed pulses and spans with a shared legal support


PROBE_PROTOCOL_REVISION = 2


def signed_span_perturbations(
    indices: torch.Tensor,
    coordinate: int,
    *,
    num_levels: int = 9,
    magnitude: int = 1,
    sign: int = 1,
    start: int = 0,
    stop: int | None = None,
) -> TokenPerturbation:
    """One *signed* level step applied over a fixed frame span.

    ``sign=+1, magnitude=1`` is the near perturbation and ``sign=+1, magnitude=d``
    the far one; both cover the same coordinate and the same frames, so the two
    differ only in how far the level moved.  Positions whose shift would leave
    ``[0, num_levels)`` are flagged in ``valid`` and left at the clamped level --
    the caller intersects the valid masks instead of comparing different supports.
    """
    _validate_levels(indices, num_levels, coordinate)
    frames = int(indices.shape[1])
    start = int(start)
    stop = frames if stop is None else int(stop)
    if not 0 <= start < stop <= frames:
        raise ValueError(f"span [{start}, {stop}) is not inside [0, {frames})")
    sign = int(sign)
    magnitude = int(magnitude)
    if sign not in (-1, 1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if magnitude < 1 or int(magnitude) >= int(num_levels):
        raise ValueError(f"magnitude must be in [1, {int(num_levels) - 1}], got {magnitude}")
    offsets = torch.zeros(
        indices.shape[:2], dtype=torch.long, device=indices.device
    )
    offsets[:, start:stop] = sign * magnitude
    return _apply_offsets(indices, coordinate, offsets, num_levels)


def signed_pulse_perturbations(
    indices: torch.Tensor,
    coordinate: int,
    *,
    frame: int,
    num_levels: int = 9,
    magnitude: int = 1,
    sign: int = 1,
) -> TokenPerturbation:
    """A single-frame signed step: the pulse form of :func:`signed_span_perturbations`."""
    return signed_span_perturbations(
        indices, coordinate, num_levels=num_levels, magnitude=magnitude, sign=sign,
        start=int(frame), stop=int(frame) + 1,
    )


def shared_legal_support(*perturbations: TokenPerturbation) -> torch.Tensor:
    """Frames where *every* perturbation was legal.

    Comparing a near and a far move over different supports measures the support,
    not the distance; this is the intersection the caller must use.
    """
    if not perturbations:
        raise ValueError("shared_legal_support needs at least one perturbation")
    support = perturbations[0].valid.clone()
    for perturbation in perturbations[1:]:
        if perturbation.valid.shape != support.shape:
            raise ValueError("perturbations must share a shape")
        support = support & perturbation.valid
    return support


def influence_profile(
    model: torch.nn.Module,
    indices: torch.Tensor,
    perturbation: TokenPerturbation,
    *,
    kinematic: KinematicContext | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Where one perturbation lands: feature L1 per frame, and world FK when asked.

    Two tails are reported separately because they obey different contracts:
    ``feature_*`` is the decoder's own token-to-feature influence (bounded by the
    decoder receptive field), while ``world_*`` is the root-integrated world-space
    motion, which integrates that influence over time and therefore has no such
    bound.  Sharing one receptive-field ceiling between them would be a category
    error.  ``valid_mask`` is the comparison support: statistics are computed there,
    and everything after its last frame is the tail.
    """
    module = getattr(model, "module", model)
    device = module_device(module)
    tokens = indices.detach().to(device).long()
    edited = perturbation.tokens.detach().to(device).long()
    legal = perturbation.valid.detach().to(device).bool()
    if valid_mask is not None:
        mask = valid_mask.to(device).bool()
        if mask.shape != legal.shape:
            raise ValueError(
                f"valid_mask must match the perturbation {tuple(legal.shape)}, "
                f"got {tuple(mask.shape)}"
            )
        legal = legal & mask
    with torch.no_grad():
        baseline = module.decode_from_indices(tokens)
        changed = module.decode_from_indices(edited)
    per_frame = (changed - baseline).abs().mean(dim=-1)  # [B, T]
    support = legal.any(dim=0)  # [T]
    inside = per_frame[legal]
    per_frame_max = per_frame.amax(dim=0)
    changed_frames = torch.nonzero(per_frame_max > 0).flatten()
    report: dict[str, Any] = {
        "unit_feature_l1": "mean |decoded difference| over feature dims",
        "feature_l1_mean": float(per_frame.mean()),
        "feature_l1_max": float(per_frame_max.max()),
        "feature_l1_inside_support": float(inside.mean()) if bool(legal.any()) else None,
        "feature_first_changed_frame": int(changed_frames.min()) if changed_frames.numel() else None,
        "feature_last_changed_frame": int(changed_frames.max()) if changed_frames.numel() else None,
        "support_frames": int(support.sum()),
        "world_fk_max": None,
        "world_fk_tail_max": None,
        "unit_world_fk": "metres",
    }
    if kinematic is not None:
        if not bool(legal.any()):
            report["world_fk_reason"] = "no legal frame to measure"
            return report
        context = kinematic.to(device)
        base_positions, base_root, _ = _world_kinematics(baseline.detach(), context)
        edited_positions, edited_root, _ = _world_kinematics(changed.detach(), context)
        joint_distance = (edited_positions - base_positions).norm(dim=-1).amax(dim=-1)  # [B, T]
        root_distance = (edited_root - base_root).norm(dim=-1)
        report["world_fk_max"] = float(joint_distance[:, support].max())
        report["world_root_max"] = float(root_distance[:, support].max())
        # The tail is everything after the support ends: the point of reporting it
        # separately is that the decoder's own influence has stopped there while the
        # integrated world-space motion still carries the change.
        after = torch.zeros_like(support)
        last = int(torch.nonzero(support).flatten().max())
        if last + 1 < int(support.shape[0]):
            after[last + 1 :] = True
        report["world_fk_tail_max"] = (
            float(joint_distance[:, after].max()) if bool(after.any()) else None
        )
        report["world_root_tail_max"] = (
            float(root_distance[:, after].max()) if bool(after.any()) else None
        )
        report["tail_frames"] = int(after.sum())
    return report


def stratified_span_probe(
    model: torch.nn.Module,
    indices: torch.Tensor,
    *,
    coordinate: int,
    frame: int,
    span: int = 1,
    magnitudes: Sequence[int] = (1, 3),
    signs: Sequence[int] = (1, -1),
    kinematic: KinematicContext | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Near/far influence over the *same* source, coordinate, time support and mask.

    ``magnitudes[0]`` is the near step and the rest are the far steps; every move is
    applied to the same window and compared only on the frames inside the span where
    *all* of them stayed inside the level range, so a difference cannot come from a
    different support.  ``temporal_probe_complete`` states whether the window even
    contains the decoder's full influence tail: a 128-frame edit at frame 32 of a
    shorter window cannot claim the effect stopped at the window end.
    """
    module = getattr(model, "module", model)
    device = module_device(module)
    tokens = indices.detach().to(device).long()
    frames = int(tokens.shape[1])
    frame = int(frame)
    span = int(span)
    if span < 1:
        raise ValueError("span must be positive")
    stop = min(frames, frame + span)
    if not 0 <= frame < stop:
        raise ValueError(f"span [{frame}, {stop}) is not inside [0, {frames})")
    receptive_field = int(getattr(module, "receptive_field", 0) or 0)
    tail_required = max(0, receptive_field - 1)
    tail_available = max(0, frames - stop)
    plan: dict[tuple[int, int], TokenPerturbation] = {}
    for magnitude in magnitudes:
        for sign in signs:
            plan[(int(sign), int(magnitude))] = signed_span_perturbations(
                tokens,
                int(coordinate),
                num_levels=int(getattr(module, "num_levels", 9)),
                magnitude=int(magnitude),
                sign=int(sign),
                start=frame,
                stop=stop,
            )
    legality = shared_legal_support(*plan.values())
    in_span = torch.zeros_like(legality)
    in_span[:, frame:stop] = True
    support = legality & in_span
    if valid_mask is not None:
        mask = valid_mask.to(device).bool()
        if mask.shape != support.shape:
            raise ValueError(
                f"valid_mask must be {tuple(support.shape)}, got {tuple(mask.shape)}"
            )
        support = support & mask
    rows: list[dict[str, Any]] = []
    for (sign, magnitude), perturbation in sorted(plan.items()):
        profile = influence_profile(module, tokens, perturbation, kinematic=kinematic,
                                    valid_mask=support)
        rows.append(
            {
                "sign": int(sign),
                "magnitude": int(magnitude),
                "kind": "near" if int(magnitude) == int(magnitudes[0]) else "far",
                "coordinate": int(coordinate),
                "frame_range": [frame, stop],
                **profile,
            }
        )
    return {
        "kind": "stratified_span",
        "protocol_revision": int(PROBE_PROTOCOL_REVISION),
        "coordinate": int(coordinate),
        "frame_range": [frame, stop],
        "span_frames": int(stop - frame),
        "magnitudes": [int(value) for value in magnitudes],
        "signs": [int(value) for value in signs],
        "legal_frames_in_span": int(support.sum()),
        "excluded_by_level_range": int((in_span & ~legality).sum()),
        "outside_span_frames": int((~in_span).sum()),
        "decoder_receptive_field": receptive_field,
        "decoder_influence_frames": int(DECODER_INFLUENCE_FRAMES),
        "tail_frames_available": int(tail_available),
        "tail_frames_required": int(tail_required),
        # A window that ends before the decoder's influence could have died out
        # cannot support a claim about where the effect stopped.
        "temporal_probe_complete": bool(tail_available >= tail_required),
        "units": {"feature": "mean |decoded difference|", "world": "metres"},
        "kinematics": kinematic is not None,
        "seed": None,
        "rows": rows,
    }


def probe_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flattens a level-geometry report into one row per coordinate."""
    rows: list[dict[str, Any]] = []
    for record in report.get("per_coordinate", []):
        adjacent = record.get("adjacent", {})
        far = record.get("far", {})
        rows.append(
            {
                "coordinate": record["coordinate"],
                "stream": record["stream"],
                "family": record.get("family", ""),
                "adjacent_distance": record["adjacent_distance"],
                "far_distance": record["far_distance"],
                "adjacent_to_far_ratio": record["adjacent_to_far_ratio"],
                "direction_consistency": record["direction_consistency"],
                "feature_l1_adjacent": record["adjacent_distance"],
                "feature_l1_far": record["far_distance"],
                "fk_owned_adjacent": adjacent.get("fk_owned_mean", ""),
                "fk_owned_far": far.get("fk_owned_mean", ""),
                "fk_offtarget_max_adjacent": adjacent.get("fk_offtarget_max", ""),
                "fk_offtarget_max_far": far.get("fk_offtarget_max", ""),
                "valid_adjacent_frames": record.get("valid_adjacent_frames", 0),
                "valid_far_frames": record.get("valid_far_frames", 0),
            }
        )
    return rows


def write_probe_csv(path: str | Path, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = probe_csv_rows(report)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# locality probe


def kinematic_descendants(parents: Sequence[int], joints: Sequence[int]) -> set[int]:
    """All joints below ``joints`` in the skeleton tree."""
    children: dict[int, list[int]] = {index: [] for index in range(len(parents))}
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    result: set[int] = set()
    stack = list(joints)
    while stack:
        for child in children[stack.pop()]:
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def decoder_influence_frames(module: torch.nn.Module) -> int:
    """The decoder's actual influence width: ``receptive_field - 1``.

    Reading it from the module instead of hard-coding 33 keeps the probe honest
    when the architecture changes: a decoder with a different RF would otherwise be
    judged against someone else's contract.
    """
    for attribute in ("decoder_receptive_field", "receptive_field"):
        value = getattr(module, attribute, None)
        if value is not None and int(value) > 0:
            return int(value) - 1
    return int(DECODER_INFLUENCE_FRAMES)


def temporal_influence_width(
    model: torch.nn.Module,
    indices: torch.Tensor,
    *,
    stream: str,
    frame: int,
    replacement: int | None = None,
) -> dict[str, Any]:
    """How far one edited token frame propagates through the causal decoder.

    The measured width must not exceed the decoder's own ``receptive_field - 1``; a
    wider span means the decoder is not the causal module the contract claims.
    """
    module = getattr(model, "module", model)
    layout = module.get_token_layout()
    slices = layout.stream_slices
    device = module_device(module)
    indices = indices.detach().to(device).long()
    if indices.ndim != 3:
        raise ValueError("temporal_influence_width expects tokens [B, T, 40]")
    frames = indices.shape[1]
    if not 0 <= int(frame) < frames:
        raise ValueError(f"frame must be in [0, {frames}), got {frame}")
    num_levels = int(getattr(module, "num_levels"))
    if replacement is not None and not 0 <= int(replacement) < num_levels:
        raise ValueError("replacement must be a legal FSQ level")
    edited = indices.clone()
    edited[:, int(frame), slices[stream]] = (
        torch.full_like(indices[:, int(frame), slices[stream]], int(replacement))
        if replacement is not None
        else (indices[:, int(frame), slices[stream]] + 1) % num_levels
    )

    with torch.no_grad():
        baseline = module.decode_from_indices(indices)
        changed = module.decode_from_indices(edited)
    difference = (changed - baseline).abs().amax(dim=(0, 2))
    affected = torch.nonzero(difference > 0).flatten()
    first = int(affected.min()) if affected.numel() else int(frame)
    last = int(affected.max()) if affected.numel() else int(frame)
    influence = decoder_influence_frames(module)
    return {
        "kind": "temporal_influence",
        "stream": stream,
        "frame": int(frame),
        "decoder_receptive_field": int(getattr(module, "decoder_receptive_field", influence + 1)),
        "decoder_influence_frames": int(influence),
        "first_changed_frame": first,
        "last_changed_frame": last,
        "frames_before": int(frame) - first,
        "frames_after": last - int(frame),
        # A window shorter than frame + influence cannot observe the whole tail, so
        # "the effect stopped" is not a statement this measurement can make.
        "temporal_probe_complete": bool(last + 1 <= frames and int(frame) + influence < frames),
        "within_contract": bool(last - int(frame) <= int(influence)),
    }


def locality_report(
    model: torch.nn.Module,
    target_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    *,
    slices: Sequence[slice],
    start: int,
    stop: int,
    target_joints: Sequence[int],
    feature_support: Sequence[int] | torch.Tensor | None = None,
    kinematic: KinematicContext | None = None,
    influence_frames: int = DECODER_INFLUENCE_FRAMES,
) -> dict[str, Any]:
    """Measures what a locally supported token swap actually changes.

    ``slices`` is the coordinate support replaced by the donor inside the
    half-open frame interval ``[start, stop)``; ``target_joints`` is the
    representation-independent body part the swap is supposed to edit (an empty
    sequence means every joint is the target, i.e. a whole-body edit);
    ``feature_support`` lists the motion features those coordinates own, so the
    off-target feature change is measured against the right complement.  When
    it is omitted the feature-side report is skipped.
    """
    module = getattr(model, "module", model)
    device = module_device(module)
    if kinematic is not None:
        kinematic = kinematic.to(device)
    target_indices = target_indices.detach().to(device).long()
    donor_indices = donor_indices.detach().to(device).long()
    if target_indices.shape != donor_indices.shape or target_indices.ndim != 3:
        raise ValueError("locality_report needs matching [B, T, 40] token tensors")
    frames = target_indices.shape[1]
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= frames:
        raise ValueError(f"edit interval [{start}, {stop}) must be inside [0, {frames})")

    edited = target_indices.clone()
    for coordinate_slice in slices:
        edited[:, start:stop, coordinate_slice] = donor_indices[:, start:stop, coordinate_slice]
    support = torch.zeros(target_indices.shape[-1], dtype=torch.bool, device=device)
    for coordinate_slice in slices:
        support[coordinate_slice] = True

    with torch.no_grad():
        decoded = module.decode_from_indices(torch.cat((target_indices, edited), dim=0))
    batch = target_indices.shape[0]
    target_decoded, edited_decoded = decoded[:batch], decoded[batch:]
    change = (edited_decoded - target_decoded).abs()

    influence_stop = min(frames, stop + int(influence_frames))
    report: dict[str, Any] = {
        "kind": "locality",
        "support_coordinates": int(support.sum()),
        "support_fraction": float(support.float().mean()),
        # How much of the support actually differs from the donor: a zero
        # decoded change is only informative next to this number.
        "support_token_change_fraction": float(
            (edited != target_indices)[..., support].float().mean()
        )
        if bool(support.any())
        else 0.0,
        "edit_interval": [start, stop],
        "influence_interval": [start, influence_stop],
        "decoder_influence_frames": int(influence_frames),
        "pre_edit_unchanged": bool(float(change[:, :start].max() if start > 0 else 0.0) == 0.0),
        "post_influence_unchanged": bool(
            float(change[:, influence_stop:].max()) == 0.0 if influence_stop < frames else True
        ),
    }
    if feature_support is not None:
        feature_mask = _as_feature_mask(feature_support, change.shape[-1], device)
        off_target = change[..., ~feature_mask]
        in_support = change[..., feature_mask]
        report["edit_feature_mean"] = float(in_support.mean()) if in_support.numel() else 0.0
        report["off_target_feature_max"] = float(off_target.max()) if off_target.numel() else 0.0
        report["off_target_feature_mean"] = float(off_target.mean()) if off_target.numel() else 0.0
    else:
        report["edit_feature_mean"] = None
        report["off_target_feature_max"] = None
        report["off_target_feature_mean"] = None
    nonzero_frames = torch.nonzero(change.amax(dim=(0, 2)) > 0).flatten()
    report["changed_frame_span"] = (
        [int(nonzero_frames.min()), int(nonzero_frames.max())]
        if nonzero_frames.numel()
        else [start, start]
    )
    velocity_step = (
        ((edited_decoded[:, 1:] - edited_decoded[:, :-1]) - (target_decoded[:, 1:] - target_decoded[:, :-1]))
        .abs()
        .amax(dim=(0, 2))
    )
    report["boundary_velocity_step_at_start"] = (
        float(velocity_step[start - 1]) if start > 0 else 0.0
    )
    report["boundary_velocity_step_at_stop"] = float(velocity_step[stop - 1]) if stop >= 2 else 0.0
    lower = max(start - 1, 0)
    upper = max(influence_stop - 1, lower + 1)
    report["max_velocity_step_in_influence"] = float(velocity_step[lower:upper].max())

    if kinematic is not None:
        positions, root_positions, root_rotations = _world_kinematics(
            torch.cat((target_decoded, edited_decoded), dim=0), kinematic
        )
        target_positions, edited_positions = positions[:batch], positions[batch:]
        joint_change = (edited_positions - target_positions).norm(dim=-1)  # [B, T, J]
        # An empty target set means the whole body is the edit target.
        target = sorted(set(int(joint) for joint in target_joints))
        if not target:
            target = list(range(joint_change.shape[-1]))
        descendants = sorted(kinematic_descendants(kinematic.parents, target) - set(target))
        non_target = sorted(set(range(joint_change.shape[-1])) - set(target) - set(descendants))
        joints = torch.as_tensor(target, dtype=torch.long, device=device)
        report["kinematics"] = {
            "target_joints": [kinematic.names[index] for index in joints.tolist()],
            "target_joint_change": float(joint_change[..., joints].mean()),
            "descendant_joints": [kinematic.names[index] for index in descendants],
            "descendant_joint_change": float(joint_change[..., descendants].mean())
            if descendants
            else 0.0,
            "non_target_joints": [kinematic.names[index] for index in non_target],
            "non_target_joint_change_mean": float(joint_change[..., non_target].mean())
            if non_target
            else 0.0,
            "non_target_joint_change_max": float(joint_change[..., non_target].max())
            if non_target
            else 0.0,
            "root_position_change": float(
                (root_positions[batch:, 1:] - root_positions[:batch, 1:]).abs().mean()
            ),
            "root_rotation_change": float(
                quat.torch_quat_angle(
                    root_rotations[batch:, 1:], root_rotations[:batch, 1:]
                ).mean()
            ),
        }
        target_contacts = _inferred_contacts(target_positions, kinematic)
        edited_contacts = _inferred_contacts(edited_positions, kinematic)
        if target_contacts is not None and edited_contacts is not None:
            report["kinematics"]["contact_flip_rate"] = float(
                (edited_contacts != target_contacts).any(dim=-1).float().mean()
            )
        report["kinematics"]["contact_feature_flip_rate"] = float(
            (change[..., -2:] > 0).any(dim=-1).float().mean()
        )
    return report


def json_dumps(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=False, default=str)


# ---------------------------------------------------------------------------
# store access: the shared, version-agnostic readers live in nef_data


def is_packed_store(store: Any) -> bool:
    """True for the schema-v4 packed reader, False for the v3 row store."""
    from stylized_motion.learning.nef_data import is_packed_store as _is_packed

    return _is_packed(store)


def split_clip_geometry(store: Any, clip_idx: int) -> tuple[int, int, int]:
    from stylized_motion.learning.nef_data import split_clip_geometry as _geometry

    return _geometry(store, clip_idx)


def read_probe_window(
    store: Any,
    request: Any,
    *,
    history: int,
    shards: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Reads ``[history + target_frames]`` frames of one sampler request.

    Delegates to the shared reader so the probes, the NEF report and the training
    loader cannot drift apart.
    """
    from stylized_motion.learning.nef_data import read_clip_window

    return read_clip_window(
        store,
        int(request.variant_idx),
        int(request.target_start),
        int(request.target_frames),
        history=int(history),
        shards=shards,
    )[0]


def model_space_window(
    window: np.ndarray,
    store: Any,
    feature_stats: Mapping[str, object],
) -> torch.Tensor:
    """Re-normalizes a *store-normalized* window into the checkpoint's space."""
    from stylized_motion.learning.nef_data import model_space_window as _model_space

    return _model_space(window, store, feature_stats)


def store_normalized_window(store: Any, window: np.ndarray) -> np.ndarray:
    """Raw on-disk frames -> the store's own normalized space (see nef_data)."""
    from stylized_motion.learning.nef_data import store_normalized_window as _normalized

    return _normalized(store, window)


def module_device(model: torch.nn.Module) -> torch.device:
    """Device of a module that may hold no parameters (buffers only)."""
    from stylized_motion.learning.nef_data import module_device as _device

    return _device(model)


def json_dumps(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=False, default=str)


# ---------------------------------------------------------------------------
# store access shared by the probe scripts (schema v3 and v4)


def is_packed_store(store: Any) -> bool:
    """True for the schema-v4 packed reader, False for the v3 row store."""
    return hasattr(store, "clip_offset")


def split_clip_geometry(store: Any, clip_idx: int) -> tuple[int, int, int]:
    """``(shard_idx, offset, length)`` of one logical clip/range row."""
    clip_idx = int(clip_idx)
    if is_packed_store(store):
        if not 0 <= clip_idx < len(store.clip_shard):
            raise IndexError(f"Invalid clip index {clip_idx}")
        return (
            int(store.clip_shard[clip_idx]),
            int(store.clip_offset[clip_idx]),
            int(store.clip_length[clip_idx]),
        )
    if not 0 <= clip_idx < len(store.range_starts):
        raise IndexError(f"Invalid clip index {clip_idx}")
    return (
        int(store.range_shard_indices[clip_idx]),
        int(store.range_starts[clip_idx]),
        int(store.range_stops[clip_idx]) - int(store.range_starts[clip_idx]),
    )


def read_probe_window(
    store: Any,
    request: Any,
    *,
    history: int,
    shards: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Reads ``[history + target_frames]`` frames of one sampler request.

    The window never leaves its logical clip; frames before the clip start are
    left-padded by repeating the first stored frame, exactly like the training
    loader's history handling.  Padding frames are copies, so a probe measures
    decoder behaviour, not fabricated motion.
    """
    clip_idx = int(request.variant_idx)
    shard_idx, offset, length = split_clip_geometry(store, clip_idx)
    start = int(request.target_start)
    frames = int(request.target_frames)
    if start < offset or start + frames > offset + length:
        raise IndexError(
            f"Window [{start}, {start + frames}) leaves clip {clip_idx} interval "
            f"[{offset}, {offset + length})"
        )
    read_start = max(offset, start - int(history))
    read_frames = start + frames - read_start
    array = None if shards is None else shards.get(shard_idx)
    if array is None:
        files = getattr(store, "shard_files", None) or getattr(store, "motion_files")
        array = np.load(files[shard_idx], mmap_mode="r", allow_pickle=False)
        if shards is not None:
            shards[shard_idx] = array
    window = np.asarray(array[read_start : read_start + read_frames], dtype=np.float32).copy()
    left_pad = read_start - (start - int(history))
    if left_pad > 0:
        window = np.concatenate((np.repeat(window[:1], left_pad, axis=0), window), axis=0)
    return window


def model_space_window(
    window: np.ndarray,
    store: Any,
    feature_stats: Mapping[str, object],
) -> torch.Tensor:
    """Re-normalizes a raw store window into the checkpoint's feature space."""
    from stylized_motion.learning.nef_eval import model_space

    return model_space(window, store, feature_stats)


__all__ = [
    "PROBE_PROTOCOL_REVISION",
    "influence_profile",
    "shared_legal_support",
    "signed_pulse_perturbations",
    "signed_span_perturbations",
    "stratified_span_probe",
    "CSV_COLUMNS",
    "DECODER_INFLUENCE_FRAMES",
    "KinematicContext",
    "LevelGeometryProbe",
    "TokenPerturbation",
    "adjacent_perturbations",
    "decoder_influence_frames",
    "far_perturbations",
    "is_packed_store",
    "json_dumps",
    "kinematic_descendants",
    "model_space_window",
    "module_device",
    "read_probe_window",
    "split_clip_geometry",
    "store_normalized_window",
    "locality_report",
    "probe_csv_rows",
    "temporal_influence_width",
    "write_probe_csv",
]
