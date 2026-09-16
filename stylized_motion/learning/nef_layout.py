"""Node-Edge Factorized FSQ layout: skeleton-aware streams, token slices and feature ownership.

The tables below are the executable form of the NEF-FSQ design: joints are
resolved by skeleton *name* and the topology is checked against the recorded
chains, so Geno numeric joint indices are never reused for SOMA.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


# Coordinate order is a persisted interface: token slice s is stable across
# checkpoints, token stores and generator contracts.
NEF_STREAM_NAMES: tuple[str, ...] = (
    "global",
    "torso_node",
    "head_node",
    "left_arm_node",
    "right_arm_node",
    "left_leg_node",
    "right_leg_node",
    "hips_edge",
    "head_edge",
    "left_shoulder_edge",
    "right_shoulder_edge",
    "left_leg_edge",
    "right_leg_edge",
)

NEF_STREAM_COORDINATES: dict[str, int] = {
    "global": 4,
    "torso_node": 4,
    "head_node": 2,
    "left_arm_node": 4,
    "right_arm_node": 4,
    "left_leg_node": 4,
    "right_leg_node": 4,
    "hips_edge": 4,
    "head_edge": 2,
    "left_shoulder_edge": 2,
    "right_shoulder_edge": 2,
    "left_leg_edge": 2,
    "right_leg_edge": 2,
}

# Left/right streams share input projection, FSQ and output head per family.
NEF_FAMILY_STREAMS: dict[str, tuple[str, ...]] = {
    "global": ("global",),
    "torso_node": ("torso_node",),
    "head_node": ("head_node",),
    "arm_node": ("left_arm_node", "right_arm_node"),
    "leg_node": ("left_leg_node", "right_leg_node"),
    "hips_edge": ("hips_edge",),
    "head_edge": ("head_edge",),
    "shoulder_edge": ("left_shoulder_edge", "right_shoulder_edge"),
    "leg_edge": ("left_leg_edge", "right_leg_edge"),
}
NEF_FAMILIES: tuple[str, ...] = tuple(NEF_FAMILY_STREAMS)
NEF_STREAM_FAMILY: dict[str, str] = {
    stream: family for family, streams in NEF_FAMILY_STREAMS.items() for stream in streams
}

# Child region -> (Node stream, unique incoming Edge stream).
NEF_EDIT_PARTS: dict[str, tuple[str, str]] = {
    "torso": ("torso_node", "hips_edge"),
    "head": ("head_node", "head_edge"),
    "left_arm": ("left_arm_node", "left_shoulder_edge"),
    "right_arm": ("right_arm_node", "right_shoulder_edge"),
    "left_leg": ("left_leg_node", "left_leg_edge"),
    "right_leg": ("right_leg_node", "right_leg_edge"),
}

NEF_ARCHITECTURE_VERSION = 1
NEF_VARIANT = "independent"

# The only region token accepted next to the edit parts: the full 13-stream body.
NEF_WHOLE_BODY_REGION = "whole_body"


@dataclass(frozen=True)
class NEFSkeletonSpec:
    """Name-keyed Node/Edge partition for one known skeleton."""

    name: str
    chains: tuple[tuple[str, ...], ...]
    streams: Mapping[str, tuple[str, ...]]

    @property
    def joints(self) -> frozenset[str]:
        required = {"Simulation", "Hips"}
        required.update(joint for chain in self.chains for joint in chain)
        required.update(joint for joints in self.streams.values() for joint in joints)
        return frozenset(required)


GENO_SKELETON = NEFSkeletonSpec(
    name="geno",
    chains=(
        ("Simulation", "Hips", "Spine", "Spine1", "Spine2", "Spine3"),
        ("Spine3", "Neck", "Neck1", "Head"),
        ("Spine3", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"),
        ("Spine3", "RightShoulder", "RightArm", "RightForeArm", "RightHand"),
        ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"),
        ("Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"),
    ),
    streams={
        "torso_node": ("Spine", "Spine1", "Spine2", "Spine3"),
        "head_node": ("Neck1", "Head"),
        "left_arm_node": ("LeftArm", "LeftForeArm", "LeftHand"),
        "right_arm_node": ("RightArm", "RightForeArm", "RightHand"),
        "left_leg_node": ("LeftLeg", "LeftFoot", "LeftToeBase"),
        "right_leg_node": ("RightLeg", "RightFoot", "RightToeBase"),
        "hips_edge": ("Hips",),
        "head_edge": ("Neck",),
        "left_shoulder_edge": ("LeftShoulder",),
        "right_shoulder_edge": ("RightShoulder",),
        "left_leg_edge": ("LeftUpLeg",),
        "right_leg_edge": ("RightUpLeg",),
    },
)

SOMA_SKELETON = NEFSkeletonSpec(
    name="soma",
    chains=(
        ("Simulation", "Hips", "Spine1", "Spine2", "Chest"),
        ("Chest", "Neck1", "Neck2", "Head"),
        ("Head", "Jaw"),
        ("Head", "LeftEye"),
        ("Head", "RightEye"),
        ("Chest", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"),
        ("Chest", "RightShoulder", "RightArm", "RightForeArm", "RightHand"),
        ("Hips", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"),
        ("Hips", "RightLeg", "RightShin", "RightFoot", "RightToeBase"),
    ),
    streams={
        "torso_node": ("Spine1", "Spine2", "Chest"),
        "head_node": ("Neck2", "Head", "Jaw", "LeftEye", "RightEye"),
        "left_arm_node": ("LeftArm", "LeftForeArm", "LeftHand"),
        "right_arm_node": ("RightArm", "RightForeArm", "RightHand"),
        "left_leg_node": ("LeftShin", "LeftFoot", "LeftToeBase"),
        "right_leg_node": ("RightShin", "RightFoot", "RightToeBase"),
        "hips_edge": ("Hips",),
        "head_edge": ("Neck1",),
        "left_shoulder_edge": ("LeftShoulder",),
        "right_shoulder_edge": ("RightShoulder",),
        "left_leg_edge": ("LeftLeg",),
        "right_leg_edge": ("RightLeg",),
    },
)

NEF_SKELETON_SPECS: dict[str, NEFSkeletonSpec] = {
    GENO_SKELETON.name: GENO_SKELETON,
    SOMA_SKELETON.name: SOMA_SKELETON,
}


@dataclass(frozen=True)
class NEFLayout:
    skeleton: str
    names: tuple[str, ...]
    parents: tuple[int, ...]
    stream_joint_indices: tuple[tuple[int, ...], ...]

    @classmethod
    def from_skeleton(
        cls,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        *,
        skeleton: str | None = None,
    ) -> "NEFLayout":
        names_tuple = tuple(str(name) for name in names)
        parents_tuple = tuple(int(parent) for parent in parents)
        cls._validate_skeleton_structure(names_tuple, parents_tuple)
        spec = NEF_SKELETON_SPECS[skeleton] if skeleton is not None else cls._match_spec(names_tuple)
        if skeleton is not None and set(names_tuple) != set(spec.joints):
            raise ValueError(
                f"Skeleton {skeleton!r} joints do not match the recorded NEF partition; "
                f"missing={sorted(spec.joints - set(names_tuple))}, extra={sorted(set(names_tuple) - spec.joints)}"
            )
        cls._validate_chains(spec, names_tuple, parents_tuple)
        stream_indices = cls._resolve_streams(spec, names_tuple)
        cls._validate_streams(spec, names_tuple, stream_indices)
        return cls(
            skeleton=spec.name,
            names=names_tuple,
            parents=parents_tuple,
            stream_joint_indices=tuple(
                () if stream == "global" else stream_indices[stream] for stream in NEF_STREAM_NAMES
            ),
        )

    @staticmethod
    def _validate_skeleton_structure(names: tuple[str, ...], parents: tuple[int, ...]) -> None:
        if len(names) != len(parents):
            raise ValueError(f"names has {len(names)} entries but parents has {len(parents)}")
        if len(set(names)) != len(names):
            raise ValueError("NEF-FSQ rejects duplicated joint names")
        roots = [index for index, parent in enumerate(parents) if parent < 0]
        if roots != [0]:
            raise ValueError(f"NEF-FSQ requires exactly one root at index 0, got root indices {roots}")
        if names[0] != "Simulation":
            raise ValueError(f"NEF-FSQ requires joint 0 to be 'Simulation', got {names[0]!r}")
        if len(names) < 2 or names[1] != "Hips" or parents[1] != 0:
            raise ValueError("NEF-FSQ requires joint 1 to be 'Hips' parented to 'Simulation'")
        for joint, parent in enumerate(parents):
            if parent >= joint and parent >= 0:
                raise ValueError(f"parents must be parent-ordered; joint {joint} has parent {parent}")

    @staticmethod
    def _match_spec(names: tuple[str, ...]) -> NEFSkeletonSpec:
        name_set = set(names)
        for spec in NEF_SKELETON_SPECS.values():
            if name_set == set(spec.joints):
                return spec
        known = {spec.name: sorted(spec.joints) for spec in NEF_SKELETON_SPECS.values()}
        raise ValueError(
            "Unknown skeleton for NEF-FSQ; no automatic remapping is performed. "
            f"got={sorted(name_set)} known={ {name: len(joints) for name, joints in known.items()} }"
        )

    @staticmethod
    def _validate_chains(
        spec: NEFSkeletonSpec,
        names: tuple[str, ...],
        parents: tuple[int, ...],
    ) -> None:
        index_of = {name: index for index, name in enumerate(names)}
        for chain in spec.chains:
            for parent_name, child_name in zip(chain[:-1], chain[1:]):
                if parents[index_of[child_name]] != index_of[parent_name]:
                    raise ValueError(
                        f"{spec.name} chain {chain} expects {child_name!r} parented to {parent_name!r}, "
                        f"got parent {names[parents[index_of[child_name]]]!r}"
                    )

    @staticmethod
    def _resolve_streams(
        spec: NEFSkeletonSpec,
        names: tuple[str, ...],
    ) -> dict[str, tuple[int, ...]]:
        index_of = {name: index for index, name in enumerate(names)}
        missing = sorted(joint for joints in spec.streams.values() for joint in joints if joint not in index_of)
        if missing:
            raise ValueError(f"{spec.name} NEF partition references missing joints: {missing}")
        expected = set(spec.streams)
        supplied = {stream for stream in NEF_STREAM_NAMES if stream != "global"}
        if expected != supplied:
            raise ValueError(
                f"{spec.name} NEF partition must define every non-global stream; "
                f"missing={sorted(supplied - expected)}, extra={sorted(expected - supplied)}"
            )
        return {
            stream: tuple(index_of[joint] for joint in spec.streams[stream])
            for stream in supplied
        }

    @staticmethod
    def _validate_streams(
        spec: NEFSkeletonSpec,
        names: tuple[str, ...],
        stream_indices: Mapping[str, tuple[int, ...]],
    ) -> None:
        seen: dict[int, str] = {}
        for stream, joints in stream_indices.items():
            if not joints:
                raise ValueError(f"NEF stream {stream!r} owns no joints")
            for joint in joints:
                if joint in seen:
                    raise ValueError(
                        f"Joint {names[joint]!r} is owned by both {seen[joint]!r} and {stream!r}"
                    )
                seen[joint] = stream
        expected = set(range(1, len(names)))  # every non-root joint
        if set(seen) != expected:
            raise ValueError(
                f"NEF partition must cover every non-root joint; missing={sorted(expected - set(seen))}"
            )
        for left, right in (
            ("left_arm_node", "right_arm_node"),
            ("left_leg_node", "right_leg_node"),
            ("left_shoulder_edge", "right_shoulder_edge"),
            ("left_leg_edge", "right_leg_edge"),
        ):
            if len(stream_indices[left]) != len(stream_indices[right]):
                raise ValueError(
                    f"{left} and {right} must own the same joint count for shared family weights"
                )

    @property
    def num_joints(self) -> int:
        return len(self.names)

    def stream_joints(self, stream: str) -> tuple[int, ...]:
        index = NEF_STREAM_NAMES.index(stream)
        return self.stream_joint_indices[index]

    @property
    def stream_slices(self) -> dict[str, slice]:
        start = 0
        result: dict[str, slice] = {}
        for stream in NEF_STREAM_NAMES:
            end = start + NEF_STREAM_COORDINATES[stream]
            result[stream] = slice(start, end)
            start = end
        return result

    @property
    def num_coordinates(self) -> int:
        return sum(NEF_STREAM_COORDINATES.values())

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return NEF_STREAM_NAMES

    @property
    def coordinate_counts(self) -> dict[str, int]:
        return {stream: NEF_STREAM_COORDINATES[stream] for stream in NEF_STREAM_NAMES}

    def validate_motion_dim(self, motion_dim: int) -> None:
        expected = 9 * self.num_joints + 5
        if int(motion_dim) != expected:
            raise ValueError(f"NEF-FSQ feature layout requires motion_dim={expected}, got {motion_dim}")

    def feature_indices(self, motion_dim: int) -> dict[str, torch.Tensor]:
        """Returns a disjoint, complete partition of the local motion features."""
        self.validate_motion_dim(motion_dim)
        num_joints = self.num_joints
        rotation_start = 9
        hips_velocity_start = rotation_start + (num_joints - 1) * 6
        angular_start = hips_velocity_start + 3
        contact_start = angular_start + (num_joints - 1) * 3

        def joint_features(joint: int) -> list[int]:
            feature_joint = joint - 1
            return [
                *range(rotation_start + 6 * feature_joint, rotation_start + 6 * (feature_joint + 1)),
                *range(angular_start + 3 * feature_joint, angular_start + 3 * (feature_joint + 1)),
            ]

        result: dict[str, torch.Tensor] = {
            "global": torch.tensor(
                [0, 1, 2, 3, 4, 5, contact_start, contact_start + 1], dtype=torch.long
            )
        }
        for stream in NEF_STREAM_NAMES:
            if stream == "global":
                continue
            joints = self.stream_joints(stream)
            if stream == "hips_edge":
                if joints != (1,):
                    raise ValueError("The Simulation->Hips Edge must own the Hips joint alone")
                indices = [
                    6, 7, 8,
                    hips_velocity_start, hips_velocity_start + 1, hips_velocity_start + 2,
                    *joint_features(1),
                ]
            else:
                indices = [feature for joint in joints for feature in joint_features(joint)]
            result[stream] = torch.tensor(indices, dtype=torch.long)

        flattened = torch.cat([result[stream] for stream in NEF_STREAM_NAMES])
        if flattened.numel() != motion_dim or not torch.equal(
            torch.sort(flattened).values, torch.arange(motion_dim, dtype=torch.long)
        ):
            raise RuntimeError("NEF streams must be a disjoint complete motion feature partition")
        global_indices = set(result["global"].tolist())
        if {6, 7, 8, hips_velocity_start, hips_velocity_start + 1, hips_velocity_start + 2} & global_indices:
            raise RuntimeError("Hips position/velocity must belong to the Simulation->Hips Edge only")
        if {contact_start, contact_start + 1} != {index for index in global_indices if index >= contact_start}:
            raise RuntimeError("Contacts must belong to Global only")
        for family, streams in NEF_FAMILY_STREAMS.items():
            dims = {result[stream].numel() for stream in streams}
            if len(dims) != 1:
                raise ValueError(f"NEF family {family!r} streams must share one feature width, got {sorted(dims)}")
        return result

    def to_dict(self) -> dict[str, Any]:
        slices = self.stream_slices
        return {
            "skeleton": self.skeleton,
            "names": list(self.names),
            "parents": list(self.parents),
            "stream_order": list(NEF_STREAM_NAMES),
            "stream_coordinates": self.coordinate_counts,
            "stream_slices": {stream: [slices[stream].start, slices[stream].stop] for stream in NEF_STREAM_NAMES},
            "stream_joint_indices": {
                stream: list(self.stream_joints(stream)) for stream in NEF_STREAM_NAMES
            },
            "families": {family: list(streams) for family, streams in NEF_FAMILY_STREAMS.items()},
        }

    def layout_hash(self) -> str:
        """Stable sha256 over the persisted layout payload.

        Two layouts hash equally exactly when their skeleton, stream order,
        coordinate spans and ownership tables agree, so a tokenizer, a token
        store and a generator can fingerprint the same alphabet without
        comparing dictionaries field by field.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stream_graph(self) -> tuple[tuple[str, str, str], ...]:
        """Stream adjacency implied by the recorded skeleton chains.

        Each entry is ``(parent_stream, child_stream, relation)`` where the
        relation is one of:

        - ``global_to_root``: the root stream to the stream owning the body root;
        - ``node_to_edge``: a Node stream to the Edge stream of a child region;
        - ``edge_to_child``: an Edge stream to the Node stream it parents;
        - ``edge_to_edge`` / ``node_to_node``: same-kind hand-off between regions.

        These are *relations*, not a causal claim: an Edge stream owns the
        joint between two regions, so a token in it constrains both sides.
        """
        owner = {
            self.names[joint]: stream
            for stream in NEF_STREAM_NAMES
            for joint in self.stream_joints(stream)
        }
        spec = NEF_SKELETON_SPECS[self.skeleton]
        relations: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for chain in spec.chains:
            for parent_name, child_name in zip(chain[:-1], chain[1:]):
                parent_stream = owner.get(parent_name, "global" if parent_name == "Simulation" else None)
                child_stream = owner.get(child_name)
                if parent_stream is None or child_stream is None or parent_stream == child_stream:
                    continue
                relation = (parent_stream, child_stream, _stream_relation(parent_stream, child_stream))
                if relation not in seen:
                    seen.add(relation)
                    relations.append(relation)
        return tuple(relations)

    def coordinate_metadata(self) -> tuple[dict[str, object], ...]:
        """One JSON-safe record per FSQ coordinate, in canonical coordinate order."""
        slices = self.stream_slices
        feature_indices = self.feature_indices(9 * self.num_joints + 5)
        records: list[dict[str, object]] = []
        for stream_index, stream in enumerate(NEF_STREAM_NAMES):
            span = slices[stream]
            joints = [self.names[joint] for joint in self.stream_joints(stream)]
            for offset in range(span.stop - span.start):
                records.append(
                    {
                        "coordinate": span.start + offset,
                        "stream": stream,
                        "stream_index": stream_index,
                        "stream_slice": [span.start, span.stop],
                        "coordinate_in_stream": offset,
                        "family": NEF_STREAM_FAMILY[stream],
                        "joints": joints,
                        "feature_count": int(feature_indices[stream].numel()),
                    }
                )
        return tuple(records)

    def region_streams(self, regions: str | Sequence[str], *, graph_radius: int = 0) -> tuple[str, ...]:
        """Streams covered by named edit regions under a graph radius.

        The region vocabulary is the toolkit's own: the parts of
        :data:`NEF_EDIT_PARTS` plus ``whole_body``.  Radii follow the design:

        - ``0``: the Node stream of each part only;
        - ``1``: the designed part, i.e. its Node stream and incoming Edge stream;
        - ``>= 2``: that part expanded by ``radius - 1`` undirected hops over
          :meth:`stream_graph`.

        Streams are returned in canonical :data:`NEF_STREAM_NAMES` order.
        """
        names = [regions] if isinstance(regions, str) else [str(name) for name in regions]
        if not names:
            raise ValueError("regions must name at least one NEF edit part")
        radius = int(graph_radius)
        if radius < 0:
            raise ValueError(f"graph_radius must be non-negative, got {graph_radius}")
        unknown = sorted(
            name for name in names if name != NEF_WHOLE_BODY_REGION and name not in NEF_EDIT_PARTS
        )
        if unknown:
            raise ValueError(
                f"Unknown NEF region(s) {unknown}; expected {sorted(NEF_EDIT_PARTS)} or "
                f"{NEF_WHOLE_BODY_REGION!r}"
            )
        if NEF_WHOLE_BODY_REGION in names:
            return tuple(NEF_STREAM_NAMES)
        selected: set[str] = set()
        for name in names:
            selected.update(nef_edit_streams(name, full_part=radius >= 1))
        if radius >= 2:
            selected = self._expand_streams(selected, radius - 1)
        return tuple(stream for stream in NEF_STREAM_NAMES if stream in selected)

    def _expand_streams(self, streams: set[str], hops: int) -> set[str]:
        adjacency: dict[str, set[str]] = {}
        for parent, child, _ in self.stream_graph():
            adjacency.setdefault(parent, set()).add(child)
            adjacency.setdefault(child, set()).add(parent)
        selected = set(streams)
        frontier = set(streams)
        for _ in range(hops):
            frontier = {n for stream in frontier for n in adjacency.get(stream, set())} - selected
            if not frontier:
                break
            selected |= frontier
        return selected

    def make_region_mask(
        self,
        regions: str | Sequence[str],
        *,
        graph_radius: int = 0,
        frame_range: tuple[int, int] | None = None,
        length: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Boolean ``[length, 40]`` support mask for the named regions.

        ``frame_range`` is a half-open ``(start, stop)`` interval inside
        ``length``; ``None`` keeps every frame.  Callers never hand-write
        coordinate slices: the stream span comes from this layout.
        """
        length = int(length)
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        streams = self.region_streams(regions, graph_radius=graph_radius)
        slices = self.stream_slices
        mask = torch.zeros((length, self.num_coordinates), dtype=torch.bool, device=device)
        for stream in streams:
            mask[:, slices[stream]] = True
        if frame_range is not None:
            start, stop = (int(frame_range[0]), int(frame_range[1]))
            if not 0 <= start < stop <= length:
                raise ValueError(
                    f"frame_range {frame_range} must be a non-empty half-open interval inside [0, {length})"
                )
            frames = torch.zeros(length, dtype=torch.bool, device=mask.device)
            frames[start:stop] = True
            mask &= frames.unsqueeze(-1)
        return mask


def _stream_relation(parent_stream: str, child_stream: str) -> str:
    """Names how a stream hands off to the next one along a skeleton chain."""
    if parent_stream == "global":
        return "global_to_root"
    if child_stream == "global":
        raise ValueError("No skeleton chain may enter the global stream")
    parent_kind = "node" if parent_stream.endswith("_node") else "edge"
    child_kind = "node" if child_stream.endswith("_node") else "edge"
    if (parent_kind, child_kind) == ("node", "edge"):
        return "node_to_edge"
    if (parent_kind, child_kind) == ("edge", "node"):
        return "edge_to_child"
    return f"{parent_kind}_to_{child_kind}"


def nef_edit_streams(part: str, *, full_part: bool) -> tuple[str, ...]:
    """Strict edits replace the Node stream only; full-part edits add the incoming Edge."""
    if part not in NEF_EDIT_PARTS:
        raise ValueError(f"Unknown NEF edit part {part!r}; expected one of {sorted(NEF_EDIT_PARTS)}")
    node_stream, edge_stream = NEF_EDIT_PARTS[part]
    return (node_stream, edge_stream) if full_part else (node_stream,)


__all__ = [
    "GENO_SKELETON",
    "NEF_ARCHITECTURE_VERSION",
    "NEF_EDIT_PARTS",
    "NEF_FAMILIES",
    "NEF_FAMILY_STREAMS",
    "NEF_SKELETON_SPECS",
    "NEF_STREAM_COORDINATES",
    "NEF_STREAM_FAMILY",
    "NEF_STREAM_NAMES",
    "NEF_VARIANT",
    "NEF_WHOLE_BODY_REGION",
    "NEFLayout",
    "NEFSkeletonSpec",
    "SOMA_SKELETON",
    "nef_edit_streams",
]
