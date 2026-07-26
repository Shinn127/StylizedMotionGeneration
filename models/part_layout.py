from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


PART_NAMES = ("torso", "left_leg", "right_leg", "left_arm", "right_arm")
FEATURE_GROUP_NAMES = ("global", *PART_NAMES)
GROUP_NAMES = ("global", "sync", *PART_NAMES)
GROUP_COORDINATES = {
    "global": 6,
    "sync": 4,
    "torso": 6,
    "left_leg": 7,
    "right_leg": 7,
    "left_arm": 5,
    "right_arm": 5,
}


def _normalized_name(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def _side_of(name: str) -> str | None:
    normalized = _normalized_name(name)
    if normalized.startswith("left"):
        return "left"
    if normalized.startswith("right"):
        return "right"
    return None


def _part_for_joint(name: str) -> str:
    normalized = _normalized_name(name)
    side = _side_of(name)
    if side is None:
        return "torso"
    if any(token in normalized for token in ("shoulder", "arm", "forearm", "hand", "wrist", "finger", "thumb")):
        return f"{side}_arm"
    if any(token in normalized for token in ("upleg", "leg", "thigh", "calf", "foot", "toe", "ankle")):
        return f"{side}_leg"
    raise ValueError(
        f"Cannot infer a body part for side-specific joint {name!r}. "
        "Pass part_membership explicitly for this skeleton."
    )


def _coerce_joint_index(value: int | str, names: tuple[str, ...]) -> int:
    if isinstance(value, str):
        try:
            return names.index(value)
        except ValueError as exc:
            raise ValueError(f"Unknown joint name in part_membership: {value!r}") from exc
    index = int(value)
    if index < 0 or index >= len(names):
        raise ValueError(f"Joint index {index} is outside [0, {len(names)})")
    return index


@dataclass(frozen=True)
class PartFSQLayout:
    """Static, non-overlapping feature partition for Hierarchical Part-FSQ."""

    names: tuple[str, ...]
    parents: tuple[int, ...]
    root_index: int
    hips_index: int
    toe_indices: tuple[int, int]
    part_joint_indices: tuple[tuple[int, ...], ...]

    @classmethod
    def from_skeleton(
        cls,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        part_membership: Mapping[str, Sequence[int | str]] | None = None,
    ) -> "PartFSQLayout":
        names_tuple = tuple(str(name) for name in names)
        parents_tuple = tuple(int(parent) for parent in parents)
        cls._validate_skeleton(names_tuple, parents_tuple)
        root_index = parents_tuple.index(-1)
        if part_membership is None:
            part_indices = {part: [] for part in PART_NAMES}
            for joint_index, name in enumerate(names_tuple):
                if joint_index != root_index:
                    part_indices[_part_for_joint(name)].append(joint_index)
        else:
            expected = set(PART_NAMES)
            supplied = set(part_membership)
            if supplied != expected:
                raise ValueError(
                    f"part_membership must contain exactly {PART_NAMES}; "
                    f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
                )
            part_indices = {
                part: [_coerce_joint_index(value, names_tuple) for value in part_membership[part]]
                for part in PART_NAMES
            }
        cls._validate_parts(part_indices, root_index, len(names_tuple))
        hips_index = cls._find_hips(names_tuple, parents_tuple, root_index, part_indices["torso"])
        toe_indices = (
            cls._find_foot_endpoint(names_tuple, part_indices["left_leg"], "left"),
            cls._find_foot_endpoint(names_tuple, part_indices["right_leg"], "right"),
        )
        return cls(
            names=names_tuple,
            parents=parents_tuple,
            root_index=root_index,
            hips_index=hips_index,
            toe_indices=toe_indices,
            part_joint_indices=tuple(tuple(part_indices[part]) for part in PART_NAMES),
        )

    @staticmethod
    def _validate_skeleton(names: tuple[str, ...], parents: tuple[int, ...]) -> None:
        if len(names) < 7:
            raise ValueError("Part-FSQ requires a root and at least one joint for each of five body parts")
        if len(names) != len(parents):
            raise ValueError(f"names has {len(names)} entries but parents has {len(parents)}")
        if len(set(names)) != len(names):
            raise ValueError("Joint names must be unique")
        roots = [index for index, parent in enumerate(parents) if parent < 0]
        if roots != [0]:
            raise ValueError(f"Feature layout requires exactly one root at index 0, got root indices {roots}")
        for joint, parent in enumerate(parents):
            if parent >= joint and parent >= 0:
                raise ValueError(f"parents must be parent-ordered; joint {joint} has parent {parent}")

    @staticmethod
    def _validate_parts(part_indices: Mapping[str, Sequence[int]], root_index: int, num_joints: int) -> None:
        seen: set[int] = set()
        for part in PART_NAMES:
            indices = list(part_indices[part])
            if not indices:
                raise ValueError(f"Body part {part!r} has no joints")
            for joint in indices:
                if joint == root_index:
                    raise ValueError("The feature root must not belong to a body part")
                if joint in seen:
                    raise ValueError(f"Joint {joint} belongs to multiple body parts")
                seen.add(joint)
        expected = set(range(num_joints)) - {root_index}
        if seen != expected:
            raise ValueError(f"Part membership must cover all non-root joints; missing={sorted(expected - seen)}")

    @staticmethod
    def _find_hips(
        names: tuple[str, ...],
        parents: tuple[int, ...],
        root_index: int,
        torso_indices: Sequence[int],
    ) -> int:
        for candidate in ("hips", "pelvis"):
            matches = [index for index, name in enumerate(names) if _normalized_name(name) == candidate]
            if len(matches) == 1:
                return matches[0]
        root_children = [index for index, parent in enumerate(parents) if parent == root_index and index in torso_indices]
        if len(root_children) == 1:
            return root_children[0]
        raise ValueError("Could not identify a unique Hips/Pelvis joint")

    @staticmethod
    def _find_foot_endpoint(names: tuple[str, ...], part_indices: Sequence[int], side: str) -> int:
        for token in ("toebase", "toe", "foot", "ankle"):
            matches = [
                joint
                for joint in part_indices
                if _side_of(names[joint]) == side and token in _normalized_name(names[joint])
            ]
            if matches:
                return matches[-1]
        raise ValueError(f"Could not identify a contact endpoint for {side}_leg")

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def group_slices(self) -> dict[str, slice]:
        start = 0
        result = {}
        for group in GROUP_NAMES:
            end = start + GROUP_COORDINATES[group]
            result[group] = slice(start, end)
            start = end
        return result

    @property
    def num_coordinates(self) -> int:
        return sum(GROUP_COORDINATES.values())

    def validate_motion_dim(self, motion_dim: int) -> None:
        expected = 9 * self.num_joints + 5
        if int(motion_dim) != expected:
            raise ValueError(f"Part-FSQ feature layout requires motion_dim={expected}, got {motion_dim}")

    def feature_indices(self, motion_dim: int) -> dict[str, torch.Tensor]:
        """Returns a disjoint, complete partition of the current motion features."""
        self.validate_motion_dim(motion_dim)
        rotation_start = 9
        rotation_end = rotation_start + (self.num_joints - 1) * 6
        hips_velocity_start = rotation_end
        angular_start = hips_velocity_start + 3
        contact_start = angular_start + (self.num_joints - 1) * 3

        def joint_features(joints: Sequence[int]) -> list[int]:
            result: list[int] = []
            for joint in joints:
                feature_joint = joint - 1
                result.extend(range(rotation_start + 6 * feature_joint, rotation_start + 6 * (feature_joint + 1)))
                result.extend(range(angular_start + 3 * feature_joint, angular_start + 3 * (feature_joint + 1)))
            return result

        result = {
            "global": torch.tensor(
                list(range(0, 9))
                + list(range(hips_velocity_start, hips_velocity_start + 3))
                + list(range(contact_start, contact_start + 2)),
                dtype=torch.long,
            ),
        }
        result.update(
            {
                part: torch.tensor(joint_features(joints), dtype=torch.long)
                for part, joints in zip(PART_NAMES, self.part_joint_indices)
            }
        )
        flattened = torch.cat([result[group] for group in FEATURE_GROUP_NAMES])
        expected = torch.arange(motion_dim, dtype=torch.long)
        if flattened.numel() != motion_dim or not torch.equal(torch.sort(flattened).values, expected):
            raise RuntimeError("Part feature groups must be a disjoint complete motion feature partition")
        if result["left_leg"].numel() != result["right_leg"].numel():
            raise ValueError("Left and right leg feature dimensions must match for shared Part-FSQ weights")
        if result["left_arm"].numel() != result["right_arm"].numel():
            raise ValueError("Left and right arm feature dimensions must match for shared Part-FSQ weights")
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "parents": list(self.parents),
            "root_index": self.root_index,
            "hips_index": self.hips_index,
            "toe_indices": list(self.toe_indices),
            "part_joint_indices": {part: list(joints) for part, joints in zip(PART_NAMES, self.part_joint_indices)},
        }
