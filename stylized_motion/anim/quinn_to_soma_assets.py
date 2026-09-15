"""Bind the Quinn (UE5 female mannequin) appearance mesh to the SOMA skeleton.

Inputs:
  * the Quinn GLB -- one skinned mesh, 80 skin joints, two embedded JPEG
    textures and a single A-pose bind pose
  * an existing SOMA resource directory, whose ``SOMA.bin`` carries the
    authoritative 78-bone bind pose (bone names, parents, bind transforms)

Outputs (into ``--output-dir``):
  * ``SOMA.bin``            -- Quinn geometry on the SOMA bones
  * ``SOMA_bind.bvh``       -- copied from the SOMA resource directory
  * ``quinn_base_color.jpg``/``quinn_normal.jpg`` -- extracted GLB textures
  * ``conversion_report.json`` -- every mapping/alignment decision and check
  * the viewer shaders     -- copied from the SOMA resource directory

Quinn's skin weights are remapped onto SOMA bones through the explicit table
below; its twist chains are merged into the parent limbs and its five-segment
spine onto ``Spine1``/``Spine2``/``Chest``.  The mesh is then placed in SOMA's
meter bind frame by a similarity transform per body region, and written through
the same binary contract as ``soma_assets``.

A single global similarity leaves Quinn's wrist joints 8.5 cm and its knees
5.5 cm away from SOMA's: the two rigs carry different proportions and a slightly
different A-pose.  ``--align segment`` (the default) fits the pelvis, torso and
each arm and leg separately and blends them with the skin weights, which brings
every align joint within 4.4 cm and the hands within 1.7 cm; ``--align global``
keeps the single body-wide transform for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stylized_motion.anim.soma_assets import (
    INFLUENCES_PER_VERTEX,
    MAX_BONE_NUM,
    SHADER_FILES,
    write_soma_bin,
)

GLB_MAGIC = 0x46546C67
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
TRIANGLES_MODE = 4
MAX_VERTEX_COUNT = 65_536  # SOMA.bin stores uint16 triangle indices

_COMPONENT_DTYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_COMPONENT_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_SHAPES = {"SCALAR": (), "VEC2": (2,), "VEC3": (3,), "VEC4": (4,), "MAT4": (4, 4)}

_NODE_INDEX_SUFFIX = re.compile(r"_\d+$")  # Sketchfab appends the source node id to every joint name

FINGER_NAMES = ("Index", "Middle", "Ring", "Pinky")
FINGER_SEGMENTS = ("metacarpal", "01", "02", "03")  # Quinn metacarpal -> SOMA <finger>1

# Quinn joint -> SOMA bone.  Rows are explicit; only the regular fingers are
# generated below.  ``spine_05`` and ``GLTF_created_0_rootJoint`` carry no
# weight in this GLB but keep a defined rule so the table stays total.
_MANUAL_MAPPING = {
    "pelvis": "Hips",
    "spine_01": "Spine1",
    "spine_02": "Spine1",
    "spine_03": "Spine2",
    "spine_04": "Chest",
    "spine_05": "Chest",
    "neck_01": "Neck1",
    "neck_02": "Neck2",
    "head": "Head",
    "clavicle_l": "LeftShoulder",
    "upperarm_l": "LeftArm",
    "upperarm_twist_01_l": "LeftArm",
    "upperarm_twist_02_l": "LeftArm",
    "lowerarm_l": "LeftForeArm",
    "lowerarm_twist_01_l": "LeftForeArm",
    "lowerarm_twist_02_l": "LeftForeArm",
    "hand_l": "LeftHand",
    "thumb_01_l": "LeftHandThumb1",
    "thumb_02_l": "LeftHandThumb2",
    "thumb_03_l": "LeftHandThumb3",
    "clavicle_r": "RightShoulder",
    "upperarm_r": "RightArm",
    "upperarm_twist_01_r": "RightArm",
    "upperarm_twist_02_r": "RightArm",
    "lowerarm_r": "RightForeArm",
    "lowerarm_twist_01_r": "RightForeArm",
    "lowerarm_twist_02_r": "RightForeArm",
    "hand_r": "RightHand",
    "thumb_01_r": "RightHandThumb1",
    "thumb_02_r": "RightHandThumb2",
    "thumb_03_r": "RightHandThumb3",
    "thigh_l": "LeftLeg",
    "thigh_twist_01_l": "LeftLeg",
    "thigh_twist_02_l": "LeftLeg",
    "calf_l": "LeftShin",
    "calf_twist_01_l": "LeftShin",
    "calf_twist_02_l": "LeftShin",
    "foot_l": "LeftFoot",
    "ball_l": "LeftToeBase",
    "thigh_r": "RightLeg",
    "thigh_twist_01_r": "RightLeg",
    "thigh_twist_02_r": "RightLeg",
    "calf_r": "RightShin",
    "calf_twist_01_r": "RightShin",
    "calf_twist_02_r": "RightShin",
    "foot_r": "RightFoot",
    "ball_r": "RightToeBase",
}


def _build_mapping() -> dict[str, str]:
    mapping = dict(_MANUAL_MAPPING)
    for side, prefix in (("l", "Left"), ("r", "Right")):
        for finger in FINGER_NAMES:
            stem = finger.lower()
            for slot, source in enumerate(FINGER_SEGMENTS, start=1):
                mapping[f"{stem}_{source}_{side}"] = f"{prefix}Hand{finger}{slot}"
    return mapping


QUINN_TO_SOMA = _build_mapping()

# Joints used to fit the alignment; twist chains and fingers are left out
# because SOMA has no counterpart joint to place them on.
_ALIGN_JOINTS = (
    "pelvis", "spine_02", "spine_03", "spine_04", "neck_01", "neck_02", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l", "ball_l",
    "thigh_r", "calf_r", "foot_r", "ball_r",
)

_SEGMENT_JOINTS = {
    "pelvis": ("pelvis",),
    "torso": ("spine_02", "spine_03", "spine_04", "neck_01", "neck_02", "head", "clavicle_l", "clavicle_r"),
    "left_arm": ("upperarm_l", "lowerarm_l", "hand_l"),
    "right_arm": ("upperarm_r", "lowerarm_r", "hand_r"),
    "left_leg": ("thigh_l", "calf_l", "foot_l", "ball_l"),
    "right_leg": ("thigh_r", "calf_r", "foot_r", "ball_r"),
}
ALIGN_MODES = ("global", "segment")

# Joints whose placement the report spells out (SOMA bind meters).
REPORT_JOINTS = (
    "Hips", "Chest", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot",
)

TEXTURE_OUTPUT_NAMES = {"base_color": "quinn_base_color", "normal": "quinn_normal"}


@dataclass
class QuinnMesh:
    """Geometry, skin and textures of the Quinn GLB in its own bind space."""

    positions: np.ndarray  # [V, 3] float32, Y-up, feet near y=0
    normals: np.ndarray  # [V, 3] float32
    texcoords: np.ndarray  # [V, 2] float32
    indices: np.ndarray  # [3T] int64
    joint_indices: np.ndarray  # [V, 4] int64, skin joint slots
    joint_weights: np.ndarray  # [V, 4] float32
    joint_names: list[str]  # skin order, Sketchfab node suffix stripped
    joint_parents: np.ndarray  # [J] int64, skin slot parent or -1
    joint_positions: np.ndarray  # [J, 3] float64, bind joint centers
    textures: list[tuple[str, bytes]]  # (mime type, payload) per glTF image


@dataclass
class SomaBind:
    """The 78-bone mesh bind pose read back from an existing ``SOMA.bin``."""

    names: list[str]
    parents: np.ndarray  # [B] int32, stored values (full-skeleton index space)
    positions: np.ndarray  # [B, 3] float32 meters
    rotations: np.ndarray  # [B, 4] float32 wxyz
    source_path: Path


def load_glb(path: Path) -> tuple[dict, bytes]:
    """Read the JSON and binary chunks of a glTF-binary file."""
    with open(path, "rb") as handle:
        magic, version, _ = struct.unpack("<III", handle.read(12))
        if magic != GLB_MAGIC:
            raise ValueError(f"{path} is not a GLB file")
        if version != 2:
            raise ValueError(f"{path} uses glTF version {version}; only 2 is supported")
        gltf: dict | None = None
        binary = b""
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            length, kind = struct.unpack("<II", header)
            payload = handle.read(length)
            if kind == GLB_JSON_CHUNK:
                gltf = json.loads(payload)
            elif kind == GLB_BIN_CHUNK:
                binary = payload
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk")
    return gltf, binary


def read_accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    """Materialize one accessor, honoring bufferView byteOffset and byteStride."""
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
    shape = _TYPE_SHAPES[accessor["type"]]
    count = int(accessor["count"])
    width = int(np.prod(shape)) if shape else 1
    item_size = _COMPONENT_SIZES[accessor["componentType"]] * width
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride") or item_size)
    if stride == item_size:
        flat = np.frombuffer(binary, dtype=dtype, count=count * width, offset=offset)
        values = flat.reshape(count, width)
    else:
        raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + item_size, offset=offset)
        values = np.lib.stride_tricks.as_strided(
            raw.view(dtype), shape=(count, width), strides=(stride, dtype.itemsize)
        ).copy()
    return values.reshape((count, *shape)) if shape else values[:, 0]


def _node_world_matrices(gltf: dict) -> dict[int, np.ndarray]:
    """World transform of every node reachable from the scene roots."""
    nodes = gltf["nodes"]
    scene_roots = gltf["scenes"][gltf.get("scene", 0)]["nodes"]
    worlds: dict[int, np.ndarray] = {}

    def visit(index: int, parent: np.ndarray) -> None:
        node = nodes[index]
        if "matrix" in node:
            local = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        else:
            local = np.eye(4)
            if "translation" in node:
                local[:3, 3] = node["translation"]
            if "rotation" in node:
                x, y, z, w = node["rotation"]
                local[:3, :3] = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
            if "scale" in node:
                local[:3, :3] = local[:3, :3] @ np.diag(node["scale"])
        worlds[index] = parent @ local
        for child in node.get("children", []):
            visit(child, worlds[index])

    for root in scene_roots:
        visit(root, np.eye(4))
    return worlds


def _skin_joint_parents(gltf: dict, joint_nodes: list[int]) -> np.ndarray:
    parent_of_node: dict[int, int] = {}
    for index, node in enumerate(gltf["nodes"]):
        for child in node.get("children", []):
            parent_of_node[child] = index
    slot_of_node = {node: slot for slot, node in enumerate(joint_nodes)}
    parents = np.full(len(joint_nodes), -1, dtype=np.int64)
    for slot, node in enumerate(joint_nodes):
        ancestor = parent_of_node.get(node)
        # Walk up through helper nodes (Sketchfab inserts a few) to the next joint.
        while ancestor is not None and ancestor not in slot_of_node:
            ancestor = parent_of_node.get(ancestor)
        if ancestor is not None:
            parents[slot] = slot_of_node[ancestor]
    return parents


def load_quinn_mesh(path: Path) -> QuinnMesh:
    """Extract the single skinned primitive, its skin and its textures."""
    gltf, binary = load_glb(path)
    meshes = gltf.get("meshes", [])
    if len(meshes) != 1 or len(meshes[0]["primitives"]) != 1:
        raise ValueError(f"expected exactly one mesh with one primitive, got {len(meshes)} mesh(es)")
    primitive = meshes[0]["primitives"][0]
    if primitive.get("mode", TRIANGLES_MODE) != TRIANGLES_MODE:
        raise ValueError(f"primitive mode {primitive.get('mode')} is not triangles")
    skins = gltf.get("skins", [])
    if len(skins) != 1:
        raise ValueError(f"expected exactly one skin, got {len(skins)}")

    attributes = primitive["attributes"]
    missing = [name for name in ("POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0") if name not in attributes]
    if missing:
        raise ValueError(f"primitive is missing attributes: {missing}")
    positions = read_accessor(gltf, binary, attributes["POSITION"]).astype(np.float32)
    normals = read_accessor(gltf, binary, attributes["NORMAL"]).astype(np.float32)
    texcoords = read_accessor(gltf, binary, attributes["TEXCOORD_0"]).astype(np.float32)
    joint_indices = read_accessor(gltf, binary, attributes["JOINTS_0"]).astype(np.int64)
    joint_weights = read_accessor(gltf, binary, attributes["WEIGHTS_0"]).astype(np.float32)
    indices = read_accessor(gltf, binary, primitive["indices"]).astype(np.int64).reshape(-1)
    _validate_quinn_primitive(positions, normals, texcoords, joint_indices, joint_weights, indices)

    joint_nodes = list(skins[0]["joints"])
    joint_names = [_NODE_INDEX_SUFFIX.sub("", gltf["nodes"][node].get("name", f"joint{slot}")) for slot, node in enumerate(joint_nodes)]
    inverse_bind = read_accessor(gltf, binary, skins[0]["inverseBindMatrices"]).astype(np.float64)
    # glTF matrices are column-major; the inverted matrix maps mesh space into
    # joint space, so its translation is the joint center in mesh space.
    bind_frames = np.linalg.inv(np.transpose(inverse_bind.reshape(-1, 4, 4), (0, 2, 1)))

    textures = []
    for image in gltf.get("images", []):
        view = gltf["bufferViews"][image["bufferView"]]
        start = int(view.get("byteOffset", 0))
        textures.append((image.get("mimeType", "image/jpeg"), binary[start : start + int(view["byteLength"])]))

    joint_weights = joint_weights / np.maximum(joint_weights.sum(axis=1, keepdims=True), 1e-12)
    return QuinnMesh(
        positions=positions,
        normals=normals,
        texcoords=texcoords,
        indices=indices,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
        joint_names=joint_names,
        joint_parents=_skin_joint_parents(gltf, joint_nodes),
        joint_positions=bind_frames[:, :3, 3],
        textures=textures,
    )


def _validate_quinn_primitive(positions, normals, texcoords, joint_indices, joint_weights, indices) -> None:
    counts = {len(positions), len(normals), len(texcoords), len(joint_indices), len(joint_weights)}
    if len(counts) != 1:
        raise ValueError(f"primitive attribute counts disagree: {sorted(counts)}")
    if indices.size % 3 or indices.size == 0:
        raise ValueError(f"index count {indices.size} is not a whole number of triangles")
    if indices.min() < 0 or indices.max() >= len(positions):
        raise ValueError("triangle indices are out of the vertex range")
    if joint_indices.min() < 0:
        raise ValueError("skin joint indices are negative")
    if not np.isfinite(joint_weights).all() or joint_weights.min() < 0.0:
        raise ValueError("skin weights must be finite and non-negative")
    totals = joint_weights.sum(axis=1)
    if np.abs(totals - 1.0).max() > 1e-3:
        raise ValueError(f"skin weight sums deviate from 1 by {np.abs(totals - 1.0).max():.2e}")


def soma_bin_layout(vertex_count: int, triangle_count: int, bone_count: int) -> dict[str, tuple[int, int]]:
    """Byte ranges of ``SOMA.bin``, matching ``genoview.load_geno_model``."""
    sections = {
        "vertices": (3, "<f4"),
        "texcoords": (2, "<f4"),
        "normals": (3, "<f4"),
        "bone_ids": (4, "u1"),
        "bone_weights": (4, "<f4"),
        "indices": (3, "<u2"),
    }
    layout: dict[str, tuple[int, int]] = {"header": (0, 12)}
    offset = 12
    for name, (width, dtype) in sections.items():
        count = vertex_count if name != "indices" else triangle_count
        size = count * width * np.dtype(dtype).itemsize
        layout[name] = (offset, offset + size)
        offset += size
    layout["bones"] = (offset, offset + bone_count * 36)
    offset += bone_count * 36
    layout["bind_pose"] = (offset, offset + bone_count * 40)
    return layout


def read_soma_bin(path: Path) -> tuple[dict[str, np.ndarray], SomaBind]:
    """Read a viewer ``SOMA.bin``: mesh arrays plus the 78-bone bind pose."""
    raw = path.read_bytes()
    vertex_count, triangle_count, bone_count = struct.unpack_from("<III", raw, 0)
    layout = soma_bin_layout(vertex_count, triangle_count, bone_count)
    if layout["bind_pose"][1] != len(raw):
        raise ValueError(f"{path} has {len(raw)} bytes, expected {layout['bind_pose'][1]}")

    def section(name: str, width: int, dtype: str) -> np.ndarray:
        start, stop = layout[name]
        return np.frombuffer(raw, dtype=dtype, count=(stop - start) // np.dtype(dtype).itemsize, offset=start).reshape(-1, width)

    mesh = {
        "vertices": section("vertices", 3, "<f4"),
        "texcoords": section("texcoords", 2, "<f4"),
        "normals": section("normals", 3, "<f4"),
        "bone_ids": section("bone_ids", 4, "u1"),
        "bone_weights": section("bone_weights", 4, "<f4"),
        "indices": section("indices", 3, "<u2").reshape(-1),
        "vertex_count": np.int32(vertex_count),
        "triangle_count": np.int32(triangle_count),
    }

    names, parents = [], []
    bone_start = layout["bones"][0]
    for index in range(bone_count):
        offset = bone_start + index * 36
        names.append(raw[offset : offset + 32].split(b"\x00")[0].decode("utf-8"))
        parents.append(struct.unpack_from("<i", raw, offset + 32)[0])
    positions, rotations = [], []
    bind_start = layout["bind_pose"][0]
    for index in range(bone_count):
        # Transform = float3 translation, quat xyzw, float3 scale, two pads.
        transform = struct.unpack_from("<10f", raw, bind_start + index * 40)
        positions.append(transform[0:3])
        rotations.append((transform[6], transform[3], transform[4], transform[5]))
    bind = SomaBind(
        names=names,
        parents=np.asarray(parents, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float32),
        rotations=np.asarray(rotations, dtype=np.float32),
        source_path=path,
    )
    return mesh, bind


def load_soma_bind(resource_dir: Path) -> SomaBind:
    """Read the authoritative bind pose from an existing SOMA resource directory."""
    return read_soma_bin(resource_dir / "SOMA.bin")[1]


def build_bone_index(quinn: QuinnMesh, soma: SomaBind) -> tuple[np.ndarray, dict[str, object]]:
    """Resolve every Quinn skin joint to a SOMA bone index by name."""
    bone_of_name = {name: index for index, name in enumerate(soma.names)}
    targets_used = {QUINN_TO_SOMA[name] for name in quinn.joint_names if name in QUINN_TO_SOMA}
    unknown_bones = sorted(targets_used - set(bone_of_name))
    if unknown_bones:
        raise ValueError(f"mapping targets bones that SOMA.bin does not define: {unknown_bones}")

    bone_index = np.full(len(quinn.joint_names), -1, dtype=np.int64)
    for slot, name in enumerate(quinn.joint_names):
        target = QUINN_TO_SOMA.get(name)
        if target is not None:
            bone_index[slot] = bone_of_name[target]

    totals = np.zeros(len(quinn.joint_names), dtype=np.float64)
    np.add.at(totals, quinn.joint_indices.reshape(-1), quinn.joint_weights.reshape(-1).astype(np.float64))
    unmapped = np.nonzero((bone_index < 0) & (totals > 0.0))[0]
    if unmapped.size:
        offenders = {quinn.joint_names[slot]: float(totals[slot]) for slot in unmapped}
        raise ValueError(f"{unmapped.size} Quinn joint(s) with skin weight have no SOMA target: {offenders}")

    rules: dict[str, float] = {}
    for slot, name in enumerate(quinn.joint_names):
        target = QUINN_TO_SOMA.get(name)
        if target is not None:
            key = f"{name} -> {target}"
            rules[key] = float(totals[slot])
    report = {
        "quinn_joint_input_weight": {name: float(totals[slot]) for slot, name in enumerate(quinn.joint_names)},
        "rule_weight": rules,
        "unmapped_joints": [quinn.joint_names[slot] for slot in np.nonzero(bone_index < 0)[0]],
    }
    return bone_index, report


def map_quinn_weights(quinn: QuinnMesh, bone_index: np.ndarray, bone_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Merge Quinn influences onto SOMA bones, keep four, and renormalize."""
    vertex_count = len(quinn.positions)
    slots = quinn.joint_indices.shape[1]
    dense = np.zeros((vertex_count, len(bone_names)), dtype=np.float64)
    rows = np.repeat(np.arange(vertex_count), slots)
    targets = bone_index[quinn.joint_indices.reshape(-1)]
    weights_flat = quinn.joint_weights.reshape(-1).astype(np.float64)
    usable = targets >= 0
    np.add.at(dense, (rows[usable], targets[usable]), weights_flat[usable])

    totals = dense.sum(axis=1)
    if (totals <= 1e-8).any():
        raise ValueError(f"{int((totals <= 1e-8).sum())} vertices have no usable skin weight")

    keep = np.argsort(-dense, axis=1)[:, :INFLUENCES_PER_VERTEX]
    kept = np.take_along_axis(dense, keep, axis=1)
    dropped = totals - kept.sum(axis=1)
    kept = kept / kept.sum(axis=1, keepdims=True)

    bone_ids = keep.astype(np.uint8)
    weights = kept.astype(np.float32)
    # Fill unused slots with bone 0 at zero weight so every slot stays valid.
    weights[weights < 1e-8] = 0.0
    weights = weights / weights.sum(axis=1, keepdims=True)

    influence_counts = (weights > 0.0).sum(axis=1)
    report = {
        "soma_bone_output_weight": {name: float(dense[:, index].sum()) for index, name in enumerate(bone_names)},
        "dropped_weight": {
            "max": float(dropped.max()),
            "mean": float(dropped.mean()),
            "sum": float(dropped.sum()),
        },
        "zero_weight_vertices": int((weights.max(axis=1) <= 0.0).sum()),
        "influence_count_histogram": {str(count): int((influence_counts == count).sum()) for count in range(1, INFLUENCES_PER_VERTEX + 1)},
    }
    return bone_ids, weights, report


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares uniform scale, rotation and translation mapping source onto target."""
    if len(source) < 3:
        raise ValueError("a similarity fit needs at least three correspondences")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_local = source - source_center
    target_local = target - target_center
    covariance = target_local.T @ source_local / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = float(np.sign(np.linalg.det(u @ vt)))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    variance = float((source_local**2).sum() / len(source))
    if variance <= 0.0:
        raise ValueError(f"alignment correspondences are degenerate for {len(source)} point(s)")
    scale = float((singular * np.array([1.0, 1.0, sign])).sum() / variance)
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def _rotation_from_vector(omega: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3)
    axis = omega / angle
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def fit_region_similarity(
    source: np.ndarray,
    target: np.ndarray,
    base: tuple[float, np.ndarray, np.ndarray],
    iterations: int = 4,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Align one body region onto its SOMA counterpart, relative to ``base``.

    The scale is solved in closed form about the centroids and the rotation by a
    least-squares small-rotation refinement of the base; the translation then
    follows from the centroids.  Both steps stay well posed when the
    correspondences are nearly collinear (the SOMA spine, a straight limb) or
    reduce to a single joint (the pelvis), where a free similarity fit would spin
    the region about its own axis or collapse its scale.
    """
    base_scale, rotation, translation = base
    scale = base_scale
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_local = source - source_center
    for _ in range(iterations):
        if len(source) > 1:
            rotated = source_local @ rotation.T
            denominator = float((rotated**2).sum())
            scale = float((rotated * (target - target_center)).sum() / denominator) if denominator > 1e-12 else base_scale
        translation = target_center - scale * rotation @ source_center
        moved = scale * source @ rotation.T + translation
        cross = np.zeros((len(moved), 3, 3))
        cross[:, 0, 1], cross[:, 0, 2] = -moved[:, 2], moved[:, 1]
        cross[:, 1, 0], cross[:, 1, 2] = moved[:, 2], -moved[:, 0]
        cross[:, 2, 0], cross[:, 2, 1] = -moved[:, 1], moved[:, 0]
        design = np.concatenate([-cross, np.broadcast_to(np.eye(3), (len(moved), 3, 3))], axis=2).reshape(-1, 6)
        solution, *_ = np.linalg.lstsq(design, (target - moved).reshape(-1), rcond=None)
        delta = _rotation_from_vector(solution[0:3])
        translation = delta @ translation + solution[3:6]
        rotation = delta @ rotation
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def _segment_of(name: str) -> str:
    if name == "pelvis":
        return "pelvis"
    if name.startswith(("spine_", "neck_", "head", "clavicle")):
        return "torso"
    side = "left" if name.endswith("_l") else "right" if name.endswith("_r") else ""
    if name.startswith(("thigh", "calf", "foot", "ball")):
        return f"{side}_leg"
    if name.startswith(("upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")):
        return f"{side}_arm"
    raise ValueError(f"Quinn joint {name!r} has no alignment segment")


def align_mesh_to_soma_bind(quinn: QuinnMesh, soma: SomaBind, bone_index: np.ndarray, mode: str = "global") -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Place Quinn's bind mesh in SOMA's meter bind frame.

    ``global`` fits one similarity over every align joint; ``segment`` fits one
    per pelvis/torso/arm/leg region and blends them with the skin weights, which
    absorbs the proportion differences between the two rigs.
    """
    if mode not in ALIGN_MODES:
        raise ValueError(f"unknown alignment mode {mode!r}; expected one of {ALIGN_MODES}")
    bone_of_name = {name: index for index, name in enumerate(soma.names)}
    slot_of_name = {name: slot for slot, name in enumerate(quinn.joint_names)}
    missing = [name for name in _ALIGN_JOINTS if name not in slot_of_name]
    if missing:
        raise ValueError(f"Quinn skin is missing align joints: {missing}")

    regions = {"global": _ALIGN_JOINTS} if mode == "global" else _SEGMENT_JOINTS
    correspondence = {
        region: (
            np.array([quinn.joint_positions[slot_of_name[name]] for name in names]),
            np.array([soma.positions[bone_of_name[QUINN_TO_SOMA[name]]] for name in names], dtype=np.float64),
        )
        for region, names in regions.items()
    }
    global_source = np.array([quinn.joint_positions[slot_of_name[name]] for name in _ALIGN_JOINTS])
    global_target = np.array(
        [soma.positions[bone_of_name[QUINN_TO_SOMA[name]]] for name in _ALIGN_JOINTS], dtype=np.float64
    )
    base = fit_similarity(global_source, global_target)
    transforms = {
        region: base if mode == "global" else fit_region_similarity(source, target, base)
        for region, (source, target) in correspondence.items()
    }

    region_report: dict[str, object] = {}
    joint_alignment: dict[str, dict[str, object]] = {}
    for region, names in regions.items():
        source, target = correspondence[region]
        scale, rotation, translation = transforms[region]
        aligned = scale * source @ rotation.T + translation
        error = np.linalg.norm(aligned - target, axis=1) * 100.0
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        correction = np.linalg.inv(base[1]) @ rotation
        region_report[region] = {
            "joints": list(names),
            "scale": scale,
            "rotation_axis_angle_deg": (rotation.tolist(), angle),
            "translation_m": translation.tolist(),
            "correction_axis_angle_deg": (
                correction.tolist(),
                float(np.degrees(np.arccos(np.clip((np.trace(correction) - 1.0) / 2.0, -1.0, 1.0)))),
            ),
            "joint_error_cm": {name: float(value) for name, value in zip(names, error)},
            "max_joint_error_cm": float(error.max()),
        }
        for index, name in enumerate(names):
            bone_name = QUINN_TO_SOMA[name]
            joint_alignment[bone_name] = {
                "quinn_bind": source[index].tolist(),
                "aligned": aligned[index].tolist(),
                "soma_bind": target[index].tolist(),
                "error_cm": float(error[index]),
                "region": region,
            }

    if mode == "global":
        region_of_slot = np.full(len(quinn.joint_names), "global", dtype=object)
    else:
        # Unmapped joints carry no weight, so their region never contributes.
        region_of_slot = np.array(
            ["pelvis" if bone_index[slot] < 0 else _segment_of(name) for slot, name in enumerate(quinn.joint_names)],
            dtype=object,
        )
    region_names = list(regions)
    blend = np.zeros((len(quinn.positions), len(region_names)), dtype=np.float64)
    slot_regions = region_of_slot[quinn.joint_indices]  # [V, influences]
    weights = quinn.joint_weights.astype(np.float64)
    for index, region in enumerate(region_names):
        blend[:, index] = (weights * (slot_regions == region)).sum(axis=1)
    blend /= np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)

    linear = np.zeros((len(quinn.positions), 3, 3), dtype=np.float64)
    offset = np.zeros((len(quinn.positions), 3), dtype=np.float64)
    displacement: dict[str, float] = {}
    for index, region in enumerate(region_names):
        scale, rotation, translation = transforms[region]
        weight = blend[:, index : index + 1]
        linear += weight[:, :, None] * (scale * rotation)
        offset += weight * translation
        single = scale * quinn.positions @ rotation.T + translation
        moved = np.linalg.norm(single - quinn.positions, axis=1)
        mask = blend[:, index] > 0.5
        displacement[region] = float(moved[mask].max()) if mask.any() else 0.0

    vertices = np.einsum("vij,vj->vi", linear, quinn.positions.astype(np.float64)) + offset
    normals = np.einsum("vij,vj->vi", np.linalg.inv(linear).transpose(0, 2, 1), quinn.normals.astype(np.float64))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    for region in region_names:
        region_report[region]["max_displacement_m"] = displacement[region]

    report = {
        "mode": mode,
        "regions": region_report,
        "joint_alignment": joint_alignment,
        "max_vertex_displacement_m": float(np.linalg.norm(vertices - quinn.positions, axis=1).max()),
    }
    return vertices, normals, report


def _bounding_box(points: np.ndarray) -> dict[str, object]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "extent": (maximum - minimum).tolist(),
        "height_m": float(maximum[1] - minimum[1]),
    }


def extract_textures(gltf: dict, quinn: QuinnMesh, output_dir: Path) -> dict[str, str]:
    """Write Quinn's base-color and normal maps, following the material binding."""
    material = gltf["materials"][0]
    sources = {
        "base_color": material.get("pbrMetallicRoughness", {}).get("baseColorTexture", {}).get("index"),
        "normal": material.get("normalTexture", {}).get("index"),
    }
    written: dict[str, str] = {}
    for role, texture_index in sources.items():
        if texture_index is None:
            continue
        image_index = gltf["textures"][texture_index]["source"]
        mime, payload = quinn.textures[image_index]
        suffix = ".png" if "png" in mime else ".jpg"
        target = output_dir / f"{TEXTURE_OUTPUT_NAMES[role]}{suffix}"
        target.write_bytes(payload)
        written[role] = target.name
    if set(written) != set(TEXTURE_OUTPUT_NAMES):
        raise ValueError(f"GLB material does not bind both textures: {written}")
    return written


def _validate_viewer_mesh(viewer_mesh: dict[str, np.ndarray], bone_count: int) -> dict[str, object]:
    vertex_count = int(viewer_mesh["vertex_count"])
    triangle_count = int(viewer_mesh["triangle_count"])
    checks = {
        "vertex_count_below_uint16": vertex_count < MAX_VERTEX_COUNT,
        "bone_count_within_shader_cap": bone_count <= MAX_BONE_NUM,
        "attributes_agree": len({len(viewer_mesh[key]) for key in ("vertices", "texcoords", "normals", "bone_ids", "bone_weights")}) == 1,
        "triangle_indices_in_range": bool(viewer_mesh["indices"].max() < vertex_count) and triangle_count * 3 == len(viewer_mesh["indices"]),
        "bone_ids_in_range": bool(viewer_mesh["bone_ids"].max() < bone_count),
    }
    weights = viewer_mesh["bone_weights"].astype(np.float64)
    checks["weights_finite_non_negative"] = bool(np.isfinite(weights).all() and weights.min() >= 0.0)
    checks["weights_normalized"] = bool(np.abs(weights.sum(axis=1) - 1.0).max() < 1e-5)
    checks["every_vertex_weighted"] = bool((weights > 0.0).sum(axis=1).min() >= 1)
    checks["normals_unit"] = bool(np.abs(np.linalg.norm(viewer_mesh["normals"], axis=1) - 1.0).max() < 1e-3)
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"viewer mesh validation failed: {failed}")
    return checks


def _check_weighted_bind_rotations(viewer_mesh: dict[str, np.ndarray], soma: SomaBind) -> bool:
    """Every bone the mesh actually uses needs a unit bind rotation.

    SOMA ships a placeholder rotation on the weightless ``Root``, so the check
    covers only the bones the skin references; a misread bind blob shows up here
    instead of as an exploded render.
    """
    used = np.unique(viewer_mesh["bone_ids"][viewer_mesh["bone_weights"] > 0.0])
    deviations = np.abs(np.linalg.norm(soma.rotations[used], axis=1) - 1.0)
    if deviations.max() > 1e-3:
        offenders = [soma.names[index] for index in used[deviations > 1e-3]]
        raise ValueError(f"weighted bones have non-unit bind rotations: {offenders[:5]}")
    return True


def validate_written_soma_bin(path: Path, soma: SomaBind, viewer_mesh: dict[str, np.ndarray]) -> dict[str, object]:
    """Re-read the written file and compare it against the source contract."""
    mesh, written = read_soma_bin(path)
    if written.names != soma.names:
        raise ValueError("written bone names do not match the SOMA source")
    if not np.array_equal(written.parents, soma.parents):
        raise ValueError("written bone parents do not match the SOMA source")
    if np.abs(written.positions - soma.positions).max() > 1e-6:
        raise ValueError("written bind positions do not match the SOMA source")
    if np.abs(written.rotations - soma.rotations).max() > 1e-6:
        raise ValueError("written bind rotations do not match the SOMA source")
    if not np.array_equal(mesh["bone_ids"], viewer_mesh["bone_ids"]):
        raise ValueError("written bone ids do not round-trip")
    if np.abs(mesh["bone_weights"] - viewer_mesh["bone_weights"]).max() > 1e-6:
        raise ValueError("written bone weights do not round-trip")
    if np.abs(mesh["vertices"] - viewer_mesh["vertices"]).max() > 1e-6:
        raise ValueError("written vertices do not round-trip")
    expected_size = soma_bin_layout(
        int(viewer_mesh["vertex_count"]), int(viewer_mesh["triangle_count"]), len(soma.names)
    )["bind_pose"][1]
    if path.stat().st_size != expected_size:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {expected_size}")
    return {
        "round_trip": True,
        "file_size": path.stat().st_size,
        "expected_size": expected_size,
        "bone_count": len(written.names),
    }


def build_quinn_assets(
    glb_path: Path,
    soma_resources: Path,
    output_dir: Path,
    align_mode: str = "segment",
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert the Quinn GLB into a SomaView resource directory."""
    bin_path = output_dir / "SOMA.bin"
    bind_path = output_dir / "SOMA_bind.bvh"
    if not overwrite and (bin_path.exists() or bind_path.exists()):
        raise FileExistsError(f"{bin_path} or {bind_path} already exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    gltf, _ = load_glb(glb_path)
    quinn = load_quinn_mesh(glb_path)
    soma = load_soma_bind(soma_resources)
    bone_index, mapping_report = build_bone_index(quinn, soma)
    bone_ids, bone_weights, weight_report = map_quinn_weights(quinn, bone_index, soma.names)
    vertices, normals, align_report = align_mesh_to_soma_bind(quinn, soma, bone_index, align_mode)

    viewer_mesh = {
        "vertices": vertices.astype(np.float32),
        "texcoords": quinn.texcoords,
        "normals": normals.astype(np.float32),
        "bone_ids": bone_ids,
        "bone_weights": bone_weights,
        "indices": quinn.indices.astype(np.uint16),
        "triangle_count": quinn.indices.size // 3,
        "vertex_count": len(vertices),
    }
    checks = _validate_viewer_mesh(viewer_mesh, len(soma.names))
    checks["weighted_bind_rotations_unit"] = _check_weighted_bind_rotations(viewer_mesh, soma)

    # write_soma_bin consumes the "Simulation"-rooted full skeleton; inverting
    # its parent remap keeps the stored values byte-identical to the source.
    bind = {
        "names": ["Simulation", *soma.names],
        "parents": np.concatenate([np.array([-1], dtype=np.int64), soma.parents + 1]).tolist(),
        "global_positions": np.concatenate([np.zeros((1, 3), dtype=np.float32), soma.positions]),
        "global_rotations": np.concatenate([np.eye(4, dtype=np.float32)[0][None], soma.rotations]),
    }
    bind["local_positions"] = bind["global_positions"]
    bind["local_rotations"] = bind["global_rotations"]
    write_soma_bin(bin_path, viewer_mesh, bind)

    shutil.copyfile(soma_resources / "SOMA_bind.bvh", bind_path)
    for shader in SHADER_FILES:
        shutil.copyfile(soma_resources / shader, output_dir / shader)
    textures = extract_textures(gltf, quinn, output_dir)
    written = validate_written_soma_bin(bin_path, soma, viewer_mesh)

    report = {
        "inputs": {
            "glb": str(glb_path),
            "glb_sha256": hashlib.sha256(glb_path.read_bytes()).hexdigest(),
            "soma_bin": str(soma.source_path),
        },
        "outputs": {
            "soma_bin": str(bin_path),
            "bind_bvh": str(bind_path),
            "textures": textures,
            "shaders": list(SHADER_FILES),
        },
        "mesh": {
            "source_vertices": int(len(quinn.positions)),
            "source_triangles": int(quinn.indices.size // 3),
            "source_bounding_box": _bounding_box(quinn.positions.astype(np.float64)),
            "output_bounding_box": _bounding_box(vertices),
            "soma_bounding_box": _bounding_box(soma.positions.astype(np.float64)),
        },
        "mapping": mapping_report,
        "weights": weight_report,
        "alignment": align_report,
        "joint_targets": {name: align_report["joint_alignment"][name] for name in REPORT_JOINTS if name in align_report["joint_alignment"]},
        "validation": {"viewer_mesh": checks, "written_file": written},
        "license": {
            "title": gltf["asset"].get("extras", {}).get("title"),
            "author": gltf["asset"].get("extras", {}).get("author"),
            "license": gltf["asset"].get("extras", {}).get("license"),
            "note": "Quinn geometry re-skinned onto the SOMA skeleton.",
        },
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {bin_path} ({bin_path.stat().st_size} bytes): {viewer_mesh['vertex_count']} vertices, "
        f"{viewer_mesh['triangle_count']} triangles, {len(soma.names)} bones, align={align_mode}"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glb", type=Path, required=True, help="Path to the Quinn skinned GLB")
    parser.add_argument("--soma-resources", type=Path, required=True, help="SOMA resource directory holding the reference SOMA.bin")
    parser.add_argument("--output-dir", type=Path, required=True, help="Resource directory to populate")
    parser.add_argument(
        "--align",
        choices=ALIGN_MODES,
        default="segment",
        help="segment: one similarity per body region (pelvis/torso/arms/legs); global: one similarity for the whole body",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_quinn_assets(args.glb, args.soma_resources, args.output_dir, align_mode=args.align, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
