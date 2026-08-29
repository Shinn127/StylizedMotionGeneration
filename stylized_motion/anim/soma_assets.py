"""Convert the BONES-SEED SOMA base rig (usda + bind BVH) into somaview assets.

Inputs (from ``soma_shapes/soma_base_rig``):
  * ``soma_base_skel_minimal.usd``  -- skeleton (78 joints), skinned mesh
    (~18k vertices, 8 influences/vertex), face-varying UVs
  * ``soma_base_skel_minimal.bvh``  -- the same skeleton as a bind BVH

Outputs (into the output directory):
  * ``SOMA.bin``       -- GenoView-compatible binary mesh + bind pose
  * ``SOMA_bind.bvh``  -- copy of the source bind BVH
  * the viewer shaders -- copied from the genoview resource directory

The .bin layout mirrors what ``genoview.load_geno_model`` reads, including raw
raylib ``BoneInfo``/``Transform`` structs written through cffi so the byte
layout cannot drift between raylib builds.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import cffi
import numpy as np

from stylized_motion.anim import bvh, quat
from stylized_motion.util.paths import RESOURCE_DIR

ffi = cffi.FFI()

MAX_BONE_NUM = 128  # skinnedBasic.vs / skinnedShadow.vs uniform array cap
INFLUENCES_PER_VERTEX = 4  # viewer shaders read vec4 boneIds/boneWeights


@dataclass
class SomaMesh:
    """Skinned mesh extracted from the SOMA base-rig usda."""

    points: np.ndarray  # [V, 3] float32, centimeters (usda metersPerUnit = 0.01)
    face_vertex_counts: np.ndarray  # [F] int
    face_vertex_indices: np.ndarray  # [sum(counts)] int
    joint_indices: np.ndarray  # [V, K] int (K = influences per vertex in the usda)
    joint_weights: np.ndarray  # [V, K] float32
    uv_values: np.ndarray  # [U, 2] float32 (faceVarying values)
    uv_indices: np.ndarray | None  # [sum(counts)] int, may be absent


@dataclass
class SomaSkeleton:
    """Skeleton joint paths in usda order plus their bind transforms."""

    joint_paths: list[str]  # e.g. "Root/Hips/Spine1"
    bind_transforms: np.ndarray  # [J, 4, 4] float64, world-space at bind


def _extract_payload(text: str, decl: str) -> str:
    pattern = re.compile(re.escape(decl) + r"\s*=\s*\[(.*?)\]", re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"USD is missing attribute {decl!r}")
    return match.group(1)


def _extract_array(text: str, decl: str) -> np.ndarray:
    body = _extract_payload(text, decl).replace("(", " ").replace(")", " ").replace(",", " ")
    return np.fromstring(body, sep=" ")


def _extract_token_array(text: str, decl: str) -> list[str]:
    return re.findall(r'"([^"]+)"', _extract_payload(text, decl))


def parse_soma_usd(usd_path: Path) -> tuple[SomaSkeleton, SomaMesh]:
    """Parse the skinned mesh and skeleton from the SOMA base-rig usda text."""
    text = usd_path.read_text(encoding="utf-8")

    skeleton = SomaSkeleton(
        joint_paths=_extract_token_array(text, "uniform token[] joints"),
        bind_transforms=_extract_array(text, "uniform matrix4d[] bindTransforms").reshape(-1, 16).reshape(-1, 4, 4),
    )

    points = _extract_array(text, "point3f[] points").reshape(-1, 3)
    face_vertex_counts = _extract_array(text, "int[] faceVertexCounts").astype(np.int64)
    face_vertex_indices = _extract_array(text, "int[] faceVertexIndices").astype(np.int64)
    uv_values = _extract_array(text, "texCoord2f[] primvars:st").reshape(-1, 2)

    uv_indices_payload = re.search(r"int\[\] primvars:st:indices\s*=\s*\[(.*?)\]", text, re.DOTALL)
    uv_indices = (
        np.fromstring(
            uv_indices_payload.group(1).replace("(", " ").replace(")", " ").replace(",", " "), sep=" "
        ).astype(np.int64)
        if uv_indices_payload is not None
        else None
    )

    vertex_count = points.shape[0]
    joint_flat = _extract_array(text, "int[] primvars:skel:jointIndices")
    weight_flat = _extract_array(text, "float[] primvars:skel:jointWeights")
    if joint_flat.shape[0] != weight_flat.shape[0] or joint_flat.shape[0] % vertex_count != 0:
        raise ValueError("USD skinning arrays are not a whole multiple of the vertex count")
    influences = joint_flat.shape[0] // vertex_count

    mesh = SomaMesh(
        points=points.astype(np.float32),
        face_vertex_counts=face_vertex_counts,
        face_vertex_indices=face_vertex_indices,
        joint_indices=joint_flat.reshape(vertex_count, influences).astype(np.int64),
        joint_weights=weight_flat.reshape(vertex_count, influences).astype(np.float32),
        uv_values=uv_values.astype(np.float32),
        uv_indices=uv_indices,
    )
    _validate_soma_usd(skeleton, mesh)
    return skeleton, mesh


def _validate_soma_usd(skeleton: SomaSkeleton, mesh: SomaMesh) -> None:
    if len(skeleton.joint_paths) != skeleton.bind_transforms.shape[0]:
        raise ValueError("USD joints and bindTransforms lengths disagree")
    if mesh.points.shape[0] != mesh.joint_indices.shape[0] or mesh.points.shape[0] != mesh.joint_weights.shape[0]:
        raise ValueError("USD skinning arrays do not match the vertex count")
    if mesh.joint_indices.shape[1] != mesh.joint_weights.shape[1]:
        raise ValueError("USD jointIndices/jointWeights influence counts disagree")
    if int(mesh.face_vertex_indices.shape[0]) != int(mesh.face_vertex_counts.sum()):
        raise ValueError("USD faceVertexIndices length does not match faceVertexCounts")
    if mesh.uv_indices is not None and int(mesh.uv_indices.shape[0]) != int(mesh.face_vertex_counts.sum()):
        raise ValueError("USD primvars:st:indices length does not match faceVertexCounts")
    if mesh.points.shape[0] > 65535:
        raise ValueError("Viewer .bin indices are uint16; mesh has too many vertices")
    if len(skeleton.joint_paths) > MAX_BONE_NUM:
        raise ValueError(f"Skeleton has {len(skeleton.joint_paths)} joints; shader cap is {MAX_BONE_NUM}")


def _build_bvh_bind_pose(bind_bvh_path: Path, unit_scale: float) -> dict[str, object]:
    """Bind skeleton in the viewer's sim-root convention plus its frame-0 globals."""
    from stylized_motion.anim.genoview import build_simulation_root_skeleton_from_bind

    names, parents, local_positions, local_rotations = build_simulation_root_skeleton_from_bind(bind_bvh_path)
    global_rotations, global_positions = quat.fk(local_rotations[None], local_positions[None], parents)
    return {
        "names": names,
        "parents": parents,
        "local_positions": local_positions,
        "local_rotations": local_rotations,
        "global_rotations": global_rotations[0],
        "global_positions": global_positions[0],
    }


def build_bind_from_usd(skeleton: SomaSkeleton, bind: dict[str, object]) -> None:
    """Overwrite the bind pose globals with the USD ``bindTransforms`` skinning rest pose.

    The BVH's frame 0 is a T-pose used only to define the skeleton hierarchy;
    the mesh was skinned against a different rest pose (relaxed arms), whose
    world transforms are authoritative in ``bindTransforms``. The re-rooted
    skeleton's FK globals live in the same raw BVH world frame (the Simulation
    joint cancels out), so the USD transforms are written through unchanged
    apart from the unit scale and the USD row-vector matrix convention.
    """
    bvh_names = [str(name) for name in bind["names"]]
    usda_to_full = np.full(len(skeleton.joint_paths), -1, dtype=np.int64)
    for usda_idx, path in enumerate(skeleton.joint_paths):
        usda_to_full[usda_idx] = bvh_names.index(path.rsplit("/", 1)[-1])
    if np.any(usda_to_full < 0):
        missing = [skeleton.joint_paths[i] for i in np.nonzero(usda_to_full < 0)[0]]
        raise ValueError(f"USD joints missing from the bind BVH: {missing[:5]}")

    matrices = skeleton.bind_transforms  # raw-BVH-world, USD row-vector convention
    rotations = quat.from_xform(np.transpose(matrices[:, :3, :3], (0, 2, 1)).astype(np.float32))
    positions = matrices[:, 3, :3].astype(np.float32) * 0.01  # cm -> m

    global_rotations = np.zeros_like(bind["global_rotations"])
    global_positions = np.zeros_like(bind["global_positions"])
    global_rotations[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    global_positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    global_rotations[usda_to_full] = rotations
    global_positions[usda_to_full] = positions

    bind["global_rotations"] = global_rotations
    bind["global_positions"] = global_positions


def build_viewer_mesh(skeleton: SomaSkeleton, mesh: SomaMesh, bind: dict[str, object]) -> dict[str, np.ndarray]:
    """Triangulate, reduce influences, and remap bones to the bind-BVH order.

    Bone order in the .bin follows the bind BVH minus its synthetic Simulation
    root (viewer convention: model bone i == full skeleton joint i+1), so the
    per-vertex usda joint indices are remapped by joint name.
    """
    bvh_names = list(bind["names"])[1:]  # drop "Simulation"
    usda_to_bvh = np.full(len(skeleton.joint_paths), -1, dtype=np.int64)
    for usda_idx, path in enumerate(skeleton.joint_paths):
        name = path.rsplit("/", 1)[-1]
        if name in bvh_names:
            usda_to_bvh[usda_idx] = bvh_names.index(name)
    if np.any(usda_to_bvh < 0):
        missing = [skeleton.joint_paths[i] for i in np.nonzero(usda_to_bvh < 0)[0]]
        raise ValueError(f"USD joints missing from the bind BVH: {missing[:5]}")

    vertex_count = mesh.points.shape[0]
    weights = mesh.joint_weights.copy()
    joints = usda_to_bvh[mesh.joint_indices]
    # Zero-weight influences are dropped by the top-K selection below; a joint
    # index of -1 cannot survive because its weight is zero by construction.
    top = np.argsort(-weights, axis=1)[:, :INFLUENCES_PER_VERTEX]
    top_weights = np.take_along_axis(weights, top, axis=1)
    top_joints = np.take_along_axis(joints, top, axis=1)
    total = top_weights.sum(axis=1, keepdims=True)
    degenerate = (total <= 1e-8).sum()
    if degenerate:
        raise ValueError(f"{degenerate} vertices have zero total skin weight")
    top_weights = top_weights / total

    # Fan triangulation of each polygon.
    tri_counts = mesh.face_vertex_counts - 2
    tri_counts = np.maximum(tri_counts, 0)
    triangle_count = int(tri_counts.sum())
    indices = np.empty(triangle_count * 3, dtype=np.int64)
    write = 0
    read = 0
    for count in mesh.face_vertex_counts:
        count = int(count)
        if count < 3:
            read += count
            continue
        fan = mesh.face_vertex_indices[read : read + count]
        for corner in range(1, count - 1):
            indices[write : write + 3] = (fan[0], fan[corner], fan[corner + 1])
            write += 3
        read += count

    normals = _vertex_normals(mesh.points, indices, triangle_count)
    texcoords = _vertex_texcoords(mesh, vertex_count)

    return {
        "vertices": (mesh.points * 0.01).astype(np.float32),  # cm -> m to match the mesh unit the viewer expects
        "texcoords": texcoords,
        "normals": normals,
        "bone_ids": top_joints.astype(np.uint8),
        "bone_weights": top_weights.astype(np.float32),
        "indices": indices.astype(np.uint16),
        "triangle_count": triangle_count,
        "vertex_count": vertex_count,
    }


def _vertex_normals(points: np.ndarray, indices: np.ndarray, triangle_count: int) -> np.ndarray:
    """Area-weighted vertex normals; the usda ships no normals."""
    tri = indices.reshape(triangle_count, 3)
    v0 = points[tri[:, 0]]
    v1 = points[tri[:, 1]]
    v2 = points[tri[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, tri[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return (normals / norm).astype(np.float32)


def _vertex_texcoords(mesh: SomaMesh, vertex_count: int) -> np.ndarray:
    """Per-vertex UVs averaged from the faceVarying set.

    The viewer materials do not sample a texture; UVs are written only because
    the mesh format has the slot.
    """
    per_corner = mesh.uv_indices if mesh.uv_indices is not None else mesh.face_vertex_indices
    if per_corner.shape[0] != mesh.face_vertex_indices.shape[0]:
        per_corner = mesh.face_vertex_indices
    corner_uvs = mesh.uv_values[per_corner]
    texcoords = np.zeros((vertex_count, 2), dtype=np.float32)
    np.add.at(texcoords, mesh.face_vertex_indices, corner_uvs)
    counts = np.zeros(vertex_count, dtype=np.float32)
    np.add.at(counts, mesh.face_vertex_indices, 1.0)
    counts[counts == 0] = 1.0
    return (texcoords / counts[:, None]).astype(np.float32)


def _pack_bone_info(name: str, parent: int) -> bytes:
    """BoneInfo: char name[32] + int parent (36 B total in raylib 5.5)."""
    encoded = name.encode()[:31]
    return encoded.ljust(32, b"\x00") + struct.pack("<i", parent)


def _pack_transform(position: np.ndarray, rotation_wxyz: np.ndarray) -> bytes:
    """Transform: float3 translation + quat xyzw + float3 scale + 2 pad (40 B)."""
    values = [
        float(position[0]), float(position[1]), float(position[2]),
        float(rotation_wxyz[1]), float(rotation_wxyz[2]), float(rotation_wxyz[3]), float(rotation_wxyz[0]),
        1.0, 1.0, 1.0,
    ]
    return struct.pack("<10f", *values)


def write_soma_bin(path: Path, viewer_mesh: dict[str, np.ndarray], bind: dict[str, object]) -> None:
    """Write the GenoView-compatible binary for the SOMA mesh and bind pose."""
    bone_count = len(bind["names"]) - 1
    bvh_parents = np.asarray(bind["parents"], dtype=np.int32)[1:]  # drop Simulation's parent slot
    global_rotations = bind["global_rotations"]
    global_positions = bind["global_positions"]

    if bone_count > MAX_BONE_NUM:
        raise ValueError(f"{bone_count} bones exceeds the shader cap {MAX_BONE_NUM}")

    bone_blob = bytearray()
    bind_blob = bytearray()
    for index in range(bone_count):
        # Model bone i mirrors full-skeleton joint i+1 (the synthetic Simulation
        # root lives only in the skeleton, not in the mesh bones).
        bone_blob += _pack_bone_info(str(bind["names"][index + 1]), int(bvh_parents[index]))
        bind_blob += _pack_transform(global_positions[index + 1], global_rotations[index + 1])

    vertex_count = int(viewer_mesh["vertex_count"])
    triangle_count = int(viewer_mesh["triangle_count"])
    with open(path, "wb") as f:
        f.write(struct.pack("<III", vertex_count, triangle_count, bone_count))
        f.write(np.ascontiguousarray(viewer_mesh["vertices"], dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(viewer_mesh["texcoords"], dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(viewer_mesh["normals"], dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(viewer_mesh["bone_ids"], dtype=np.uint8).tobytes())
        f.write(np.ascontiguousarray(viewer_mesh["bone_weights"], dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(viewer_mesh["indices"], dtype="<u2").tobytes())
        f.write(bytes(bone_blob))
        f.write(bytes(bind_blob))


SHADER_FILES = (
    "basic.fs", "basic.vs", "blur.fs", "fxaa.fs", "lighting.fs", "post.vs",
    "shadow.fs", "shadow.vs", "skinnedBasic.vs", "skinnedShadow.vs", "ssao.fs",
)


def _quat_to_zyx_euler_degrees(rotations_wxyz: np.ndarray) -> np.ndarray:
    """Decompose quaternions into the BVH ZYX euler channels (degrees).

    Inverse of ``quat.from_euler(e, order='zyx')``: channels are stored as
    (Zrotation, Yrotation, Xrotation) per joint.
    """
    matrices = quat.to_xform(rotations_wxyz.astype(np.float32))  # row-major R
    clamp = np.clip(-matrices[:, 2, 0], -1.0, 1.0)
    z = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
    y = np.arcsin(clamp)
    x = np.arctan2(matrices[:, 2, 1], matrices[:, 2, 2])
    gimbal = np.abs(clamp) > 0.99999
    if np.any(gimbal):
        z[gimbal] = np.arctan2(-matrices[gimbal, 0, 1], matrices[gimbal, 1, 1])
        x[gimbal] = 0.0
    return np.degrees(np.stack([z, y, x], axis=1)).astype(np.float32)


def write_apose_bvh(path: Path, skeleton: SomaSkeleton, bind_bvh_path: Path) -> None:
    """Write the USD ``bindTransforms`` pose as a playable single-frame BVH.

    The bind BVH's frame 0 is a T-pose that only defines the hierarchy; the
    skinned mesh rests in the relaxed pose carried by the USD bindTransforms.
    This pose is exported with local translations derived from the USD world
    transforms so it can be inspected directly in the viewer.
    """
    from stylized_motion.anim import bvh as bvh_io

    data = bvh_io.load(str(bind_bvh_path))
    bvh_names = [str(name) for name in data["names"]]
    parents = np.asarray(data["parents"], dtype=np.int64)

    matrices = skeleton.bind_transforms  # world transforms, USD row-vector convention
    world_rot = quat.from_xform(np.transpose(matrices[:, :3, :3], (0, 2, 1)).astype(np.float32))
    world_pos = matrices[:, 3, :3].astype(np.float32)

    usda_to_bvh = np.full(len(skeleton.joint_paths), -1, dtype=np.int64)
    bvh_to_usda = np.full(len(bvh_names), -1, dtype=np.int64)
    for usda_idx, joint_path in enumerate(skeleton.joint_paths):
        bvh_idx = bvh_names.index(joint_path.rsplit("/", 1)[-1])
        usda_to_bvh[usda_idx] = bvh_idx
        bvh_to_usda[bvh_idx] = usda_idx
    if np.any(usda_to_bvh < 0):
        raise ValueError("USD joints missing from the bind BVH cannot be exported to an A-pose clip")

    local_rot = np.zeros((len(bvh_names), 4), dtype=np.float32)
    local_rot[:, 0] = 1.0
    local_pos = np.zeros((len(bvh_names), 3), dtype=np.float32)
    for bvh_idx in range(len(bvh_names)):
        usda_idx = int(bvh_to_usda[bvh_idx])
        parent_bvh = int(parents[bvh_idx])
        if parent_bvh == -1:
            local_rot[bvh_idx] = world_rot[usda_idx]
            local_pos[bvh_idx] = world_pos[usda_idx]
            continue
        parent_usda = int(bvh_to_usda[parent_bvh])
        local_rot[bvh_idx] = quat.inv_mul(world_rot[parent_usda][None], world_rot[usda_idx][None])[0]
        local_pos[bvh_idx] = quat.inv_mul_vec(
            world_rot[parent_usda][None], world_pos[usda_idx][None] - world_pos[parent_usda][None]
        )[0]

    euler = _quat_to_zyx_euler_degrees(local_rot)
    # USD translations are centimeters (metersPerUnit = 0.01), which is also
    # the BVH unit; no further scaling is needed.
    _write_bvh_frames(path, bvh_names, parents, local_pos, euler[None])


def _write_bvh_frames(
    path: Path,
    names: list[str],
    parents: np.ndarray,
    offsets_cm: np.ndarray,
    euler_frames: np.ndarray,
    frame_time: float = 1.0 / 24.0,
) -> None:
    children: dict[int, list[int]] = {}
    for index, parent in enumerate(parents):
        children.setdefault(int(parent), []).append(index)

    lines = ["HIERARCHY"]

    def emit(index: int, depth: int) -> None:
        indent = "\t" * depth
        keyword = "ROOT" if parents[index] == -1 else "JOINT"
        lines.append(f"{indent}{keyword} {names[index]}")
        lines.append(f"{indent}{{")
        offset = offsets_cm[index]
        lines.append(f"{indent}\tOFFSET {offset[0]:.8f} {offset[1]:.8f} {offset[2]:.8f}")
        if parents[index] == -1:
            lines.append(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{indent}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        for child in children.get(index, []):
            emit(child, depth + 1)
        lines.append(f"{indent}}}")

    emit(0, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(euler_frames)}")
    lines.append(f"Frame Time: {frame_time:.17f}")
    for frame in euler_frames:
        # Root position comes from its A-pose world position (already in offsets);
        # every joint contributes only its three rotation channels.
        values = [0.0, 0.0, 0.0, *frame[0].tolist()]
        for index in range(1, len(names)):
            values.extend(frame[index].tolist())
        lines.append(" ".join(f"{value:.8f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_soma_assets(usd_path: Path, bind_bvh_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / "SOMA.bin"
    bind_out = output_dir / "SOMA_bind.bvh"
    if not overwrite and (bin_path.exists() or bind_out.exists()):
        raise FileExistsError(f"{bin_path} or {bind_out} already exists; pass --overwrite")

    skeleton, mesh = parse_soma_usd(usd_path)
    bind = _build_bvh_bind_pose(bind_bvh_path, unit_scale=0.01)
    build_bind_from_usd(skeleton, bind)
    viewer_mesh = build_viewer_mesh(skeleton, mesh, bind)
    write_soma_bin(bin_path, viewer_mesh, bind)
    shutil.copyfile(bind_bvh_path, bind_out)
    apose_path = output_dir / "SOMA_apose.bvh"
    write_apose_bvh(apose_path, skeleton, bind_bvh_path)
    for shader in SHADER_FILES:
        shutil.copyfile(RESOURCE_DIR / shader, output_dir / shader)

    print(
        f"wrote {bin_path} ({bin_path.stat().st_size} bytes): "
        f"{viewer_mesh['vertex_count']} vertices, {viewer_mesh['triangle_count']} triangles, "
        f"{len(bind['names']) - 1} bones"
    )
    return bin_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True, help="Path to soma_base_skel_minimal.usd")
    parser.add_argument("--bvh", type=Path, required=True, help="Path to soma_base_skel_minimal.bvh")
    parser.add_argument("--output-dir", type=Path, required=True, help="somaview resource directory to populate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_soma_assets(args.usd, args.bvh, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
