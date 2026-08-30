import struct
from pathlib import Path

import numpy as np
import pytest

from stylized_motion.anim import quat
from stylized_motion.anim.soma_assets import (
    build_bind_from_usd,
    build_viewer_mesh,
    parse_soma_usd,
    write_apose_bvh,
    write_soma_bin,
    _build_bvh_bind_pose,
    _quat_to_zyx_euler_degrees,
)

SOMA_ASSETS = Path(__file__).resolve().parents[1] / "data" / "assets" / "somaview"


MINI_JOINT_INDICES = (
    np.array(
        [
            [1, 1, 0, 0, 2, 2, 0, 0],  # v0: Hips + Root + Head zero-weight fillers
            [1, 1, 0, 0, 2, 2, 0, 0],  # v1
            [1, 1, 0, 0, 2, 2, 0, 0],  # v2
            [2, 2, 1, 1, 0, 0, 0, 0],  # v3: Head + Hips
            [2, 2, 1, 1, 0, 0, 0, 0],  # v4
        ],
        dtype=np.int64,
    )
    .reshape(-1)
    .tolist()
)

MINI_JOINT_WEIGHTS = (
    np.array(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.2, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    .reshape(-1)
    .tolist()
)

MINI_USD = f"""#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Y"
)

def SkelRoot "OUTPUT"
{{
    def Xform "c_skeleton_grp"
    {{
        def Skeleton "Root"
        {{
            uniform token[] joints = ["Root", "Root/Hips", "Root/Head"]
            uniform matrix4d[] bindTransforms = [( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) ), ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 100, 0, 1) ), ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 160, 0, 1) )]
        }}
    }}

    def Xform "c_geometry_grp"
    {{
        def Mesh "Mesh"
        {{
            int[] faceVertexCounts = [4, 3]
            int[] faceVertexIndices = [0, 1, 2, 3, 1, 4, 2]
            point3f[] points = [(-10, 100, 0), (10, 100, 0), (10, 160, 0), (-10, 160, 0), (0, 170, 0)]
            int[] primvars:skel:jointIndices = [{", ".join(str(v) for v in MINI_JOINT_INDICES)}]
                elementSize = 8
                interpolation = "vertex"
            float[] primvars:skel:jointWeights = [{", ".join(repr(float(v)) for v in MINI_JOINT_WEIGHTS)}]
                elementSize = 8
                interpolation = "vertex"
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
                interpolation = "faceVarying"
        }}
    }}
}}
"""

MINI_BVH = """HIERARCHY
ROOT Root
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Hips
    {
        OFFSET 0.0 100.0 0.0
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT Spine2
        {
            OFFSET 0.0 20.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT Head
            {
                OFFSET 0.0 40.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                End Site
                {
                    OFFSET 0.0 10.0 0.0
                }
            }
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.041666666666666664
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"""


@pytest.fixture
def mini_rig(tmp_path: Path) -> tuple[Path, Path]:
    usd = tmp_path / "mini.usd"
    usd.write_text(MINI_USD, encoding="utf-8")
    bvh = tmp_path / "mini.bvh"
    bvh.write_text(MINI_BVH, encoding="utf-8")
    return usd, bvh


def test_parse_and_mesh_conversion(mini_rig):
    usd, bvh = mini_rig
    skeleton, mesh = parse_soma_usd(usd)
    assert len(skeleton.joint_paths) == 3
    assert mesh.points.shape == (5, 3)

    bind = _build_bvh_bind_pose(bvh, unit_scale=0.01)
    build_bind_from_usd(skeleton, bind)
    viewer = build_viewer_mesh(skeleton, mesh, bind)

    # quad -> 2 triangles + one native triangle = 3 triangles
    assert viewer["triangle_count"] == 3
    assert viewer["indices"].shape == (9,)
    # top-4 weights renormalized per vertex
    assert np.allclose(viewer["bone_weights"].sum(axis=1), 1.0, atol=1e-6)
    assert viewer["bone_ids"].shape == (5, 4)
    # ids index the full skeleton minus Simulation: Root, Hips, Spine2, Head
    assert viewer["bone_ids"].max() < 4
    # vertices converted to meters
    assert np.allclose(viewer["vertices"], mesh.points * 0.01)
    # bind pose joints mapped by name: Hips at 1.0 m, Head at 1.6 m (identity rotations)
    names = [str(n) for n in bind["names"]]
    assert names == ["Simulation", "Root", "Hips", "Spine2", "Head"]
    assert np.allclose(bind["global_positions"][names.index("Hips")], [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(bind["global_positions"][names.index("Head")], [0.0, 1.6, 0.0], atol=1e-6)

    output = usd.parent / "mini.bin"
    write_soma_bin(output, viewer, bind)
    raw = output.read_bytes()
    vertex_count, triangle_count, bone_count = struct.unpack_from("<III", raw)
    bone_offset = 12 + vertex_count * (12 + 8 + 12 + 4 + 16) + triangle_count * 6
    parents = [struct.unpack_from("<i", raw, bone_offset + index * 36 + 32)[0] for index in range(bone_count)]
    assert parents == [-1, 0, 1, 2]


def test_bind_pose_is_identity_skinnable(mini_rig):
    """LBS with anim == bind must reproduce the mesh (the rest-pose identity)."""
    usd, bvh = mini_rig
    skeleton, mesh = parse_soma_usd(usd)
    bind = _build_bvh_bind_pose(bvh, unit_scale=0.01)
    build_bind_from_usd(skeleton, bind)
    viewer = build_viewer_mesh(skeleton, mesh, bind)

    global_rotations = bind["global_rotations"]
    global_positions = bind["global_positions"]
    skinned = np.zeros_like(viewer["vertices"])
    for k in range(4):
        weights = viewer["bone_weights"][:, k]
        bones = viewer["bone_ids"][:, k].astype(np.int64)
        local = quat.inv_mul_vec(global_rotations[bones], viewer["vertices"] - global_positions[bones])
        skinned += weights[:, None] * (quat.mul_vec(global_rotations[bones], local) + global_positions[bones])
    assert np.abs(skinned - viewer["vertices"]).max() < 1e-5


def test_missing_usd_joint_in_bvh_raises(mini_rig):
    usd, bvh = mini_rig
    skeleton, mesh = parse_soma_usd(usd)
    text = bvh.read_text(encoding="utf-8").replace("JOINT Head", "JOINT Skull")
    bvh.write_text(text, encoding="utf-8")
    bind = _build_bvh_bind_pose(bvh, unit_scale=0.01)
    with pytest.raises(ValueError, match="Head"):
        build_bind_from_usd(skeleton, bind)


def test_quat_to_zyx_euler_round_trip():
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(64, 3)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = (rng.uniform(-3.0, 3.0, size=(64,))).astype(np.float32)
    quaternions = np.stack(
        [np.cos(angles / 2), *(np.sin(angles / 2)[:, None] * axes).T], axis=1
    ).astype(np.float32)

    euler = _quat_to_zyx_euler_degrees(quaternions)
    recovered = quat.from_euler(np.radians(euler), order="zyx")
    dots = np.abs(np.sum(recovered * quaternions, axis=1))
    assert dots.min() > 0.999999


def test_apose_bvh_round_trips_to_usd_bind_pose(mini_rig):
    """The exported A-pose BVH must FK back to the USD bindTransforms pose."""
    from stylized_motion.anim import bvh as bvh_io

    usd, bvh = mini_rig
    skeleton, _ = parse_soma_usd(usd)
    out = usd.parent / "mini_apose.bvh"
    write_apose_bvh(out, skeleton, bvh)

    data = bvh_io.load(str(out))
    names = [str(n) for n in data["names"]]
    assert names == ["Root", "Hips", "Spine2", "Head"]

    # BVH positions parse in centimeters; USD bindTransforms are also
    # centimeters (metersPerUnit = 0.01) -- compare like with like.
    rotations = quat.unroll(quat.from_euler(np.radians(data["rotations"]), order=data["order"])).astype(np.float32)
    grot, gpos = quat.fk(rotations, data["positions"].astype(np.float32), np.asarray(data["parents"], dtype=np.int32))

    matrices = skeleton.bind_transforms
    usd_pos = matrices[:, 3, :3].astype(np.float32)
    usd_rot = quat.from_xform(np.transpose(matrices[:, :3, :3], (0, 2, 1)).astype(np.float32))
    for usda_idx, path in enumerate(skeleton.joint_paths):
        name = path.rsplit("/", 1)[-1]
        bvh_idx = names.index(name)
        assert np.abs(gpos[0][bvh_idx] - usd_pos[usda_idx]).max() < 1e-3
        assert abs(abs(float(np.sum(grot[0][bvh_idx] * usd_rot[usda_idx]))) - 1.0) < 1e-5


@pytest.mark.skipif(not (SOMA_ASSETS / "SOMA.bin").exists(), reason="SOMA assets have not been generated")
def test_generated_soma_bin_layout_matches_viewer_contract():
    raw = (SOMA_ASSETS / "SOMA.bin").read_bytes()
    vertex_count, triangle_count, bone_count = struct.unpack("<III", raw[:12])
    assert (vertex_count, triangle_count, bone_count) == (18056, 36108, 78)
    expected = 12 + vertex_count * (12 + 8 + 12 + 4 + 16) + triangle_count * 6 + bone_count * (36 + 40)
    assert len(raw) == expected

    bind_pose = np.frombuffer(raw, "<f4", bone_count * 10, len(raw) - bone_count * 40).reshape(bone_count, 10)
    quaternions = np.stack(
        [bind_pose[:, 6], bind_pose[:, 3], bind_pose[:, 4], bind_pose[:, 5]], axis=1
    )
    norms = np.linalg.norm(quaternions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
