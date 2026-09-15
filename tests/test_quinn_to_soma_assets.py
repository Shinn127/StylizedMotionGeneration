"""Contracts for the Quinn -> SOMA asset conversion.

The fixture is a miniature GLB built in-process: the full set of alignment
joints plus one triangle, driven by the real mapping table.  It covers accessor
addressing (byteOffset/byteStride), twist/duplicate/finger weight merging,
four-influence truncation, region-fit degeneracy, the failure path for weighted
joints without a mapping target, and the round trip of the written ``SOMA.bin``.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from stylized_motion.anim.quinn_to_soma_assets import (
    QUINN_TO_SOMA,
    _ALIGN_JOINTS,
    align_mesh_to_soma_bind,
    build_bone_index,
    build_quinn_assets,
    fit_region_similarity,
    load_quinn_mesh,
    load_soma_bind,
    map_quinn_weights,
    read_soma_bin,
    soma_bin_layout,
)
from stylized_motion.anim.soma_assets import SHADER_FILES, _write_bvh_frames, write_soma_bin

GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942

# Quinn joints the fixture carries: every alignment joint plus the twist,
# metacarpal and unmapped cases the mapping rules have to cover.
QUINN_JOINTS = (
    "rootJoint_0", "pelvis_1", "spine_01_2", "spine_02_3", "spine_03_4", "spine_04_5",
    "neck_01_6", "neck_02_7", "head_8", "clavicle_l_9", "upperarm_l_10", "upperarm_twist_01_l_11",
    "lowerarm_l_12", "hand_l_13", "index_metacarpal_l_14", "index_01_l_15", "index_02_l_16",
    "index_03_l_17", "clavicle_r_18", "upperarm_r_19", "lowerarm_r_20", "hand_r_21",
    "thigh_l_22", "calf_l_23", "foot_l_24", "ball_l_25",
    "thigh_r_26", "calf_r_27", "foot_r_28", "ball_r_29", "mystery_30",
)

# Quinn bind joint centers, Keyed by the suffix-stripped name.
QUINN_BIND = {
    "rootJoint": (0.0, 0.0, 0.0),
    "pelvis": (0.0, 1.00, 0.0),
    "spine_01": (0.0, 1.02, 0.0),
    "spine_02": (0.0, 1.05, 0.0),
    "spine_03": (0.0, 1.12, 0.0),
    "spine_04": (0.0, 1.20, 0.0),
    "neck_01": (0.0, 1.45, 0.0),
    "neck_02": (0.0, 1.52, 0.0),
    "head": (0.0, 1.60, 0.0),
    "clavicle_l": (0.02, 1.42, 0.0),
    "clavicle_r": (-0.02, 1.42, 0.0),
    "upperarm_l": (0.18, 1.42, 0.0),
    "upperarm_r": (-0.18, 1.42, 0.0),
    "lowerarm_l": (0.36, 1.20, 0.0),
    "lowerarm_r": (-0.36, 1.20, 0.0),
    "hand_l": (0.50, 1.02, 0.0),
    "hand_r": (-0.50, 1.02, 0.0),
    "thigh_l": (0.10, 0.92, 0.0),
    "thigh_r": (-0.10, 0.92, 0.0),
    "calf_l": (0.13, 0.48, 0.0),
    "calf_r": (-0.13, 0.48, 0.0),
    "foot_l": (0.15, 0.07, 0.0),
    "foot_r": (-0.15, 0.07, 0.0),
    "ball_l": (0.17, 0.02, 0.09),
    "ball_r": (-0.17, 0.02, 0.09),
    "mystery": (0.0, 1.30, 0.0),
}

# The synthetic SOMA bind pose is this similarity image of the Quinn skeleton,
# so the alignment has a known exact answer.
SOMA_SCALE, SOMA_OFFSET = 1.05, np.array([0.01, -0.02, 0.03])
SOMA_ROTATION = np.array([[0.9986, 0.0, 0.0523], [0.0, 1.0, 0.0], [-0.0523, 0.0, 0.9986]])  # 3 deg about Y


def _stem(name: str) -> str:
    return name.rsplit("_", 1)[0] if name[-1].isdigit() else name


JOINT_SLOT = {_stem(name): index for index, name in enumerate(QUINN_JOINTS)}
SOMA_BONES = sorted({QUINN_TO_SOMA[_stem(name)] for name in QUINN_JOINTS if _stem(name) in QUINN_TO_SOMA} | {"Root"})
SOMA_INDEX = {name: index for index, name in enumerate(SOMA_BONES)}


def _soma_source_stem(bone_name: str) -> str | None:
    """The Quinn joint a fixture SOMA bone mirrors: align joints win, since the
    duplicated spine and twist chains also map onto the same bones."""
    for stem in _ALIGN_JOINTS:
        if QUINN_TO_SOMA[stem] == bone_name:
            return stem
    return next((stem for stem in QUINN_TO_SOMA if QUINN_TO_SOMA[stem] == bone_name), None)


def soma_bind_positions() -> np.ndarray:
    """SOMA bind centers for the fixture bones, as a similarity image of Quinn."""
    positions = np.zeros((len(SOMA_BONES), 3), dtype=np.float64)
    for name in SOMA_BONES:
        source = _soma_source_stem(name)
        base = np.array(QUINN_BIND.get(source, (0.0, 0.0, 0.0)), dtype=np.float64)
        positions[SOMA_INDEX[name]] = SOMA_SCALE * (SOMA_ROTATION @ base) + SOMA_OFFSET
    return positions


def quinn_bind_positions() -> np.ndarray:
    # Joints outside the align set (twists, metacarpals, the unmapped ones) only
    # need a deterministic position; they never enter the similarity fit.
    return np.stack([QUINN_BIND.get(_stem(name), (0.0, 1.0, 0.0)) for name in QUINN_JOINTS]).astype(np.float64)


def _glb(json_body: dict, binary: bytes) -> bytes:
    payload = json.dumps(json_body).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    padded = binary + b"\x00" * (-len(binary) % 4)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(payload) + 8 + len(padded)),
            struct.pack("<II", len(payload), GLB_JSON_CHUNK),
            payload,
            struct.pack("<II", len(padded), GLB_BIN_CHUNK),
            padded,
        ]
    )


def _build_fixture(tmp_path: Path, weights: np.ndarray) -> Path:
    """Write a GLB whose single triangle carries the supplied skin weights."""
    positions = np.array([[0.0, 0.5, 0.0], [0.1, 0.5, 0.0], [0.0, 0.6, 0.0]], dtype=np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (3, 1))
    texcoords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    blobs: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(data: bytes, pad: int = 0, stride: int | None = None) -> int:
        blobs.append(b"\x00" * pad + data)
        views.append({"buffer": 0, "byteOffset": sum(len(blob) for blob in blobs) - len(data) - pad, "byteLength": len(data) + pad, **({"byteStride": stride} if stride else {})})
        return len(views) - 1

    def add_accessor(view: int, component_type: int, kind: str, count: int, offset: int = 0) -> int:
        accessors.append(
            {"bufferView": view, "componentType": component_type, "type": kind, "count": count, **({"byteOffset": offset} if offset else {})}
        )
        return len(accessors) - 1

    index_accessor = add_accessor(add_view(np.arange(3, dtype="<u4").tobytes()), 5125, "SCALAR", 3)
    # POSITION/NORMAL sit behind a four-byte pad inside their views, so the
    # accessor byteOffset has to be honored on top of the bufferView offset.
    position_accessor = add_accessor(add_view(positions.astype("<f4").tobytes(), pad=4), 5126, "VEC3", 3, offset=4)
    normal_accessor = add_accessor(add_view(normals.astype("<f4").tobytes(), pad=4), 5126, "VEC3", 3, offset=4)
    uv_accessor = add_accessor(add_view(texcoords.astype("<f4").tobytes()), 5126, "VEC2", 3)
    joint_accessor = add_accessor(add_view(np.full((3, 4), JOINT_SLOT["pelvis"], dtype="<u2").tobytes()), 5123, "VEC4", 3)
    weight_accessor = add_accessor(add_view(weights.astype("<f4").tobytes()), 5126, "VEC4", 3)
    # Inverse bind matrices are padded to an 80-byte stride (64 bytes plus
    # padding); their inverses put every joint at its QUINN_BIND center.
    bind_frames = np.tile(np.eye(4), (len(QUINN_JOINTS), 1, 1))
    bind_frames[:, :3, 3] = quinn_bind_positions()
    padded_bind = np.zeros((len(QUINN_JOINTS), 20), dtype="<f4")
    # glTF stores matrices column-major: transpose before the row-major flatten.
    padded_bind[:, :16] = np.transpose(np.linalg.inv(bind_frames), (0, 2, 1)).reshape(len(QUINN_JOINTS), 16)
    bind_view = add_view(padded_bind.tobytes())
    views[bind_view]["byteStride"] = 80
    bind_accessor = add_accessor(bind_view, 5126, "MAT4", len(QUINN_JOINTS))

    # Two one-pixel JPEG payloads stand in for Quinn's embedded textures.
    base_color_view = add_view(b"\xff\xd8\xff\xd9")
    normal_view = add_view(b"\xff\xd8\xff\xd9")

    nodes = [
        {"name": name, **({"children": [index + 1]} if index + 1 < len(QUINN_JOINTS) else {})}
        for index, name in enumerate(QUINN_JOINTS)
    ]
    nodes.append({"name": "SkinnedMesh", "mesh": 0, "skin": 0})
    nodes[0]["children"] = [1, len(nodes) - 1]

    body = {
        "asset": {"version": "2.0", "generator": "quinn_to_soma_assets fixture"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "TEXCOORD_0": uv_accessor,
                            "JOINTS_0": joint_accessor,
                            "WEIGHTS_0": weight_accessor,
                        },
                        "indices": index_accessor,
                        "mode": 4,
                    }
                ]
            }
        ],
        "skins": [{"joints": list(range(len(QUINN_JOINTS))), "inverseBindMatrices": bind_accessor}],
        "images": [{"bufferView": base_color_view, "mimeType": "image/jpeg"}, {"bufferView": normal_view, "mimeType": "image/jpeg"}],
        "textures": [{"source": 0}, {"source": 1}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}, "normalTexture": {"index": 1}}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(blob) for blob in blobs)}],
    }
    path = tmp_path / "quinn_fixture.glb"
    path.write_bytes(_glb(body, b"".join(blobs)))
    return path


def _write_soma_resources(root: Path):
    """A miniature SOMA resource directory using the real binary contract."""
    viewer_mesh = {
        "vertices": np.zeros((3, 3), dtype=np.float32),
        "texcoords": np.zeros((3, 2), dtype=np.float32),
        "normals": np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (3, 1)),
        "bone_ids": np.zeros((3, 4), dtype=np.uint8),
        "bone_weights": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
        "indices": np.zeros(3, dtype=np.uint16),
        "triangle_count": 1,
        "vertex_count": 3,
    }
    parents = np.array([-1, *range(len(SOMA_BONES))], dtype=np.int64)
    bind = {
        "names": ["Simulation", *SOMA_BONES],
        "parents": parents.tolist(),
        "global_positions": np.concatenate([np.zeros((1, 3)), soma_bind_positions()]).astype(np.float32),
        "global_rotations": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(SOMA_BONES) + 1, 1)),
    }
    bind["local_positions"] = bind["global_positions"]
    bind["local_rotations"] = bind["global_rotations"]
    root.mkdir(parents=True, exist_ok=True)
    write_soma_bin(root / "SOMA.bin", viewer_mesh, bind)
    _write_bvh_frames(
        root / "SOMA_bind.bvh",
        bind["names"],
        parents,
        np.zeros((len(SOMA_BONES) + 1, 3)),
        np.zeros((1, len(SOMA_BONES) + 1, 3)),
    )
    for shader in SHADER_FILES:
        (root / shader).write_text("// fixture\n", encoding="utf-8")
    return load_soma_bind(root)


def _weights_for(slots: list[int], values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """One row of influences per vertex, zero-padded to four slots."""
    padded_slots = [*slots, *([0] * (4 - len(slots)))]
    padded_values = [*values, *([0.0] * (4 - len(values)))]
    joints = np.tile(np.array([padded_slots], dtype=np.int64), (3, 1))
    weights = np.tile(np.array([padded_values], dtype=np.float32), (3, 1))
    return joints, weights


def _unit_weights() -> np.ndarray:
    return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1))


def test_load_quinn_mesh_reads_padded_accessors(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    np.testing.assert_allclose(mesh.positions, [[0.0, 0.5, 0.0], [0.1, 0.5, 0.0], [0.0, 0.6, 0.0]])
    np.testing.assert_allclose(mesh.texcoords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(mesh.joint_positions, quinn_bind_positions())
    assert mesh.joint_parents[1] == 0 and mesh.joint_parents[0] == -1


def test_joint_names_lose_their_sketchfab_suffix(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    assert mesh.joint_names[1] == "pelvis"
    assert mesh.joint_names[11] == "upperarm_twist_01_l"
    assert mesh.joint_names[14] == "index_metacarpal_l"
    assert mesh.joint_names[0] == "rootJoint"


def test_twist_and_duplicate_influences_merge(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    mesh.joint_indices, mesh.joint_weights = _weights_for(
        [JOINT_SLOT["upperarm_twist_01_l"], JOINT_SLOT["upperarm_l"], JOINT_SLOT["pelvis"]], [0.3, 0.2, 0.5]
    )
    bone_index, report = build_bone_index(mesh, _write_soma_resources(tmp_path / "soma"))
    bone_ids, bone_weights, _ = map_quinn_weights(mesh, bone_index, SOMA_BONES)
    assert bone_ids[0, 0] == SOMA_INDEX["LeftArm"]
    assert bone_weights[0, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(bone_weights.sum(axis=1), 1.0, atol=1e-6)
    assert report["rule_weight"]["upperarm_twist_01_l -> LeftArm"] == pytest.approx(0.9)  # three vertices x 0.3


def test_finger_metacarpal_maps_one_slot_ahead(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    mesh.joint_indices, mesh.joint_weights = _weights_for(
        [JOINT_SLOT["index_metacarpal_l"], JOINT_SLOT["index_01_l"], JOINT_SLOT["index_02_l"]], [0.5, 0.3, 0.2]
    )
    bone_index, _ = build_bone_index(mesh, _write_soma_resources(tmp_path / "soma"))
    bone_ids, _, _ = map_quinn_weights(mesh, bone_index, SOMA_BONES)
    assert bone_ids[0, 0] == SOMA_INDEX["LeftHandIndex1"]
    assert bone_ids[0, 1] == SOMA_INDEX["LeftHandIndex2"]
    assert bone_ids[0, 2] == SOMA_INDEX["LeftHandIndex3"]


def test_top_four_truncation_drops_the_remainder(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    # Five influences on one vertex: the set the four-slot contract has to cut.
    mesh.joint_indices = np.tile(
        np.array([[JOINT_SLOT[name] for name in ("upperarm_l", "lowerarm_l", "hand_l", "pelvis", "head")]]), (3, 1)
    )
    mesh.joint_weights = np.tile(np.array([[0.4, 0.3, 0.15, 0.1, 0.05]], dtype=np.float32), (3, 1))
    bone_index, _ = build_bone_index(mesh, _write_soma_resources(tmp_path / "soma"))
    _, bone_weights, report = map_quinn_weights(mesh, bone_index, SOMA_BONES)
    assert report["dropped_weight"]["max"] == pytest.approx(0.05, abs=1e-6)
    assert int((bone_weights[0] > 0.0).sum()) == 4
    np.testing.assert_allclose(bone_weights.sum(axis=1), 1.0, atol=1e-6)
    assert report["influence_count_histogram"]["4"] == 3


def test_weighted_joint_without_mapping_target_fails(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    mesh.joint_indices, mesh.joint_weights = _weights_for([JOINT_SLOT["mystery"]], [1.0])
    with pytest.raises(ValueError, match="mystery"):
        build_bone_index(mesh, _write_soma_resources(tmp_path / "soma"))


def test_unweighted_joint_without_mapping_is_reported(tmp_path):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    mesh.joint_indices, mesh.joint_weights = _weights_for([JOINT_SLOT["pelvis"]], [1.0])
    _, report = build_bone_index(mesh, _write_soma_resources(tmp_path / "soma"))
    assert report["unmapped_joints"] == ["rootJoint", "mystery"]


def test_region_fit_leaves_collinear_points_unrotated():
    # A free similarity spins a straight line about its own axis; the region fit
    # has to keep the base rotation instead of flipping the region.
    source = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.01, 0.5, 0.0]])
    target = np.array([[0.0, 0.1, 0.0], [0.0, 0.4, 0.0], [0.0, 0.7, 0.0], [0.01, 0.4, 0.0]])
    scale, rotation, translation = fit_region_similarity(source, target, (1.0, np.eye(3), np.zeros(3)))
    assert np.trace(rotation) > 2.9
    assert scale == pytest.approx(0.6, abs=0.05)
    np.testing.assert_allclose(scale * source @ rotation.T + translation, target, atol=0.02)


@pytest.mark.parametrize("mode", ["global", "segment"])
def test_alignment_recovers_a_known_similarity(tmp_path, mode):
    mesh = load_quinn_mesh(_build_fixture(tmp_path, _unit_weights()))
    soma = _write_soma_resources(tmp_path / "soma")
    bone_index, _ = build_bone_index(mesh, soma)
    vertices, normals, report = align_mesh_to_soma_bind(mesh, soma, bone_index, mode=mode)
    for name, entry in report["joint_alignment"].items():
        assert entry["error_cm"] < 0.01, (name, entry["error_cm"])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert len(vertices) == len(mesh.positions)


def test_build_quinn_assets_writes_a_loadable_resource_directory(tmp_path):
    glb = _build_fixture(tmp_path, _unit_weights())
    resources = tmp_path / "soma"
    _write_soma_resources(resources)
    output = tmp_path / "out"
    report = build_quinn_assets(glb, resources, output)

    mesh, bind = read_soma_bin(output / "SOMA.bin")
    assert bind.names == SOMA_BONES
    assert (output / "SOMA_bind.bvh").read_bytes() == (resources / "SOMA_bind.bvh").read_bytes()
    assert (output / "pbrLighting.fs").exists()
    assert report["validation"]["written_file"]["round_trip"] is True
    assert report["validation"]["viewer_mesh"]["weighted_bind_rotations_unit"] is True
    assert (output / "SOMA.bin").stat().st_size == soma_bin_layout(3, 1, len(SOMA_BONES))["bind_pose"][1]
    assert mesh["bone_ids"].max() < len(bind.names)
    np.testing.assert_allclose(mesh["bone_weights"].sum(axis=1), 1.0, atol=1e-6)
    assert json.loads((output / "conversion_report.json").read_text())["alignment"]["mode"] == "segment"


def test_every_alignment_joint_has_a_mapping_rule():
    for name in _ALIGN_JOINTS:
        assert name in QUINN_TO_SOMA, name
