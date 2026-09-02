import argparse
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cffi
import numpy as np
import torch
from pyray import BoneInfo, Camera3D, Color, Matrix, Mesh, Model, Rectangle, Transform, Vector2, Vector3, Vector4, take_screenshot
from raylib import *
from raylib.defines import *

from stylized_motion.anim.features import deserialize_motion_feature_stats, reconstruct_motion_state_from_features
from stylized_motion.anim import quat
from stylized_motion.anim.environment import IBLResources
from stylized_motion.anim.materials import Material, load_material_texture
from stylized_motion.anim.render_targets import RenderTargets
from stylized_motion.anim.scene import DirectionalLight, RenderObject, Scene
from stylized_motion.data import open_feature_store
from stylized_motion.util.paths import RESOURCE_DIR


ffi = cffi.FFI()


DEBUG_MODES = (
    "final",
    "base_color",
    "metallic",
    "roughness",
    "normal",
    "depth",
    "ao",
    "shadow",
    "diffuse",
    "specular",
    "ibl",
    "hdr",
)

# debugMode consumed by pbrLighting.fs. Only shadow/diffuse/specular/ibl exist
# inside the lighting pass; every other mode renders the final image and the
# debug display shader reads the GBuffer/SSAO attachments directly.
LIGHTING_DEBUG_MODES = {
    "final": 0,
    "base_color": 0,
    "metallic": 0,
    "roughness": 0,
    "normal": 0,
    "depth": 0,
    "ao": 0,
    "shadow": 1,
    "diffuse": 2,
    "specular": 3,
    "ibl": 4,
    "hdr": 0,
}


# Lighting rig inherited from GenoViewPython, frozen so `--shading legacy`
# keeps rendering exactly the way it always has.
LEGACY_LIGHT_RIG = {
    "light_dir": (0.35, -1.0, -0.35),
    "sun_color": (253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0),
    "sky_color": (174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0),
    "sun_strength": 0.25,
    "sky_strength": 0.15,
    "ground_strength": 0.1,
    "ambient_strength": 1.0,
    "ground_albedo": (190, 190, 190),
}

# PBR rig, retuned for the physically-based path. Targets: direct:ambient
# roughly 3.5:1 on white albedo (sun-dominant, outdoor), a shadow side around
# a fifth of the lit side in linear terms so form stays readable, a warm sun
# against a cool sky for temperature contrast, and a concrete-like ground
# albedo that does not outshine the character.
PBR_LIGHT_RIG = {
    "light_dir": (0.45, -0.8, -0.35),
    "sun_color": (1.0, 240.0 / 255.0, 214.0 / 255.0),
    "sky_color": (150.0 / 255.0, 180.0 / 255.0, 220.0 / 255.0),
    "sun_strength": 0.55,
    "sky_strength": 0.35,
    "ground_strength": 0.25,
    "ambient_strength": 0.15,
    "ground_albedo": (160, 160, 160),
}


@dataclass(frozen=True)
class RigSpec:
    """Character assets and skeleton conventions used by the viewer.

    ``sim_position_joint``/``sim_rotation_joint`` define the simulation-root
    convention shared with the preprocess pipeline; ``unit_scale`` converts
    BVH offsets to the mesh's unit (Geno.bin and SOMA meshes are in meters).
    ``skeleton_root_joint`` is where the skeleton overlay starts: joints above
    it (the simulation root and static rig nodes such as SOMA's origin-pinned
    ``Root``) are excluded so no overlay line reaches the world origin.
    """

    model_filename: str
    bind_bvh_filename: str
    sim_position_joint: str
    sim_rotation_joint: str
    unit_scale: float
    window_title: bytes
    skeleton_root_joint: str = "Hips"


GENO_RIG = RigSpec(
    model_filename="Geno.bin",
    bind_bvh_filename="Geno_bind.bvh",
    sim_position_joint="Spine2",
    sim_rotation_joint="Hips",
    unit_scale=0.01,
    window_title=b"GenoView",
    skeleton_root_joint="Hips",
)


class Camera:
    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()

    def update(self, target, azimuth_delta, altitude_delta, offset_delta_x, offset_delta_y, mouse_wheel, dt):
        self.azimuth = self.azimuth + 1.0 * dt * -azimuth_delta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitude_delta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance + 20.0 * dt * -mouse_wheel, 0.1, 100.0)

        rotation_azimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotation_azimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))
        rotation_altitude = QuaternionFromAxisAngle(axis, self.altitude)

        local_offset = Vector3(dt * offset_delta_x, dt * -offset_delta_y, 0.0)
        local_offset = Vector3RotateByQuaternion(local_offset, rotation_azimuth)
        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(local_offset, rotation_altitude))

        camera_target = Vector3Add(self.offset, target)
        eye = Vector3Add(camera_target, Vector3RotateByQuaternion(position, rotation_altitude))
        self.cam3d.target = camera_target
        self.cam3d.position = eye


class ShadowLight:
    def __init__(self):
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.width = 0.0
        self.height = 0.0
        self.near = 0.0
        self.far = 1.0


class PlaybackController:
    def __init__(self, frame_count, frame_time, speeds=(0.25, 0.5, 1.0, 1.5, 2.0), default_speed_index=2, playing=True):
        self.frame_count = max(1, int(frame_count))
        self.frame_time = float(frame_time)
        self.speeds = list(speeds)
        self.speed_index = int(np.clip(default_speed_index, 0, len(self.speeds) - 1))
        self.playing = bool(playing)
        self.frame = 0.0
        self.scrubbing = False

    @staticmethod
    def _key_pressed_or_repeat(key):
        if IsKeyPressed(key):
            return True
        repeat_fn = globals().get("IsKeyPressedRepeat")
        return bool(callable(repeat_fn) and repeat_fn(key))

    @property
    def current_frame(self):
        return int(self.frame) % self.frame_count

    @property
    def current_speed(self):
        return self.speeds[self.speed_index]

    def _clamp_frame(self, frame):
        return min(max(int(frame), 0), self.frame_count - 1)

    def set_current_frame(self, frame):
        self.frame = float(self._clamp_frame(frame))

    def toggle_playing(self):
        self.playing = not self.playing

    def step_frames(self, delta):
        self.playing = False
        self.set_current_frame(self.current_frame + int(delta))

    def nudge_speed(self, delta):
        self.speed_index = int(np.clip(self.speed_index + int(delta), 0, len(self.speeds) - 1))

    def handle_shortcuts(self):
        shift_down = IsKeyDown(KEY_LEFT_SHIFT) or IsKeyDown(KEY_RIGHT_SHIFT)
        step_size = 10 if shift_down else 1

        if IsKeyPressed(KEY_SPACE):
            self.toggle_playing()
        if self._key_pressed_or_repeat(KEY_LEFT):
            self.step_frames(-step_size)
        if self._key_pressed_or_repeat(KEY_RIGHT):
            self.step_frames(step_size)
        if self._key_pressed_or_repeat(KEY_UP):
            self.nudge_speed(1)
        if self._key_pressed_or_repeat(KEY_DOWN):
            self.nudge_speed(-1)
        if IsKeyPressed(KEY_HOME):
            self.playing = False
            self.set_current_frame(0)
        if IsKeyPressed(KEY_END):
            self.playing = False
            self.set_current_frame(self.frame_count - 1)

    def update(self, dt):
        if self.playing and not self.scrubbing:
            self.frame = (self.frame + self.current_speed * dt / self.frame_time) % self.frame_count
        return self.current_frame

    def timeline_rect(self, screen_width, screen_height):
        margin = 24
        return Rectangle(margin, screen_height - 34, screen_width - 2 * margin, 10)

    def _frame_from_mouse_x(self, rect, mouse_x):
        alpha = (float(mouse_x) - float(rect.x)) / max(float(rect.width), 1.0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return int(round(alpha * float(self.frame_count - 1)))

    def handle_scrub(self, screen_width, screen_height):
        rect = self.timeline_rect(screen_width, screen_height)
        mouse = GetMousePosition()
        hovered = CheckCollisionPointRec(mouse, rect)
        left_button = globals().get("MOUSE_BUTTON_LEFT", 0)

        if IsMouseButtonPressed(left_button) and hovered:
            self.scrubbing = True
            self.playing = False
            self.set_current_frame(self._frame_from_mouse_x(rect, mouse.x))
        elif self.scrubbing and IsMouseButtonDown(left_button):
            self.set_current_frame(self._frame_from_mouse_x(rect, mouse.x))
        elif self.scrubbing and IsMouseButtonReleased(left_button):
            self.set_current_frame(self._frame_from_mouse_x(rect, mouse.x))
            self.scrubbing = False

    def draw_ui(self, screen_width, screen_height, label):
        rect = self.timeline_rect(screen_width, screen_height)
        progress = self.current_frame / max(self.frame_count - 1, 1)
        fill_width = int(round(float(rect.width) * progress))
        knob_x = int(round(float(rect.x) + float(rect.width) * progress))
        readout = f"{label} | {self.current_frame + 1}/{self.frame_count} | {self.current_speed:.2f}x"

        DrawRectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), Color(30, 30, 30, 95))
        DrawRectangle(int(rect.x), int(rect.y), fill_width, int(rect.height), Color(45, 132, 255, 220))
        DrawRectangleLines(int(rect.x), int(rect.y), int(rect.width), int(rect.height), Color(20, 20, 20, 180))
        DrawCircle(knob_x, int(rect.y + rect.height * 0.5), 7.0, Color(20, 82, 180, 245))
        DrawText(readout.encode(), int(rect.x), int(rect.y - 24), 18, BLACK)


def file_read(out, size, f):
    ffi.memmove(out, f.read(size), size)


def load_geno_model(filename: Path):
    material_size = ffi.sizeof(Mesh())
    mesh_size = ffi.sizeof(Mesh())
    int_size = ffi.sizeof("int")
    float_size = ffi.sizeof("float")
    boneinfo_size = ffi.sizeof(BoneInfo())
    transform_size = ffi.sizeof(Transform())
    matrix_size = ffi.sizeof(Matrix())
    uchar_size = ffi.sizeof("unsigned char")
    ushort_size = ffi.sizeof("unsigned short")

    model = Model()
    model.transform = MatrixIdentity()

    with open(filename, "rb") as f:
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * material_size)
        model.materials[0] = LoadMaterialDefault()
        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * int_size)
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * mesh_size)
        model.meshes[0].vertexCount = struct.unpack("I", f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack("I", f.read(4))[0]
        model.boneCount = struct.unpack("I", f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * float_size)
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * float_size)
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * float_size)
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * uchar_size)
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * float_size)
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ushort_size)
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * float_size)
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * float_size)
        model.bones = MemAlloc(model.boneCount * boneinfo_size)
        model.bindPose = MemAlloc(model.boneCount * transform_size)

        file_read(model.meshes[0].vertices, float_size * model.meshes[0].vertexCount * 3, f)
        file_read(model.meshes[0].texcoords, float_size * model.meshes[0].vertexCount * 2, f)
        file_read(model.meshes[0].normals, float_size * model.meshes[0].vertexCount * 3, f)
        file_read(model.meshes[0].boneIds, uchar_size * model.meshes[0].vertexCount * 4, f)
        file_read(model.meshes[0].boneWeights, float_size * model.meshes[0].vertexCount * 4, f)
        file_read(model.meshes[0].indices, ushort_size * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, float_size * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, float_size * model.meshes[0].vertexCount * 3)
        file_read(model.bones, boneinfo_size * model.boneCount, f)
        file_read(model.bindPose, transform_size * model.boneCount, f)

        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * matrix_size)
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()

    GenMeshTangents(ffi.addressof(model.meshes[0]))
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    return model


def get_model_bind_pose_as_numpy_arrays(model):
    bind_pos = np.zeros([model.boneCount, 3], dtype=np.float32)
    bind_rot = np.zeros([model.boneCount, 4], dtype=np.float32)
    for bone_id in range(model.boneCount):
        bind_transform = model.bindPose[bone_id]
        bind_pos[bone_id] = (bind_transform.translation.x, bind_transform.translation.y, bind_transform.translation.z)
        bind_rot[bone_id] = (
            bind_transform.rotation.w,
            bind_transform.rotation.x,
            bind_transform.rotation.y,
            bind_transform.rotation.z,
        )
    return bind_pos, bind_rot


def update_model_pose_from_numpy_arrays(model, bind_pos, bind_rot, anim_pos, anim_rot):
    mesh_pos = quat.mul_vec(anim_rot, quat.inv_mul_vec(bind_rot, -bind_pos)) + anim_pos
    mesh_rot = quat.mul_inv(anim_rot, bind_rot)
    mat_array = np.frombuffer(
        ffi.buffer(model.meshes[0].boneMatrices, model.boneCount * 4 * 4 * 4),
        dtype=np.float32,
    ).reshape([model.boneCount, 4, 4])
    mat_array.fill(0.0)
    mat_array[:, 3, 3] = 1.0
    mat_array[:, :3, :3] = quat.to_xform(mesh_rot)
    mat_array[:, :3, 3] = mesh_pos


def build_simulation_root_skeleton_from_bind(bind_bvh_path: Path, rig: RigSpec = GENO_RIG):
    bind_data = load_bvh_data(bind_bvh_path)
    positions = bind_data["positions"].astype(np.float32) * rig.unit_scale
    rotations = quat.unroll(quat.from_euler(np.radians(bind_data["rotations"]), order=bind_data["order"])).astype(np.float32)

    global_rotations, global_positions = quat.fk(rotations, positions, bind_data["parents"])
    sim_position_joint = bind_data["names"].index(rig.sim_position_joint)
    sim_rotation_joint = bind_data["names"].index(rig.sim_rotation_joint)

    sim_position = np.array([1.0, 0.0, 1.0], dtype=np.float32) * global_positions[:, sim_position_joint : sim_position_joint + 1]
    sim_direction = np.array([1.0, 0.0, 1.0], dtype=np.float32) * quat.mul_vec(
        global_rotations[:, sim_rotation_joint : sim_rotation_joint + 1], np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    sim_direction = sim_direction / np.maximum(
        np.sqrt(np.sum(np.square(sim_direction), axis=-1))[..., np.newaxis], 1e-8
    )
    sim_rotation = quat.normalize(quat.between(np.array([0.0, 0.0, 1.0], dtype=np.float32), sim_direction))

    positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
    rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])

    positions = np.concatenate([sim_position, positions], axis=1)
    rotations = np.concatenate([sim_rotation, rotations], axis=1)
    parents = np.concatenate([[-1], bind_data["parents"] + 1]).astype(np.int32)
    names = ["Simulation"] + bind_data["names"]
    return names, parents, positions[0].astype(np.float32), rotations[0].astype(np.float32)


def load_bvh_data(path: Path):
    from stylized_motion.anim import bvh

    return bvh.load(str(path))


def build_database_from_bvh(bvh_path: Path, range_name: str | None = None, rig: RigSpec = GENO_RIG) -> dict[str, np.ndarray]:
    """Convert one BVH clip into the database contract consumed by GenoView."""
    data = load_bvh_data(Path(bvh_path))
    positions = data["positions"].astype(np.float32) * rig.unit_scale
    rotations = quat.unroll(
        quat.from_euler(np.radians(data["rotations"]), order=data["order"])
    ).astype(np.float32)
    names = list(data["names"])
    parents = np.asarray(data["parents"], dtype=np.int32)

    required = (rig.sim_position_joint, rig.sim_rotation_joint)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            f"BVH {bvh_path} is incompatible with the {rig.window_title.decode()} skeleton; missing joints: {missing}"
        )

    # Match the project's simulation-root convention without importing the
    # dataset-building pipeline (which would also pull in its CLI machinery).
    global_rotations, global_positions = quat.fk(rotations, positions, parents)
    sim_position = np.array([1.0, 0.0, 1.0], dtype=np.float32) * global_positions[
        :, names.index(rig.sim_position_joint) : names.index(rig.sim_position_joint) + 1
    ]
    sim_direction = quat.mul_vec(
        global_rotations[:, names.index(rig.sim_rotation_joint) : names.index(rig.sim_rotation_joint) + 1],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    sim_direction = sim_direction / np.maximum(
        np.linalg.norm(sim_direction, axis=-1, keepdims=True), 1e-8
    )
    sim_rotation = quat.normalize(
        quat.between(np.array([0.0, 0.0, 1.0], dtype=np.float32), sim_direction)
    )
    positions[:, :1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, :1] - sim_position)
    rotations[:, :1] = quat.mul(quat.inv(sim_rotation), rotations[:, :1])
    positions = np.concatenate([sim_position, positions], axis=1)
    rotations = np.concatenate([sim_rotation, rotations], axis=1)
    parents = np.concatenate([np.asarray([-1], dtype=np.int32), parents + 1])
    names = ["Simulation", *names]

    nframes = int(len(positions))
    if range_name is None:
        range_name = Path(bvh_path).stem
    return {
        "positions": positions.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "velocities": np.zeros_like(positions, dtype=np.float32),
        "angular_velocities": np.zeros_like(positions, dtype=np.float32),
        "contacts": np.zeros((nframes, 2), dtype=np.uint8),
        "parents": parents,
        "names": np.asarray(names, dtype=object),
        "range_starts": np.asarray([0], dtype=np.int32),
        "range_stops": np.asarray([nframes], dtype=np.int32),
        "range_names": np.asarray([range_name], dtype=object),
        "range_mirror": np.asarray([False], dtype=bool),
        "joint_subset": np.asarray("full", dtype=object),
        "frame_time": np.asarray(float(data["frametime"] or (1.0 / 60.0)), dtype=np.float32),
    }


def load_feature_array(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        features = np.load(path)
    else:
        data = np.load(path, allow_pickle=True)
        if key in data.files:
            features = data[key]
        elif len(data.files) == 1:
            features = data[data.files[0]]
        else:
            raise KeyError(f"Could not find key {key!r} in {path}. Available keys: {list(data.files)}")

    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(f"Expected feature shape [T, D] or [1, T, D], got {features.shape}")
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"Expected feature shape [T, D], got {features.shape}")
    return features


def load_feature_stats(stats_source: Path):
    if stats_source.is_dir():
        store = open_feature_store(stats_source)
        try:
            metadata = {
                "names": list(store.names),
                "parents": store.parents.astype(np.int32, copy=True),
                "joint_subset": store.joint_subset,
            }
            return store.stats, metadata
        finally:
            store.close()
    if stats_source.suffix != ".pt":
        raise ValueError("stats_source must be a schema-v3 feature database or checkpoint .pt")
    checkpoint = torch.load(stats_source, map_location="cpu", weights_only=False)
    if "feature_stats" not in checkpoint:
        raise KeyError(f"Checkpoint {stats_source} does not contain feature_stats")
    payload = dict(checkpoint["feature_stats"])
    if "dist" not in payload:
        payload["dist"] = np.ones_like(np.asarray(payload["offset"], dtype=np.float32))

    stats, metadata = deserialize_motion_feature_stats(payload)
    for key in ("names", "parents", "joint_subset"):
        if key not in metadata:
            raise KeyError(f"Stats source {stats_source} does not contain {key}")
    return stats, metadata


def build_database_from_features(
    features_path: Path,
    stats_source: Path,
    feature_key: str,
    normalized: bool,
    range_name: str,
    root_position0: list[float] | None,
    root_rotation0: list[float] | None,
) -> dict[str, np.ndarray]:
    features = load_feature_array(features_path, feature_key)
    return build_database_from_feature_array(
        features=features,
        stats_source=stats_source,
        normalized=normalized,
        range_name=range_name,
        root_position0=root_position0,
        root_rotation0=root_rotation0,
    )


def build_database_from_feature_array(
    features: np.ndarray,
    stats_source: Path,
    normalized: bool,
    range_name: str,
    root_position0: list[float] | np.ndarray | None = None,
    root_rotation0: list[float] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(f"Expected feature shape [T, D] or [1, T, D], got {features.shape}")
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"Expected feature shape [T, D], got {features.shape}")

    stats, metadata = load_feature_stats(stats_source)
    if features.shape[1] != stats.offset.shape[0]:
        raise ValueError(f"Feature dim {features.shape[1]} does not match stats dim {stats.offset.shape[0]}")

    state = reconstruct_motion_state_from_features(
        x=features,
        stats=stats,
        parents=np.asarray(metadata["parents"], dtype=np.int32),
        normalized=normalized,
        root_position0=None if root_position0 is None else np.asarray(root_position0, dtype=np.float32),
        root_rotation0=None if root_rotation0 is None else np.asarray(root_rotation0, dtype=np.float32),
    )
    nframes = int(len(state.local_positions))
    return {
        "positions": state.local_positions.astype(np.float32),
        "rotations": state.local_rotations.astype(np.float32),
        "velocities": state.local_velocities.astype(np.float32),
        "angular_velocities": state.local_angular_velocities.astype(np.float32),
        "contacts": np.asarray(state.contacts > 0.5, dtype=np.uint8),
        "parents": np.asarray(metadata["parents"], dtype=np.int32),
        "names": np.asarray(metadata["names"], dtype=object),
        "range_starts": np.asarray([0], dtype=np.int32),
        "range_stops": np.asarray([nframes], dtype=np.int32),
        "range_names": np.asarray([range_name], dtype=object),
        "range_mirror": np.asarray([False], dtype=bool),
        "joint_subset": np.asarray(str(metadata["joint_subset"]), dtype=object),
    }


def load_database_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def draw_trajectory(root_pos, root_rot, tpos, tdir):
    if tpos is None or tdir is None:
        return
    world_pos = quat.mul_vec(root_rot[None], tpos) + root_pos[None]
    world_dir = quat.mul_vec(root_rot[None], tdir)
    for i in range(len(world_pos)):
        point = world_pos[i]
        DrawSphere(Vector3(point[0], point[1], point[2]), 0.035, RED)
        if i > 0:
            prev = world_pos[i - 1]
            DrawLine3D(Vector3(prev[0], prev[1], prev[2]), Vector3(point[0], point[1], point[2]), MAROON)
        tip = point + 0.2 * world_dir[i]
        DrawLine3D(Vector3(point[0], point[1], point[2]), Vector3(tip[0], tip[1], tip[2]), ORANGE)


def skeleton_axis_endpoints(position, rotation, scale):
    """World endpoints of a joint's local X/Y/Z axes for the skeleton overlay.

    Project quaternions are (w, x, y, z) while raylib's ``QuaternionToMatrix``
    expects (x, y, z, w), so the components are swizzled explicitly before the
    call. This corrects the component-order bug in GenoViewPython's debug
    draw, which passes the (w, x, y, z) quaternion through unchanged and
    renders every axis frame from mismatched quaternion columns.
    """
    quat_wxyz = np.asarray(rotation, dtype=np.float32)
    rot_matrix = QuaternionToMatrix(
        Vector4(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]))
    )
    origin = np.asarray(position, dtype=np.float32)
    columns = (
        (rot_matrix.m0, rot_matrix.m1, rot_matrix.m2),
        (rot_matrix.m4, rot_matrix.m5, rot_matrix.m6),
        (rot_matrix.m8, rot_matrix.m9, rot_matrix.m10),
    )
    return [origin + scale * np.asarray(column, dtype=np.float32) for column in columns]


def draw_transform_axes(position, rotation, scale):
    origin = Vector3(float(position[0]), float(position[1]), float(position[2]))
    for endpoint, color in zip(skeleton_axis_endpoints(position, rotation, scale), (RED, GREEN, BLUE)):
        DrawLine3D(origin, Vector3(float(endpoint[0]), float(endpoint[1]), float(endpoint[2])), color)


def draw_skeleton(positions, rotations, parents, color):
    """Draw joints, parent links, and per-joint frames, mirroring GenoViewPython's Debug Draw."""
    for joint in range(len(positions)):
        pos = positions[joint]
        point = Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
        DrawSphereWires(point, 0.01, 4, 6, color)
        draw_transform_axes(pos, rotations[joint], 0.1)
        parent = int(parents[joint])
        if parent != -1:
            parent_pos = positions[parent]
            DrawLine3D(
                point,
                Vector3(float(parent_pos[0]), float(parent_pos[1]), float(parent_pos[2])),
                color,
            )


def skeleton_overlay_pose(global_pos, global_rot, parents, root_index):
    """Character-skeleton slice of a full simulation-root pose for the overlay.

    The overlay draws the subtree attached to the character root joint.
    Joints above ``root_index`` — the virtual simulation root plus static rig
    nodes like SOMA's origin-pinned ``Root`` — are excluded, and so is any
    joint hanging off those excluded nodes, so no overlay joint or bone line
    ever touches the world origin while the character walks away from it.
    """
    parents = np.asarray(parents, dtype=np.int32)
    kept = []
    for joint in range(root_index, len(parents)):
        ancestor = joint
        while ancestor > root_index:
            ancestor = int(parents[ancestor])
        if ancestor == root_index:
            kept.append(joint)
    remap = {old: new for new, old in enumerate(kept)}
    overlay_parents = np.asarray(
        [-1 if int(parents[old]) == -1 else remap.get(int(parents[old]), -1) for old in kept],
        dtype=np.int32,
    )
    positions = np.asarray(global_pos)[kept]
    rotations = np.asarray(global_rot)[kept]
    return positions, rotations, overlay_parents


class GenoView:
    def __init__(
        self,
        database: dict[str, np.ndarray],
        trajectory_path: Path | None,
        resources_root: Path,
        fps: int | None = None,
        compare_database: dict[str, np.ndarray] | None = None,
        left_label: str = "Source",
        right_label: str = "Recon",
        compare_spacing: float = 2.0,
        shading: str = "pbr",
        metallic: float = 0.0,
        roughness: float = 0.58,
        exposure: float = 0.9,
        ssao_intensity: float = 0.15,
        ibl_strength: float = 0.35,
        ibl_enabled: bool = True,
        shadow_resolution: int = 2048,
        output_video: Path | None = None,
        draw_skeleton: bool = False,
        debug_view: str = "final",
        normal_map: Path | None = None,
        metallic_roughness_map: Path | None = None,
        sun_strength: float | None = None,
        rig: RigSpec = GENO_RIG,
    ):
        if debug_view not in DEBUG_MODES:
            raise ValueError(f"Unknown debug view {debug_view!r}; expected one of {DEBUG_MODES}")
        self.rig = rig
        self.skeleton_enabled = bool(draw_skeleton)
        self.skeleton_color = GRAY
        self.debug_view = debug_view
        self.normal_map_path = normal_map
        self.metallic_roughness_map_path = metallic_roughness_map
        self.shading = shading
        self.metallic = float(metallic)
        self.roughness = float(roughness)
        self.exposure = float(exposure)
        self.ssao_intensity = float(ssao_intensity)
        self.ibl_strength = float(ibl_strength)
        self.ibl_enabled = bool(ibl_enabled)
        self.shadow_resolution = int(shadow_resolution)
        self.output_video = Path(output_video).resolve() if output_video is not None else None
        self.database = database
        self.positions = self.database["positions"].astype(np.float32)
        self.rotations = self.database["rotations"].astype(np.float32)
        self.parents = self.database["parents"].astype(np.int32)
        self.names = self.database["names"]
        self.joint_subset = self.database["joint_subset"].item() if "joint_subset" in self.database else "full"
        self.range_names = self.database["range_names"]
        self.range_starts = self.database["range_starts"].astype(np.int32)
        self.range_stops = self.database["range_stops"].astype(np.int32)
        frame_time = float(np.asarray(self.database.get("frame_time", 1.0 / 60.0)).item())
        self.fps = int(fps) if fps is not None else int(round(1.0 / frame_time))
        self.resources_root = resources_root
        self.compare_database = compare_database
        self.compare_mode = compare_database is not None
        self.left_label = left_label
        self.right_label = right_label
        self.compare_spacing = float(compare_spacing)
        self.left_model_offset = Vector3(-0.5 * self.compare_spacing, 0.0, 0.0) if self.compare_mode else Vector3(0.0, 0.0, 0.0)
        self.right_model_offset = Vector3(0.5 * self.compare_spacing, 0.0, 0.0)

        if self.compare_mode:
            self.compare_positions = self.compare_database["positions"].astype(np.float32)
            self.compare_rotations = self.compare_database["rotations"].astype(np.float32)
            self.compare_parents = self.compare_database["parents"].astype(np.int32)
            self.compare_names = self.compare_database["names"]
            self.compare_joint_subset = (
                self.compare_database["joint_subset"].item() if "joint_subset" in self.compare_database else "full"
            )
            self.compare_range_names = self.compare_database["range_names"]
            if len(self.compare_positions) != len(self.positions):
                raise ValueError(f"Compare databases must have the same frame count, got {len(self.positions)} and {len(self.compare_positions)}")
            if self.compare_joint_subset != self.joint_subset:
                raise ValueError(f"Compare joint_subset mismatch: {self.joint_subset} vs {self.compare_joint_subset}")
            if self.compare_parents.shape != self.parents.shape or not np.array_equal(self.compare_parents, self.parents):
                raise ValueError("Compare database parents do not match")
            if self.compare_names.shape != self.names.shape or any(str(a) != str(b) for a, b in zip(self.compare_names, self.names)):
                raise ValueError("Compare database joint names do not match")
        else:
            self.compare_positions = None
            self.compare_rotations = None
            self.compare_parents = None
            self.compare_names = None
            self.compare_joint_subset = None
            self.compare_range_names = None

        self.trajectory = None
        if trajectory_path is not None:
            if self.compare_mode:
                raise ValueError("Trajectory overlay is not supported in compare mode yet")
            self.trajectory = np.load(trajectory_path, allow_pickle=True)
            self.indices = self.trajectory["indices"].astype(np.int32)
            self.tpos = self.trajectory["Tpos"].astype(np.float32)
            self.tdir = self.trajectory["Tdir"].astype(np.float32)
            self.sample_range_names = self.trajectory["sample_range_names"]
            self.sample_mirror = self.trajectory["sample_mirror"].astype(bool)
        else:
            self.indices = None
            self.tpos = None
            self.tdir = None
            self.sample_range_names = None
            self.sample_mirror = None

        playback_count = len(self.indices) if self.indices is not None else len(self.positions)
        self.playback = PlaybackController(playback_count, 1.0 / float(self.fps), playing=True)
        self.sample_index = 0
        self.frame_index = int(self.indices[0]) if self.indices is not None else 0

        self.camera = Camera()
        # Legacy keeps its frozen GenoViewPython rig; PBR uses the retuned one.
        self.light_rig = PBR_LIGHT_RIG if shading == "pbr" else LEGACY_LIGHT_RIG
        self.light_dir = Vector3Normalize(Vector3(*self.light_rig["light_dir"]))
        self.sun_color = Vector3(*self.light_rig["sun_color"])
        self.sky_color = Vector3(*self.light_rig["sky_color"])
        self.sun_strength = float(sun_strength) if sun_strength is not None else float(self.light_rig["sun_strength"])
        self.default_material = Material(metallic=self.metallic, roughness=self.roughness)
        self.scene = Scene(
            directional_light=DirectionalLight(
                direction=self.light_dir,
                color=self.sun_color,
                intensity=self.sun_strength,
            )
        )
        self.shadow_light = ShadowLight()
        self.shadow_light.target = Vector3Zero()
        self.shadow_light.position = Vector3Scale(self.light_dir, -5.0)
        self.shadow_light.up = Vector3(0.0, 1.0, 0.0)
        self.shadow_light.width = 5.0
        self.shadow_light.height = 5.0
        self.shadow_light.near = 0.01
        self.shadow_light.far = 10.0

        self.shadow_map = None
        self.gbuffer = None
        self.lighted = None
        self.tonemapped = None
        self.ssao_front = None
        self.ssao_back = None
        self.render_targets = None
        self.ibl = None
        self.ground_model = None
        self.geno_model = None
        self.compare_model = None
        self.bind_pos = None
        self.bind_rot = None
        self.compare_bind_pos = None
        self.compare_bind_rot = None
        self.shaders = {}
        self.full_names = None
        self.full_parents = None
        self.skeleton_root_index = 1
        self.full_bind_local_positions = None
        self.full_bind_local_rotations = None
        self.database_name_to_index = {str(name): idx for idx, name in enumerate(self.names.tolist())}
        self.full_name_to_index = None
        self.use_pruned_reconstruction = False
        self.shader_locs = {}
        self.ground_position = Vector3(0.0, -0.01, 0.0)

        self.light_clip_near_ptr = ffi.new("float*")
        self.light_clip_far_ptr = ffi.new("float*")
        self.cam_clip_near_ptr = ffi.new("float*")
        self.cam_clip_far_ptr = ffi.new("float*")
        self.specularity_ptr = ffi.new("float*")
        self.glossiness_ptr = ffi.new("float*")
        self.metallic_ptr = ffi.new("float*")
        self.roughness_ptr = ffi.new("float*")
        self.sun_strength_ptr = ffi.new("float*")
        self.sky_strength_ptr = ffi.new("float*")
        self.ground_strength_ptr = ffi.new("float*")
        self.ambient_strength_ptr = ffi.new("float*")
        self.material_ao_ptr = ffi.new("float*")
        self.ibl_strength_ptr = ffi.new("float*")
        self.use_ibl_ptr = ffi.new("int*")
        self.exposure_ptr = ffi.new("float*")
        self.ssao_intensity_ptr = ffi.new("float*")
        self.shadow_texture_slot_ptr = ffi.new("int*")
        self.environment_texture_slot_ptr = ffi.new("int*")
        self.irradiance_texture_slot_ptr = ffi.new("int*")
        self.prefilter_texture_slot_ptr = ffi.new("int*")
        self.brdf_lut_texture_slot_ptr = ffi.new("int*")
        self.prefilter_max_lod_ptr = ffi.new("float*")
        self.material_base_color_ptr = ffi.new("float[4]")
        self.ground_pattern_ptr = ffi.new("int*")
        self.use_base_color_map_ptr = ffi.new("int*")
        self.use_metallic_roughness_map_ptr = ffi.new("int*")
        self.use_normal_map_ptr = ffi.new("int*")
        self.debug_mode_ptr = ffi.new("int*")
        self.shadow_texture_slot_ptr[0] = 10
        self.environment_texture_slot_ptr[0] = 11
        self.irradiance_texture_slot_ptr[0] = 12
        self.prefilter_texture_slot_ptr[0] = 13
        self.brdf_lut_texture_slot_ptr[0] = 14
        # Dedicated slots beyond the lighting pass's 10-14: render-target GL
        # texture names collide with SetShaderValueTexture's id-keyed samplers.
        self.material_ao_texture_slot_ptr = ffi.new("int*")
        self.material_ao_texture_slot_ptr[0] = 15
        self.ssao_texture_slot_ptr = ffi.new("int*")
        self.ssao_texture_slot_ptr[0] = 21
        self.debug_gbuffer_color_slot_ptr = ffi.new("int*")
        self.debug_gbuffer_normal_slot_ptr = ffi.new("int*")
        self.debug_gbuffer_depth_slot_ptr = ffi.new("int*")
        self.debug_ssao_slot_ptr = ffi.new("int*")
        self.debug_lighted_slot_ptr = ffi.new("int*")
        self.debug_gbuffer_color_slot_ptr[0] = 16
        self.debug_gbuffer_normal_slot_ptr[0] = 17
        self.debug_gbuffer_depth_slot_ptr[0] = 18
        self.debug_ssao_slot_ptr[0] = 19
        self.debug_lighted_slot_ptr[0] = 20
        self.prefilter_max_lod_ptr[0] = 5.0
        self.ground_pattern_ptr[0] = 0
        self.specularity_ptr[0] = 0.5
        self.glossiness_ptr[0] = 10.0
        self.metallic_ptr[0] = self.metallic
        self.roughness_ptr[0] = self.roughness
        self.sun_strength_ptr[0] = self.sun_strength
        self.sky_strength_ptr[0] = float(self.light_rig["sky_strength"])
        self.ground_strength_ptr[0] = float(self.light_rig["ground_strength"])
        self.ambient_strength_ptr[0] = float(self.light_rig["ambient_strength"])
        self.material_ao_ptr[0] = self.default_material.ao
        self.ibl_strength_ptr[0] = self.ibl_strength
        self.use_ibl_ptr[0] = int(self.ibl_enabled)
        self.exposure_ptr[0] = self.exposure
        self.ssao_intensity_ptr[0] = self.ssao_intensity
        for index, channel in enumerate(self.default_material.base_color):
            self.material_base_color_ptr[index] = channel
        self.use_base_color_map_ptr[0] = 0
        self.use_metallic_roughness_map_ptr[0] = 0
        self.use_normal_map_ptr[0] = 0
        self.blur_direction = Vector2(0.0, 0.0)
        self.blur_inv_texture_resolution = Vector2(0.0, 0.0)
        self.fxaa_inv_texture_resolution = Vector2(0.0, 0.0)

    def _res(self, name: str) -> bytes:
        return str((self.resources_root / name).resolve()).encode()

    def _initialize_rendering(self, screen_width: int, screen_height: int):
        gbuffer_fs = "pbr.fs" if self.shading == "pbr" else "basic.fs"
        lighting_fs = "pbrLighting.fs" if self.shading == "pbr" else "lighting.fs"
        self.shaders["basic"] = LoadShader(self._res("basic.vs"), self._res(gbuffer_fs))
        self.shaders["skinned_basic"] = LoadShader(self._res("skinnedBasic.vs"), self._res(gbuffer_fs))
        self.shaders["shadow"] = LoadShader(self._res("shadow.vs"), self._res("shadow.fs"))
        self.shaders["skinned_shadow"] = LoadShader(self._res("skinnedShadow.vs"), self._res("shadow.fs"))
        self.shaders["ssao"] = LoadShader(self._res("post.vs"), self._res("ssao.fs"))
        self.shaders["blur"] = LoadShader(self._res("post.vs"), self._res("blur.fs"))
        self.shaders["lighting"] = LoadShader(self._res("post.vs"), self._res(lighting_fs))
        if self.shading == "pbr":
            self.shaders["tonemap"] = LoadShader(self._res("post.vs"), self._res("tonemap.fs"))
            self.shaders["debug"] = LoadShader(self._res("post.vs"), self._res("debug.fs"))
        self.shaders["fxaa"] = LoadShader(self._res("post.vs"), self._res("fxaa.fs"))

        self.shader_locs["basic_specularity"] = GetShaderLocation(self.shaders["basic"], b"specularity")
        self.shader_locs["basic_glossiness"] = GetShaderLocation(self.shaders["basic"], b"glossiness")
        self.shader_locs["basic_cam_clip_near"] = GetShaderLocation(self.shaders["basic"], b"camClipNear")
        self.shader_locs["basic_cam_clip_far"] = GetShaderLocation(self.shaders["basic"], b"camClipFar")

        self.shader_locs["skinned_basic_specularity"] = GetShaderLocation(self.shaders["skinned_basic"], b"specularity")
        self.shader_locs["skinned_basic_glossiness"] = GetShaderLocation(self.shaders["skinned_basic"], b"glossiness")
        self.shader_locs["skinned_basic_cam_clip_near"] = GetShaderLocation(self.shaders["skinned_basic"], b"camClipNear")
        self.shader_locs["skinned_basic_cam_clip_far"] = GetShaderLocation(self.shaders["skinned_basic"], b"camClipFar")
        self.shader_locs["basic_metallic"] = GetShaderLocation(self.shaders["basic"], b"pbrMetallic")
        self.shader_locs["basic_roughness"] = GetShaderLocation(self.shaders["basic"], b"pbrRoughness")
        self.shader_locs["skinned_basic_metallic"] = GetShaderLocation(self.shaders["skinned_basic"], b"pbrMetallic")
        self.shader_locs["skinned_basic_roughness"] = GetShaderLocation(self.shaders["skinned_basic"], b"pbrRoughness")
        if self.shading == "pbr":
            for shader_key, prefix in (("basic", "basic"), ("skinned_basic", "skinned_basic")):
                self.shader_locs[f"{prefix}_material_base_color"] = GetShaderLocation(self.shaders[shader_key], b"materialBaseColor")
                self.shader_locs[f"{prefix}_ao"] = GetShaderLocation(self.shaders[shader_key], b"pbrAo")
                self.shader_locs[f"{prefix}_base_color_map"] = GetShaderLocation(self.shaders[shader_key], b"baseColorMap")
                self.shader_locs[f"{prefix}_metallic_roughness_map"] = GetShaderLocation(self.shaders[shader_key], b"metallicRoughnessMap")
                self.shader_locs[f"{prefix}_normal_map"] = GetShaderLocation(self.shaders[shader_key], b"normalMap")
                self.shader_locs[f"{prefix}_use_base_color_map"] = GetShaderLocation(self.shaders[shader_key], b"useBaseColorMap")
                self.shader_locs[f"{prefix}_use_metallic_roughness_map"] = GetShaderLocation(self.shaders[shader_key], b"useMetallicRoughnessMap")
                self.shader_locs[f"{prefix}_use_normal_map"] = GetShaderLocation(self.shaders[shader_key], b"useNormalMap")
                self.shader_locs[f"{prefix}_ground_pattern"] = GetShaderLocation(self.shaders[shader_key], b"pbrGroundPattern")

        self.shader_locs["shadow_light_clip_near"] = GetShaderLocation(self.shaders["shadow"], b"lightClipNear")
        self.shader_locs["shadow_light_clip_far"] = GetShaderLocation(self.shaders["shadow"], b"lightClipFar")
        self.shader_locs["skinned_shadow_light_clip_near"] = GetShaderLocation(self.shaders["skinned_shadow"], b"lightClipNear")
        self.shader_locs["skinned_shadow_light_clip_far"] = GetShaderLocation(self.shaders["skinned_shadow"], b"lightClipFar")

        self.shader_locs["ssao_gbuffer_normal"] = GetShaderLocation(self.shaders["ssao"], b"gbufferNormal")
        self.shader_locs["ssao_gbuffer_depth"] = GetShaderLocation(self.shaders["ssao"], b"gbufferDepth")
        self.shader_locs["ssao_cam_view"] = GetShaderLocation(self.shaders["ssao"], b"camView")
        self.shader_locs["ssao_cam_proj"] = GetShaderLocation(self.shaders["ssao"], b"camProj")
        self.shader_locs["ssao_cam_inv_proj"] = GetShaderLocation(self.shaders["ssao"], b"camInvProj")
        self.shader_locs["ssao_cam_clip_near"] = GetShaderLocation(self.shaders["ssao"], b"camClipNear")
        self.shader_locs["ssao_cam_clip_far"] = GetShaderLocation(self.shaders["ssao"], b"camClipFar")
        self.shader_locs["ssao_intensity"] = GetShaderLocation(self.shaders["ssao"], b"ssaoIntensity")

        self.shader_locs["blur_gbuffer_normal"] = GetShaderLocation(self.shaders["blur"], b"gbufferNormal")
        self.shader_locs["blur_gbuffer_depth"] = GetShaderLocation(self.shaders["blur"], b"gbufferDepth")
        self.shader_locs["blur_input_texture"] = GetShaderLocation(self.shaders["blur"], b"inputTexture")
        self.shader_locs["blur_cam_inv_proj"] = GetShaderLocation(self.shaders["blur"], b"camInvProj")
        self.shader_locs["blur_cam_clip_near"] = GetShaderLocation(self.shaders["blur"], b"camClipNear")
        self.shader_locs["blur_cam_clip_far"] = GetShaderLocation(self.shaders["blur"], b"camClipFar")
        self.shader_locs["blur_inv_texture_resolution"] = GetShaderLocation(self.shaders["blur"], b"invTextureResolution")
        self.shader_locs["blur_direction"] = GetShaderLocation(self.shaders["blur"], b"blurDirection")

        self.shader_locs["lighting_gbuffer_color"] = GetShaderLocation(self.shaders["lighting"], b"gbufferColor")
        self.shader_locs["lighting_gbuffer_normal"] = GetShaderLocation(self.shaders["lighting"], b"gbufferNormal")
        self.shader_locs["lighting_gbuffer_depth"] = GetShaderLocation(self.shaders["lighting"], b"gbufferDepth")
        self.shader_locs["lighting_ssao"] = GetShaderLocation(self.shaders["lighting"], b"ssao")
        self.shader_locs["lighting_material_ao"] = GetShaderLocation(self.shaders["lighting"], b"materialAO")
        self.shader_locs["lighting_cam_pos"] = GetShaderLocation(self.shaders["lighting"], b"camPos")
        self.shader_locs["lighting_cam_inv_view_proj"] = GetShaderLocation(self.shaders["lighting"], b"camInvViewProj")
        self.shader_locs["lighting_light_dir"] = GetShaderLocation(self.shaders["lighting"], b"lightDir")
        self.shader_locs["lighting_sun_color"] = GetShaderLocation(self.shaders["lighting"], b"sunColor")
        self.shader_locs["lighting_sun_strength"] = GetShaderLocation(self.shaders["lighting"], b"sunStrength")
        self.shader_locs["lighting_sky_color"] = GetShaderLocation(self.shaders["lighting"], b"skyColor")
        self.shader_locs["lighting_sky_strength"] = GetShaderLocation(self.shaders["lighting"], b"skyStrength")
        self.shader_locs["lighting_ground_strength"] = GetShaderLocation(self.shaders["lighting"], b"groundStrength")
        self.shader_locs["lighting_ambient_strength"] = GetShaderLocation(self.shaders["lighting"], b"ambientStrength")
        if self.shading == "legacy":
            self.shader_locs["lighting_exposure"] = GetShaderLocation(self.shaders["lighting"], b"exposure")
        self.shader_locs["lighting_cam_clip_near"] = GetShaderLocation(self.shaders["lighting"], b"camClipNear")
        self.shader_locs["lighting_cam_clip_far"] = GetShaderLocation(self.shaders["lighting"], b"camClipFar")

        self.shader_locs["lighting_shadow_map"] = GetShaderLocation(self.shaders["lighting"], b"shadowMap")
        self.shader_locs["lighting_light_view_proj"] = GetShaderLocation(self.shaders["lighting"], b"lightViewProj")
        self.shader_locs["lighting_light_clip_near"] = GetShaderLocation(self.shaders["lighting"], b"lightClipNear")
        self.shader_locs["lighting_light_clip_far"] = GetShaderLocation(self.shaders["lighting"], b"lightClipFar")
        if self.shading == "pbr":
            self.shader_locs["lighting_environment_map"] = GetShaderLocation(self.shaders["lighting"], b"environmentMap")
            self.shader_locs["lighting_irradiance_map"] = GetShaderLocation(self.shaders["lighting"], b"irradianceMap")
            self.shader_locs["lighting_prefilter_map"] = GetShaderLocation(self.shaders["lighting"], b"prefilterMap")
            self.shader_locs["lighting_brdf_lut"] = GetShaderLocation(self.shaders["lighting"], b"brdfLut")
            self.shader_locs["lighting_prefilter_max_lod"] = GetShaderLocation(self.shaders["lighting"], b"prefilterMaxLod")
            self.shader_locs["lighting_ibl_strength"] = GetShaderLocation(self.shaders["lighting"], b"iblStrength")
            self.shader_locs["lighting_use_ibl"] = GetShaderLocation(self.shaders["lighting"], b"useIBL")
            self.shader_locs["lighting_debug_mode"] = GetShaderLocation(self.shaders["lighting"], b"debugMode")

        if self.shading == "pbr":
            self.shader_locs["tonemap_input_texture"] = GetShaderLocation(self.shaders["tonemap"], b"inputTexture")
            self.shader_locs["tonemap_exposure"] = GetShaderLocation(self.shaders["tonemap"], b"exposure")
            self.shader_locs["debug_gbuffer_color"] = GetShaderLocation(self.shaders["debug"], b"texGbufferColor")
            self.shader_locs["debug_gbuffer_normal"] = GetShaderLocation(self.shaders["debug"], b"texGbufferNormal")
            self.shader_locs["debug_gbuffer_depth"] = GetShaderLocation(self.shaders["debug"], b"texGbufferDepth")
            self.shader_locs["debug_ssao"] = GetShaderLocation(self.shaders["debug"], b"texSSAO")
            self.shader_locs["debug_lighted"] = GetShaderLocation(self.shaders["debug"], b"texLighted")
            self.shader_locs["debug_mode"] = GetShaderLocation(self.shaders["debug"], b"debugMode")
            self.shader_locs["debug_exposure"] = GetShaderLocation(self.shaders["debug"], b"exposure")

        if self.shading == "pbr":
            self.ibl = IBLResources(strength=self.ibl_strength, enabled=self.ibl_enabled).initialize()
            self.prefilter_max_lod_ptr[0] = self.ibl.prefilter_max_lod

        # Material maps need the GL context, so they are loaded here instead of
        # __init__; they ride on the default material of the skinned character.
        if self.shading == "pbr":
            if self.normal_map_path is not None:
                self.default_material.normal_map = load_material_texture(self.normal_map_path, "normal_map")
            if self.metallic_roughness_map_path is not None:
                self.default_material.metallic_roughness_map = load_material_texture(
                    self.metallic_roughness_map_path, "metallic_roughness_map"
                )

        self.shader_locs["fxaa_input_texture"] = GetShaderLocation(self.shaders["fxaa"], b"inputTexture")
        self.shader_locs["fxaa_inv_texture_resolution"] = GetShaderLocation(self.shaders["fxaa"], b"invTextureResolution")

        self.render_targets = RenderTargets(
            screen_width, screen_height, self.shading, shadow_resolution=self.shadow_resolution
        ).initialize()
        self.shadow_map = self.render_targets.shadow_map
        self.gbuffer = self.render_targets.gbuffer
        self.lighted = self.render_targets.lighting
        self.tonemapped = self.render_targets.tonemapped
        self.ssao_front = self.render_targets.ssao_front
        self.ssao_back = self.render_targets.ssao_back

        ground_mesh = GenMeshPlane(20.0, 20.0, 10, 10)
        GenMeshTangents(ffi.addressof(ground_mesh))
        self.ground_model = LoadModelFromMesh(ground_mesh)

        self.geno_model = load_geno_model(self.resources_root / self.rig.model_filename)
        self.bind_pos, self.bind_rot = get_model_bind_pose_as_numpy_arrays(self.geno_model)
        if self.compare_mode:
            self.compare_model = load_geno_model(self.resources_root / self.rig.model_filename)
            self.compare_bind_pos, self.compare_bind_rot = get_model_bind_pose_as_numpy_arrays(self.compare_model)
        self.scene.clear()
        self.scene.add_object(
            RenderObject(
                model=self.ground_model,
                material=Material(),
                position=self.ground_position,
                draw_color=Color(*self.light_rig["ground_albedo"], 255),
                skinned=False,
            )
        )
        self.scene.add_object(
            RenderObject(
                model=self.geno_model,
                material=self.default_material,
                position=self.left_model_offset,
                draw_color=Color(70, 125, 255, 255) if self.compare_mode else ORANGE,
                skinned=True,
            )
        )
        if self.compare_mode:
            self.scene.add_object(
                RenderObject(
                    model=self.compare_model,
                    material=self.default_material,
                    position=self.right_model_offset,
                    draw_color=ORANGE,
                    skinned=True,
                )
            )
        (
            self.full_names,
            self.full_parents,
            self.full_bind_local_positions,
            self.full_bind_local_rotations,
        ) = build_simulation_root_skeleton_from_bind(self.resources_root / self.rig.bind_bvh_filename, self.rig)
        self.full_name_to_index = {name: idx for idx, name in enumerate(self.full_names)}
        if self.rig.skeleton_root_joint not in self.full_name_to_index:
            raise ValueError(
                f"Rig skeleton_root_joint {self.rig.skeleton_root_joint!r} is missing from the bind skeleton"
            )
        self.skeleton_root_index = self.full_name_to_index[self.rig.skeleton_root_joint]

        if len(self.full_names) - 1 != self.geno_model.boneCount:
            raise ValueError(
                f"Bind skeleton count ({len(self.full_names)} incl. Simulation) does not match Geno model bone count "
                f"({self.geno_model.boneCount})."
            )
        if self.compare_mode and self.compare_model.boneCount != self.geno_model.boneCount:
            raise ValueError("Compare model bone count does not match primary model bone count")

        unknown_names = [str(name) for name in self.names.tolist() if str(name) not in self.full_name_to_index]
        if unknown_names:
            raise ValueError(f"Database contains joints missing from Geno bind skeleton: {unknown_names}")

        self.use_pruned_reconstruction = len(self.names) != len(self.full_names)

    def _cleanup(self):
        if self.ibl is not None:
            self.ibl.cleanup()
        if self.render_targets is not None:
            self.render_targets.cleanup()
        if self.default_material is not None:
            for texture in (
                self.default_material.base_color_map,
                self.default_material.metallic_roughness_map,
                self.default_material.normal_map,
            ):
                if texture is not None:
                    UnloadTexture(texture)
        if self.geno_model is not None:
            UnloadModel(self.geno_model)
        if self.compare_model is not None:
            UnloadModel(self.compare_model)
        if self.ground_model is not None:
            UnloadModel(self.ground_model)
        for shader in self.shaders.values():
            UnloadShader(shader)

    def _frame_range_name(self):
        for name, start, stop in zip(self.range_names, self.range_starts, self.range_stops):
            if start <= self.frame_index < stop:
                return str(name)
        return "unknown"

    def _control_hint(self) -> bytes:
        return b"Space: play/pause | Left/Right: step | Up/Down: speed | Home/End | Drag timeline | B: skeleton"

    def _reconstruct_full_local_pose_for(self, positions, rotations, frame_index):
        full_positions = self.full_bind_local_positions.copy()
        full_rotations = self.full_bind_local_rotations.copy()
        db_positions = positions[frame_index]
        db_rotations = rotations[frame_index]

        for db_index, name in enumerate(self.names.tolist()):
            full_index = self.full_name_to_index[str(name)]
            full_positions[full_index] = db_positions[db_index]
            full_rotations[full_index] = db_rotations[db_index]

        return full_rotations, full_positions

    def _reconstruct_full_local_pose(self):
        return self._reconstruct_full_local_pose_for(self.positions, self.rotations, self.frame_index)

    def _current_globals_for(self, positions, rotations, parents, frame_index):
        if self.use_pruned_reconstruction:
            local_rotations, local_positions = self._reconstruct_full_local_pose_for(positions, rotations, frame_index)
            return quat.fk(local_rotations[None], local_positions[None], self.full_parents)
        global_rot, global_pos = quat.fk(rotations[frame_index][None], positions[frame_index][None], parents)
        names = self.names if positions is self.positions else self.compare_names
        if [str(name) for name in names.tolist()] != self.full_names:
            name_to_index = {str(name): index for index, name in enumerate(names.tolist())}
            order = [name_to_index[name] for name in self.full_names]
            global_rot = global_rot[:, order]
            global_pos = global_pos[:, order]
        return global_rot, global_pos

    def _current_globals(self):
        return self._current_globals_for(self.positions, self.rotations, self.parents, self.frame_index)

    def _update_model_pose_for(self, model, bind_pos, bind_rot, positions, rotations, parents, frame_index):
        global_rot, global_pos = self._current_globals_for(positions, rotations, parents, frame_index)
        update_model_pose_from_numpy_arrays(model, bind_pos, bind_rot, global_pos[0, 1:], global_rot[0, 1:])
        return global_rot[0], global_pos[0]

    def _update_model_pose(self):
        return self._update_model_pose_for(
            self.geno_model,
            self.bind_pos,
            self.bind_rot,
            self.positions,
            self.rotations,
            self.parents,
            self.frame_index,
        )

    def _update_compare_model_pose(self):
        return self._update_model_pose_for(
            self.compare_model,
            self.compare_bind_pos,
            self.compare_bind_rot,
            self.compare_positions,
            self.compare_rotations,
            self.compare_parents,
            self.frame_index,
        )

    def _sync_playback_frame(self):
        if self.indices is not None:
            self.sample_index = self.playback.current_frame
            self.frame_index = int(self.indices[self.sample_index])
        else:
            self.frame_index = self.playback.current_frame

    def run(self):
        screen_width = 1280
        screen_height = 720
        frame_dir = Path(tempfile.mkdtemp(prefix="somaview_frames_")) if self.output_video else None
        recording_cwd = Path.cwd()
        if frame_dir is not None:
            os.chdir(frame_dir)
        SetConfigFlags(FLAG_VSYNC_HINT | (FLAG_WINDOW_HIDDEN if frame_dir is not None else 0))
        InitWindow(screen_width, screen_height, self.rig.window_title)
        SetTargetFPS(self.fps)
        rlSetClipPlanes(0.01, 50.0)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._initialize_rendering(screen_width, screen_height)
        from stylized_motion.anim.renderer import Renderer

        self.renderer = Renderer(self)
        recorded_frames = 0

        try:
            while not WindowShouldClose():
                if frame_dir is not None:
                    self.playback.set_current_frame(recorded_frames)
                self.playback.handle_shortcuts()
                if IsKeyPressed(KEY_B):
                    self.skeleton_enabled = not self.skeleton_enabled
                if self.shading == "pbr" and IsKeyPressed(KEY_V):
                    step = -1 if IsKeyDown(KEY_LEFT_SHIFT) or IsKeyDown(KEY_RIGHT_SHIFT) else 1
                    self.debug_view = DEBUG_MODES[(DEBUG_MODES.index(self.debug_view) + step) % len(DEBUG_MODES)]
                if frame_dir is None:
                    self.playback.handle_scrub(screen_width, screen_height)
                    self.playback.update(GetFrameTime())
                self._sync_playback_frame()

                global_rot, global_pos = self._update_model_pose()
                if self.compare_mode:
                    compare_global_rot, compare_global_pos = self._update_compare_model_pose()
                else:
                    compare_global_rot = None
                    compare_global_pos = None

                root = global_pos[0]
                target_x = root[0] + self.left_model_offset.x
                target_z = root[2] + self.left_model_offset.z
                if self.compare_mode:
                    compare_root = compare_global_pos[0]
                    target_x = 0.5 * (target_x + compare_root[0] + self.right_model_offset.x)
                    target_z = 0.5 * (target_z + compare_root[2] + self.right_model_offset.z)
                self.shadow_light.target = Vector3(target_x, 0.0, target_z)
                self.shadow_light.position = Vector3Add(self.shadow_light.target, Vector3Scale(self.light_dir, -5.0))

                self.camera.update(
                    Vector3(target_x, 0.75, target_z),
                    GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
                    GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
                    GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
                    GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
                    GetMouseWheelMove(),
                    GetFrameTime(),
                )

                rlDisableColorBlend()
                BeginDrawing()
                self.renderer.render_frame(global_rot, global_pos, self.sample_index, compare_global_rot, compare_global_pos)

                self.renderer.draw_output()

                if frame_dir is None:
                    rlEnableColorBlend()
                    DrawFPS(10, 10)
                    play_state = "Paused" if not self.playback.playing else f"Playing {self.playback.current_speed:.2f}x"
                    status = f"{play_state} | Skeleton: {'on' if self.skeleton_enabled else 'off'}"
                    DrawText(f"Frame: {self.frame_index}".encode(), 10, 34, 20, BLACK)
                    DrawText(f"Range: {self._frame_range_name()}".encode(), 10, 58, 20, DARKGRAY)
                    DrawText(status.encode(), 10, 82, 20, BLUE)
                    mode_label = f"Skeleton: {'pruned->full reconstruction' if self.use_pruned_reconstruction else 'full direct'}"
                    DrawText(mode_label.encode(), 10, 106, 20, DARKGRAY)
                    if self.compare_mode:
                        DrawText(f"Left: {self.left_label}".encode(), 10, 130, 20, Color(40, 90, 220, 255))
                        DrawText(f"Right: {self.right_label}".encode(), 10, 154, 20, ORANGE)
                    if self.indices is not None:
                        DrawText(f"Sample: {self.sample_index}".encode(), 10, 130, 20, DARKGRAY)
                        DrawText(f"Mirror: {bool(self.sample_mirror[self.sample_index])}".encode(), 10, 154, 20, DARKGRAY)
                    DrawText(b"Ctrl+LMB/RMB+drag: camera | Wheel: zoom", 10, 184, 18, BLACK)
                    DrawText(self._control_hint(), 10, 208, 18, BLACK)
                    if self.shading == "pbr":
                        DrawText(f"Debug: {self.debug_view} (V / Shift+V)".encode(), 10, 232, 18, DARKGRAY)
                    self.playback.draw_ui(screen_width, screen_height, "Sample" if self.indices is not None else "Frame")
                EndDrawing()
                if frame_dir is not None:
                    take_screenshot(str(frame_dir / f"frame_{recorded_frames:06d}.png"))
                    recorded_frames += 1
                    if recorded_frames >= self.playback.frame_count:
                        break
        finally:
            if frame_dir is not None:
                os.chdir(recording_cwd)
            self._cleanup()
            CloseWindow()
            if frame_dir is not None:
                self.output_video.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-framerate", str(self.fps),
                        "-i", str(frame_dir / "frame_%06d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(self.output_video),
                    ],
                    check=True,
                )
                shutil.rmtree(frame_dir)


class GenoViewCompare(GenoView):
    def __init__(
        self,
        left_database: dict[str, np.ndarray],
        right_database: dict[str, np.ndarray],
        resources_root: Path,
        fps: int = 60,
        left_label: str = "Source",
        right_label: str = "Recon",
        compare_spacing: float = 2.0,
        shading: str = "pbr",
        metallic: float = 0.0,
        roughness: float = 0.58,
        exposure: float = 0.9,
        ssao_intensity: float = 0.15,
        ibl_strength: float = 0.35,
        ibl_enabled: bool = True,
        shadow_resolution: int = 2048,
        output_video: Path | None = None,
        draw_skeleton: bool = False,
        debug_view: str = "final",
        normal_map: Path | None = None,
        metallic_roughness_map: Path | None = None,
        sun_strength: float | None = None,
        rig: RigSpec = GENO_RIG,
    ):
        super().__init__(
            database=left_database,
            trajectory_path=None,
            resources_root=resources_root,
            fps=fps,
            compare_database=right_database,
            left_label=left_label,
            right_label=right_label,
            compare_spacing=compare_spacing,
            rig=rig,
            shading=shading,
            metallic=metallic,
            roughness=roughness,
            exposure=exposure,
            ssao_intensity=ssao_intensity,
            ibl_strength=ibl_strength,
            ibl_enabled=ibl_enabled,
            shadow_resolution=shadow_resolution,
            output_video=output_video,
            draw_skeleton=draw_skeleton,
            debug_view=debug_view,
            normal_map=normal_map,
            metallic_roughness_map=metallic_roughness_map,
            sun_strength=sun_strength,
        )


def main():
    parser = argparse.ArgumentParser(description="High-quality Geno viewer driven by database.npz or 230D motion features.")
    parser.add_argument("--database", type=Path, default=None, help="Path to database.npz")
    parser.add_argument("--bvh", type=Path, default=None, help="Path to a single BVH clip")
    parser.add_argument("--features", type=Path, default=None, help="Path to .npy or .npz containing 230D features with shape [T, D].")
    parser.add_argument("--feature-key", type=str, default="motion", help="Array key for .npz feature input.")
    parser.add_argument("--stats-source", type=Path, default=None, help="Checkpoint .pt or schema-v3 feature_database directory containing feature stats.")
    parser.add_argument("--normalized", action="store_true", help="Treat --features as normalized feature values.")
    parser.add_argument("--range-name", type=str, default="features", help="Range name used in feature visualization mode.")
    parser.add_argument("--root-position0", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--root-rotation0", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional path to trajectory.npz")
    parser.add_argument(
        "--resources-root",
        type=Path,
        default=RESOURCE_DIR,
        help="Directory containing Geno.bin and shader files",
    )
    parser.add_argument("--fps", type=int, default=None, help="Playback FPS (defaults to the BVH frame time)")
    parser.add_argument("--shading", choices=("legacy", "pbr"), default="pbr")
    parser.add_argument("--metallic", type=float, default=0.0)
    parser.add_argument("--roughness", type=float, default=0.58)
    parser.add_argument("--exposure", type=float, default=0.9)
    parser.add_argument("--sun-strength", type=float, default=None, help="Direct light intensity. Defaults to the per-shading light rig value (PBR: 0.55, legacy: 0.25).")
    parser.add_argument("--ssao-intensity", type=float, default=0.15)
    parser.add_argument("--ibl-strength", type=float, default=0.35)
    parser.add_argument("--disable-ibl", action="store_true")
    parser.add_argument("--shadow-resolution", type=int, default=2048)
    parser.add_argument("--output-video", type=Path, default=None, help="Render the full clip to an MP4 at this path.")
    parser.add_argument("--skeleton", action="store_true", help="Draw the character skeleton overlay (joints, bone links, per-joint XYZ frames). Toggle at runtime with B.")
    parser.add_argument("--debug-view", choices=DEBUG_MODES, default="final", help="Initial PBR debug view. Cycle at runtime with V (Shift+V to go back).")
    parser.add_argument("--normal-map", type=Path, default=None, help="Optional tangent-space normal map applied to the character's default material.")
    parser.add_argument("--metallic-roughness-map", type=Path, default=None, help="Optional linear map (R=metallic, G=roughness, B=AO) applied to the character's default material.")
    args = parser.parse_args()

    selected_inputs = [args.database is not None, args.bvh is not None, args.features is not None]
    if sum(selected_inputs) != 1:
        raise ValueError("Exactly one of --database, --bvh, or --features is required")
    if args.features is not None and args.stats_source is None:
        raise ValueError("--stats-source is required when using --features")

    database = (
        load_database_dict(args.database)
        if args.database is not None
        else build_database_from_bvh(args.bvh, range_name=args.range_name)
        if args.bvh is not None
        else build_database_from_features(
            features_path=args.features,
            stats_source=args.stats_source,
            feature_key=args.feature_key,
            normalized=args.normalized,
            range_name=args.range_name,
            root_position0=args.root_position0,
            root_rotation0=args.root_rotation0,
        )
    )

    viewer = GenoView(
        database=database,
        trajectory_path=args.trajectory,
        resources_root=args.resources_root,
        fps=args.fps,
        shading=args.shading,
        metallic=args.metallic,
        roughness=args.roughness,
        exposure=args.exposure,
        ssao_intensity=args.ssao_intensity,
        ibl_strength=args.ibl_strength,
        ibl_enabled=not args.disable_ibl,
        shadow_resolution=args.shadow_resolution,
        output_video=args.output_video,
        draw_skeleton=args.skeleton,
        debug_view=args.debug_view,
        normal_map=args.normal_map,
        metallic_roughness_map=args.metallic_roughness_map,
        sun_strength=args.sun_strength,
    )
    viewer.run()


if __name__ == "__main__":
    main()
