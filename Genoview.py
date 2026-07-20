import argparse
import struct
from pathlib import Path

import cffi
import numpy as np
import torch
from pyray import BoneInfo, Camera3D, Color, Matrix, Mesh, Model, Rectangle, RenderTexture, Texture, Transform, Vector2, Vector3
from raylib import *
from raylib.defines import *

from motion_features import deserialize_motion_feature_stats, reconstruct_motion_state_from_features
from preprocess import quat


ffi = cffi.FFI()


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


class GBuffer:
    def __init__(self):
        self.id = 0
        self.color = Texture()
        self.normal = Texture()
        self.depth = Texture()


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


def load_shadow_map(width, height):
    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0

    rlEnableFramebuffer(target.id)
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)
    rlDisableFramebuffer()
    return target


def unload_shadow_map(target):
    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def begin_shadow_map(target, shadow_light):
    BeginTextureMode(target)
    ClearBackground(WHITE)
    rlDrawRenderBatchActive()
    rlMatrixMode(RL_PROJECTION)
    rlPushMatrix()
    rlLoadIdentity()
    rlOrtho(
        -shadow_light.width / 2,
        shadow_light.width / 2,
        -shadow_light.height / 2,
        shadow_light.height / 2,
        shadow_light.near,
        shadow_light.far,
    )
    rlMatrixMode(RL_MODELVIEW)
    rlLoadIdentity()
    mat_view = MatrixLookAt(shadow_light.position, shadow_light.target, shadow_light.up)
    rlMultMatrixf(MatrixToFloatV(mat_view).v)
    rlEnableDepthTest()


def end_shadow_map():
    rlDrawRenderBatchActive()
    rlMatrixMode(RL_PROJECTION)
    rlPopMatrix()
    rlMatrixMode(RL_MODELVIEW)
    rlLoadIdentity()
    rlDisableDepthTest()
    EndTextureMode()


def set_shader_value_shadow_map(shader, loc_index, target):
    if loc_index > -1:
        rlEnableShader(shader.id)
        slot_ptr = ffi.new("int*")
        slot_ptr[0] = 10
        rlActiveTextureSlot(slot_ptr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(loc_index, slot_ptr, SHADER_UNIFORM_INT, 1)


def load_gbuffer(width, height):
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)

    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)
    rlDisableFramebuffer()
    return target


def unload_gbuffer(target):
    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def begin_gbuffer(target, camera):
    rlDrawRenderBatchActive()
    rlEnableFramebuffer(target.id)
    rlActiveDrawBuffers(2)
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)
    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)
    rlPushMatrix()
    rlLoadIdentity()

    aspect = float(target.color.width) / float(target.color.height)
    top = rlGetCullDistanceNear() * np.tan(camera.fovy * 0.5 * DEG2RAD)
    right = top * aspect
    rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    rlMatrixMode(RL_MODELVIEW)
    rlLoadIdentity()
    mat_view = MatrixLookAt(camera.position, camera.target, camera.up)
    rlMultMatrixf(MatrixToFloatV(mat_view).v)
    rlEnableDepthTest()


def end_gbuffer(window_width, window_height):
    rlDrawRenderBatchActive()
    rlDisableDepthTest()
    rlActiveDrawBuffers(1)
    rlDisableFramebuffer()
    rlMatrixMode(RL_PROJECTION)
    rlPopMatrix()
    rlLoadIdentity()
    rlOrtho(0, window_width, window_height, 0, 0.0, 1.0)
    rlMatrixMode(RL_MODELVIEW)
    rlLoadIdentity()


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


def build_simulation_root_skeleton_from_bind(bind_bvh_path: Path):
    bind_data = load_bvh_data(bind_bvh_path)
    positions = bind_data["positions"].astype(np.float32) * 0.01
    rotations = quat.unroll(quat.from_euler(np.radians(bind_data["rotations"]), order=bind_data["order"])).astype(np.float32)

    global_rotations, global_positions = quat.fk(rotations, positions, bind_data["parents"])
    sim_position_joint = bind_data["names"].index("Spine2")
    sim_rotation_joint = bind_data["names"].index("Hips")

    sim_position = np.array([1.0, 0.0, 1.0], dtype=np.float32) * global_positions[:, sim_position_joint : sim_position_joint + 1]
    sim_direction = np.array([1.0, 0.0, 1.0], dtype=np.float32) * quat.mul_vec(
        global_rotations[:, sim_rotation_joint : sim_rotation_joint + 1], np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[..., np.newaxis]
    sim_rotation = quat.normalize(quat.between(np.array([0.0, 0.0, 1.0], dtype=np.float32), sim_direction))

    positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
    rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])

    positions = np.concatenate([sim_position, positions], axis=1)
    rotations = np.concatenate([sim_rotation, rotations], axis=1)
    parents = np.concatenate([[-1], bind_data["parents"] + 1]).astype(np.int32)
    names = ["Simulation"] + bind_data["names"]
    return names, parents, positions[0].astype(np.float32), rotations[0].astype(np.float32)


def load_bvh_data(path: Path):
    from preprocess import bvh

    return bvh.load(str(path))


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
        stats_source = stats_source / "metadata.npz"

    if stats_source.suffix == ".npz":
        payload_npz = np.load(stats_source, allow_pickle=True)
        payload = {key: payload_npz[key] for key in payload_npz.files}
    else:
        checkpoint = torch.load(stats_source, map_location="cpu", weights_only=False)
        if "stats" not in checkpoint:
            raise KeyError(f"Checkpoint {stats_source} does not contain stats")
        payload = checkpoint["stats"]

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


class GenoView:
    def __init__(
        self,
        database: dict[str, np.ndarray],
        trajectory_path: Path | None,
        resources_root: Path,
        fps: int = 60,
        compare_database: dict[str, np.ndarray] | None = None,
        left_label: str = "Source",
        right_label: str = "Recon",
        compare_spacing: float = 2.0,
    ):
        self.database = database
        self.positions = self.database["positions"].astype(np.float32)
        self.rotations = self.database["rotations"].astype(np.float32)
        self.parents = self.database["parents"].astype(np.int32)
        self.names = self.database["names"]
        self.joint_subset = self.database["joint_subset"].item() if "joint_subset" in self.database else "full"
        self.range_names = self.database["range_names"]
        self.range_starts = self.database["range_starts"].astype(np.int32)
        self.range_stops = self.database["range_stops"].astype(np.int32)
        self.fps = fps
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
        self.light_dir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
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
        self.ssao_front = None
        self.ssao_back = None
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
        self.full_bind_local_positions = None
        self.full_bind_local_rotations = None
        self.database_name_to_index = {str(name): idx for idx, name in enumerate(self.names.tolist())}
        self.full_name_to_index = None
        self.use_pruned_reconstruction = False
        self.shader_locs = {}
        self.shadow_inv_resolution = Vector2(1.0 / 1024.0, 1.0 / 1024.0)
        self.ground_position = Vector3(0.0, -0.01, 0.0)

    def _res(self, name: str) -> bytes:
        return str((self.resources_root / name).resolve()).encode()

    def _initialize_rendering(self, screen_width: int, screen_height: int):
        self.shaders["basic"] = LoadShader(self._res("basic.vs"), self._res("basic.fs"))
        self.shaders["skinned_basic"] = LoadShader(self._res("skinnedBasic.vs"), self._res("basic.fs"))
        self.shaders["shadow"] = LoadShader(self._res("shadow.vs"), self._res("shadow.fs"))
        self.shaders["skinned_shadow"] = LoadShader(self._res("skinnedShadow.vs"), self._res("shadow.fs"))
        self.shaders["ssao"] = LoadShader(self._res("post.vs"), self._res("ssao.fs"))
        self.shaders["blur"] = LoadShader(self._res("post.vs"), self._res("blur.fs"))
        self.shaders["lighting"] = LoadShader(self._res("post.vs"), self._res("lighting.fs"))
        self.shaders["fxaa"] = LoadShader(self._res("post.vs"), self._res("fxaa.fs"))

        self.shader_locs["basic_specularity"] = GetShaderLocation(self.shaders["basic"], b"specularity")
        self.shader_locs["basic_glossiness"] = GetShaderLocation(self.shaders["basic"], b"glossiness")
        self.shader_locs["basic_cam_clip_near"] = GetShaderLocation(self.shaders["basic"], b"camClipNear")
        self.shader_locs["basic_cam_clip_far"] = GetShaderLocation(self.shaders["basic"], b"camClipFar")

        self.shader_locs["skinned_basic_specularity"] = GetShaderLocation(self.shaders["skinned_basic"], b"specularity")
        self.shader_locs["skinned_basic_glossiness"] = GetShaderLocation(self.shaders["skinned_basic"], b"glossiness")
        self.shader_locs["skinned_basic_cam_clip_near"] = GetShaderLocation(self.shaders["skinned_basic"], b"camClipNear")
        self.shader_locs["skinned_basic_cam_clip_far"] = GetShaderLocation(self.shaders["skinned_basic"], b"camClipFar")

        self.shader_locs["shadow_light_clip_near"] = GetShaderLocation(self.shaders["shadow"], b"lightClipNear")
        self.shader_locs["shadow_light_clip_far"] = GetShaderLocation(self.shaders["shadow"], b"lightClipFar")
        self.shader_locs["skinned_shadow_light_clip_near"] = GetShaderLocation(self.shaders["skinned_shadow"], b"lightClipNear")
        self.shader_locs["skinned_shadow_light_clip_far"] = GetShaderLocation(self.shaders["skinned_shadow"], b"lightClipFar")

        self.shader_locs["ssao_gbuffer_normal"] = GetShaderLocation(self.shaders["ssao"], b"gbufferNormal")
        self.shader_locs["ssao_gbuffer_depth"] = GetShaderLocation(self.shaders["ssao"], b"gbufferDepth")
        self.shader_locs["ssao_cam_view"] = GetShaderLocation(self.shaders["ssao"], b"camView")
        self.shader_locs["ssao_cam_proj"] = GetShaderLocation(self.shaders["ssao"], b"camProj")
        self.shader_locs["ssao_cam_inv_proj"] = GetShaderLocation(self.shaders["ssao"], b"camInvProj")
        self.shader_locs["ssao_cam_inv_view_proj"] = GetShaderLocation(self.shaders["ssao"], b"camInvViewProj")
        self.shader_locs["ssao_light_view_proj"] = GetShaderLocation(self.shaders["ssao"], b"lightViewProj")
        self.shader_locs["ssao_shadow_map"] = GetShaderLocation(self.shaders["ssao"], b"shadowMap")
        self.shader_locs["ssao_shadow_inv_resolution"] = GetShaderLocation(self.shaders["ssao"], b"shadowInvResolution")
        self.shader_locs["ssao_cam_clip_near"] = GetShaderLocation(self.shaders["ssao"], b"camClipNear")
        self.shader_locs["ssao_cam_clip_far"] = GetShaderLocation(self.shaders["ssao"], b"camClipFar")
        self.shader_locs["ssao_light_clip_near"] = GetShaderLocation(self.shaders["ssao"], b"lightClipNear")
        self.shader_locs["ssao_light_clip_far"] = GetShaderLocation(self.shaders["ssao"], b"lightClipFar")
        self.shader_locs["ssao_light_dir"] = GetShaderLocation(self.shaders["ssao"], b"lightDir")

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
        self.shader_locs["lighting_cam_pos"] = GetShaderLocation(self.shaders["lighting"], b"camPos")
        self.shader_locs["lighting_cam_inv_view_proj"] = GetShaderLocation(self.shaders["lighting"], b"camInvViewProj")
        self.shader_locs["lighting_light_dir"] = GetShaderLocation(self.shaders["lighting"], b"lightDir")
        self.shader_locs["lighting_sun_color"] = GetShaderLocation(self.shaders["lighting"], b"sunColor")
        self.shader_locs["lighting_sun_strength"] = GetShaderLocation(self.shaders["lighting"], b"sunStrength")
        self.shader_locs["lighting_sky_color"] = GetShaderLocation(self.shaders["lighting"], b"skyColor")
        self.shader_locs["lighting_sky_strength"] = GetShaderLocation(self.shaders["lighting"], b"skyStrength")
        self.shader_locs["lighting_ground_strength"] = GetShaderLocation(self.shaders["lighting"], b"groundStrength")
        self.shader_locs["lighting_ambient_strength"] = GetShaderLocation(self.shaders["lighting"], b"ambientStrength")
        self.shader_locs["lighting_exposure"] = GetShaderLocation(self.shaders["lighting"], b"exposure")
        self.shader_locs["lighting_cam_clip_near"] = GetShaderLocation(self.shaders["lighting"], b"camClipNear")
        self.shader_locs["lighting_cam_clip_far"] = GetShaderLocation(self.shaders["lighting"], b"camClipFar")

        self.shader_locs["fxaa_input_texture"] = GetShaderLocation(self.shaders["fxaa"], b"inputTexture")
        self.shader_locs["fxaa_inv_texture_resolution"] = GetShaderLocation(self.shaders["fxaa"], b"invTextureResolution")

        self.shadow_map = load_shadow_map(1024, 1024)
        self.gbuffer = load_gbuffer(screen_width, screen_height)
        self.lighted = LoadRenderTexture(screen_width, screen_height)
        self.ssao_front = LoadRenderTexture(screen_width, screen_height)
        self.ssao_back = LoadRenderTexture(screen_width, screen_height)

        ground_mesh = GenMeshPlane(20.0, 20.0, 10, 10)
        self.ground_model = LoadModelFromMesh(ground_mesh)

        self.geno_model = load_geno_model(self.resources_root / "Geno.bin")
        self.bind_pos, self.bind_rot = get_model_bind_pose_as_numpy_arrays(self.geno_model)
        if self.compare_mode:
            self.compare_model = load_geno_model(self.resources_root / "Geno.bin")
            self.compare_bind_pos, self.compare_bind_rot = get_model_bind_pose_as_numpy_arrays(self.compare_model)
        (
            self.full_names,
            self.full_parents,
            self.full_bind_local_positions,
            self.full_bind_local_rotations,
        ) = build_simulation_root_skeleton_from_bind(self.resources_root / "Geno_bind.bvh")
        self.full_name_to_index = {name: idx for idx, name in enumerate(self.full_names)}

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
        if self.lighted is not None:
            UnloadRenderTexture(self.lighted)
        if self.ssao_back is not None:
            UnloadRenderTexture(self.ssao_back)
        if self.ssao_front is not None:
            UnloadRenderTexture(self.ssao_front)
        if self.gbuffer is not None:
            unload_gbuffer(self.gbuffer)
        if self.shadow_map is not None:
            unload_shadow_map(self.shadow_map)
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
        return b"Space: play/pause | Left/Right: step | Up/Down: speed | Home/End | Drag timeline"

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
        return quat.fk(rotations[frame_index][None], positions[frame_index][None], parents)

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
        SetConfigFlags(FLAG_VSYNC_HINT)
        InitWindow(screen_width, screen_height, b"GenoView")
        SetTargetFPS(self.fps)
        rlSetClipPlanes(0.01, 50.0)
        self._initialize_rendering(screen_width, screen_height)

        try:
            while not WindowShouldClose():
                self.playback.handle_shortcuts()
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

                begin_shadow_map(self.shadow_map, self.shadow_light)
                light_view_proj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
                light_clip_near = rlGetCullDistanceNear()
                light_clip_far = rlGetCullDistanceFar()
                light_clip_near_ptr = ffi.new("float*")
                light_clip_far_ptr = ffi.new("float*")
                light_clip_near_ptr[0] = light_clip_near
                light_clip_far_ptr[0] = light_clip_far

                SetShaderValue(self.shaders["shadow"], self.shader_locs["shadow_light_clip_near"], light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["shadow"], self.shader_locs["shadow_light_clip_far"], light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(
                    self.shaders["skinned_shadow"],
                    self.shader_locs["skinned_shadow_light_clip_near"],
                    light_clip_near_ptr,
                    SHADER_UNIFORM_FLOAT,
                )
                SetShaderValue(
                    self.shaders["skinned_shadow"],
                    self.shader_locs["skinned_shadow_light_clip_far"],
                    light_clip_far_ptr,
                    SHADER_UNIFORM_FLOAT,
                )

                self.ground_model.materials[0].shader = self.shaders["shadow"]
                DrawModel(self.ground_model, self.ground_position, 1.0, WHITE)
                self.geno_model.materials[0].shader = self.shaders["skinned_shadow"]
                DrawModel(self.geno_model, self.left_model_offset, 1.0, WHITE)
                if self.compare_mode:
                    self.compare_model.materials[0].shader = self.shaders["skinned_shadow"]
                    DrawModel(self.compare_model, self.right_model_offset, 1.0, WHITE)
                end_shadow_map()

                begin_gbuffer(self.gbuffer, self.camera.cam3d)
                cam_view = rlGetMatrixModelview()
                cam_proj = rlGetMatrixProjection()
                cam_inv_proj = MatrixInvert(cam_proj)
                cam_inv_view_proj = MatrixInvert(MatrixMultiply(cam_view, cam_proj))
                cam_clip_near = rlGetCullDistanceNear()
                cam_clip_far = rlGetCullDistanceFar()
                cam_clip_near_ptr = ffi.new("float*")
                cam_clip_far_ptr = ffi.new("float*")
                cam_clip_near_ptr[0] = cam_clip_near
                cam_clip_far_ptr[0] = cam_clip_far
                specularity_ptr = ffi.new("float*")
                glossiness_ptr = ffi.new("float*")
                specularity_ptr[0] = 0.5
                glossiness_ptr[0] = 10.0

                SetShaderValue(self.shaders["basic"], self.shader_locs["basic_specularity"], specularity_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["basic"], self.shader_locs["basic_glossiness"], glossiness_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["basic"], self.shader_locs["basic_cam_clip_near"], cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["basic"], self.shader_locs["basic_cam_clip_far"], cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(
                    self.shaders["skinned_basic"],
                    self.shader_locs["skinned_basic_specularity"],
                    specularity_ptr,
                    SHADER_UNIFORM_FLOAT,
                )
                SetShaderValue(
                    self.shaders["skinned_basic"],
                    self.shader_locs["skinned_basic_glossiness"],
                    glossiness_ptr,
                    SHADER_UNIFORM_FLOAT,
                )
                SetShaderValue(
                    self.shaders["skinned_basic"],
                    self.shader_locs["skinned_basic_cam_clip_near"],
                    cam_clip_near_ptr,
                    SHADER_UNIFORM_FLOAT,
                )
                SetShaderValue(
                    self.shaders["skinned_basic"],
                    self.shader_locs["skinned_basic_cam_clip_far"],
                    cam_clip_far_ptr,
                    SHADER_UNIFORM_FLOAT,
                )

                self.ground_model.materials[0].shader = self.shaders["basic"]
                DrawModel(self.ground_model, self.ground_position, 1.0, Color(190, 190, 190, 255))
                self.geno_model.materials[0].shader = self.shaders["skinned_basic"]
                DrawModel(self.geno_model, self.left_model_offset, 1.0, Color(70, 125, 255, 255) if self.compare_mode else ORANGE)
                if self.compare_mode:
                    self.compare_model.materials[0].shader = self.shaders["skinned_basic"]
                    DrawModel(self.compare_model, self.right_model_offset, 1.0, ORANGE)
                end_gbuffer(screen_width, screen_height)

                BeginTextureMode(self.ssao_front)
                BeginShaderMode(self.shaders["ssao"])
                SetShaderValueTexture(self.shaders["ssao"], self.shader_locs["ssao_gbuffer_normal"], self.gbuffer.normal)
                SetShaderValueTexture(self.shaders["ssao"], self.shader_locs["ssao_gbuffer_depth"], self.gbuffer.depth)
                SetShaderValueMatrix(self.shaders["ssao"], self.shader_locs["ssao_cam_view"], cam_view)
                SetShaderValueMatrix(self.shaders["ssao"], self.shader_locs["ssao_cam_proj"], cam_proj)
                SetShaderValueMatrix(self.shaders["ssao"], self.shader_locs["ssao_cam_inv_proj"], cam_inv_proj)
                SetShaderValueMatrix(self.shaders["ssao"], self.shader_locs["ssao_cam_inv_view_proj"], cam_inv_view_proj)
                SetShaderValueMatrix(self.shaders["ssao"], self.shader_locs["ssao_light_view_proj"], light_view_proj)
                set_shader_value_shadow_map(self.shaders["ssao"], self.shader_locs["ssao_shadow_map"], self.shadow_map)
                SetShaderValue(
                    self.shaders["ssao"],
                    self.shader_locs["ssao_shadow_inv_resolution"],
                    ffi.addressof(self.shadow_inv_resolution),
                    SHADER_UNIFORM_VEC2,
                )
                SetShaderValue(self.shaders["ssao"], self.shader_locs["ssao_cam_clip_near"], cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["ssao"], self.shader_locs["ssao_cam_clip_far"], cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["ssao"], self.shader_locs["ssao_light_clip_near"], light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["ssao"], self.shader_locs["ssao_light_clip_far"], light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["ssao"], self.shader_locs["ssao_light_dir"], ffi.addressof(self.light_dir), SHADER_UNIFORM_VEC3)

                ClearBackground(WHITE)
                DrawTextureRec(
                    self.ssao_front.texture,
                    Rectangle(0, 0, self.ssao_front.texture.width, -self.ssao_front.texture.height),
                    Vector2(0.0, 0.0),
                    WHITE,
                )
                EndShaderMode()
                EndTextureMode()

                BeginTextureMode(self.ssao_back)
                BeginShaderMode(self.shaders["blur"])
                blur_direction = Vector2(1.0, 0.0)
                blur_inv_texture_resolution = Vector2(1.0 / self.ssao_front.texture.width, 1.0 / self.ssao_front.texture.height)
                SetShaderValueTexture(self.shaders["blur"], self.shader_locs["blur_gbuffer_normal"], self.gbuffer.normal)
                SetShaderValueTexture(self.shaders["blur"], self.shader_locs["blur_gbuffer_depth"], self.gbuffer.depth)
                SetShaderValueTexture(self.shaders["blur"], self.shader_locs["blur_input_texture"], self.ssao_front.texture)
                SetShaderValueMatrix(self.shaders["blur"], self.shader_locs["blur_cam_inv_proj"], cam_inv_proj)
                SetShaderValue(self.shaders["blur"], self.shader_locs["blur_cam_clip_near"], cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["blur"], self.shader_locs["blur_cam_clip_far"], cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(
                    self.shaders["blur"],
                    self.shader_locs["blur_inv_texture_resolution"],
                    ffi.addressof(blur_inv_texture_resolution),
                    SHADER_UNIFORM_VEC2,
                )
                SetShaderValue(
                    self.shaders["blur"],
                    self.shader_locs["blur_direction"],
                    ffi.addressof(blur_direction),
                    SHADER_UNIFORM_VEC2,
                )
                DrawTextureRec(
                    self.ssao_back.texture,
                    Rectangle(0, 0, self.ssao_back.texture.width, -self.ssao_back.texture.height),
                    Vector2(0, 0),
                    WHITE,
                )
                EndShaderMode()
                EndTextureMode()

                BeginTextureMode(self.ssao_front)
                BeginShaderMode(self.shaders["blur"])
                blur_direction = Vector2(0.0, 1.0)
                SetShaderValueTexture(self.shaders["blur"], self.shader_locs["blur_input_texture"], self.ssao_back.texture)
                SetShaderValue(
                    self.shaders["blur"],
                    self.shader_locs["blur_direction"],
                    ffi.addressof(blur_direction),
                    SHADER_UNIFORM_VEC2,
                )
                DrawTextureRec(
                    self.ssao_front.texture,
                    Rectangle(0, 0, self.ssao_front.texture.width, -self.ssao_front.texture.height),
                    Vector2(0, 0),
                    WHITE,
                )
                EndShaderMode()
                EndTextureMode()

                BeginTextureMode(self.lighted)
                BeginShaderMode(self.shaders["lighting"])
                sun_color = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
                sun_strength_ptr = ffi.new("float*")
                sky_strength_ptr = ffi.new("float*")
                ground_strength_ptr = ffi.new("float*")
                ambient_strength_ptr = ffi.new("float*")
                exposure_ptr = ffi.new("float*")
                sun_strength_ptr[0] = 0.25
                sky_color = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
                sky_strength_ptr[0] = 0.15
                ground_strength_ptr[0] = 0.1
                ambient_strength_ptr[0] = 1.0
                exposure_ptr[0] = 0.9

                SetShaderValueTexture(self.shaders["lighting"], self.shader_locs["lighting_gbuffer_color"], self.gbuffer.color)
                SetShaderValueTexture(self.shaders["lighting"], self.shader_locs["lighting_gbuffer_normal"], self.gbuffer.normal)
                SetShaderValueTexture(self.shaders["lighting"], self.shader_locs["lighting_gbuffer_depth"], self.gbuffer.depth)
                SetShaderValueTexture(self.shaders["lighting"], self.shader_locs["lighting_ssao"], self.ssao_front.texture)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_cam_pos"], ffi.addressof(self.camera.cam3d.position), SHADER_UNIFORM_VEC3)
                SetShaderValueMatrix(self.shaders["lighting"], self.shader_locs["lighting_cam_inv_view_proj"], cam_inv_view_proj)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_light_dir"], ffi.addressof(self.light_dir), SHADER_UNIFORM_VEC3)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_sun_color"], ffi.addressof(sun_color), SHADER_UNIFORM_VEC3)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_sun_strength"], sun_strength_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_sky_color"], ffi.addressof(sky_color), SHADER_UNIFORM_VEC3)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_sky_strength"], sky_strength_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_ground_strength"], ground_strength_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_ambient_strength"], ambient_strength_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_exposure"], exposure_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_cam_clip_near"], cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
                SetShaderValue(self.shaders["lighting"], self.shader_locs["lighting_cam_clip_far"], cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)

                ClearBackground(RAYWHITE)
                DrawTextureRec(
                    self.gbuffer.color,
                    Rectangle(0, 0, self.gbuffer.color.width, -self.gbuffer.color.height),
                    Vector2(0, 0),
                    WHITE,
                )
                EndShaderMode()

                if self.tpos is not None and self.tdir is not None:
                    BeginMode3D(self.camera.cam3d)
                    draw_trajectory(global_pos[0], global_rot[0], self.tpos[self.sample_index], self.tdir[self.sample_index])
                    EndMode3D()
                EndTextureMode()

                BeginShaderMode(self.shaders["fxaa"])
                fxaa_inv_texture_resolution = Vector2(1.0 / self.lighted.texture.width, 1.0 / self.lighted.texture.height)
                SetShaderValueTexture(self.shaders["fxaa"], self.shader_locs["fxaa_input_texture"], self.lighted.texture)
                SetShaderValue(
                    self.shaders["fxaa"],
                    self.shader_locs["fxaa_inv_texture_resolution"],
                    ffi.addressof(fxaa_inv_texture_resolution),
                    SHADER_UNIFORM_VEC2,
                )
                DrawTextureRec(
                    self.lighted.texture,
                    Rectangle(0, 0, self.lighted.texture.width, -self.lighted.texture.height),
                    Vector2(0, 0),
                    WHITE,
                )
                EndShaderMode()

                rlEnableColorBlend()
                DrawFPS(10, 10)
                status = "Paused" if not self.playback.playing else f"Playing {self.playback.current_speed:.2f}x"
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
                self.playback.draw_ui(screen_width, screen_height, "Sample" if self.indices is not None else "Frame")
                EndDrawing()
        finally:
            self._cleanup()
            CloseWindow()


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
        )


def main():
    parser = argparse.ArgumentParser(description="High-quality Geno viewer driven by database.npz or 230D motion features.")
    parser.add_argument("--database", type=Path, default=None, help="Path to database.npz")
    parser.add_argument("--features", type=Path, default=None, help="Path to .npy or .npz containing 230D features with shape [T, D].")
    parser.add_argument("--feature-key", type=str, default="motion", help="Array key for .npz feature input.")
    parser.add_argument("--stats-source", type=Path, default=None, help="Checkpoint .pt, metadata.npz, or feature_database directory containing feature stats.")
    parser.add_argument("--normalized", action="store_true", help="Treat --features as normalized feature values.")
    parser.add_argument("--range-name", type=str, default="features", help="Range name used in feature visualization mode.")
    parser.add_argument("--root-position0", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--root-rotation0", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional path to trajectory.npz")
    parser.add_argument(
        "--resources-root",
        type=Path,
        default=Path(__file__).resolve().parent / "resources",
        help="Directory containing Geno.bin and shader files",
    )
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS")
    args = parser.parse_args()

    if (args.database is None) == (args.features is None):
        raise ValueError("Exactly one of --database or --features is required")
    if args.features is not None and args.stats_source is None:
        raise ValueError("--stats-source is required when using --features")

    database = (
        load_database_dict(args.database)
        if args.database is not None
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
    )
    viewer.run()


if __name__ == "__main__":
    main()
