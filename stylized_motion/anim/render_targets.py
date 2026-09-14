"""Render-target allocation and pass-boundary helpers."""

from __future__ import annotations

import cffi
from pathlib import Path
import numpy as np
from pyray import RenderTexture, Texture
from raylib import *
from raylib.defines import *


ffi = cffi.FFI()


class GBuffer:
    def __init__(self):
        self.id = 0
        self.color = Texture()
        self.normal = Texture()
        self.material_ao = Texture()
        self.depth = Texture()


def load_color_target(width, height, pixel_format):
    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    rlEnableFramebuffer(target.id)
    target.texture.id = rlLoadTexture(ffi.NULL, width, height, pixel_format, 1)
    target.texture.width = width
    target.texture.height = height
    target.texture.format = pixel_format
    target.texture.mipmaps = 1
    rlFramebufferAttach(target.id, target.texture.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)
    rlDisableFramebuffer()
    return target


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
        _unload_texture(target.depth)
        rlUnloadFramebuffer(target.id)
        target.id = 0


def _unload_texture(texture: Texture) -> None:
    """Release a manually-owned texture once and mark its handle invalid."""
    if texture.id > 0:
        rlUnloadTexture(texture.id)
        texture.id = 0


def unload_color_target(target: RenderTexture) -> None:
    """Release a color-only framebuffer created by :func:`load_color_target`."""
    if target.id > 0:
        _unload_texture(target.texture)
        rlUnloadFramebuffer(target.id)
        target.id = 0


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


def set_shader_value_shadow_map(shader, loc_index, target, slot_ptr):
    if loc_index > -1:
        rlEnableShader(shader.id)
        rlActiveTextureSlot(slot_ptr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(loc_index, slot_ptr, SHADER_UNIFORM_INT, 1)


def set_shader_value_texture_slot(shader, loc_index, texture, slot_ptr):
    """Bind a 2D texture to an explicit texture slot.

    raylib's SetShaderValueTexture keys samplers off texture.id, which collides
    with the manually managed slots (shadow maps 10-12, environment 13-15,
    BRDF LUT 16) whenever a render target's GL texture name lands on those
    numbers. Multi-texture passes must bind their textures to dedicated slots
    instead.
    """
    if loc_index > -1:
        rlEnableShader(shader.id)
        rlActiveTextureSlot(slot_ptr[0])
        rlEnableTexture(texture.id)
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
    # Phase 4 revision: a dedicated R8 attachment for baked per-material AO.
    # SSAO never enters the GBuffer; this channel is the material-level
    # occlusion term that scales indirect light in the lighting pass.
    target.material_ao.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, 1)
    target.material_ao.width = width
    target.material_ao.height = height
    target.material_ao.format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
    target.material_ao.mipmaps = 1
    rlFramebufferAttach(target.id, target.material_ao.id, RL_ATTACHMENT_COLOR_CHANNEL2, RL_ATTACHMENT_TEXTURE2D, 0)
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
        _unload_texture(target.color)
        _unload_texture(target.normal)
        _unload_texture(target.material_ao)
        _unload_texture(target.depth)
        rlUnloadFramebuffer(target.id)
        target.id = 0


def export_render_target(target: RenderTexture, output_path: Path) -> None:
    """Write a render target using raylib's established texture readback path."""
    image = LoadImageFromTexture(target.texture)
    try:
        # Render-target memory has the OpenGL origin; files and the window
        # framebuffer use a top-left origin. Keep every export path upright.
        ImageFlipVertical(ffi.addressof(image))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ExportImage(image, str(output_path).encode("utf-8"))
    finally:
        UnloadImage(image)


def begin_gbuffer(target, camera):
    rlDrawRenderBatchActive()
    rlEnableFramebuffer(target.id)
    rlActiveDrawBuffers(3)
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


class RenderTargets:
    """Own all screen-sized render resources for one viewer instance."""

    def __init__(self, width: int, height: int, shading: str, shadow_resolution: int = 1024):
        self.width = width
        self.height = height
        self.shading = shading
        self.shadow_resolution = shadow_resolution
        self.shadow_maps = []
        self.shadow_map = None
        self.gbuffer = None
        self.lighting = None
        self.tonemapped = None
        self.final = None
        self.ssao_front = None
        self.ssao_back = None

    def initialize(self) -> "RenderTargets":
        shadow_count = 1
        self.shadow_maps = [
            load_shadow_map(self.shadow_resolution, self.shadow_resolution)
            for _ in range(shadow_count)
        ]
        self.shadow_map = self.shadow_maps[0]
        self.gbuffer = load_gbuffer(self.width, self.height)
        self.lighting = (
            load_color_target(self.width, self.height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16)
            if self.shading == "pbr"
            else LoadRenderTexture(self.width, self.height)
        )
        self.tonemapped = LoadRenderTexture(self.width, self.height) if self.shading == "pbr" else None
        # This is the exact image shown in the window: FXAA is already applied
        # for normal views, while debug views intentionally remain unfiltered.
        self.final = LoadRenderTexture(self.width, self.height)
        self.ssao_front = LoadRenderTexture(self.width, self.height)
        self.ssao_back = LoadRenderTexture(self.width, self.height)
        return self

    def cleanup(self) -> None:
        if self.lighting is not None:
            if self.shading == "pbr":
                unload_color_target(self.lighting)
            else:
                UnloadRenderTexture(self.lighting)
            self.lighting = None
        if self.tonemapped is not None:
            UnloadRenderTexture(self.tonemapped)
            self.tonemapped = None
        if self.final is not None:
            UnloadRenderTexture(self.final)
            self.final = None
        if self.ssao_back is not None:
            UnloadRenderTexture(self.ssao_back)
            self.ssao_back = None
        if self.ssao_front is not None:
            UnloadRenderTexture(self.ssao_front)
            self.ssao_front = None
        if self.gbuffer is not None:
            unload_gbuffer(self.gbuffer)
            self.gbuffer = None
        for shadow_map in self.shadow_maps:
            unload_shadow_map(shadow_map)
        self.shadow_maps = []
        self.shadow_map = None
