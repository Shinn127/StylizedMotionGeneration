"""Procedural environment resources used by the deferred PBR pass."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import cffi
import numpy as np
from pyray import Color, Image, Texture
from raylib import *
from raylib.defines import *


ffi = cffi.FFI()


def _cubemap_atlas(face_size: int, colors: tuple[Color, ...]):
    image = GenImageColor(face_size * 6, face_size, colors[0])
    for face, color in enumerate(colors[1:], 1):
        ImageDrawRectangle(ffi.addressof(image), face * face_size, 0, face_size, face_size, color)
    return image


def _load_cubemap(face_size: int, colors: tuple[Color, ...]) -> Texture:
    image = _cubemap_atlas(face_size, colors)
    texture = LoadTextureCubemap(image, CUBEMAP_LAYOUT_LINE_HORIZONTAL)
    UnloadImage(image)
    return texture


def _integrate_brdf_lut(size: int = 128, samples: int = 128) -> Image:
    """Generate the GGX split-sum BRDF integration texture on the CPU."""
    pixels = np.empty((size, size, 4), dtype=np.uint8)
    sample_index = np.arange(samples, dtype=np.float32)
    bits = sample_index.astype(np.uint32)
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    radical_inverse = bits.astype(np.float32) * 2.3283064365386963e-10
    xi_x = (sample_index + 0.5) / samples
    xi_y = radical_inverse

    for y in range(size):
        roughness = (y + 0.5) / size
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        for x in range(size):
            n_dot_v = (x + 0.5) / size
            v = np.array([sqrt(max(1.0 - n_dot_v * n_dot_v, 0.0)), 0.0, n_dot_v], dtype=np.float32)
            phi = 2.0 * pi * xi_x
            cos_theta = np.sqrt((1.0 - xi_y) / (1.0 + (alpha2 - 1.0) * xi_y))
            sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
            h = np.stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=1)
            v_dot_h = np.maximum(h @ v, 0.0)
            l = 2.0 * v_dot_h[:, None] * h - v
            n_dot_l = np.maximum(l[:, 2], 0.0)
            n_dot_h = np.maximum(h[:, 2], 0.0)
            valid = n_dot_l > 0.0
            k = (roughness * roughness) * 0.5
            g_v = n_dot_v / (n_dot_v * (1.0 - k) + k)
            g_l = n_dot_l / (n_dot_l * (1.0 - k) + k)
            g_vis = (g_v * g_l * v_dot_h) / np.maximum(n_dot_h * n_dot_v, 1e-5)
            fresnel = np.power(1.0 - v_dot_h, 5.0)
            a = np.sum(np.where(valid, (1.0 - fresnel) * g_vis, 0.0)) / samples
            b = np.sum(np.where(valid, fresnel * g_vis, 0.0)) / samples
            pixels[y, x, 0] = int(np.clip(a, 0.0, 1.0) * 255.0 + 0.5)
            pixels[y, x, 1] = int(np.clip(b, 0.0, 1.0) * 255.0 + 0.5)
            pixels[y, x, 2] = 0
            pixels[y, x, 3] = 255

    image = Image()
    image.data = ffi.from_buffer("unsigned char[]", pixels)
    image.width = size
    image.height = size
    image.mipmaps = 1
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    return image


@dataclass
class IBLResources:
    """One viewer's environment, irradiance, prefilter and BRDF resources."""

    strength: float = 0.35
    enabled: bool = True
    environment: Texture | None = None
    irradiance: Texture | None = None
    prefilter: Texture | None = None
    brdf_lut: Texture | None = None
    prefilter_max_lod: float = 0.0

    def initialize(self) -> "IBLResources":
        # Sky-driven palettes: cool blue-biased upper hemisphere, darker and
        # neutral below the horizon, so indirect light carries the same
        # directional temperature cue as a real outdoor environment.
        self.environment = _load_cubemap(
            32,
            (
                Color(184, 192, 202, 255),
                Color(172, 180, 190, 255),
                Color(170, 190, 216, 255),
                Color(104, 108, 112, 255),
                Color(178, 186, 196, 255),
                Color(170, 178, 188, 255),
            ),
        )
        GenTextureMipmaps(ffi.addressof(self.environment))
        self.irradiance = _load_cubemap(
            16,
            (
                Color(152, 166, 186, 255),
                Color(148, 162, 182, 255),
                Color(152, 174, 206, 255),
                Color(96, 100, 106, 255),
                Color(150, 164, 184, 255),
                Color(146, 160, 180, 255),
            ),
        )
        self.prefilter = _load_cubemap(
            32,
            (
                Color(180, 188, 198, 255),
                Color(168, 176, 186, 255),
                Color(164, 184, 210, 255),
                Color(100, 104, 108, 255),
                Color(174, 182, 192, 255),
                Color(166, 174, 184, 255),
            ),
        )
        GenTextureMipmaps(ffi.addressof(self.prefilter))
        self.prefilter_max_lod = 5.0
        lut_image = _integrate_brdf_lut()
        self.brdf_lut = LoadTextureFromImage(lut_image)
        GenTextureMipmaps(ffi.addressof(self.brdf_lut))
        lut_image.data = ffi.NULL
        UnloadImage(lut_image)
        return self

    def cleanup(self) -> None:
        UnloadTexture(self.brdf_lut)
        UnloadTexture(self.prefilter)
        UnloadTexture(self.irradiance)
        UnloadTexture(self.environment)
        self.brdf_lut = None
        self.prefilter = None
        self.irradiance = None
        self.environment = None


def set_shader_value_cubemap(shader, loc_index, texture: Texture, slot_ptr) -> None:
    rlEnableShader(shader.id)
    rlActiveTextureSlot(slot_ptr[0])
    rlEnableTextureCubemap(texture.id)
    rlSetUniform(loc_index, slot_ptr, SHADER_UNIFORM_INT, 1)
