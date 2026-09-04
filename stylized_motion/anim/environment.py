"""Procedural environment resources used by the deferred PBR pass."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

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


# Cubemap face axes for CUBEMAP_LAYOUT_LINE_HORIZONTAL (+X,-X,+Y,-Y,+Z,-Z).
_FACE_AXES = (
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, -1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
)


def _procedural_sky_array(face_size: int = 32) -> np.ndarray:
    """Source environment for the IBL chain: a linear-radiance sky dome.

    Values are linear radiance (not display sRGB), consumed as-is by the
    lighting pass — the background and the IBL terms must agree on that
    contract. Bright blue zenith (~HDR, B>1) into a pale warm horizon with a
    darker neutral ground bounce.
    """

    zenith = np.array([0.58, 0.74, 1.32], dtype=np.float32)
    horizon = np.array([0.92, 0.96, 1.06], dtype=np.float32)
    ground = np.array([0.42, 0.43, 0.46], dtype=np.float32)
    faces = np.empty((6, face_size, face_size, 3), dtype=np.float32)
    for face in range(6):
        ys, xs = np.mgrid[0:face_size, 0:face_size]
        u = (xs + 0.5) / face_size * 2.0 - 1.0
        v = (ys + 0.5) / face_size * 2.0 - 1.0
        right, up, forward = (np.asarray(axis, dtype=np.float32) for axis in _FACE_AXES[face])
        d = right[None, None] * u[..., None] + up[None, None] * v[..., None] + forward
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        height = d[..., 1]
        sky_mix = np.clip(height, 0.0, 1.0)[..., None]
        ground_mix = np.clip(-height, 0.0, 1.0)[..., None]
        faces[face] = horizon * (1.0 - sky_mix) * (1.0 - ground_mix) + zenith * sky_mix + ground * ground_mix
    return faces


def _array_to_cubemap(faces: np.ndarray) -> Texture:
    """Upload a (6, size, size, 3) linear-radiance array as a cubemap."""
    return _array_to_cubemap_mipped([faces])


def _array_to_cubemap_mipped(levels: list[np.ndarray]) -> Texture:
    """Upload prefiltered roughness levels as one cubemap with a manual mip chain.

    rlLoadTextureCubemap expects the byte stream as: for each mip level
    (base first), the six square faces (+X,-X,+Y,-Y,+Z,-Z) as contiguous
    s_k x s_k RGBA16F blocks; the level sizes halve per mip. Going through the
    image path instead cannot inject custom per-level content (LoadCubemap
    only converts the base level and re-derives the rest), so the raw data
    pointer path is used here.
    """

    mip_count = len(levels)
    assert mip_count >= 1
    size = levels[0].shape[1]
    chunks = []
    for faces in levels:
        level_size = faces.shape[1]
        for face in range(6):
            rgba = np.empty((level_size, level_size, 4), dtype=np.float16)
            rgba[..., :3] = np.clip(faces[face], 0.0, 65504.0)
            rgba[..., 3] = np.float16(1.0)
            chunks.append(np.ascontiguousarray(rgba).tobytes())
    data = b"".join(chunks)

    texture_id = rlLoadTextureCubemap(data, size, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, mip_count)
    assert texture_id != 0, "rlLoadTextureCubemap failed for the prefilter chain"
    texture = Texture()
    texture.id = texture_id
    texture.width = size
    texture.height = size
    texture.mipmaps = mip_count
    texture.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    return texture


def _cubemap_to_array(texture: Texture, face_size: int) -> np.ndarray:
    """Read a line-horizontal cubemap back into a (6, size, size, 3) array."""
    image = LoadImageFromTexture(texture)
    try:
        pixels = np.frombuffer(ffi.buffer(image.data, image.width * image.height * 4), dtype=np.uint8)
        pixels = pixels.reshape(image.height, image.width, 4).astype(np.float32) / 255.0
    finally:
        UnloadImage(image)
    faces = np.empty((6, face_size, face_size, 3), dtype=np.float32)
    for face in range(6):
        faces[face] = pixels[:, face * face_size : (face + 1) * face_size, :3]
    return faces


def _build_env_lookup(environment: np.ndarray, face_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (dirs, colors) lookup over every environment texel."""
    dirs = np.empty((6, face_size, face_size, 3), dtype=np.float32)
    for face in range(6):
        ys, xs = np.mgrid[0:face_size, 0:face_size]
        u = (xs + 0.5) / face_size * 2.0 - 1.0
        v = (ys + 0.5) / face_size * 2.0 - 1.0
        right, up, forward = (np.asarray(axis, dtype=np.float32) for axis in _FACE_AXES[face])
        d = right[None, None] * u[..., None] + up[None, None] * v[..., None] + forward
        dirs[face] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return dirs.reshape(-1, 3), environment.reshape(-1, 3)


def _sample_env(env_dirs: np.ndarray, env_colors: np.ndarray, light_dirs: np.ndarray) -> np.ndarray:
    """Nearest-texel environment color for each light direction."""
    # Chunked argmax over the (pixels x samples) dot to bound memory.
    best = np.empty(light_dirs.shape[0], dtype=np.int64)
    chunk = 256
    for start in range(0, light_dirs.shape[0], chunk):
        block = light_dirs[start : start + chunk]
        dots = env_dirs @ block.T
        best[start : start + chunk] = np.argmax(dots, axis=0)
    return env_colors[best]


def _convolve_irradiance(environment: np.ndarray, face_size: int, output_size: int = 16) -> np.ndarray:
    """Cosine-weighted hemisphere convolution (diffuse IBL).

    irradiance(n) = sum_i env(l_i) * max(n·l_i, 0) / sum_i max(n·l_i, 0)
    over a quasi-uniform Fibonacci hemisphere; this is the Lambert integral
    the lighting pass needs, evaluated once per output texel.
    """

    sample_count = 2048
    indices = np.arange(sample_count, dtype=np.float32)
    golden = 0.5 * (1.0 + np.sqrt(5.0))
    # Uniform sphere: cos(theta) uniform in [-1, 1] (the 2x stretch). Without
    # it the samples only cover the upper hemisphere and downward normals
    # convolve to numerical garbage.
    theta = np.arccos(np.clip(1.0 - 2.0 * (indices + 0.5) / sample_count, -1.0, 1.0))
    phi = 2.0 * np.pi * golden * indices
    light_dirs = np.stack(
        (np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)), axis=1
    ).astype(np.float32)

    env_dirs, env_colors = _build_env_lookup(environment, face_size)
    sampled = _sample_env(env_dirs, env_colors, light_dirs)

    irradiance = np.empty((6, output_size, output_size, 3), dtype=np.float32)
    for face in range(6):
        ys, xs = np.mgrid[0:output_size, 0:output_size]
        u = (xs + 0.5) / output_size * 2.0 - 1.0
        v = (ys + 0.5) / output_size * 2.0 - 1.0
        right, up, forward = (np.asarray(axis, dtype=np.float32) for axis in _FACE_AXES[face])
        n = right[None, None] * u[..., None] + up[None, None] * v[..., None] + forward
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        weights = np.clip(n.reshape(-1, 1, 3) @ light_dirs.T, 0.0, None).reshape(-1, sample_count)
        denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
        irradiance[face] = (weights @ sampled / denom).reshape(output_size, output_size, 3)
    return irradiance


def _prefilter_environment(environment: np.ndarray, face_size: int, output_size: int = 32, mips: int = 6) -> list[np.ndarray]:
    """GGX importance-sampled prefilter mips (split-sum, spec 17/18).

    prefilter(r, n) = mean_i env(l_i) where l_i = reflect about H_i drawn from
    the GGX NDF for roughness r. The split-sum trick (Karis 13) needs no pdf
    weight here because the Fresnel/visibility terms live in the BRDF LUT.

    Performance: the environment is consulted through a pre-downsampled 8^2
    per-face proxy (384 texels) instead of the full 32^2 map — the IS noise
    at 256 samples dominates that quantization — which keeps the whole chain
    to a couple of seconds of one-off init CPU.
    """

    proxy_size = 8
    proxy = _downsample_faces(environment, face_size, proxy_size)
    sample_count = 256
    indices = np.arange(sample_count, dtype=np.float32)
    # Van der Corput radical inverse: reverse the (up to) 16 significant bits
    # of the index. The 32-bit-wide swap pair first is required; a second
    # 16-bit swap would undo the reversal and collapse xi to ~0 for small N.
    bits = indices.astype(np.uint32)
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    xi_y = bits.astype(np.float32) * 2.3283064365386963e-10
    xi_x = (indices + 0.5) / sample_count

    env_dirs, env_colors = _build_env_lookup(proxy, proxy_size)

    mips_out = []
    for mip in range(mips):
        roughness = mip / (mips - 1)
        # GL mip chains halve per level; render each level at its native size
        # so the uploaded chain is a valid complete texture.
        level_size = max(output_size >> mip, 1)
        alpha = max(roughness * roughness, 2e-3)
        alpha2 = alpha * alpha
        cos_theta = np.sqrt((1.0 - xi_y) / (1.0 + (alpha2 - 1.0) * xi_y)).astype(np.float32)
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
        phi = 2.0 * np.pi * xi_x
        half_dirs = np.stack(
            (sin_theta * np.cos(phi), cos_theta, sin_theta * np.sin(phi)), axis=1
        ).astype(np.float32)

        # H in tangent space (x=right, y=up/normal, z=bitangent) per output texel.
        prefiltered = np.empty((6, level_size, level_size, 3), dtype=np.float32)
        for face in range(6):
            ys, xs = np.mgrid[0:level_size, 0:level_size]
            u = (xs + 0.5) / level_size * 2.0 - 1.0
            v = (ys + 0.5) / level_size * 2.0 - 1.0
            right, up, forward = (np.asarray(axis, dtype=np.float32) for axis in _FACE_AXES[face])
            n = right[None, None] * u[..., None] + up[None, None] * v[..., None] + forward
            n = n / np.linalg.norm(n, axis=-1, keepdims=True)
            n_flat = n.reshape(-1, 3)
            # Tangent frame: pick an orthogonal helper axis.
            helper = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n_flat.shape[0], 1))
            flip = np.abs(n_flat[:, 1]) > 0.999
            helper[flip] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            t = np.cross(helper, n_flat)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
            b = np.cross(n_flat, t)
            # l = 2(n·h)h - n for each (texel, sample) pair, vectorized in chunks.
            chunk = 4096
            acc = np.empty((n_flat.shape[0], 3), dtype=np.float32)
            for start in range(0, n_flat.shape[0], chunk):
                nf = n_flat[start : start + chunk]                       # (p,3)
                tf = t[start : start + chunk]                            # (p,3)
                bf = b[start : start + chunk]                            # (p,3)
                # h_world: (p, s, 3)
                hw = (
                    half_dirs[None, :, 0:1] * tf[:, None, :]
                    + half_dirs[None, :, 1:2] * nf[:, None, :]
                    + half_dirs[None, :, 2:3] * bf[:, None, :]
                )
                hw /= np.maximum(np.linalg.norm(hw, axis=2, keepdims=True), 1e-8)
                l = 2.0 * np.einsum("psc,pc->ps", hw, nf)[..., None] * hw - nf[:, None, :]
                l /= np.maximum(np.linalg.norm(l, axis=2, keepdims=True), 1e-8)
                colors = _sample_env(env_dirs, env_colors, l.reshape(-1, 3)).reshape(nf.shape[0], -1, 3)
                acc[start : start + chunk] = colors.mean(axis=1)
            prefiltered[face] = acc.reshape(level_size, level_size, 3)
        mips_out.append(prefiltered)
    return mips_out


def _downsample_faces(faces: np.ndarray, source_size: int, target_size: int) -> np.ndarray:
    """Box-downsample a (6, s, s, 3) cubemap array to (6, t, t, 3)."""
    step = source_size // target_size
    return faces.reshape(6, target_size, step, target_size, step, 3).mean(axis=(2, 4))


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


def _ibl_cache_path(sky: np.ndarray) -> Path:
    """Cache file path keyed by the source sky content and generator version."""
    import hashlib
    import os

    digest = hashlib.sha256(np.ascontiguousarray(sky).tobytes()).hexdigest()[:16]
    base = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    cache_dir = Path(base) / "stylized_motion" / "ibl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"procedural_sky_{digest}_v2.npz"


def _load_or_build_ibl_arrays(sky: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Disk cache around the irradiance + prefilter CPU integration."""
    cache_path = _ibl_cache_path(sky)
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                irradiance = data["irradiance"]
                levels = [data[f"level_{i}"] for i in range(int(data["level_count"]))]
            return irradiance, levels
        except Exception:
            pass  # Corrupt cache: rebuild below.
    irradiance = _convolve_irradiance(sky, sky.shape[1], output_size=16)
    levels = _prefilter_environment(sky, sky.shape[1], output_size=32, mips=6)
    try:
        payload = {"irradiance": irradiance, "level_count": len(levels)}
        payload.update({f"level_{i}": level for i, level in enumerate(levels)})
        np.savez(cache_path, **payload)
    except Exception:
        pass  # Cache write is best-effort; the textures are already built.
    return irradiance, levels


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
        # Full split-sum chain from one procedural sky: environment cubemap,
        # cosine-convolved irradiance, GGX importance-sampled prefilter mips,
        # and the CPU-integrated BRDF LUT. The face arrays are cached on disk
        # keyed by the source sky, so repeat startups skip the CPU integration.
        sky = _procedural_sky_array(32)
        irradiance_faces, prefilter_faces = _load_or_build_ibl_arrays(sky)
        self.environment = _array_to_cubemap(sky)
        GenTextureMipmaps(ffi.addressof(self.environment))
        self.irradiance = _array_to_cubemap(irradiance_faces)
        # One cubemap whose mip chain *is* the roughness sequence, so the
        # shader's textureLod(prefilterMap, r, roughness * maxLod) samples
        # exact importance-sampled levels instead of box-filtered mips.
        self.prefilter = _array_to_cubemap_mipped(prefilter_faces)
        self.prefilter_max_lod = 5.0
        lut_image = _integrate_brdf_lut()
        self.brdf_lut = LoadTextureFromImage(lut_image)
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
