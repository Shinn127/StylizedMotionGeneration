"""Material data and texture conventions for the deferred renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cffi
from pyray import Texture
from raylib import GenTextureMipmaps, LoadTexture


ffi = cffi.FFI()


TEXTURE_CONTRACT = {
    "base_color_map": {"color_space": "sRGB", "channels": ("base_color",)},
    "metallic_roughness_map": {
        "color_space": "linear",
        "channels": ("metallic", "roughness", "ao"),
    },
    "normal_map": {"color_space": "linear", "channels": ("normal_x", "normal_y")},
}


@dataclass
class Material:
    """Opaque material inputs consumed by the GBuffer pass.

    Texture fields intentionally remain raylib ``Texture`` values so the
    viewer can bind them without introducing a second resource wrapper.
    ``normal_map`` is part of the contract; tangent-space application is a
    later normal-mapping phase.
    """

    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.58
    ao: float = 1.0
    base_color_map: Texture | None = None
    normal_map: Texture | None = None
    metallic_roughness_map: Texture | None = None

    def __post_init__(self):
        self.metallic = min(max(float(self.metallic), 0.0), 1.0)
        self.roughness = min(max(float(self.roughness), 0.04), 1.0)
        self.ao = min(max(float(self.ao), 0.0), 1.0)
        self.base_color = tuple(float(channel) for channel in self.base_color)

    @property
    def uses_base_color_map(self) -> bool:
        return self.base_color_map is not None

    @property
    def uses_metallic_roughness_map(self) -> bool:
        return self.metallic_roughness_map is not None

    @property
    def uses_normal_map(self) -> bool:
        return self.normal_map is not None


def load_material_texture(path: Path, semantic: str) -> Texture:
    """Load a material texture and generate its sampling mip chain.

    ``semantic`` is kept explicit at the call site to make the color-space
    contract visible; GPU upload itself is handled by raylib.
    """

    TEXTURE_CONTRACT[semantic]
    texture = LoadTexture(str(path).encode("utf-8"))
    GenTextureMipmaps(ffi.addressof(texture))
    return texture


def material_texture_contract(semantic: str) -> dict[str, Any]:
    return TEXTURE_CONTRACT[semantic]
