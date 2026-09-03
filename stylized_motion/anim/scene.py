"""Scene-side render data shared by GenoView and SomaView."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyray import Vector3
from raylib import Vector3Zero

from stylized_motion.anim.materials import Material


@dataclass
class RenderObject:
    model: Any
    material: Material
    position: Vector3 = field(default_factory=Vector3Zero)
    draw_color: Any = None
    scale: float = 1.0
    skinned: bool = False


@dataclass
class DirectionalLight:
    direction: Vector3
    color: Vector3
    intensity: float = 1.0


@dataclass
class Scene:
    objects: list[RenderObject] = field(default_factory=list)
    directional_light: DirectionalLight | None = None

    def add_object(self, render_object: RenderObject) -> RenderObject:
        self.objects.append(render_object)
        return render_object

    def clear(self) -> None:
        self.objects.clear()


# Standard probe materials for the material test scene (spec section 14).
MATERIAL_GRID_PROBES = {
    "white_diffuse": {"base_color": (0.82, 0.82, 0.82, 1.0), "metallic": 0.0, "roughness": 0.95},
    "plastic": {"base_color": (0.9, 0.28, 0.2, 1.0), "metallic": 0.0, "roughness": 0.18},
    "rough_plastic": {"base_color": (0.9, 0.28, 0.2, 1.0), "metallic": 0.0, "roughness": 0.55},
    "gold": {"base_color": (1.0, 0.71, 0.29, 1.0), "metallic": 1.0, "roughness": 0.22},
    "chrome": {"base_color": (0.94, 0.94, 0.94, 1.0), "metallic": 1.0, "roughness": 0.05},
}

GRID_METALLIC_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_ROUGHNESS_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_SPACING = 1.1


def material_grid_positions() -> list[tuple[float, float, float, float]]:
    """(x, z, metallic, roughness) placements for the 5x5 material grid.

    Rows increase metallic toward -Z, columns increase roughness toward +X.
    A sway animation rig walks the +Z direction, so the grid spreads
    perpendicular to it and sits behind the character's default pose.
    """

    return [
        (
            (column - 2) * GRID_SPACING,
            -2.0 - (row - 2) * GRID_SPACING,
            row_metallic,
            column_roughness,
        )
        for row, row_metallic in enumerate(GRID_METALLIC_STEPS)
        for column, column_roughness in enumerate(GRID_ROUGHNESS_STEPS)
    ]


def build_material_grid_object(model: Any, x: float, z: float, metallic: float, roughness: float) -> RenderObject:
    """One grid cell: a neutral albedo so only metallic/roughness vary.

    Albedo keeps a mild per-row tint difference (cool -> warm with metallic)
    so adjacent cells stay distinguishable while diffuse/specular behaviour
    remains readable in the debug views.
    """

    albedo = 0.75 + 0.05 * metallic
    tint = (albedo * (1.0 - 0.15 * metallic), albedo, albedo * (1.0 + 0.05 * metallic))
    return RenderObject(
        model=model,
        material=Material(
            base_color=(tint[0], tint[1], tint[2], 1.0),
            metallic=metallic,
            roughness=max(roughness, 0.04),
        ),
        position=Vector3(x, 0.45, z),
    )
