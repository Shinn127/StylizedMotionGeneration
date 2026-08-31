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
