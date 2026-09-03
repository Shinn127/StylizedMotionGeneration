"""Offscreen still renderer for the viewers.

Renders one frame through the production Renderer and writes the display
target to a PNG. Frames are captured from the render-target texture instead
of the window framebuffer, so the result is independent of window
compositing (macOS hidden/short-lived windows capture black through
``take_screenshot``, which made ``--output-video`` useless for stills).

The final view is captured pre-FXAA; everything else (debug views, legacy
lighting, tone curve, overlays) is pixel-identical to the interactive path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyray import Vector3
from raylib import (
    CloseWindow,
    ExportImage,
    FLAG_WINDOW_HIDDEN,
    InitWindow,
    LoadImageFromTexture,
    SetConfigFlags,
    rlDisableColorBlend,
    Vector3Add,
    Vector3Scale,
)

from stylized_motion.anim.genoview import (
    DEBUG_MODES,
    GENO_RIG,
    SCENE_MODES,
    TONE_CURVES,
    GenoView,
    build_database_from_bvh,
    load_database_dict,
)
from stylized_motion.anim.renderer import Renderer
from stylized_motion.anim.somaview import SOMA_RIG
from stylized_motion.util.paths import RESOURCE_DIR, SOMA_RESOURCE_DIR


def render_still(args: argparse.Namespace) -> Path:
    rig = SOMA_RIG if args.pipeline == "somaview" else GENO_RIG
    resources_root = args.resources_root or (SOMA_RESOURCE_DIR if args.pipeline == "somaview" else RESOURCE_DIR)
    database = (
        load_database_dict(args.database)
        if args.database is not None
        else build_database_from_bvh(args.bvh, rig=rig)
    )
    viewer = GenoView(
        database=database,
        trajectory_path=None,
        resources_root=resources_root,
        shading=args.shading,
        debug_view=args.debug_view,
        metallic=args.metallic,
        roughness=args.roughness,
        exposure=args.exposure,
        ssao_intensity=args.ssao_intensity,
        ibl_strength=args.ibl_strength,
        ibl_enabled=not args.disable_ibl,
        draw_skeleton=args.skeleton,
        normal_map=args.normal_map,
        metallic_roughness_map=args.metallic_roughness_map,
        sun_strength=args.sun_strength,
        tone_curve=args.tone_curve,
        scene_mode=args.scene,
        rig=rig,
    )

    SetConfigFlags(FLAG_WINDOW_HIDDEN)
    InitWindow(args.width, args.height, b"render_stills")
    viewer.screen_width = args.width
    viewer.screen_height = args.height
    try:
        viewer._initialize_rendering(args.width, args.height)
        viewer.renderer = Renderer(viewer)

        viewer.playback.set_current_frame(args.frame)
        viewer._sync_playback_frame()
        global_rot, global_pos = viewer._update_model_pose()
        # Mirror run()'s follow logic: the character scene tracks the root,
        # the material grid stays centered on the grid.
        root = global_pos[0]
        if viewer.scene_mode == "grid":
            target_x, target_z = 0.0, -2.0
        else:
            target_x, target_z = root[0], root[2]
        viewer.shadow_light.target = Vector3(target_x, 0.0, target_z)
        viewer.shadow_light.position = Vector3Add(
            viewer.shadow_light.target, Vector3Scale(viewer.light_dir, -5.0)
        )
        viewer.camera.distance = 8.5 if viewer.scene_mode == "grid" else 4.0
        viewer.camera.update(
            Vector3(target_x, 0.75 if viewer.scene_mode == "character" else 0.4, target_z),
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 60.0,
        )

        rlDisableColorBlend()
        viewer.renderer.render_frame(global_rot, global_pos, viewer.sample_index, None, None)
        viewer.renderer.draw_output()

        target = viewer.tonemapped if viewer.shading == "pbr" else viewer.lighted
        image = LoadImageFromTexture(target.texture)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ExportImage(image, str(args.output).encode("utf-8"))
    finally:
        viewer._cleanup()
        CloseWindow()
    return args.output


def main():
    parser = argparse.ArgumentParser(description="Render one viewer frame to a PNG without visible-window compositing.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bvh", type=Path, help="Path to a single BVH clip.")
    inputs.add_argument("--database", type=Path, help="Path to database.npz.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--pipeline", choices=("genoview", "somaview"), default="genoview")
    parser.add_argument("--resources-root", type=Path, default=None)
    parser.add_argument("--shading", choices=("legacy", "pbr"), default="pbr")
    parser.add_argument("--debug-view", choices=DEBUG_MODES, default="final")
    parser.add_argument("--frame", type=int, default=0, help="Playback frame index to render.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--metallic", type=float, default=0.0)
    parser.add_argument("--roughness", type=float, default=0.58)
    parser.add_argument("--exposure", type=float, default=0.9)
    parser.add_argument("--sun-strength", type=float, default=None)
    parser.add_argument("--ssao-intensity", type=float, default=0.15)
    parser.add_argument("--ibl-strength", type=float, default=0.35)
    parser.add_argument("--tone-curve", choices=TONE_CURVES, default="aces")
    parser.add_argument("--scene", choices=SCENE_MODES, default="character")
    parser.add_argument("--normal-map", type=Path, default=None)
    parser.add_argument("--metallic-roughness-map", type=Path, default=None)
    parser.add_argument("--disable-ibl", action="store_true")
    parser.add_argument("--skeleton", action="store_true")
    args = parser.parse_args()

    output = render_still(args)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
