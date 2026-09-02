from pathlib import Path

import numpy as np

from stylized_motion.anim.genoview import GENO_RIG, RigSpec, build_simulation_root_skeleton_from_bind
from stylized_motion.anim.somaview import SOMA_FPS, SOMA_RIG
from stylized_motion.run import COMMANDS
from stylized_motion.util.paths import RESOURCE_DIR, SOMA_RESOURCE_DIR


def test_somaview_pipelines_are_registered():
    assert COMMANDS[("visualize", "somaview")] == "stylized_motion.anim.somaview"
    assert COMMANDS[("preprocess", "soma-assets")] == "stylized_motion.anim.soma_assets"


def test_soma_rig_differs_from_geno_only_in_assets():
    assert SOMA_RIG.model_filename == "SOMA.bin"
    assert SOMA_RIG.bind_bvh_filename == "SOMA_bind.bvh"
    assert SOMA_RIG.window_title == b"SomaView"
    assert SOMA_FPS == 120
    # Both rigs share the simulation-root convention and centimeter BVHs.
    assert SOMA_RIG.sim_position_joint == GENO_RIG.sim_position_joint == "Spine2"
    assert SOMA_RIG.sim_rotation_joint == GENO_RIG.sim_rotation_joint == "Hips"
    assert SOMA_RIG.unit_scale == GENO_RIG.unit_scale == 0.01


def test_soma_resource_dir_is_separate_from_genoview():
    assert SOMA_RESOURCE_DIR == RESOURCE_DIR.parent / "somaview"
    assert SOMA_RESOURCE_DIR != RESOURCE_DIR


def test_pbr_shader_contract_is_shared_between_viewers():
    for resource_dir in (RESOURCE_DIR, SOMA_RESOURCE_DIR):
        pbr_gbuffer = (resource_dir / "pbr.fs").read_text(encoding="utf-8")
        pbr_lighting = (resource_dir / "pbrLighting.fs").read_text(encoding="utf-8")
        ssao = (resource_dir / "ssao.fs").read_text(encoding="utf-8")
        assert (resource_dir / "tonemap.fs").exists()
        assert "uniform sampler2D baseColorMap" in pbr_gbuffer
        assert "uniform sampler2D metallicRoughnessMap" in pbr_gbuffer
        assert "uniform int useBaseColorMap" in pbr_gbuffer
        assert "uniform int useMetallicRoughnessMap" in pbr_gbuffer
        assert "uniform int pbrGroundPattern" in pbr_gbuffer
        assert "uniform sampler2D shadowMap" in pbr_lighting
        assert "uniform float prefilterMaxLod" in pbr_lighting
        assert "textureLod(prefilterMap" in pbr_lighting
        assert "finalColor = vec4(direct + ambient, 1.0)" in pbr_lighting
        assert "uniform float ssaoIntensity" in ssao
        assert "shadowMap" not in ssao
    for shader in ("pbr.fs", "lighting.fs", "pbrLighting.fs", "ssao.fs", "tonemap.fs", "debug.fs"):
        assert (RESOURCE_DIR / shader).read_bytes() == (SOMA_RESOURCE_DIR / shader).read_bytes()


def test_debug_view_contract_is_shared_between_viewers():
    from stylized_motion.anim.genoview import DEBUG_MODES, LIGHTING_DEBUG_MODES

    assert DEBUG_MODES[0] == "final"
    assert len(set(DEBUG_MODES)) == len(DEBUG_MODES)
    assert set(LIGHTING_DEBUG_MODES) == set(DEBUG_MODES)
    # Only quantities that exist inside the lighting pass are lighting-side
    # debug modes; everything else reads the GBuffer/SSAO attachments directly.
    assert LIGHTING_DEBUG_MODES["shadow"] == 1
    assert LIGHTING_DEBUG_MODES["diffuse"] == 2
    assert LIGHTING_DEBUG_MODES["specular"] == 3
    assert LIGHTING_DEBUG_MODES["ibl"] == 4
    assert LIGHTING_DEBUG_MODES["final"] == 0
    for resource_dir in (RESOURCE_DIR, SOMA_RESOURCE_DIR):
        debug_fs = (resource_dir / "debug.fs").read_text(encoding="utf-8")
        assert "uniform int debugMode" in debug_fs
        assert "uniform sampler2D texGbufferDepth" in debug_fs
        assert "uniform sampler2D texLighted" in debug_fs
        assert "vec3(1.0 - depth)" in debug_fs
    pbr_lighting = (RESOURCE_DIR / "pbrLighting.fs").read_text(encoding="utf-8")
    assert "uniform int debugMode" in pbr_lighting
    assert "finalColor = vec4(vec3(shadow), 1.0)" in pbr_lighting
    assert "finalColor = vec4(ambient, 1.0)" in pbr_lighting


def test_material_scene_and_texture_contract():
    from stylized_motion.anim.materials import Material, TEXTURE_CONTRACT
    from stylized_motion.anim.scene import DirectionalLight, RenderObject, Scene

    material = Material(metallic=2.0, roughness=0.0, ao=-1.0)
    assert material.metallic == 1.0
    assert material.roughness == 0.04
    assert material.ao == 0.0
    assert TEXTURE_CONTRACT["base_color_map"]["color_space"] == "sRGB"
    assert TEXTURE_CONTRACT["metallic_roughness_map"]["channels"] == ("metallic", "roughness", "ao")
    assert TEXTURE_CONTRACT["normal_map"]["color_space"] == "linear"

    light = DirectionalLight(direction=None, color=None, intensity=0.25)
    scene = Scene(directional_light=light)
    obj = RenderObject(model=None, material=material)
    assert scene.add_object(obj) is obj
    assert scene.objects == [obj]


def test_default_rig_stays_geno_for_existing_consumers():
    assert build_simulation_root_skeleton_from_bind.__defaults__ == (GENO_RIG,)
    import stylized_motion.anim.genoview as genoview

    assert genoview.GenoView.__init__.__defaults__[-1] is GENO_RIG


def test_renderer_owns_render_passes_and_run_only_schedules_them():
    import ast

    import stylized_motion.anim.genoview as genoview
    import stylized_motion.anim.render_targets as render_targets

    renderer_source = (Path(genoview.__file__).parent / "renderer.py").read_text(encoding="utf-8")
    renderer_tree = ast.parse(renderer_source)
    renderer_methods = {
        node.name
        for node in ast.walk(renderer_tree)
        if isinstance(node, ast.FunctionDef) and node.args.args and node.args.args[0].arg == "self"
    }
    assert {"render_shadow", "render_gbuffer", "render_ssao", "render_lighting", "render_tonemap", "render_debug", "render_frame"} <= renderer_methods

    genoview_source = Path(genoview.__file__).read_text(encoding="utf-8")
    genoview_tree = ast.parse(genoview_source)
    run_node = next(node for node in ast.walk(genoview_tree) if isinstance(node, ast.FunctionDef) and node.name == "run")
    run_calls = {
        node.func.id
        for node in ast.walk(run_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "render_frame" not in run_calls
    assert "BeginTextureMode" not in run_calls
    assert "ffi.new" not in genoview_source[genoview_source.index("    def run(self):"):]
    render_target_methods = {
        node.name
        for node in ast.walk(ast.parse(Path(render_targets.__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.args.args and node.args.args[0].arg == "self"
    }
    assert {"initialize", "cleanup"} <= render_target_methods
    assert "RenderTargets" in genoview_source


SOMA_BIND = Path(__file__).resolve().parents[1] / "data" / "assets" / "somaview" / "SOMA_bind.bvh"


def test_soma_bind_skeleton_parses_with_simulation_root():
    if not SOMA_BIND.exists():
        import pytest

        pytest.skip("SOMA assets have not been generated")
    names, parents, positions, rotations = build_simulation_root_skeleton_from_bind(SOMA_BIND, SOMA_RIG)
    assert len(names) == 79  # 78 SOMA joints + Simulation
    assert names[0] == "Simulation"
    assert parents[0] == -1
    assert "Hips" in [str(n) for n in names]
    # Hips sits about one meter up in the T-pose bind frame (meters).
    hips = positions[list(map(str, names)).index("Hips")]
    assert abs(hips[1] - 1.01) < 0.05
    assert abs(np.linalg.norm(rotations[0]) - 1.0) < 1e-5
