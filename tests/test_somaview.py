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
        assert "in vec4 fragTangent" in pbr_gbuffer
        assert "layout (location = 2) out float gbufferMaterialAO" in pbr_gbuffer
        assert "gbufferMaterialAO = ao;" in pbr_gbuffer
        assert "uniform sampler2D shadowMap" in pbr_lighting
        assert "uniform sampler2D materialAO" in pbr_lighting
        assert "texture(materialAO, fragTexCoord).r" in pbr_lighting
        assert "uniform float prefilterMaxLod" in pbr_lighting
        assert "textureLod(prefilterMap" in pbr_lighting
        # 3x3 PCF shadow sampling and the procedural sky background
        assert "uniform vec2 shadowTexelSize" in pbr_lighting
        assert "shadow / 9.0" in pbr_lighting
        # Sky background samples linear-radiance environment data directly.
        assert "textureLod(environmentMap, viewDir, 0.0)" in pbr_lighting
        assert "SRGBToLinear(textureLod(prefilterMap" not in pbr_lighting
        assert "finalColor = vec4(direct + ambient, 1.0)" in pbr_lighting
        assert "uniform float ssaoIntensity" in ssao
        assert "shadowMap" not in ssao
        assert "sampleTexCoord.x <= 0.0" in ssao
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


def test_gbuffer_carries_material_ao_attachment():
    import stylized_motion.anim.render_targets as render_targets

    source = Path(render_targets.__file__).read_text(encoding="utf-8")
    assert "RL_ATTACHMENT_COLOR_CHANNEL2" in source
    assert "PIXELFORMAT_UNCOMPRESSED_GRAYSCALE" in source
    assert "rlActiveDrawBuffers(3)" in source
    gbuffer_init = render_targets.GBuffer()
    assert hasattr(gbuffer_init, "material_ao")


def test_multi_texture_passes_bind_explicit_slots():
    # raylib keys SetShaderValueTexture samplers off texture.id, which collides
    # with the manually managed slots 10-14 whenever a render target's GL
    # texture name lands on those numbers (see spec 20.6).
    import stylized_motion.anim.render_targets as render_targets

    assert "def set_shader_value_texture_slot" in Path(render_targets.__file__).read_text(encoding="utf-8")

    import stylized_motion.anim.genoview as genoview_module

    genoview_source = Path(genoview_module.__file__).read_text(encoding="utf-8")
    for slot_ptr in ("material_ao_texture_slot_ptr", "ssao_texture_slot_ptr", "debug_lighted_slot_ptr"):
        assert slot_ptr in genoview_source


def test_pbr_light_rig_is_retuned_while_legacy_stays_frozen():
    from stylized_motion.anim.genoview import LEGACY_LIGHT_RIG, PBR_LIGHT_RIG

    # The legacy rig must keep the exact GenoViewPython values.
    assert LEGACY_LIGHT_RIG["sun_strength"] == 0.25
    assert LEGACY_LIGHT_RIG["ambient_strength"] == 1.0
    assert LEGACY_LIGHT_RIG["sky_strength"] == 0.15
    assert LEGACY_LIGHT_RIG["ground_strength"] == 0.1
    # The PBR rig is sun-dominant with a much weaker omni ambient term.
    assert PBR_LIGHT_RIG["sun_strength"] > LEGACY_LIGHT_RIG["sun_strength"]
    assert PBR_LIGHT_RIG["ambient_strength"] < LEGACY_LIGHT_RIG["ambient_strength"]
    # Warm sun against a cool sky carries the temperature contrast.
    assert PBR_LIGHT_RIG["sun_color"][2] < PBR_LIGHT_RIG["sun_color"][0]
    assert PBR_LIGHT_RIG["sky_color"][2] > PBR_LIGHT_RIG["sky_color"][0]


def test_white_background_sentinel_contract_is_shared_between_viewers():
    # --white-background marks sky pixels with an unreachable radiance sentinel
    # that tonemap.fs swaps for exact display white (paper figure mode). The
    # sentinel must survive RGBA16F storage (half max 65504) and stay far above
    # scene radiance, so both shaders must agree on the same constant.
    for resource_dir in (RESOURCE_DIR, SOMA_RESOURCE_DIR):
        lighting = (resource_dir / "pbrLighting.fs").read_text(encoding="utf-8")
        tonemap = (resource_dir / "tonemap.fs").read_text(encoding="utf-8")
        assert "uniform int whiteBackground" in lighting
        assert "uniform int whiteBackground" in tonemap
        assert "#define BACKGROUND_SENTINEL 6.0e4" in lighting
        assert "BACKGROUND_SENTINEL_MIN 3.0e4" in tonemap
        assert "if (whiteBackground == 1) { finalColor = vec4(vec3(BACKGROUND_SENTINEL), 1.0); }" in lighting
        assert "hdr.r > BACKGROUND_SENTINEL_MIN" in tonemap
    # Debug views share the flat white background for paper figures.
    for resource_dir in (RESOURCE_DIR, SOMA_RESOURCE_DIR):
        debug_fs = (resource_dir / "debug.fs").read_text(encoding="utf-8")
        assert "Flat white background" in debug_fs
    import stylized_motion.anim.genoview as genoview_module

    source = Path(genoview_module.__file__).read_text(encoding="utf-8")
    assert "white_background: bool = False" in source
    assert '--white-background' in source


def test_tone_curve_selection_is_shared_between_viewers():
    from stylized_motion.anim.genoview import TONE_CURVES

    assert TONE_CURVES[0] == "aces"
    assert TONE_CURVES == ("aces", "reinhard", "agx")
    for resource_dir in (RESOURCE_DIR, SOMA_RESOURCE_DIR):
        tonemap = (resource_dir / "tonemap.fs").read_text(encoding="utf-8")
        assert "uniform int toneCurve" in tonemap
        assert "ACESApprox" in tonemap
        assert "Reinhard" in tonemap
        assert "AgXInverse(AgX(" in tonemap
        # The default branch must stay the historical ACES fit.
        assert "return ACESApprox(color);" in tonemap

    import stylized_motion.anim.genoview as genoview_module

    try:
        genoview_module.GenoView(database=None, trajectory_path=None, resources_root=Path("."), tone_curve="bogus")
    except ValueError as e:
        assert "tone curve" in str(e)
    else:
        raise AssertionError("expected ValueError for bogus tone curve")


def test_material_grid_scene_contract():
    from stylized_motion.anim.genoview import SCENE_MODES
    from stylized_motion.anim.scene import (
        GRID_METALLIC_STEPS,
        GRID_ROUGHNESS_STEPS,
        build_material_grid_object,
        material_grid_positions,
    )

    assert SCENE_MODES == ("character", "grid")
    assert GRID_METALLIC_STEPS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert GRID_ROUGHNESS_STEPS == (0.0, 0.25, 0.5, 0.75, 1.0)

    placements = material_grid_positions()
    assert len(placements) == 25
    metals = sorted({p[2] for p in placements})
    roughs = sorted({p[3] for p in placements})
    assert metals == list(GRID_METALLIC_STEPS)
    assert roughs == list(GRID_ROUGHNESS_STEPS)

    cell = build_material_grid_object(None, *placements[0])
    # Pure metal cell: metallic clamps through Material, diffuse must vanish
    # in the lighting pass (verified visually), roughness honors the floor.
    assert cell.material.metallic == 0.0
    rough_cell = build_material_grid_object(None, *placements[4])
    assert rough_cell.material.roughness == 1.0
    # The roughness floor keeps the zero-roughness column physically valid.
    floor_cell = build_material_grid_object(None, 0.0, 0.0, 0.0, 0.0)
    assert floor_cell.material.roughness == 0.04

    try:
        from stylized_motion.anim import genoview as genoview_module

        genoview_module.GenoView(database=None, trajectory_path=None, resources_root=Path("."), scene_mode="bogus")
    except ValueError as e:
        assert "scene" in str(e)
    else:
        raise AssertionError("expected ValueError for bogus scene mode")


def test_ibl_chain_uses_convolution_and_importance_sampling():
    import numpy as np

    import stylized_motion.anim.environment as environment

    source = Path(environment.__file__).read_text(encoding="utf-8")
    # The irradiance map is convolved, not hand-painted; the prefilter mips
    # come from GGX importance sampling with a real mip chain.
    assert "def _convolve_irradiance" in source
    assert "def _prefilter_environment" in source
    assert "def _array_to_cubemap_mipped" in source
    assert "1.0 - 2.0 * (indices + 0.5) / sample_count" in source  # full-sphere sampling

    sky = environment._procedural_sky_array(16)
    irradiance = environment._convolve_irradiance(sky, 16, output_size=8)
    up = irradiance[2][4, 4]
    down = irradiance[3][4, 4]
    # The sky dome is brighter and bluer above the horizon.
    assert up.sum() > down.sum()
    assert up[2] > up[0]
    # +-X face centers sample symmetric environment regions.
    assert float(np.abs(irradiance[0][4, 4] - irradiance[1][4, 4]).max()) < 0.01

    levels = environment._prefilter_environment(sky, 16, output_size=16, mips=3)
    assert len(levels) == 3
    assert levels[0].shape[1] == 16 and levels[1].shape[1] == 8 and levels[2].shape[1] == 4
    # Rougher levels integrate a wider lobe: variance must drop.
    variances = [float(level[2].var()) for level in levels]
    assert variances[0] > variances[-1]


def test_ibl_upload_preserves_hdr_precision():
    import stylized_motion.anim.environment as environment

    source = Path(environment.__file__).read_text(encoding="utf-8")
    assert "np.float16" in source
    assert "PIXELFORMAT_UNCOMPRESSED_R16G16B16A16" in source
    assert "np.clip(faces[face], 0.0, 65504.0)" in source


def test_base_color_map_is_exposed_across_viewer_entrypoints():
    geno_source = Path(__file__).resolve().parents[1] / "stylized_motion" / "anim" / "genoview.py"
    source = geno_source.read_text(encoding="utf-8")
    assert "base_color_map: Path | None = None" in source
    assert "--base-color-map" in source
    assert "base_color_map=base_color_map" in source

    soma_source = (Path(__file__).resolve().parents[1] / "stylized_motion" / "anim" / "somaview.py").read_text(encoding="utf-8")
    still_source = (Path(__file__).resolve().parents[1] / "stylized_motion" / "anim" / "render_stills.py").read_text(encoding="utf-8")
    assert "--base-color-map" in soma_source
    assert "--base-color-map" in still_source


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
