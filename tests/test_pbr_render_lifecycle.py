from types import SimpleNamespace

import pytest

from stylized_motion.anim import render_targets
from stylized_motion.anim.environment import IBLResources
from stylized_motion.anim.renderer import _require_distinct_source_destination
from stylized_motion.anim.renderer import Renderer
from stylized_motion.anim.soma_assets import RESOURCE_DIR, SHADER_FILES
from stylized_motion.util.paths import SOMA_RESOURCE_DIR


def _texture(identifier):
    return SimpleNamespace(id=identifier)


def _color_target(framebuffer_id, texture_id):
    return SimpleNamespace(id=framebuffer_id, texture=_texture(texture_id))


def test_fullscreen_pass_rejects_framebuffer_feedback():
    destination = _color_target(11, 41)
    with pytest.raises(ValueError, match="cannot sample"):
        _require_distinct_source_destination(_texture(41), destination)

    _require_distinct_source_destination(_texture(42), destination)


def test_render_target_cleanup_releases_manual_attachments_once(monkeypatch):
    """PBR targets own every texture attachment, not only their FBO handles."""
    released_textures = []
    released_framebuffers = []
    released_render_textures = []

    monkeypatch.setattr(render_targets, "rlUnloadTexture", lambda identifier: released_textures.append(identifier))
    monkeypatch.setattr(render_targets, "rlUnloadFramebuffer", lambda identifier: released_framebuffers.append(identifier))
    monkeypatch.setattr(render_targets, "UnloadRenderTexture", lambda target: released_render_textures.append(target.id))

    targets = render_targets.RenderTargets(8, 8, "pbr")
    targets.lighting = _color_target(1, 101)
    targets.tonemapped = _color_target(2, 102)
    targets.final = _color_target(3, 103)
    targets.ssao_front = _color_target(4, 104)
    targets.ssao_back = _color_target(5, 105)
    targets.gbuffer = SimpleNamespace(
        id=6,
        color=_texture(106),
        normal=_texture(107),
        material_ao=_texture(108),
        depth=_texture(109),
    )
    targets.shadow_maps = [
        SimpleNamespace(id=7, texture=_texture(110), depth=_texture(111)),
    ]
    targets.shadow_blurred = [_color_target(8, 112)]
    targets.evsm_scratch = _color_target(9, 113)

    targets.cleanup()
    targets.cleanup()

    assert sorted(released_textures) == [101, *range(106, 114)]
    assert sorted(released_framebuffers) == [1, 6, 7, 8, 9]
    assert sorted(released_render_textures) == [2, 3, 4, 5]


def test_soma_shader_manifest_copies_every_runtime_shader():
    assert {"debug.fs", "evsmBlur.fs"} <= set(SHADER_FILES)
    for shader in SHADER_FILES:
        assert (RESOURCE_DIR / shader).is_file()
        assert (SOMA_RESOURCE_DIR / shader).is_file()


def test_fxaa_writes_opaque_alpha_in_both_viewers():
    for resource_dir in (RESOURCE_DIR, SOMA_RESOURCE_DIR):
        shader = (resource_dir / "fxaa.fs").read_text(encoding="utf-8")
        assert "finalColor = vec4(" in shader
        assert ", 1.0);" in shader


def test_disabled_ibl_skips_generation_upload_and_binding(monkeypatch):
    import stylized_motion.anim.environment as environment

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("disabled IBL must not build or upload resources")

    for name in ("_procedural_sky_array", "_load_or_build_ibl_arrays", "_array_to_cubemap", "_array_to_cubemap_mipped", "_integrate_brdf_lut", "LoadTextureFromImage"):
        monkeypatch.setattr(environment, name, unexpected_call)

    resources = IBLResources(enabled=False).initialize()
    assert resources.enabled is False
    assert resources.environment is None
    assert resources.irradiance is None
    assert resources.prefilter is None
    assert resources.brdf_lut is None
    resources.cleanup()
    resources.cleanup()

    released = []
    monkeypatch.setattr(environment, "UnloadTexture", lambda texture: released.append(texture.id))
    stale = IBLResources(
        enabled=False,
        environment=_texture(201),
        irradiance=_texture(202),
        prefilter=_texture(203),
        brdf_lut=_texture(204),
    ).initialize()
    assert stale.environment is None
    assert released == [204, 203, 202, 201]

    # Renderer sees the disabled state and returns before touching any sampler
    # or texture handle, so fallback sky/ambient shader code remains usable.
    assert Renderer(SimpleNamespace(ibl=resources))._bind_ibl_textures() is False
