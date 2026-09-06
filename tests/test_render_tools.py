"""Static contracts for the still-rendering and baseline tooling."""

from pathlib import Path

from stylized_motion.util.paths import PROJECT_ROOT

BASELINE_DIR = PROJECT_ROOT / "docs" / "assets" / "pbr_baseline"
BASELINE_NAMES = (
    "genoview_final.png",
    "genoview_base_color.png",
    "genoview_metallic.png",
    "genoview_roughness.png",
    "genoview_normal.png",
    "genoview_depth.png",
    "genoview_ao.png",
    "genoview_shadow.png",
    "genoview_diffuse.png",
    "genoview_specular.png",
    "genoview_ibl.png",
    "genoview_hdr.png",
    "genoview_legacy.png",
    "somaview_final.png",
)


def test_render_stills_supports_viewer_and_debug_options():
    source = (PROJECT_ROOT / "stylized_motion" / "anim" / "render_stills.py").read_text(encoding="utf-8")
    for flag in (
        "--debug-view",
        "--pipeline",
        "--shading",
        "--tone-curve",
        "--normal-map",
        "--metallic-roughness-map",
        "--base-color-map",
        "--disable-ibl",
        "--sun-strength",
        "--shadow-resolution",
    ):
        assert flag in source, flag
    assert "render_frame" in source and "render_presentation" in source
    assert "rlSetClipPlanes(0.01, 50.0)" in source
    # The reusable helper reads the same final target as video export.
    targets_source = (PROJECT_ROOT / "stylized_motion" / "anim" / "render_targets.py").read_text(encoding="utf-8")
    assert "def export_render_target" in targets_source
    assert "ImageFlipVertical" in targets_source


def test_compare_stills_has_thresholded_comparison():
    source = (PROJECT_ROOT / "stylized_motion" / "anim" / "compare_stills.py").read_text(encoding="utf-8")
    assert "def compare_images" in source
    assert "--threshold" in source
    assert "sys.exit(1)" in source


def test_pbr_baseline_set_is_committed():
    for name in BASELINE_NAMES:
        path = BASELINE_DIR / name
        assert path.exists(), path
        assert path.stat().st_size > 1000, path
