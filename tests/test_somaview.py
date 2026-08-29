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


def test_default_rig_stays_geno_for_existing_consumers():
    assert build_simulation_root_skeleton_from_bind.__defaults__ == (GENO_RIG,)
    import stylized_motion.anim.genoview as genoview

    assert genoview.GenoView.__init__.__defaults__[-1] is GENO_RIG


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
