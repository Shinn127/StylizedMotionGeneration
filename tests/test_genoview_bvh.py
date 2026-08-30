from stylized_motion.anim.genoview import build_database_from_bvh
from stylized_motion.util.paths import RESOURCE_DIR


def test_build_database_from_bvh_matches_genoview_contract():
    database = build_database_from_bvh(RESOURCE_DIR / "Geno_bind.bvh")

    assert database["positions"].shape == (2, 76, 3)
    assert database["rotations"].shape == (2, 76, 4)
    assert database["names"].tolist()[0] == "Simulation"
    assert database["parents"].tolist()[0] == -1
    assert database["range_names"].tolist() == ["Geno_bind"]
    assert abs(float(database["frame_time"]) - 1.0 / 60.0) < 1e-6
