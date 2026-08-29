import numpy as np
import pytest

from stylized_motion.anim import bvh


MIXED_BVH = """HIERARCHY
ROOT Root
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Hips
    {
        OFFSET 0.0 100.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
        JOINT Spine2
        {
            OFFSET 0.0 20.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT Head
            {
                OFFSET 0.0 40.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                End Site
                {
                    OFFSET 0.0 10.0 0.0
                }
            }
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.008333333333333333
1.0 2.0 3.0 0.0 0.0 0.0 0.5 100.5 0.5 10.0 0.0 0.0 20.0 0.0 0.0 30.0 0.0 0.0
4.0 5.0 6.0 1.0 1.0 1.0 0.5 101.0 0.5 11.0 1.0 0.0 21.0 0.0 1.0 31.0 0.0 1.0
"""


def test_mixed_channel_layout_parses_per_joint_channels():
    """soma_uniform motions mix 6-channel Root/Hips with 3-channel joints."""
    import pathlib

    path = pathlib.Path("/tmp/_mixed_test.bvh")
    path.write_text(MIXED_BVH, encoding="utf-8")
    data = bvh.load(str(path))
    path.unlink()

    assert [str(n) for n in data["names"]] == ["Root", "Hips", "Spine2", "Head"]
    assert data["order"] == "zyx"

    # Frame 0: Root translation (1,2,3); Hips carries its own absolute local
    # translation (0.5, 100.5, 0.5); joints carry only rotations.
    assert np.allclose(data["positions"][0, 0], [1.0, 2.0, 3.0])
    assert np.allclose(data["positions"][0, 1], [0.5, 100.5, 0.5])
    # 3-channel joints keep their static offsets as positions.
    assert np.allclose(data["positions"][0, 2], [0.0, 20.0, 0.0])
    assert np.allclose(data["rotations"][0, 0], [0.0, 0.0, 0.0])
    assert np.allclose(data["rotations"][0, 1], [10.0, 0.0, 0.0])
    assert np.allclose(data["rotations"][0, 2], [20.0, 0.0, 0.0])
    assert np.allclose(data["rotations"][0, 3], [30.0, 0.0, 0.0])

    assert np.allclose(data["positions"][1, 0], [4.0, 5.0, 6.0])
    assert np.allclose(data["rotations"][1, 3], [31.0, 0.0, 1.0])
