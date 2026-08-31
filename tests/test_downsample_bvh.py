from pathlib import Path

from scripts.downsample_bvh import downsample_bvh


BVH = """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
}
MOTION
Frames: 5
Frame Time: 0.008333
0 0 0 0 0 0
1 0 0 1 1 1
2 0 0 2 2 2
3 0 0 3 3 3
4 0 0 4 4 4
"""


def test_downsample_bvh_updates_headers_and_keeps_every_nth_frame(tmp_path: Path):
    source = tmp_path / "input.bvh"
    target = tmp_path / "out" / "output.bvh"
    source.write_text(BVH, encoding="utf-8")

    assert downsample_bvh(source, target, factor=2) == (5, 3)
    result = target.read_text(encoding="utf-8")

    assert "Frames: 3" in result
    assert "Frame Time: 0.016666" in result
    assert result.splitlines()[-3:] == [
        "0 0 0 0 0 0",
        "2 0 0 2 2 2",
        "4 0 0 4 4 4",
    ]
