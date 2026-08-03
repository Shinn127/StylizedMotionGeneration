import numpy as np
import torch

from stylized_motion.data.preprocess import _encode_feature_shard
from stylized_motion.learning.fsq import FSQMotionAutoencoder


def test_chunked_fsq_encoding_matches_full_causal_encoding():
    torch.manual_seed(13)
    model = FSQMotionAutoencoder(
        motion_dim=12,
        code_dim=16,
        width=16,
        num_coordinates=5,
        num_levels=9,
    )
    model.eval()
    motion = np.random.default_rng(5).standard_normal((96, 12), dtype=np.float32)
    indices, codes = _encode_feature_shard(
        model,
        motion,
        chunk_size=24,
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        full_codes, full_indices = model.encode_to_codes(torch.from_numpy(motion).unsqueeze(0))
    np.testing.assert_array_equal(indices, full_indices[0].numpy().astype(np.uint8))
    np.testing.assert_allclose(codes, full_codes[0].numpy().astype(np.float16), rtol=0.0, atol=0.0)
