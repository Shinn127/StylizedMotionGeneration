import numpy as np
import torch

from encode_fsq_database import encode_shard
from models.fsq import FSQMotionAutoencoder


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
    offset = torch.zeros(12)
    scale = torch.ones(12)

    indices, codes = encode_shard(
        model,
        motion,
        offset,
        scale,
        offset,
        scale,
        chunk_size=24,
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        full_codes, full_indices = model.encode_to_codes(torch.from_numpy(motion).unsqueeze(0))
    np.testing.assert_array_equal(indices, full_indices[0].numpy().astype(np.uint8))
    np.testing.assert_allclose(codes, full_codes[0].numpy().astype(np.float16), rtol=0.0, atol=0.0)
