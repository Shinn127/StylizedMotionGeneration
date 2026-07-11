import numpy as np
import torch

from analyze_fsq_patterns import (
    block_shuffled_indices,
    coordinate_covariance_features,
    coordinate_cooccurrence_features,
    hashed_ngram_features,
    histogram_features,
    position_histogram_features,
    raw_sequence_features,
    run_length_features,
    shuffled_indices,
    spectrum_features,
    transition_features,
)
from encode_fsq_database import encode_shard
from models.fsq import FSQMotionAutoencoder


def test_pattern_features_are_normalized_and_shuffle_preserves_histogram():
    indices = np.asarray(
        [
            [[0, 1], [1, 1], [1, 2], [0, 2]],
            [[2, 0], [2, 1], [1, 1], [1, 0]],
        ],
        dtype=np.uint8,
    )
    histogram = histogram_features(indices, num_levels=3).reshape(2, 2, 3)
    transitions = transition_features(indices, num_levels=3).reshape(2, 2, 9)
    np.testing.assert_allclose(histogram.sum(axis=-1), 1.0)
    np.testing.assert_allclose(transitions.sum(axis=-1), 1.0)

    shuffled = shuffled_indices(indices, seed=7)
    shuffled_histogram = histogram_features(shuffled, num_levels=3)
    np.testing.assert_allclose(shuffled_histogram, histogram.reshape(2, -1))


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


def test_extended_pattern_features_have_stable_shapes_and_normalization():
    indices = np.asarray([[[0, 1], [0, 1], [1, 2], [1, 2], [1, 1], [2, 1]]], dtype=np.uint8)
    assert hashed_ngram_features(indices, 3, 3, 17).shape == (1, 17)
    np.testing.assert_allclose(hashed_ngram_features(indices, 3, 3, 17).sum(), 1.0)
    assert run_length_features(indices, 3).shape == (1, 2 * 3 * 7)
    assert position_histogram_features(indices, 3, 3).shape == (1, 3 * 2 * 3)
    assert spectrum_features(indices, 3).shape == (1, 2 * (3 + 2))
    assert coordinate_covariance_features(indices, 3).shape == (1, 3)
    assert coordinate_cooccurrence_features(indices, 3, 19).shape == (1, 19)
    np.testing.assert_allclose(coordinate_cooccurrence_features(indices, 3, 19).sum(), 1.0)
    assert raw_sequence_features(indices, 3).shape == (1, 6 * 2 * 3)

    block_shuffled = block_shuffled_indices(indices, seed=3, block_size=2)
    np.testing.assert_allclose(histogram_features(block_shuffled, 3), histogram_features(indices, 3))
