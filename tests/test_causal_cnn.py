import torch

from stylized_motion.learning.nets.causal_cnn import (
    CausalDecoder1D,
    CausalEncoder1D,
    FrameCausalDecoder1D,
    FrameCausalEncoder1D,
)
from stylized_motion.learning.fsq import FSQMotionAutoencoder
from stylized_motion.learning.vqvae import CausalMotionVQVAE


def test_frame_causal_network_does_not_read_outside_64_frame_window():
    torch.manual_seed(7)
    encoder = FrameCausalEncoder1D(3, 4, width=8)
    decoder = FrameCausalDecoder1D(3, 4, width=8)
    encoder.eval()
    decoder.eval()

    x = torch.randn(1, 3, 96)
    changed = x.clone()
    changed[:, :, 0] += 100.0
    changed[:, :, 65:] += 100.0
    with torch.no_grad():
        expected = decoder(encoder(x))[:, :, 64]
        actual = decoder(encoder(changed))[:, :, 64]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_both_cnn_variants_preserve_expected_sequence_length():
    x = torch.randn(2, 3, 64)
    frame_encoder = FrameCausalEncoder1D(3, 4, width=8)
    frame_decoder = FrameCausalDecoder1D(3, 4, width=8)
    causal_encoder = CausalEncoder1D(3, 4, width=8)
    causal_decoder = CausalDecoder1D(3, 4, width=8)

    assert frame_decoder(frame_encoder(x)).shape == x.shape
    assert causal_decoder(causal_encoder(x)).shape == x.shape


def test_autoencoder_models_expose_64_frame_context_metadata():
    frame_vq = CausalMotionVQVAE(
        motion_dim=12,
        code_dim=16,
        codebook_size=8,
        num_heads=4,
        width=16,
        model_type="frame_causal_cnn",
    )
    downsampled_vq = CausalMotionVQVAE(
        motion_dim=12,
        code_dim=16,
        codebook_size=8,
        num_heads=4,
        width=16,
        model_type="causal_cnn",
    )
    fsq = FSQMotionAutoencoder(
        motion_dim=12,
        code_dim=16,
        width=16,
        num_coordinates=5,
    )

    assert (frame_vq.receptive_field, frame_vq.context_left, frame_vq.lookahead_frames) == (64, 63, 0)
    assert (downsampled_vq.receptive_field, downsampled_vq.context_left, downsampled_vq.lookahead_frames) == (64, 63, 3)
    assert (fsq.receptive_field, fsq.context_left, fsq.lookahead_frames) == (64, 63, 0)


def test_rf64_autoencoders_roundtrip_indices_with_expected_shapes():
    torch.manual_seed(11)
    x = torch.randn(2, 64, 12)
    models = (
        CausalMotionVQVAE(
            motion_dim=12,
            code_dim=16,
            codebook_size=8,
            num_heads=4,
            width=16,
            model_type="causal_cnn",
        ),
        CausalMotionVQVAE(
            motion_dim=12,
            code_dim=16,
            codebook_size=8,
            num_heads=4,
            width=16,
            model_type="frame_causal_cnn",
        ),
        FSQMotionAutoencoder(
            motion_dim=12,
            code_dim=16,
            width=16,
            num_coordinates=5,
        ),
    )

    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(x)
            decoded = model.decode_from_indices(output["indices"])
        assert output["recon_state"].shape == x.shape
        assert decoded.shape == x.shape
        torch.testing.assert_close(decoded, output["recon_state"], rtol=0.0, atol=0.0)
