from pathlib import Path

import torch
import yaml

from stylized_motion.learning.nets.causal_transformer_generator import (
    FSQCanonicalTransformerGenerator,
    FSQGeneratorCache,
)


def _model(context_frames: int = 8) -> FSQCanonicalTransformerGenerator:
    return FSQCanonicalTransformerGenerator(
        num_coordinates=3,
        num_levels=4,
        trajectory_dim=5,
        trajectory_hidden_dim=8,
        coordinate_embedding_dim=4,
        dim=32,
        num_layers=1,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        dropout=0.0,
        context_frames=context_frames,
    )


def test_canonical_generator_has_no_style_parameters_and_preserves_token_shape():
    model = _model().eval()
    indices = torch.randint(0, 4, (2, 4, 3))
    trajectory = torch.randn(2, 4, 5)
    with torch.inference_mode():
        output = model(indices, trajectory=trajectory)
    assert output["logits"].shape == (2, 4, 3, 4)
    assert not any("style" in name or "film" in name for name, _ in model.named_parameters())


def test_canonical_cached_decode_matches_full_forward_with_controls():
    model = _model(context_frames=4).eval()
    indices = torch.randint(0, 4, (2, 4, 3))
    trajectory = torch.randn(2, 4, 5)
    valid = torch.ones(2, 4, dtype=torch.bool)
    with torch.inference_mode():
        full = model(indices, trajectory=trajectory, trajectory_valid=valid)["logits"]
        first = model(
            indices[:, :1], trajectory=trajectory[:, :1], trajectory_valid=valid[:, :1], use_cache=True
        )
        cached = [first["logits"]]
        cache = first["cache"]
        assert isinstance(cache, FSQGeneratorCache)
        for frame in range(1, indices.shape[1]):
            step = model(
                indices[:, frame : frame + 1],
                trajectory=trajectory[:, frame : frame + 1],
                trajectory_valid=valid[:, frame : frame + 1],
                cache=cache,
                use_cache=True,
            )
            cached.append(step["logits"])
            cache = step["cache"]
    torch.testing.assert_close(full, torch.cat(cached, dim=1), rtol=1e-5, atol=1e-6)


def test_canonical_generator_config_uses_metadata_driven_token_dimensions():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "data" / "configs" / "canonical_generator.yaml").read_text()
    )
    assert config["model"]["model_kind"] == "canonical_generator"
    assert "num_coordinates" not in config["model"]
    assert "num_levels" not in config["model"]
    assert "style_embedding_dim" not in config["model"]
