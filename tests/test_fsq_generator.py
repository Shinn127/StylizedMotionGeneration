from pathlib import Path

import torch
import yaml

from stylized_motion.learning.nets.causal_transformer_generator import FSQCausalTransformerGenerator, FSQGeneratorCache


def _model(context_frames: int = 8, coordinates: int = 3, levels: int = 4):
    return FSQCausalTransformerGenerator(
        num_coordinates=coordinates,
        num_levels=levels,
        coordinate_embedding_dim=4,
        dim=32,
        num_layers=1,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        dropout=0.0,
        context_frames=context_frames,
    )


def test_generator_uses_dynamic_store_dimensions():
    model = _model(coordinates=7, levels=5).eval()
    indices = torch.randint(0, 5, (2, 8, 7))
    with torch.inference_mode():
        output = model(indices)
    assert output["logits"].shape == (2, 8, 7, 5)


def test_cached_decode_matches_full_causal_forward():
    model = _model(context_frames=4).eval()
    indices = torch.randint(0, 4, (2, 4, 3))
    with torch.inference_mode():
        full = model(indices)["logits"]
        first = model(indices[:, :1], use_cache=True)
        cached = [first["logits"]]
        cache = first["cache"]
        assert isinstance(cache, FSQGeneratorCache)
        for frame in range(1, indices.shape[1]):
            step = model(indices[:, frame:frame + 1], cache=cache, use_cache=True)
            cached.append(step["logits"])
            cache = step["cache"]
    torch.testing.assert_close(full, torch.cat(cached, dim=1), rtol=1e-5, atol=1e-6)


def test_generator_config_is_canonical_and_does_not_name_token_width():
    config = yaml.safe_load((Path(__file__).parents[1] / "data" / "configs" / "fsq_generator.yaml").read_text())
    assert "data" in config and "model" in config and "training" in config
    assert "20" not in str(config)
    assert "40" not in str(config["model"])
