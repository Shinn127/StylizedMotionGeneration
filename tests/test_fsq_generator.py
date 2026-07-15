from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from evaluate_fsq_generator import _distribution_js, decoded_rollout_metrics
from models.fsq import FSQMotionAutoencoder
from models.fsq_generator import FSQCausalTransformerGenerator, FSQGeneratorCache


def build_small_generator(context_frames: int = 8) -> FSQCausalTransformerGenerator:
    return FSQCausalTransformerGenerator(
        num_coordinates=3,
        num_levels=4,
        coordinate_embedding_dim=4,
        dim=32,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        dropout=0.0,
        context_frames=context_frames,
    )


def test_generator_outputs_frame_level_coordinate_logits_and_hidden_states():
    torch.manual_seed(3)
    model = build_small_generator().eval()
    indices = torch.randint(0, 4, (2, 8, 3))
    with torch.inference_mode():
        output = model(indices)
    assert output["hidden"].shape == (2, 8, 32)
    assert output["logits"].shape == (2, 8, 3, 4)
    assert output["cache"] is None


def test_generator_is_strictly_causal():
    torch.manual_seed(5)
    model = build_small_generator().eval()
    first = torch.randint(0, 4, (2, 8, 3))
    second = first.clone()
    second[:, 4:] = torch.randint(0, 4, second[:, 4:].shape)
    with torch.inference_mode():
        first_logits = model(first)["logits"]
        second_logits = model(second)["logits"]
    torch.testing.assert_close(first_logits[:, :4], second_logits[:, :4], rtol=1e-5, atol=1e-6)


def test_cached_decode_matches_full_forward():
    torch.manual_seed(7)
    model = build_small_generator().eval()
    indices = torch.randint(0, 4, (2, 8, 3))
    with torch.inference_mode():
        full_logits = model(indices)["logits"]
        output = model(indices[:, :1], use_cache=True)
        cached_logits = [output["logits"]]
        cache = output["cache"]
        assert isinstance(cache, FSQGeneratorCache)
        for frame in range(1, indices.shape[1]):
            output = model(indices[:, frame : frame + 1], cache=cache, use_cache=True)
            cached_logits.append(output["logits"])
            cache = output["cache"]
            assert isinstance(cache, FSQGeneratorCache)
    torch.testing.assert_close(full_logits, torch.cat(cached_logits, dim=1), rtol=1e-5, atol=1e-6)


def test_kv_cache_is_bounded_while_absolute_position_advances():
    torch.manual_seed(11)
    model = build_small_generator(context_frames=4).eval()
    seed = torch.randint(0, 4, (1, 4, 3))
    with torch.inference_mode():
        logits, cache = model.prefill(seed)
        for _ in range(9):
            current = logits.argmax(dim=-1)
            logits, cache = model.decode_step(current, cache)
    assert cache.length == 4
    assert cache.next_position == 13
    assert all(key.shape[-2] == 4 and value.shape[-2] == 4 for key, value in cache.layers)


def test_greedy_generation_is_deterministic_and_produces_valid_levels():
    torch.manual_seed(13)
    model = build_small_generator().eval()
    seed = torch.randint(0, 4, (2, 8, 3))
    with torch.inference_mode():
        first = model.generate(seed, num_steps=6, greedy=True)
        second = model.generate(seed, num_steps=6, greedy=True)
    assert first.shape == (2, 6, 3)
    assert int(first.min()) >= 0
    assert int(first.max()) < 4
    torch.testing.assert_close(first, second)


def test_small_generator_can_overfit_a_deterministic_token_sequence():
    torch.manual_seed(17)
    model = FSQCausalTransformerGenerator(
        num_coordinates=2,
        num_levels=3,
        coordinate_embedding_dim=4,
        dim=24,
        num_layers=1,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=48,
        dropout=0.0,
        context_frames=8,
    )
    pattern = torch.tensor([[time % 3, (time + 1) % 3] for time in range(8)])
    indices = pattern.unsqueeze(0).repeat(16, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)

    def loss_value() -> torch.Tensor:
        logits = model(indices[:, :-1])["logits"]
        return F.cross_entropy(logits.reshape(-1, 3), indices[:, 1:].reshape(-1))

    initial = float(loss_value().detach())
    for _ in range(60):
        loss = loss_value()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    assert final < 0.1
    assert final < initial * 0.1


def test_default_config_targets_full_dataset_and_batch_512():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "fsq_generator.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["token_database"] == "data/processed/100style_pruned/fsq_20x9_full_loss"
    assert config["batch_size"] == 512


def test_distribution_js_handles_empty_bins_without_warnings():
    first = np.asarray([[2, 0, 1], [0, 3, 0]])
    assert _distribution_js(first, first) == 0.0


def test_decoded_rollout_metrics_are_finite_for_matching_fsq_sequences():
    torch.manual_seed(19)
    fsq = FSQMotionAutoencoder(
        motion_dim=32,
        code_dim=16,
        width=16,
        num_coordinates=3,
        num_levels=3,
    ).eval()
    indices = torch.randint(0, 3, (2, 8, 3))
    checkpoint = {
        "stats": {
            "offset": np.zeros(32, dtype=np.float32),
            "scale": np.ones(32, dtype=np.float32),
            "ref_pos": np.asarray([[0.0, 0.0, 0.0], [-0.1, -1.0, 0.0], [0.1, -1.0, 0.0]], dtype=np.float32),
            "parents": np.asarray([-1, 0, 0], dtype=np.int64),
            "names": np.asarray(["Root", "LeftToeBase", "RightToeBase"], dtype=object),
        }
    }
    with torch.inference_mode():
        metrics = decoded_rollout_metrics(
            checkpoint,
            fsq,
            generated_sequence=indices,
            target_sequence=indices,
            seed_frames=4,
            root_dt=1.0 / 60.0,
        )
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["normalized_feature_l1"] == 0.0
    assert metrics["joint_mpjpe_m"] == 0.0
