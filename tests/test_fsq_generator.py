from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from evaluate_fsq_generator import _distribution_js, decoded_rollout_metrics
from models.fsq import FSQMotionAutoencoder
from models.fsq_generator import (
    FSQCausalTransformerGenerator,
    FSQConditionalTransformerGenerator,
    FSQGeneratorCache,
    STYLE_CONDITIONING,
)


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


def build_small_conditional_generator(context_frames: int = 8) -> FSQConditionalTransformerGenerator:
    return FSQConditionalTransformerGenerator(
        num_coordinates=3,
        num_levels=4,
        num_styles=5,
        trajectory_dim=18,
        trajectory_hidden_dim=16,
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


def test_dynamic_film_conditional_cache_matches_full_forward_without_prefix():
    torch.manual_seed(23)
    model = build_small_conditional_generator(context_frames=4).eval()
    indices = torch.randint(0, 4, (2, 8, 3))
    styles = torch.tensor([1, 3])
    trajectory = torch.randn(2, 8, 18)
    valid = torch.ones(2, 8, dtype=torch.bool)
    with torch.inference_mode():
        full_logits = model(
            indices[:, :4],
            style_ids=styles,
            trajectory=trajectory[:, :4],
            trajectory_valid=valid[:, :4],
        )["logits"]
        logits, cache = model.prefill(
            indices[:, :1],
            style_ids=styles,
            seed_trajectory=trajectory[:, :1],
            seed_trajectory_valid=valid[:, :1],
        )
        cached_logits = [logits[:, None]]
        for frame in range(1, indices.shape[1]):
            logits, cache = model.decode_step(
                indices[:, frame],
                cache,
                style_ids=styles,
                trajectory=trajectory[:, frame],
                trajectory_valid=valid[:, frame],
            )
            if frame < 4:
                cached_logits.append(logits[:, None])
    torch.testing.assert_close(full_logits, torch.cat(cached_logits, dim=1), rtol=1e-5, atol=1e-6)
    assert cache.prefix_length == 0
    assert cache.motion_length == 4
    assert cache.length == 4
    assert cache.next_position == 8


def test_conditional_trajectory_changes_logits():
    torch.manual_seed(29)
    model = build_small_conditional_generator().eval()
    indices = torch.randint(0, 4, (1, 4, 3))
    style = torch.tensor([2])
    zero = torch.zeros(1, 4, 18)
    active = torch.ones(1, 4, 18)
    valid = torch.ones(1, 4, dtype=torch.bool)
    with torch.inference_mode():
        zero_logits = model(indices, style_ids=style, trajectory=zero, trajectory_valid=valid)["logits"]
        active_logits = model(indices, style_ids=style, trajectory=active, trajectory_valid=valid)["logits"]
    assert not torch.allclose(zero_logits, active_logits)


def test_conditional_style_film_changes_every_frame():
    torch.manual_seed(31)
    model = build_small_conditional_generator().eval()
    indices = torch.randint(0, 4, (1, 8, 3))
    trajectory = torch.zeros(1, 8, 18)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    with torch.inference_mode():
        first = model(
            indices,
            style_ids=torch.tensor([0]),
            trajectory=trajectory,
            trajectory_valid=valid,
        )["logits"]
        second = model(
            indices,
            style_ids=torch.tensor([1]),
            trajectory=trajectory,
            trajectory_valid=valid,
        )["logits"]
    per_frame_difference = (first - second).abs().sum(dim=(-1, -2))
    assert bool((per_frame_difference > 0.0).all())


def test_dynamic_style_switch_preserves_cached_history_and_modulates_new_frame():
    torch.manual_seed(37)
    model = build_small_conditional_generator(context_frames=4).eval()
    indices = torch.randint(0, 4, (1, 5, 3))
    trajectory = torch.randn(1, 5, 18)
    valid = torch.ones(1, 5, dtype=torch.bool)
    old_style = torch.tensor([0])
    new_style = torch.tensor([1])
    with torch.inference_mode():
        _, cache = model.prefill(
            indices[:, :4],
            style_ids=old_style,
            seed_trajectory=trajectory[:, :4],
            seed_trajectory_valid=valid[:, :4],
        )
        old_keys = [key.clone() for key, _ in cache.layers]
        old_logits, _ = model.decode_step(
            indices[:, 4],
            cache,
            style_ids=old_style,
            trajectory=trajectory[:, 4],
            trajectory_valid=valid[:, 4],
        )
        new_logits, switched_cache = model.decode_step(
            indices[:, 4],
            cache,
            style_ids=new_style,
            trajectory=trajectory[:, 4],
            trajectory_valid=valid[:, 4],
        )
    assert switched_cache.length == 4
    assert switched_cache.next_position == cache.next_position + 1
    for old_key, (switched_key, _) in zip(old_keys, switched_cache.layers):
        torch.testing.assert_close(switched_key[..., :-1, :], old_key[..., 1:, :])
    assert bool((old_logits != new_logits).any())


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


def test_conditional_config_trains_dynamic_film_model_from_scratch():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "fsq_generator_conditional.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert "base_checkpoint" not in config
    assert "base_lr" not in config
    assert "condition_lr" not in config
    assert "history_condition_dropout" not in config
    assert config["lr"] == 3e-4
    assert config["epochs"] == 100
    assert config["style_embedding_dim"] == 128
    assert config["outdir"] == "outputs/fsq_generator_conditional_dynamic_film"
    assert STYLE_CONDITIONING == "causal_dynamic_block_film_v1"


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
