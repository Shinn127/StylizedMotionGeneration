import numpy as np
import torch

from models.fsq_generator import FSQConditionalTransformerGenerator, FSQGeneratorCache
from realtime_fsq_controller import KeyboardTrajectoryControl, RealtimeFSQController


def test_keyboard_control_is_invalid_and_zero_when_idle():
    control = KeyboardTrajectoryControl()
    values, valid, label = control.build(False, False, False, False, False, False)
    assert values.shape == (18,)
    assert not valid
    assert label == "idle"
    np.testing.assert_array_equal(values, np.zeros(18, dtype=np.float32))


def test_keyboard_forward_and_left_map_to_root_local_positions():
    control = KeyboardTrajectoryControl(move_speed=1.5)
    forward, valid, forward_label = control.build(True, False, False, False, False, False)
    left, left_valid, left_label = control.build(False, False, True, False, False, False)

    assert valid and left_valid
    assert forward_label == "forward"
    assert left_label == "left"
    forward_positions = forward[:9].reshape(3, 3)
    left_positions = left[:9].reshape(3, 3)
    assert np.all(forward_positions[:, 2] > 0.0)
    assert np.all(left_positions[:, 0] > 0.0)
    np.testing.assert_allclose(forward[9:].reshape(3, 3), np.asarray([[0.0, 0.0, 1.0]] * 3))


def test_keyboard_turn_right_changes_future_facing_direction():
    control = KeyboardTrajectoryControl(turn_speed=1.0)
    values, valid, label = control.build(False, False, False, False, False, True)
    directions = values[9:].reshape(3, 3)

    assert valid
    assert label == "turn-right"
    assert np.all(directions[:, 0] < 0.0)
    assert np.all(directions[:, 2] > 0.0)


def test_style_switch_replays_only_latest_input_and_refreshes_staged_logits():
    torch.manual_seed(41)
    generator = FSQConditionalTransformerGenerator(
        num_coordinates=3,
        num_levels=4,
        num_styles=2,
        trajectory_dim=18,
        trajectory_hidden_dim=16,
        coordinate_embedding_dim=4,
        dim=32,
        num_layers=1,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        dropout=0.0,
        context_frames=4,
    ).eval()
    seed = torch.randint(0, 4, (1, 4, 3))
    trajectory = torch.randn(1, 4, 18)
    valid = torch.ones(1, 4, dtype=torch.bool)
    old_style = torch.tensor([0])
    new_style = torch.tensor([1])
    with torch.inference_mode():
        old_logits, cache = generator.prefill(
            seed,
            style_ids=old_style,
            seed_trajectory=trajectory,
            seed_trajectory_valid=valid,
        )
        previous_cache = FSQGeneratorCache(
            layers=[(key[..., :-1, :], value[..., :-1, :]) for key, value in cache.layers],
            next_position=cache.next_position - 1,
        )
        expected_logits, expected_cache = generator.decode_step(
            seed[:, -1],
            previous_cache,
            style_ids=new_style,
            trajectory=trajectory[:, -1],
            trajectory_valid=valid[:, -1],
        )
    controller = object.__new__(RealtimeFSQController)
    controller.conditional = True
    controller.generator = generator
    controller.device = torch.device("cpu")
    controller.style_id = 0
    controller._style_ids = old_style
    controller.token_history = seed
    controller.control_history = trajectory
    controller.control_valid_history = valid
    controller.next_logits = old_logits
    controller.cache = cache
    controller._rebuild_cache = lambda: (_ for _ in ()).throw(AssertionError("cache rebuild"))

    controller.set_style(1)

    assert controller.style_id == 1
    torch.testing.assert_close(controller._style_ids, new_style)
    torch.testing.assert_close(controller.next_logits, expected_logits)
    assert isinstance(controller.cache, FSQGeneratorCache)
    assert controller.cache.next_position == expected_cache.next_position
    assert bool((controller.next_logits != old_logits).any())
