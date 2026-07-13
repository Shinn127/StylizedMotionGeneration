import torch
import torch.nn.functional as F

from models.fsq_style_gate import FSQStyleGateExperiment, hard_concrete


def build_small_model() -> FSQStyleGateExperiment:
    return FSQStyleGateExperiment(
        num_coordinates=3,
        num_levels=4,
        num_styles=5,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        ff_dim=32,
        dropout=0.0,
        max_seq_len=8,
    )


def test_style_gate_outputs_coordinate_level_masks_and_logits():
    torch.manual_seed(5)
    model = build_small_model().eval()
    indices = torch.randint(0, 4, (2, 8, 3))
    with torch.no_grad():
        output = model(indices, temperature=0.5)
    assert output["mask"].shape == (2, 3, 4)
    assert output["mask_probability"].shape == (2, 3, 4)
    assert output["dynamic_logits"].shape == (2, 5)
    assert output["full_logits"].shape == (2, 5)
    assert output["random_logits"].shape == (2, 5)
    assert torch.all((output["mask"] == 0.0) | (output["mask"] == 1.0))
    assert torch.all((output["mask_probability"] >= 0.0) & (output["mask_probability"] <= 1.0))


def test_style_loss_reaches_dynamic_gate():
    torch.manual_seed(7)
    model = build_small_model().train()
    indices = torch.randint(0, 4, (3, 8, 3))
    labels = torch.tensor([0, 1, 2])
    output = model(indices, temperature=1.0)
    loss = F.cross_entropy(output["dynamic_logits"], labels) + 0.01 * output["expected_l0"].mean()
    loss.backward()
    assert model.gate.gate_head.weight.grad is not None
    assert torch.isfinite(model.gate.gate_head.weight.grad).all()
    assert model.gate.gate_head.weight.grad.abs().sum() > 0


def test_expected_l0_increases_with_gate_logits():
    low = torch.full((2, 3), -5.0)
    high = torch.full((2, 3), 5.0)
    _, _, low_l0 = hard_concrete(low, temperature=0.5, training=False)
    _, _, high_l0 = hard_concrete(high, temperature=0.5, training=False)
    assert high_l0.mean() > low_l0.mean()
