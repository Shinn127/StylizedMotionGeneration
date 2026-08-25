import torch
from torch import nn

from stylized_motion.learning.gradient_probe import compute_gradient_probe
from stylized_motion.learning.runner import _effective_loss_components


def test_gradient_probe_matches_total_gradient_without_populating_gradients():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    inputs = torch.tensor([[1.0], [2.0]])
    prediction = model(inputs)
    first = prediction.square().mean()
    second = (prediction - 1.0).square().mean() * 2.0
    total = first + second

    probe = compute_gradient_probe(
        {"first": first, "second": second},
        total,
        tuple(model.parameters()),
    )

    assert all(parameter.grad is None for parameter in model.parameters())
    assert probe["loss_recompose_error"] == 0.0
    assert probe["total_norm"] > 0.0
    assert probe["component_norm_sum"] >= probe["total_norm"]
    assert probe["components"]["first"]["norm"] > 0.0
    assert probe["components"]["second"]["norm"] > 0.0

    total.backward()
    assert model.weight.grad is not None


def test_gradient_probe_handles_non_differentiable_zero_component():
    parameter = nn.Parameter(torch.tensor(2.0))
    total = parameter.square()
    probe = compute_gradient_probe(
        {"active": total, "inactive": torch.zeros(())},
        total,
        (parameter,),
    )

    assert probe["components"]["inactive"]["norm"] == 0.0
    assert probe["components"]["inactive"]["share"] == 0.0
    assert probe["components"]["inactive"]["projection"] == 0.0


def test_effective_loss_components_apply_common_weights_once():
    values = {
        "loss": torch.tensor(7.0),
        "recon": torch.tensor(1.0),
        "delta": torch.tensor(2.0),
        "root_pos": torch.tensor(0.0),
        "root_rot": torch.tensor(0.0),
        "joint": torch.tensor(0.0),
        "contact": torch.tensor(0.0),
        "foot_slide": torch.tensor(0.0),
        "foot_height": torch.tensor(0.0),
        "base_recon": torch.tensor(2.0),
        "part_edit_transfer": torch.tensor(1.0),
    }
    context = {
        "delta_weight": 1.5,
        "root_pos_weight": 0.1,
        "root_rot_weight": 0.1,
        "joint_weight": 0.5,
        "contact_weight": 0.1,
        "foot_slide_weight": 0.1,
        "foot_height_weight": 0.1,
    }
    components = _effective_loss_components(values, context)
    assert set(components) == {
        "recon", "delta", "root_pos", "root_rot", "joint", "contact",
        "foot_slide", "foot_height", "base_recon", "part_edit_transfer",
    }
    torch.testing.assert_close(sum(components.values()), values["loss"])
