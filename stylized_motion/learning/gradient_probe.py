"""Gradient diagnostics for weighted representation losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _gradient_norms(
    gradients: Sequence[torch.Tensor | None],
) -> tuple[float, torch.Tensor]:
    squared = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squared:
        return 0.0, torch.zeros((), dtype=torch.float32)
    total_squared = torch.stack(squared).sum()
    return float(total_squared.sqrt().cpu()), total_squared


def _gradient_dot(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    products = [
        (left_gradient.detach().float() * right_gradient.detach().float()).sum()
        for left_gradient, right_gradient in zip(left, right)
        if left_gradient is not None and right_gradient is not None
    ]
    if not products:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(products).sum()


def _autograd_grad(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )


def compute_gradient_probe(
    components: Mapping[str, torch.Tensor],
    total_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> dict[str, Any]:
    """Measure component gradients without populating parameter ``.grad``.

    ``components`` must contain the effective, scalar contributions to
    ``total_loss``. The caller remains responsible for the normal backward
    pass after this function returns.
    """
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if total_loss.ndim != 0:
        raise ValueError("total_loss must be scalar")
    for name, value in components.items():
        if value.ndim != 0:
            raise ValueError(f"Gradient component {name!r} must be scalar")

    component_sum = total_loss.new_zeros(())
    component_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    component_norms: dict[str, float] = {}
    for name, value in components.items():
        component_sum = component_sum + value
        gradients = _autograd_grad(value, params)
        component_gradients[name] = gradients
        component_norms[name], _ = _gradient_norms(gradients)

    total_gradients = _autograd_grad(total_loss, params)
    total_norm, total_squared = _gradient_norms(total_gradients)
    norm_sum = sum(component_norms.values())
    result: dict[str, Any] = {
        "total_norm": total_norm,
        "component_norm_sum": float(norm_sum),
        "loss_recompose_error": float((component_sum.detach() - total_loss.detach()).abs().cpu()),
        "components": {},
    }
    denominator = max(norm_sum, 1e-12)
    for name, gradients in component_gradients.items():
        norm = component_norms[name]
        dot = _gradient_dot(gradients, total_gradients)
        projection = float((dot / total_squared.clamp_min(1e-12)).cpu())
        cosine_denominator = max(norm * total_norm, 1e-12)
        cosine = float((dot / cosine_denominator).cpu())
        result["components"][name] = {
            "norm": norm,
            "share": norm / denominator,
            "projection": projection,
            "cosine": cosine,
            "value": float(components[name].detach().float().cpu()),
        }
    return result


__all__ = ["compute_gradient_probe"]
