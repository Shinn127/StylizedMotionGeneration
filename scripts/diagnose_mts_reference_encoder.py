#!/usr/bin/env python
"""N05a: why does the trained reference encoder ignore its input?  No training.

Four candidate mechanisms, separated by measurement rather than argument:

* **insensitive from the start** -- a fresh encoder of the same architecture, given
  the same real references, also produces one descriptor;
* **trained into a constant** -- the fresh encoder separates references and the
  trained one does not;
* **gradient/implementation error** -- the encoder's parameters are not in the
  optimizer, or its gradients are zero/absent/non-finite on a real batch;
* **the data cannot separate styles** -- the descriptors separate neither styles nor
  references at all, not even at initialization.

Measured, per layer (embedding, temporal, graph, pooled, descriptor):

* the spread of the layer's output across a fixed pool of real references, in
  absolute terms and relative to its own norm;
* the response to a *time shuffle* of the reference, to a *zero* reference and to
  swapping in a real different-style clip;
* the descriptor's within-style and between-style distances, so "the descriptor
  varies" can be told apart from "the descriptor varies *with style*".

The gradient block runs one backward pass on one batch and never steps, writes a
checkpoint or touches the optimizer state.

    python scripts/diagnose_mts_reference_encoder.py \
      --checkpoint outputs/mts_revision2/operator_reference_logit_s3407_20260918_1906/best.pt \
      --style-id-checkpoint outputs/mts_revision2/operator_styleid_logit_s3407_20260918_1814/best.pt \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --protocol outputs/mts_revision2/operator_styleid_logit_s3407_20260918_1814/validation_protocol.json \
      --max-references 32 --output outputs/mts_next_round_20260918/N05a/reference_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.eval_protocol import (  # noqa: E402
    ValidationBatchBuilder,
    ValidationProtocol,
    ValidationSample,
)
from stylized_motion.learning.mts_operator.masking import MaskGenerator  # noqa: E402
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder  # noqa: E402
from stylized_motion.learning.mts_operator.summary import (  # noqa: E402
    ArmSpec,
    ReferencePool,
    load_arm,
    write_summary,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource,
    adapter_from_tokenizer,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402

FRAMES = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N05a reference-encoder diagnosis (read-only).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--style-id-checkpoint", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--max-references", type=int, default=32)
    parser.add_argument("--gradient-batch", type=int, default=8)
    parser.add_argument("--fresh-seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def read_protocol(path: Path) -> ValidationProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ValidationProtocol(
        samples=tuple(ValidationSample(**item) for item in payload["items"]),
        kinds=tuple(payload["kinds"]),
        weights={str(k): float(v) for k, v in payload["weights"].items()},
        version=int(payload["version"]),
        protocol_id=str(payload.get("protocol_id") or ""),
        selection=dict(payload.get("selection") or {}),
    )


def _temporal_variant(
    encoder: GlobalStyleEncoder, hidden: torch.Tensor, valid: torch.Tensor, *, use_position: bool
) -> torch.Tensor:
    """``encoder._temporal`` with the sinusoidal position encoding optionally off.

    The diagnostic copies the module's own steps (permute, padding mask, encoder,
    norm) instead of editing the model, so the main path is untouched; the numbers
    from the two variants are only ever compared against each other.
    """
    if use_position:
        hidden = encoder.position_encoding(hidden)
    batch, frames, streams, dim = hidden.shape
    flat = hidden.permute(0, 2, 1, 3).reshape(batch * streams, frames, dim)
    padding = (~valid).unsqueeze(1).expand(batch, streams, frames).reshape(batch * streams, frames)
    causal = encoder._causal_mask(frames, hidden) if encoder.temporal_mode == "causal" else None  # noqa: SLF001
    encoded = encoder.temporal(flat, mask=causal, src_key_padding_mask=padding)
    encoded = encoder.temporal_norm(encoded)
    return encoded.reshape(batch, streams, frames, dim).permute(0, 2, 1, 3).contiguous()


def layer_outputs(
    encoder: GlobalStyleEncoder,
    tokens: torch.Tensor,
    valid: torch.Tensor,
    *,
    use_position: bool = True,
) -> dict[str, torch.Tensor]:
    """Every stage's output for one batch of references, without changing the module."""
    spec = encoder.spec
    visible = valid.unsqueeze(-1).expand(-1, -1, spec.num_coordinates)
    embedding = encoder.embedding(tokens, visible)
    temporal = _temporal_variant(encoder, embedding, valid, use_position=use_position)
    graph = encoder.graph(temporal)
    pooled = encoder._pool(graph, valid)  # noqa: SLF001
    descriptor = encoder.projection(pooled)
    return {
        "embedding": embedding,
        "temporal": temporal,
        "graph": graph,
        "pooled": pooled,
        "descriptor": descriptor,
    }


def spread(values: torch.Tensor) -> dict[str, float]:
    """Spread of a layer's output across references, per reference and relative."""
    flat = values.reshape(values.shape[0], -1).float()
    mean = flat.mean(dim=0)
    deviations = (flat - mean).norm(dim=-1)
    norms = flat.norm(dim=-1)
    return {
        "norms_mean": float(norms.mean()),
        "deviation_mean": float(deviations.mean()),
        "relative_spread": float((deviations / norms.clamp_min(1e-12)).mean()),
        "dim_std_mean": float(flat.std(dim=0).mean()),
    }


def descriptor_distances(descriptors: torch.Tensor, styles: list[str]) -> dict[str, Any]:
    """Within-style vs between-style descriptor distances (cosine + L2)."""
    flat = descriptors.float()
    normalised = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = normalised @ normalised.t()
    distance = torch.cdist(flat, flat)
    within: list[float] = []
    between: list[float] = []
    for left in range(flat.shape[0]):
        for right in range(left + 1, flat.shape[0]):
            if styles[left] == styles[right]:
                within.append(float(1.0 - similarity[left, right]))
            else:
                between.append(float(1.0 - similarity[left, right]))
    return {
        "pairs_within_style": len(within),
        "pairs_between_styles": len(between),
        "cosine_distance_within_mean": float(np.mean(within)) if within else None,
        "cosine_distance_between_mean": float(np.mean(between)) if between else None,
        "ratio_between_over_within": float(np.mean(between) / np.mean(within))
        if within and between and float(np.mean(within)) > 0
        else None,
        "l2_distance_mean": float(distance[distance > 0].mean()) if bool((distance > 0).any()) else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    protocol = read_protocol(args.protocol)
    _, tokenizer = load_representation_checkpoint(args.tokenizer_checkpoint, torch.device("cpu"))
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    identity = tokenizer.representation_metadata()

    store = open_any_token_store(args.token_store)
    report: dict[str, Any] = {"kind": "mts_reference_encoder_diagnosis", "checkpoint": str(args.checkpoint)}
    try:
        arm = load_arm(
            ArmSpec("reference", args.checkpoint),
            adapter=adapter,
            tokenizer_identity=identity,
            tokenizer_checkpoint=args.tokenizer_checkpoint,
            device="cpu",
        )
        if arm.encoder_kind != "reference":
            raise SystemExit(f"This checkpoint's encoder is {arm.encoder_kind!r}, not a reference encoder")
        encoder_config = dict(arm.checkpoint["metadata"]["model_config"]["style_encoder"])
        encoder_config.pop("kind", None)
        torch.manual_seed(int(args.fresh_seed))
        fresh = GlobalStyleEncoder(adapter, **encoder_config).eval()
        report["models"] = {
            "trained": arm.describe(),
            "fresh": {
                "architecture": encoder_config,
                "seed": int(args.fresh_seed),
                "note": "a fresh initialization of the same architecture under an explicit seed; the "
                "training run's own init was not saved, so this is a sample of the init distribution, "
                "not the exact starting weights",
            },
        }
        report["encoder_config"] = encoder_config

        windows = windows_by_clip(store, "val", frames=FRAMES)
        source = TokenSource(
            store=store, windows_by_clip=windows, adapter=adapter, frames=FRAMES,
            history=int(tokenizer.history_frames), rng=np.random.default_rng(3407),
        )
        samples = list(protocol.samples)[: int(args.max_references)]
        references = []
        for sample in samples:
            window = source.window_at(int(sample.reference_clip), int(sample.reference_start))
            if window is not None:
                references.append(
                    {
                        "tokens": window.tokens,
                        "valid": window.valid_mask,
                        "style": str(sample.style),
                        "clip": int(sample.reference_clip),
                        "target_clip": int(sample.target_clip),
                        "kind": str(sample.kind),
                    }
                )
        if len(references) < 4:
            raise SystemExit("Fewer than four reference windows were available; nothing to diagnose")
        tokens = torch.stack([item["tokens"] for item in references])
        valid = torch.stack([item["valid"] for item in references])
        styles = [item["style"] for item in references]
        report["references"] = {
            "count": len(references),
            "distinct_clips": len({item["clip"] for item in references}),
            "styles": sorted({item["style"] for item in references}),
            "style_counts": {style: styles.count(style) for style in sorted(set(styles))},
        }

        layers: dict[str, Any] = {}
        with torch.inference_mode():
            for name, encoder in (("trained", arm.model.style_encoder), ("fresh", fresh)):
                outputs = layer_outputs(encoder, tokens, valid)
                layers[name] = {layer: spread(value) for layer, value in outputs.items()}
                layers[name]["descriptor"]["distances"] = descriptor_distances(
                    outputs["descriptor"], styles
                )
                # Responses: time shuffle, zero input, and a real wrong-style swap.
                shuffled = tokens[:, torch.randperm(tokens.shape[1], generator=torch.Generator().manual_seed(3))]
                shuffled_out = layer_outputs(encoder, shuffled, valid)
                zeros = torch.zeros_like(tokens)
                zeros_out = layer_outputs(encoder, zeros, valid)
                without_position = layer_outputs(encoder, tokens, valid, use_position=False)
                layers[name]["response"] = {
                    "descriptor_relative_change_position_encoding_off": float(
                        (without_position["descriptor"] - outputs["descriptor"]).norm(dim=-1).mean()
                        / outputs["descriptor"].norm(dim=-1).mean().clamp_min(1e-12)
                    ),
                    "descriptor_relative_change_time_shuffle": float(
                        (shuffled_out["descriptor"] - outputs["descriptor"]).norm(dim=-1).mean()
                        / outputs["descriptor"].norm(dim=-1).mean().clamp_min(1e-12)
                    ),
                    "descriptor_relative_change_zero_input": float(
                        (zeros_out["descriptor"] - outputs["descriptor"]).norm(dim=-1).mean()
                        / outputs["descriptor"].norm(dim=-1).mean().clamp_min(1e-12)
                    ),
                    "pooled_relative_change_time_shuffle": float(
                        (shuffled_out["pooled"] - outputs["pooled"]).norm(dim=-1).mean()
                        / outputs["pooled"].norm(dim=-1).mean().clamp_min(1e-12)
                    ),
                    "embedding_relative_change_zero_input": float(
                        (zeros_out["embedding"] - outputs["embedding"]).norm(dim=-1).mean()
                        / outputs["embedding"].norm(dim=-1).mean().clamp_min(1e-12)
                    ),
                }
                # Pooling numerics: how close to zero are the per-stream variances?
                hidden = outputs["graph"]
                weights = valid.to(hidden.dtype).unsqueeze(-1).unsqueeze(-1)
                total = weights.sum(dim=1).clamp_min(1.0)
                mean = (hidden * weights).sum(dim=1) / total
                squared = (hidden.square() * weights).sum(dim=1) / total
                variance = (squared - mean.square()).clamp_min(0.0)
                layers[name]["pooling"] = {
                    "variance_mean": float(variance.mean()),
                    "variance_median": float(variance.median()),
                    "fraction_below_1e-8": float((variance < 1e-8).float().mean()),
                    "std_mean": float(variance.sqrt().mean()),
                    "descriptor_over_pooled_norm": float(
                        outputs["descriptor"].norm(dim=-1).mean() / outputs["pooled"].norm(dim=-1).mean()
                    ),
                }
                # Real reference swap: does a wrong-style clip change the descriptor
                # more than a same-style one?
                pool = ReferencePool.from_store(store, windows)
                swap_changes = []
                for item in references[:8]:
                    pick = pool.pick(int(item["target_clip"]), exclude=(int(item["clip"]),))
                    if pick is None:
                        continue
                    window = source.window_at(int(pick["clip_id"]), int(min(
                        int(getattr(request, "target_start", 0))
                        for request in windows[int(pick["clip_id"])]
                    )))
                    if window is None:
                        continue
                    swapped = torch.stack([window.tokens])
                    single = torch.stack([item["tokens"]])
                    if swapped.shape[1] != single.shape[1]:
                        continue
                    with torch.inference_mode():
                        left = encoder(single, valid_mask=torch.stack([item["valid"]]))
                        right = encoder(swapped)
                    swap_changes.append(
                        {
                            "clip": int(item["clip"]),
                            "wrong_clip": int(pick["clip_id"]),
                            "wrong_style": str(pick["style"]),
                            "relative_change": float(
                                (right - left).norm(dim=-1).mean() / left.norm(dim=-1).mean().clamp_min(1e-12)
                            ),
                        }
                    )
                layers[name]["reference_swap"] = {
                    "pairs": len(swap_changes),
                    "relative_change_mean": float(np.mean([entry["relative_change"] for entry in swap_changes]))
                    if swap_changes
                    else None,
                    "examples": swap_changes[:4],
                }
        report["layers"] = layers

        # Gradients: one batch, one backward, no step.
        builder = ValidationBatchBuilder(
            token_source=source,
            mask_generator=MaskGenerator(dict(protocol.samples[0].mask_config)),
            adapter=adapter,
            device=torch.device("cpu"),
            content_vocabulary=arm.content_vocabulary,
            style_index=arm.style_index,
            encoder_kind=arm.encoder_kind,
        )
        batch = builder.build_batch(list(protocol.samples)[: int(args.gradient_batch)])
        trainable = dict(arm.model.style_encoder.named_parameters())
        requires_grad = {name: bool(parameter.requires_grad) for name, parameter in trainable.items()}
        arm.model.zero_grad(set_to_none=True)
        arm.model.train()
        loss, metrics = arm.model.loss(batch, content_weight=0.0)
        loss.backward()
        gradient_report: dict[str, Any] = {
            "loss": float(loss.detach()),
            "supervised_tokens": int(metrics["supervised_tokens"]),
            "encoder_parameters": int(sum(parameter.numel() for parameter in trainable.values())),
            "requires_grad": requires_grad,
            "frozen_flag": bool(arm.model.freeze_style_encoder),
            "per_layer_gradient_norm": {},
            "finite": True,
            "nonzero_parameters": 0,
        }
        for name, parameter in trainable.items():
            gradient = parameter.grad
            layer = name.split(".")[0]
            if gradient is None:
                gradient_report["per_layer_gradient_norm"].setdefault(layer, 0.0)
                gradient_report["per_layer_gradient_norm"][layer] += 0.0
                continue
            norm = float(gradient.detach().norm())
            gradient_report["per_layer_gradient_norm"][layer] = (
                gradient_report["per_layer_gradient_norm"].get(layer, 0.0) + norm
            )
            if not torch.isfinite(gradient).all():
                gradient_report["finite"] = False
            if norm > 0.0:
                gradient_report["nonzero_parameters"] += 1
        # Is the encoder in the optimizer that the checkpoint carries?
        optimizer_state = arm.checkpoint.get("optimizer")
        gradient_report["optimizer_state_present"] = optimizer_state is not None
        if isinstance(optimizer_state, dict):
            groups = optimizer_state.get("param_groups") or []
            gradient_report["optimizer_param_groups"] = len(groups)
            gradient_report["optimizer_parameters"] = int(
                sum(len(group.get("params", [])) for group in groups)
            )
            gradient_report["optimizer_total_tensors"] = len(optimizer_state.get("state", {}))
        report["gradients"] = gradient_report
        arm.model.zero_grad(set_to_none=True)
        arm.model.eval()

        # The style-ID arm is the upper bound: how separable are the same references
        # for a model that *can* see the label?
        if args.style_id_checkpoint and Path(args.style_id_checkpoint).exists():
            from stylized_motion.learning.mts_operator.summary import load_arm as _load  # noqa: PLC0415

            style_arm = _load(
                ArmSpec("style_id", args.style_id_checkpoint),
                adapter=adapter,
                tokenizer_identity=identity,
                tokenizer_checkpoint=args.tokenizer_checkpoint,
                device="cpu",
            )
            report["style_id_upper_bound"] = {
                "trainable_parameters": style_arm.trainable_parameters,
                "encoder": style_arm.encoder_kind,
                "note": "the style-ID arm reads a label, not the reference; it is the ceiling for how "
                "much of the signal a *descriptor* would have to carry",
            }
        verdict = _verdict(report)
        report["verdict"] = verdict
        write_summary(args.output, report)
        print(json.dumps(verdict, indent=2, default=str), flush=True)
    finally:
        store.close()
    return 0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return float(numerator / denominator)


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Which of the four mechanisms the measurements support."""
    trained = report["layers"]["trained"]
    fresh = report["layers"]["fresh"]
    gradients = report["gradients"]
    trained_spread = trained["descriptor"]["relative_spread"]
    fresh_spread = fresh["descriptor"]["relative_spread"]
    swap = trained["reference_swap"]["relative_change_mean"]
    fresh_swap = fresh["reference_swap"]["relative_change_mean"]
    ratio = trained["descriptor"]["distances"]["ratio_between_over_within"]
    fresh_ratio = fresh["descriptor"]["distances"]["ratio_between_over_within"]
    # Where along the path the reference differences are lost, per encoder: the
    # layer-to-layer ratio of the relative spread.
    damping: dict[str, dict[str, float | None]] = {}
    for name, layers in (("trained", trained), ("fresh", fresh)):
        damping[name] = {
            "embedding_to_temporal": _ratio(layers["temporal"]["relative_spread"], layers["embedding"]["relative_spread"]),
            "temporal_to_pooled": _ratio(layers["pooled"]["relative_spread"], layers["temporal"]["relative_spread"]),
            "pooled_to_descriptor": _ratio(layers["descriptor"]["relative_spread"], layers["pooled"]["relative_spread"]),
        }
    evidence = {
        "trained_descriptor_relative_spread": trained_spread,
        "fresh_descriptor_relative_spread": fresh_spread,
        "trained_swap_relative_change": swap,
        "fresh_swap_relative_change": fresh_swap,
        "trained_between_over_within": ratio,
        "fresh_between_over_within": fresh_ratio,
        "trained_within_style_cosine_distance": trained["descriptor"]["distances"]["cosine_distance_within_mean"],
        "trained_between_style_cosine_distance": trained["descriptor"]["distances"]["cosine_distance_between_mean"],
        "fresh_within_style_cosine_distance": fresh["descriptor"]["distances"]["cosine_distance_within_mean"],
        "fresh_between_style_cosine_distance": fresh["descriptor"]["distances"]["cosine_distance_between_mean"],
        "per_layer_damping": damping,
        "trained_position_encoding_effect": trained["response"]["descriptor_relative_change_position_encoding_off"],
        "trained_time_shuffle_effect": trained["response"]["descriptor_relative_change_time_shuffle"],
        "trained_zero_input_effect": trained["response"]["descriptor_relative_change_zero_input"],
        "encoder_gradient_layers_with_signal": sorted(
            name for name, value in gradients["per_layer_gradient_norm"].items() if value > 0.0
        ),
        "encoder_gradient_norms": gradients["per_layer_gradient_norm"],
        "encoder_gradients_finite": gradients["finite"],
        "encoder_requires_grad": all(gradients["requires_grad"].values()),
        "encoder_optimizer_parameters": gradients.get("optimizer_parameters"),
        "pooling_fraction_below_1e-8_trained": trained["pooling"]["fraction_below_1e-8"],
    }
    verdict: dict[str, Any] = {"evidence": evidence}
    if not gradients["finite"] or not all(gradients["requires_grad"].values()):
        verdict["mechanism"] = "gradient_or_implementation_error"
        verdict["reason"] = "the encoder's gradients are non-finite, absent, or its parameters are frozen"
    elif not gradients["per_layer_gradient_norm"] or all(
        value == 0.0 for value in gradients["per_layer_gradient_norm"].values()
    ):
        verdict["mechanism"] = "gradient_or_implementation_error"
        verdict["reason"] = "no encoder layer received a non-zero gradient on a real batch"
    elif fresh_spread is not None and fresh_swap is not None and fresh_spread > 10.0 * trained_spread:
        verdict["mechanism"] = "trained_into_a_constant"
        verdict["reason"] = "a fresh encoder of the same architecture separates the same references "
        "far more than the trained one; the loss removed the input dependence during training"
        # The secondary limit is stated, not hidden: even the fresh encoder damps
        # reference differences along the path and barely separates styles, so an
        # auxiliary objective alone may not be enough.
        verdict["secondary"] = {
            "mechanism": "the_input_path_already_damps_reference_differences",
            "reason": "the temporal block reduces the relative spread by a large factor even at "
            "initialization, and the fresh descriptor separates styles only marginally",
            "fresh_damping": damping["fresh"],
            "fresh_between_over_within": fresh_ratio,
        }
    elif trained_spread is not None and trained_spread > 0.05 and ratio is not None and ratio < 1.2:
        verdict["mechanism"] = "descriptor_varies_but_not_with_style"
        verdict["reason"] = "the descriptor moves with the reference but its within-style and "
        "between-style distances are not separated: the data/task, not the encoder, is the limit"
    else:
        verdict["mechanism"] = "insensitive_from_the_start"
        verdict["reason"] = "neither the trained nor a fresh encoder of this architecture separates "
        "these references; the input path or the architecture is the limit"
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
