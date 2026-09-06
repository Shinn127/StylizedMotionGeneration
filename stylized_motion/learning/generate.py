"""Token-generator workflow consuming the canonical data public API."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from stylized_motion.data import build_data_loaders, open_token_store, open_trajectory_store
from stylized_motion.learning.nets.causal_transformer_generator import (
    FSQCanonicalTransformerGenerator,
    FSQConditionalTransformerGenerator,
    FSQCausalTransformerGenerator,
)
from stylized_motion.learning.runner import choose_device, move_batch_to_device, set_seed


CANONICAL_REPRESENTATION_FAMILY = "latent_residual_fsq_v2"


def _model_kind(model_config: Mapping[str, object], *, conditional: bool) -> str:
    kind = str(model_config.get("model_kind", "conditional_generator" if conditional else "generator"))
    if kind not in {"generator", "conditional_generator", "canonical_generator"}:
        raise ValueError(f"Unsupported generator model_kind={kind!r}")
    if conditional and kind == "generator":
        raise ValueError("A trajectory database requires a conditional or canonical generator")
    if not conditional and kind != "generator":
        raise ValueError(f"model_kind={kind!r} requires data.trajectory_database")
    return kind


def build_generator(config: Mapping[str, object], store: Any, *, conditional: bool = False, trajectory_dim: int = 18):
    model_config = config.get("model", config)
    if not isinstance(model_config, Mapping):
        raise ValueError("generator.model must be a mapping")
    kwargs = {
        "num_coordinates": store.num_coordinates,
        "num_levels": store.num_levels,
        "coordinate_embedding_dim": int(model_config.get("coordinate_embedding_dim", 16)),
        "dim": int(model_config.get("d_model", model_config.get("dim", 256))),
        "num_layers": int(model_config.get("num_layers", 6)),
        "num_query_heads": int(model_config.get("num_query_heads", 8)),
        "num_kv_heads": int(model_config.get("num_kv_heads", 4)),
        "ff_dim": int(model_config.get("ff_dim", 768)),
        "dropout": float(model_config.get("dropout", 0.1)),
        "context_frames": int(model_config.get("context_frames", 64)),
        "rope_theta": float(model_config.get("rope_theta", 10000.0)),
        "qk_norm": bool(model_config.get("qk_norm", True)),
        "norm_eps": float(model_config.get("norm_eps", 1e-5)),
    }
    kind = _model_kind(model_config, conditional=conditional)
    if kind == "canonical_generator":
        if str(store.representation_family) != CANONICAL_REPRESENTATION_FAMILY:
            raise ValueError(
                "Canonical Generator requires a latent_residual_fsq_v2 TokenStore; "
                f"got {store.representation_family!r}"
            )
        model = FSQCanonicalTransformerGenerator(
            **kwargs,
            trajectory_dim=int(model_config.get("trajectory_dim", trajectory_dim)),
            trajectory_hidden_dim=int(model_config.get("trajectory_hidden_dim", 128)),
        )
    elif kind == "conditional_generator":
        model = FSQConditionalTransformerGenerator(
            **kwargs,
            num_styles=int(model_config.get("num_styles", 1)),
            trajectory_dim=int(model_config.get("trajectory_dim", trajectory_dim)),
            trajectory_hidden_dim=int(model_config.get("trajectory_hidden_dim", 128)),
            style_embedding_dim=int(model_config.get("style_embedding_dim", 128)),
            style_conditioning=str(model_config.get("style_conditioning", "dynamic_film")),
        )
    else:
        model = FSQCausalTransformerGenerator(**kwargs)
    if model.context_frames < 63:
        raise ValueError("generator context_frames must support the canonical 64-frame target")
    return model


def validate_generator_contract(checkpoint: Mapping[str, object], store: Any) -> None:
    if checkpoint.get("schema_version") != 2:
        raise ValueError("Generator checkpoints must use schema_version=2")
    if checkpoint.get("token_store_checkpoint_sha256") != store.checkpoint_sha256:
        raise ValueError("Generator checkpoint and TokenStore use different tokenizer checkpoints")
    representation = checkpoint.get("representation")
    if not isinstance(representation, Mapping):
        raise ValueError("Generator checkpoint is missing representation metadata")
    store.validate_contract(representation=representation)
    if checkpoint.get("model_family") == "canonical_generator":
        if str(store.representation_family) != CANONICAL_REPRESENTATION_FAMILY:
            raise ValueError("Canonical Generator checkpoint requires a latent_residual_fsq_v2 TokenStore")
    feature_schema = checkpoint.get("feature_schema")
    if not isinstance(feature_schema, Mapping) or dict(feature_schema) != store.feature_schema:
        raise ValueError("Generator checkpoint feature_schema does not match the TokenStore")


def load_generator_checkpoint(path: str | Path, store: Any, device: torch.device):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    validate_generator_contract(checkpoint, store)
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model")
    if not isinstance(model_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("Generator checkpoint requires model_config and model state_dict")
    conditional = "trajectory_dim" in model_config
    model = build_generator({"model": dict(model_config)}, store, conditional=conditional).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return checkpoint, model


def _epoch(model: torch.nn.Module, loader, device: torch.device, *, conditional: bool, optimizer=None, epoch: int = 0, grad_clip_norm: float = 1.0) -> dict[str, float]:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    training = optimizer is not None
    model.train(training)
    nll = 0.0
    correct = 0
    total = 0
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        indices = device_batch["indices"].long()
        inputs = indices[:, :-1]
        targets = indices[:, 1:]
        kwargs: dict[str, Any] = {}
        if conditional:
            kwargs["trajectory"] = device_batch["trajectory"][:, :-1]
            kwargs["trajectory_valid"] = device_batch["trajectory_valid"][:, :-1]
            if isinstance(model, FSQConditionalTransformerGenerator):
                kwargs["style_ids"] = torch.zeros(indices.shape[0], device=device, dtype=torch.long)
        output = model(inputs, **kwargs)
        logits = output["logits"]
        loss = F.cross_entropy(logits.reshape(-1, model.num_levels), targets.reshape(-1))
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        token_count = int(targets.numel())
        nll += float(loss.detach()) * token_count
        correct += int((logits.argmax(dim=-1) == targets).sum())
        total += token_count
    if total == 0:
        raise ValueError("TokenStore split has no samples")
    value = nll / total
    return {
        "nll": value,
        "perplexity": math.exp(min(value, 50.0)),
        "coordinate_accuracy": correct / total,
        "target_frames": float(total // model.num_coordinates),
    }


def train(config: dict[str, object]) -> None:
    data = config.get("data", config)
    training = config.get("training", config)
    sampling = config.get("sampling", {})
    loader_config = config.get("loader", {})
    if not isinstance(data, Mapping) or not isinstance(training, Mapping) or not isinstance(sampling, Mapping) or not isinstance(loader_config, Mapping):
        raise ValueError("generator config must contain data, sampling, loader and training mappings")
    if int(data.get("required_data_schema_version", 0)) != 3:
        raise ValueError("generator workflows require data.required_data_schema_version=3")
    token_database = data.get("token_database")
    if token_database is None:
        raise ValueError("data.token_database is required")
    seed = int(training.get("seed", sampling.get("seed", 3407)))
    set_seed(seed, bool(training.get("deterministic", False)))
    device = choose_device(str(training.get("device", "auto")))
    store = open_token_store(token_database)
    trajectory_database = data.get("trajectory_database")
    trajectory_store = open_trajectory_store(trajectory_database, token_store=store) if trajectory_database is not None else None
    conditional = trajectory_store is not None
    model = build_generator(config, store, conditional=conditional, trajectory_dim=trajectory_store.trajectory_dim if trajectory_store is not None else 18).to(device)
    if isinstance(model, FSQCanonicalTransformerGenerator) and str(store.representation_family) != CANONICAL_REPRESENTATION_FAMILY:
        raise ValueError("Canonical Generator requires an LRP-FSQ V2 TokenStore")
    assembled = build_data_loaders(
        "conditional_generator" if conditional else "generator",
        store,
        trajectory_store=trajectory_store,
        sampling_config=sampling,
        loader_config=loader_config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    output = Path(training.get("output_dir", "outputs/generator"))
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs = int(training.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        train_stats = _epoch(model, assembled.train, device, conditional=conditional, optimizer=optimizer, epoch=epoch, grad_clip_norm=float(training.get("grad_clip_norm", 1.0)))
        val_stats = _epoch(model, assembled.val, device, conditional=conditional, epoch=epoch)
        checkpoint = {
            "schema_version": 2,
            "model_family": "canonical_generator" if isinstance(model, FSQCanonicalTransformerGenerator) else "generator",
            "model_config": model.config,
            "model": model.state_dict(),
            "representation": store.representation_metadata,
            "feature_schema": store.feature_schema,
            "token_store_checkpoint_sha256": store.checkpoint_sha256,
            "style_names": (
                list(store.style_names)
                if conditional and len(store.style_names) == model.num_styles
                else [f"style_{index}" for index in range(model.num_styles)]
            ) if isinstance(model, FSQConditionalTransformerGenerator) else [],
            "epoch": epoch,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "config": config,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, output / "last.pt")
        if val_stats["nll"] < best:
            best = val_stats["nll"]
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps({"epoch": epoch, "train": train_stats, "val": val_stats}))
    store.close()
    if trajectory_store is not None:
        trajectory_store.close()


def generate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    store = open_token_store(args.token_database)
    checkpoint, model = load_generator_checkpoint(args.generator_checkpoint, store, device)
    seed = np.load(args.seed_indices, allow_pickle=False)
    if seed.ndim == 2:
        seed = seed[None]
    if seed.ndim != 3 or seed.shape[-1] != store.num_coordinates:
        raise ValueError("seed_indices must have shape [T,K] or [B,T,K] matching TokenStore")
    seed_tensor = torch.from_numpy(np.asarray(seed, dtype=np.uint8)).to(device).long()
    if isinstance(model, FSQCanonicalTransformerGenerator):
        if args.seed_trajectory is None:
            raise ValueError("Canonical generation requires --seed-trajectory")
        seed_trajectory = np.load(args.seed_trajectory, allow_pickle=False)
        if seed_trajectory.ndim == 2:
            seed_trajectory = seed_trajectory[None]
        if seed_trajectory.shape[:2] != seed_tensor.shape[:2] or seed_trajectory.shape[-1] != model.trajectory_dim:
            raise ValueError("seed_trajectory must have shape [B,T,C] aligned with seed_indices")
        future_trajectory = None
        if args.future_trajectory is not None:
            future_trajectory = np.load(args.future_trajectory, allow_pickle=False)
            if future_trajectory.ndim == 2:
                future_trajectory = future_trajectory[None]
        result = model.generate_controlled(
            seed_tensor,
            args.steps,
            seed_trajectory=torch.from_numpy(np.asarray(seed_trajectory, dtype=np.float32)).to(device),
            future_trajectory=(
                None
                if future_trajectory is None
                else torch.from_numpy(np.asarray(future_trajectory, dtype=np.float32)).to(device)
            ),
            greedy=args.greedy,
            temperature=args.temperature,
        )
    elif isinstance(model, FSQConditionalTransformerGenerator):
        style_ids = torch.zeros(seed_tensor.shape[0], device=device, dtype=torch.long)
        result = model.generate_conditioned(seed_tensor, style_ids, args.steps, greedy=args.greedy, temperature=args.temperature)
    else:
        result = model.generate(seed_tensor, args.steps, temperature=args.temperature, greedy=args.greedy)
    np.save(args.output, result.detach().cpu().numpy().astype(np.uint8))
    print(f"saved={args.output} representation={store.representation_id} shape={tuple(result.shape)}")
    store.close()


def _load_config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("generator config must contain a mapping")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or run the metadata-driven token generator.")
    parser.add_argument("--workflow-mode", choices=["train", "generate"], required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--token-database", type=Path, default=None)
    parser.add_argument("--generator-checkpoint", type=Path, default=None)
    parser.add_argument("--seed-indices", type=Path, default=None)
    parser.add_argument("--seed-trajectory", type=Path, default=None)
    parser.add_argument("--future-trajectory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_indices.npy"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args(argv)
    if args.workflow_mode == "train":
        if args.config is None:
            parser.error("--config is required for generator training")
        train(_load_config(args.config))
        return
    if args.token_database is None or args.generator_checkpoint is None or args.seed_indices is None:
        parser.error("generate requires --token-database, --generator-checkpoint, and --seed-indices")
    generate(args)


if __name__ == "__main__":
    main()
