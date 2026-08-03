"""Generic token generator workflow driven only by TokenStore metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from stylized_motion.data.token_store import TokenDataset, TokenStore, build_token_store
from stylized_motion.learning.nets.causal_transformer_generator import FSQCausalTransformerGenerator
from stylized_motion.learning.representation import load_representation_checkpoint
from stylized_motion.learning.runner import choose_device, set_seed


def build_generator(config: dict[str, object], store: TokenStore) -> FSQCausalTransformerGenerator:
    model_config = config.get("model", config)
    if not isinstance(model_config, dict):
        raise ValueError("generator.model must be a mapping")
    model = FSQCausalTransformerGenerator(
        num_coordinates=store.num_coordinates,
        num_levels=store.num_levels,
        coordinate_embedding_dim=int(model_config.get("coordinate_embedding_dim", 16)),
        dim=int(model_config.get("d_model", model_config.get("dim", 256))),
        num_layers=int(model_config.get("num_layers", 6)),
        num_query_heads=int(model_config.get("num_query_heads", 8)),
        num_kv_heads=int(model_config.get("num_kv_heads", 4)),
        ff_dim=int(model_config.get("ff_dim", 768)),
        dropout=float(model_config.get("dropout", 0.1)),
        context_frames=int(model_config.get("context_frames", store.window_size)),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        qk_norm=bool(model_config.get("qk_norm", True)),
        norm_eps=float(model_config.get("norm_eps", 1e-5)),
    )
    if model.context_frames < store.window_size - 1:
        raise ValueError("generator context_frames is smaller than the TokenStore training window")
    return model


def validate_generator_contract(checkpoint: dict[str, Any], store: TokenStore) -> None:
    if checkpoint.get("schema_version") != 2:
        raise ValueError("Generator checkpoints must use schema_version=2")
    if checkpoint.get("token_store_checkpoint_sha256") != store.checkpoint_sha256:
        raise ValueError("Generator checkpoint and TokenStore use different tokenizer checkpoints")
    representation = checkpoint.get("representation")
    if not isinstance(representation, dict):
        raise ValueError("Generator checkpoint is missing representation metadata")
    store.validate_contract(representation=representation)
    feature_schema = checkpoint.get("feature_schema")
    if not isinstance(feature_schema, dict) or feature_schema != store.feature_schema:
        raise ValueError("Generator checkpoint feature_schema does not match the TokenStore")


def load_generator_checkpoint(path: str | Path, store: TokenStore, device: torch.device):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    validate_generator_contract(checkpoint, store)
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Generator checkpoint requires model_config and model state_dict")
    model = FSQCausalTransformerGenerator(**model_config).to(device)
    if (model.num_coordinates, model.num_levels) != (store.num_coordinates, store.num_levels):
        raise ValueError("Generator model dimensions do not match TokenStore")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model


def _epoch(model, loader, device, optimizer=None, grad_clip_norm=1.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    nll = 0.0
    correct = 0
    total = 0
    for batch in loader:
        indices = batch["indices"].to(device).long()
        if indices.shape[1] < 2:
            raise ValueError("Generator requires token windows with at least two frames")
        logits = model(indices[:, :-1])["logits"]
        targets = indices[:, 1:]
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
        raise ValueError("TokenStore split has no windows")
    value = nll / total
    return {"nll": value, "perplexity": math.exp(min(value, 50.0)), "coordinate_accuracy": correct / total}


def train(config: dict[str, object]) -> None:
    data = config.get("data", config)
    training = config.get("training", config)
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("generator config must contain data/training mappings")
    token_database = data.get("token_database")
    if token_database is None:
        raise ValueError("data.token_database is required")
    if data.get("trajectory_database") is not None:
        raise ValueError("The canonical generator workflow does not implement trajectory conditioning")
    seed = int(training.get("seed", 3407))
    set_seed(seed, bool(training.get("deterministic", False)))
    device = choose_device(str(training.get("device", "auto")))
    store = build_token_store(token_database)
    model = build_generator(config, store).to(device)
    common = {"batch_size": int(training.get("batch_size", 512)), "num_workers": int(training.get("num_workers", 0)), "pin_memory": device.type == "cuda"}
    loaders = {
        split: DataLoader(TokenDataset(split, store), shuffle=split == "train", **common)
        for split in ("train", "val", "test")
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("lr", 3e-4)), weight_decay=float(training.get("weight_decay", 0.01)))
    output = Path(training.get("output_dir", "outputs/generator"))
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs = int(training.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        train_stats = _epoch(model, loaders["train"], device, optimizer, float(training.get("grad_clip_norm", 1.0)))
        val_stats = _epoch(model, loaders["val"], device)
        checkpoint = {
            "schema_version": 2,
            "model_family": "generator",
            "model_config": model.config,
            "model": model.state_dict(),
            "representation": store.representation_metadata,
            "feature_schema": store.feature_schema,
            "token_store_checkpoint_sha256": store.checkpoint_sha256,
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


def generate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    store = build_token_store(args.token_database)
    checkpoint, model = load_generator_checkpoint(args.generator_checkpoint, store, device)
    seed = np.load(args.seed_indices)
    if seed.ndim == 2:
        seed = seed[None]
    if seed.ndim != 3 or seed.shape[-1] != store.num_coordinates:
        raise ValueError("seed_indices must have shape [T,K] or [B,T,K] matching TokenStore")
    seed_tensor = torch.from_numpy(np.asarray(seed, dtype=np.int64)).to(device)
    result = model.generate(seed_tensor, args.steps, temperature=args.temperature, greedy=args.greedy)
    np.save(args.output, result.detach().cpu().numpy().astype(np.uint8))
    print(f"saved={args.output} representation={store.representation_id} shape={tuple(result.shape)}")


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
