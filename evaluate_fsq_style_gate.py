from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.fsq_token_dataset import FSQTokenDataset, build_fsq_token_store
from models.fsq_style_gate import FSQStyleGateExperiment
from train_fsq_style_gate import choose_device, classification_metrics, update_confusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and export dynamic FSQ style masks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-database", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masks-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    store = build_fsq_token_store(args.token_database)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq_style_gate":
        raise ValueError(f"Unsupported checkpoint family: {checkpoint.get('model_family')}")
    if checkpoint["tokenizer_checkpoint_sha256"] != store.checkpoint_sha256:
        raise ValueError("Gate checkpoint and token database reference different FSQ tokenizer checkpoints")
    if checkpoint["style_names"] != store.style_names:
        raise ValueError("Gate checkpoint and token database use different style vocabularies")
    model = FSQStyleGateExperiment(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    temperature = float(args.temperature if args.temperature is not None else checkpoint["temperature"])

    dataset = FSQTokenDataset(args.split, store)
    if args.max_samples is not None and args.max_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        selection = np.sort(rng.choice(len(dataset), size=args.max_samples, replace=False)).tolist()
        dataset = Subset(dataset, selection)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    num_styles = len(store.style_names)
    confusion = {name: torch.zeros(num_styles, num_styles, dtype=torch.int64) for name in ("dynamic", "full", "random")}
    ce_totals = {name: 0.0 for name in confusion}
    masks = []
    probabilities = []
    predictions = []
    style_ids = []
    action_ids = []
    range_names = []
    start_indices = []
    end_indices = []
    mirrors = []
    consistency_total = 0.0
    count = 0

    with torch.inference_mode():
        for batch in loader:
            take = len(batch["style_id"])
            indices = batch["indices"][:take].to(device)
            labels = batch["style_id"][:take].to(device).long()
            output = model(indices, temperature=temperature, stochastic=False, hard=True)
            for name in confusion:
                logits = output[f"{name}_logits"]
                ce_totals[name] += float(F.cross_entropy(logits, labels, reduction="sum"))
                update_confusion(confusion[name], labels, logits.argmax(dim=-1))

            crop_length = min(48, indices.shape[1] - 1)
            if crop_length > 0:
                first = indices[:, :crop_length]
                second = indices[:, -crop_length:]
                first_mask = model.gate_tokens(first, temperature, stochastic=False, hard=False)["mask_probability"]
                second_mask = model.gate_tokens(second, temperature, stochastic=False, hard=False)["mask_probability"]
                consistency_total += float(F.l1_loss(first_mask, second_mask, reduction="sum"))

            masks.append(output["mask"].cpu().numpy().astype(np.uint8))
            probabilities.append(output["mask_probability"].cpu().numpy().astype(np.float32))
            predictions.append(output["dynamic_logits"].argmax(dim=-1).cpu().numpy().astype(np.int32))
            style_ids.append(labels.cpu().numpy().astype(np.int32))
            action_ids.append(batch["action_id"][:take].numpy().astype(np.int32))
            range_names.extend(str(name) for name in batch["range_name"][:take])
            start_indices.append(batch["start_idx"][:take].numpy().astype(np.int32))
            end_indices.append(batch["end_idx"][:take].numpy().astype(np.int32))
            mirrors.append(batch["mirror"][:take].numpy().astype(bool))
            count += take
    if count == 0:
        raise ValueError("No samples were evaluated")

    mask_array = np.concatenate(masks, axis=0)
    probability_array = np.concatenate(probabilities, axis=0)
    prediction_array = np.concatenate(predictions)
    style_array = np.concatenate(style_ids)
    action_array = np.concatenate(action_ids)
    hard = mask_array.astype(bool)
    probability_clipped = np.clip(probability_array, 1e-7, 1.0 - 1e-7)
    entropy = -(
        probability_clipped * np.log(probability_clipped)
        + (1.0 - probability_clipped) * np.log(1.0 - probability_clipped)
    )
    metrics = {}
    for name, matrix in confusion.items():
        metrics[name] = {
            **classification_metrics(matrix),
            "cross_entropy": ce_totals[name] / count,
        }
    metrics["mask"] = {
        "mean_active_pair_ratio": float(hard.mean()),
        "mean_active_coordinates": float(hard.any(axis=-1).sum(axis=-1).mean()),
        "mean_active_levels_per_coordinate": float(hard.sum(axis=-1).mean()),
        "mean_entropy": float(entropy.mean()),
        "prefix_suffix_consistency_l1": consistency_total / (count * store.num_coordinates * store.num_levels),
    }
    mean_mask_by_style = [
        probability_array[style_array == style_id].mean(axis=0).tolist()
        if np.any(style_array == style_id)
        else None
        for style_id in range(num_styles)
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "token_database": str(args.token_database),
        "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
        "split": args.split,
        "num_samples": count,
        "temperature": temperature,
        "style_names": store.style_names,
        "action_names": store.action_names,
        "metrics": metrics,
        "mean_mask": probability_array.mean(axis=0).tolist(),
        "mean_mask_by_style": mean_mask_by_style,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    masks_output = args.masks_output or args.output.with_suffix(".masks.npz")
    masks_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        masks_output,
        masks=mask_array,
        mask_probabilities=probability_array,
        predictions=prediction_array,
        style_ids=style_array,
        action_ids=action_array,
        range_names=np.asarray(range_names, dtype=object),
        start_indices=np.concatenate(start_indices),
        end_indices=np.concatenate(end_indices),
        mirrors=np.concatenate(mirrors),
    )
    print(json.dumps(metrics, indent=2))
    print(f"report={args.output}")
    print(f"masks={masks_output}")


if __name__ == "__main__":
    main()
