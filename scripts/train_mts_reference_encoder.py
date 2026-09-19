#!/usr/bin/env python
"""N05b-2: train the reference encoder on the simplest seen-style task.

The plan's first controlled step for the reference branch: freeze everything else,
give the encoder a *classification* objective on the three trained styles, and ask
afterwards not only "is the accuracy high" but whether the descriptor keeps a
margin and responds to a real reference swap.  The operator is not touched, so a
failure here is a statement about the encoder input path and the data, not about
the operator's conditioning.

Data: balanced three-class reference windows from the *train* split, sampled per
take so a step never sees the same take twice more than the pool forces; validation
is the val split (unseen takes), reported per style and, where the data allows, on
contents the fit set never saw for that style.

Writes ``best.pt`` (kind ``style_encoder``) plus history, summary, the class list
and the training recipe, all in one fresh run directory.

    python scripts/train_mts_reference_encoder.py \
      --config data/configs/mts_reference_encoder_v1.yaml --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    build_provenance,
    code_identity,
    file_sha256,
    mts_checkpoint_payload,
    save_mts_checkpoint,
    require_token_store_binding,
    store_identity_block,
)
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder  # noqa: E402
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource,
    adapter_from_tokenizer,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402
from stylized_motion.learning.runner import choose_device  # noqa: E402

FRAMES = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MTS reference encoder on the seen-style task.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def pool_for(
    records: list[Any], windows: dict[int, list[Any]], *, split: str, classes: set[str]
) -> dict[str, list[tuple[int, str, int]]]:
    pool: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for record in records:
        if str(record.split) != str(split) or str(record.style) not in classes:
            continue
        if int(record.clip_id) not in windows:
            continue
        pool[str(record.style)].append(
            (int(record.clip_id), str(record.content), int(record.source_group))
        )
    return pool


def sample_batch(
    source: TokenSource,
    encoder: nn.Module,
    pools: dict[str, list[tuple[int, str, int]]],
    classes: list[str],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """One balanced batch: ``batch_size`` windows, round-robin over the classes."""
    per_class = max(1, int(batch_size) // len(classes))
    tokens: list[torch.Tensor] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for position, style in enumerate(classes):
        candidates = pools.get(style) or []
        if not candidates:
            raise ValueError(f"No {style!r} clip has a full window; the class cannot be trained")
        chosen = rng.choice(len(candidates), size=per_class, replace=False)
        for index in chosen:
            clip, content, take = candidates[int(index)]
            window = source.window_at(int(clip), _first_start(source, int(clip)))
            if window is None:
                continue
            tokens.append(window.tokens)
            labels.append(position)
            metadata.append({"clip": int(clip), "style": style, "content": content, "take": int(take)})
    return torch.stack(tokens), torch.tensor(labels, dtype=torch.long), metadata


def _first_start(source: TokenSource, clip: int) -> int:
    entries = source.windows_by_clip[int(clip)]
    return int(min(int(getattr(request, "target_start", 0)) for request in entries))


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    source: TokenSource,
    pools: dict[str, list[tuple[int, str, int]]],
    classes: list[str],
    *,
    per_class: int,
    rng: np.random.Generator,
    seen_contents: dict[str, set[str]],
    device: torch.device | str,
) -> dict[str, Any]:
    encoder.eval()
    rows: list[tuple[np.ndarray, int, str, int]] = []
    descriptor_norms: list[float] = []
    for position, style in enumerate(classes):
        candidates = list(pools.get(style) or [])
        rng.shuffle(candidates)
        taken = 0
        for clip, content, take in candidates:
            if taken >= int(per_class):
                break
            window = source.window_at(int(clip), _first_start(source, int(clip)))
            if window is None:
                continue
            tokens = window.tokens[None].to(device)
            valid = window.valid_mask[None].to(device)
            descriptor = encoder(tokens, valid_mask=valid)
            logits = head(descriptor)
            rows.append((logits[0].float().cpu().numpy(), position, content, int(take)))
            descriptor_norms.append(float(descriptor[0].norm()))
            taken += 1
    if not rows:
        return {"rows": 0}
    scores = np.stack([row[0] for row in rows])
    labels = np.asarray([row[1] for row in rows])
    contents = [row[2] for row in rows]
    takes = np.asarray([row[3] for row in rows])
    predicted = scores.argmax(axis=1)
    per_class_recall = {
        style: float((predicted[labels == index] == index).mean())
        for index, style in enumerate(classes)
    }
    unseen = np.asarray([content not in seen_contents.get(classes[label], set()) for content, label in zip(contents, labels)])
    report: dict[str, Any] = {
        "rows": len(rows),
        "descriptor_norm_mean": float(np.mean(descriptor_norms)) if descriptor_norms else None,
        "balanced_accuracy": float(np.mean(list(per_class_recall.values()))),
        "per_class_recall": per_class_recall,
        "per_class_support": {style: int((labels == index).sum()) for index, style in enumerate(classes)},
    }
    if bool(unseen.any()):
        unseen_predicted = predicted[unseen]
        unseen_labels = labels[unseen]
        report["unseen_content"] = {
            "rows": int(unseen.sum()),
            "balanced_accuracy": float(
                np.mean(
                    [
                        float((unseen_predicted[unseen_labels == index] == index).mean())
                        for index in range(len(classes))
                        if bool((unseen_labels == index).any())
                    ]
                )
            ),
        }
    rng_boot = np.random.default_rng(0)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, take in enumerate(takes):
        groups[int(take)].append(index)
    clusters = list(groups.values())
    draws = 1000
    values = np.empty(draws)
    for draw in range(draws):
        picked = rng_boot.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[int(p)] for p in picked])
        pred, truth = predicted[idx], labels[idx]
        values[draw] = float(
            np.mean(
                [
                    float((pred[truth == index] == index).mean())
                    for index in range(len(classes))
                    if bool((truth == index).any())
                ]
            )
        )
    report["bootstrap"] = {
        "draws": draws,
        "clusters": len(clusters),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }
    encoder.train()
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps is not None:
        config["training"]["max_steps"] = int(args.max_steps)
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if args.output is not None:
        config["training"]["output_dir"] = str(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(int(config["training"]["seed"]))
    device = choose_device(args.device)

    tokenizer_path = args.tokenizer_checkpoint or Path(config["tokenizer"]["checkpoint"])
    _, tokenizer = load_representation_checkpoint(tokenizer_path, torch.device("cpu"))
    tokenizer = tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    identity = tokenizer.representation_metadata()

    token_store_path = args.token_store or Path(config["data"]["token_store"])
    store = open_any_token_store(token_store_path)
    output = Path(config["training"]["output_dir"])
    try:
        require_token_store_binding(
            store, tokenizer_checkpoint=tokenizer_path, where="reference-encoder token store"
        )
        records = clip_records_from_store(store)
        style_split = split_styles_by_performer(records, seed=int(config["training"]["seed"]))
        # The classes are the *pairable* three (the same set the operator's style
        # index holds), unless the recipe names them: 'hurry to neutral' is in the
        # train styles but can never be paired, and a one-class-per-style pool of 96
        # clips would make "balanced three classes" untrue.
        declared = config["data"].get("classes")
        if declared:
            classes = sorted(str(value) for value in declared)
        else:
            from stylized_motion.learning.mts_operator.pairs import StylePairSampler

            sampler = StylePairSampler(
                records, style_split=style_split, seed=int(config["training"]["seed"]), window_frames=int(config["data"].get("frames", FRAMES))
            )
            classes = sorted(
                {
                    str(target.style)
                    for target in sampler.eligible_targets(stage="train")
                    if sampler.pairs_for(target, mode="same_style", count=1, stage="train")
                }
            )
        if len(classes) < 2:
            raise SystemExit(f"Fewer than two trainable styles: {classes}")
        frames = int(config["data"].get("frames", FRAMES))
        train_windows = windows_by_clip(store, "train", frames=frames)
        val_windows = windows_by_clip(store, "val", frames=frames)
        train_pools = pool_for(records, train_windows, split="train", classes=set(classes))
        val_pools = pool_for(records, val_windows, split="val", classes=set(classes))
        train_source = TokenSource(
            store=store, windows_by_clip=train_windows, adapter=adapter, frames=frames,
            history=int(tokenizer.history_frames), rng=np.random.default_rng(int(config["training"]["seed"])),
        )
        val_source = TokenSource(
            store=store, windows_by_clip=val_windows, adapter=adapter, frames=frames,
            history=int(tokenizer.history_frames), rng=np.random.default_rng(int(config["training"]["seed"]) + 1),
        )
        encoder = GlobalStyleEncoder(adapter, **dict(config["style_encoder"])).to(device).train()
        head = nn.Linear(int(encoder.output_dim), len(classes)).to(device)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(head.parameters()),
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        steps = int(config["training"]["max_steps"])
        batch_size = int(config["training"].get("batch_size", 48))
        rng = np.random.default_rng(int(config["training"]["seed"]))
        plan = {
            "config": str(args.config),
            "classes": classes,
            "train_pool": {style: len(values) for style, values in train_pools.items()},
            "val_pool": {style: len(values) for style, values in val_pools.items()},
            "encoder_config": dict(config["style_encoder"]),
            "steps": steps,
            "batch_size": batch_size,
            "lr": float(config["training"]["lr"]),
            "seed": int(config["training"]["seed"]),
            "output": str(output),
            "device": str(device),
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2, default=str))
            return 0
        if (output / "best.pt").exists():
            raise SystemExit(f"{output} already holds a run; pick a fresh directory")
        output.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        best_accuracy = -1.0
        seen_contents: dict[str, set[str]] = defaultdict(set)
        started = time.time()
        for step in range(1, steps + 1):
            tokens, labels, metadata = sample_batch(
                train_source, encoder, train_pools, classes, batch_size=batch_size, rng=rng
            )
            for entry in metadata:
                seen_contents[entry["style"]].add(entry["content"])
            descriptor = encoder(tokens.to(device))
            logits = head(descriptor)
            loss = nn.functional.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                float(config["training"].get("grad_clip_norm", 1.0)),
            )
            optimizer.step()
            if step % int(config["training"].get("log_every_steps", 50)) == 0 or step == steps:
                report = evaluate(
                    encoder, head, val_source, val_pools, classes,
                    per_class=int(config["training"].get("eval_per_class", 100)),
                    rng=np.random.default_rng(step), seen_contents=seen_contents, device=device,
                )
                entry = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "train_accuracy": float((logits.argmax(-1) == labels.to(device)).float().mean()),
                    "val": report,
                    "seconds": time.time() - started,
                }
                history.append(entry)
                print(
                    f"step {step}: train loss {entry['train_loss']:.4f} acc {entry['train_accuracy']:.3f} | "
                    f"val balanced {report.get('balanced_accuracy'):.4f} "
                    f"(unseen content {report.get('unseen_content', {}).get('balanced_accuracy')})",
                    flush=True,
                )
                if report.get("balanced_accuracy", -1.0) > best_accuracy:
                    best_accuracy = float(report["balanced_accuracy"])
                    save_mts_checkpoint(
                        output / "best.pt",
                        mts_checkpoint_payload(
                            kind="style_encoder",
                            model=encoder,
                            model_config={
                                "style_encoder": {**dict(config["style_encoder"]), "kind": "reference"},
                                "classes": classes,
                                "head": {"in_features": int(head.in_features), "out_features": int(head.out_features)},
                            },
                            token_spec=adapter.token_spec(representation_id=str(identity.get("representation_id", ""))),
                            tokenizer_metadata=identity,
                            metrics={
                                "task": "seen_style_classification",
                                "step": step,
                                "train_loss": float(loss.detach()),
                                "val_balanced_accuracy": float(report["balanced_accuracy"]),
                                "val_unseen_content_accuracy": report.get("unseen_content", {}).get("balanced_accuracy"),
                                "classes": classes,
                                "optimizer_steps": step,
                            },
                            epoch=1,
                            global_step=step,
                            tokenizer_checkpoint=tokenizer_path,
                            provenance=build_provenance(
                                store_identity=store_identity_block(
                                    store, store_kind="token", store_path=token_store_path
                                ),
                                resolved_config=config,
                                seed=int(config["training"]["seed"]),
                                training_protocol_id=str(config["training"].get("protocol_id", "mts-reference-encoder-seen-style-v1")),
                            ),
                        ),
                    )
        summary = {
            "completed": True,
            "optimizer_steps": steps,
            "steps_shortfall": 0,
            "classes": classes,
            "best_val_balanced_accuracy": best_accuracy,
            "history": history,
            "seconds": time.time() - started,
            "plan": plan,
            "code": code_identity(REPO_ROOT),
            "output": str(output),
            "best_sha256": file_sha256(output / "best.pt") if (output / "best.pt").exists() else None,
        }
        (output / "train_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (output / "history.jsonl").write_text(
            "\n".join(json.dumps(entry, default=str) for entry in history) + "\n", encoding="utf-8"
        )
        (output / "classes.json").write_text(
            json.dumps(
                {
                    "classes": classes,
                    "style_split": {
                        "train_styles": list(style_split.train_styles),
                        "val_styles": list(style_split.val_styles),
                        "unseen_styles": list(style_split.test_unseen_styles),
                    },
                    "seen_contents": {style: sorted(values) for style, values in seen_contents.items()},
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({k: summary[k] for k in ("optimizer_steps", "best_val_balanced_accuracy", "seconds")}, indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
