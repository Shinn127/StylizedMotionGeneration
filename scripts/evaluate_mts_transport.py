#!/usr/bin/env python
"""E03.1: score a base transport checkpoint on a frozen protocol (read-only).

One command, one checkpoint, one frozen row set.  It reports, for every row of the
protocol, the masked-token NLL and accuracy the checkpoint achieves under the row's
*stated* mask, plus argmax/sampled tokens and decoded motion for a fixed set of
cases.  Nothing here trains, samples data or writes a checkpoint.

Every row is checked against the store's own split table, so a "val" protocol that
contains a train clip fails instead of quietly scoring training data.  The protocol
fingerprint is recomputed and compared with the file it came from, and the
checkpoint must load against the tokenizer file (same structure is not the same
weights).

    python scripts/evaluate_mts_transport.py \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --checkpoint outputs/mts_revision2/<run>/init.pt \
      --protocol <run>/validation_protocol.json \
      --cases 8 --output <run>/eval_init.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    file_sha256,
    load_mts_checkpoint,
    require_token_store_binding,
)
from stylized_motion.learning.mts_operator.contract import masked_cross_entropy  # noqa: E402
from stylized_motion.learning.mts_operator.eval_protocol import (  # noqa: E402
    ValidationBatchBuilder,
    ValidationProtocol,
    ValidationSample,
    store_split_of_clip,
)
from stylized_motion.learning.mts_operator.masking import MaskGenerator  # noqa: E402
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer  # noqa: E402
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    ContentVocabulary,
    TokenSource,
    adapter_from_tokenizer,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402
from stylized_motion.learning.runner import choose_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a base transport on a frozen protocol.")
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="transport checkpoint (init/best/last)")
    parser.add_argument("--protocol", type=Path, required=True, help="frozen validation protocol JSON")
    parser.add_argument("--token-store", type=Path, default=None, help="default: the store the checkpoint names")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cases", type=int, default=8, help="fixed rows to decode (target, argmax, one sample)"
    )
    parser.add_argument(
        "--case-layout",
        choices=["spread", "head"],
        default="spread",
        help="'spread' (default) walks the kinds round-robin so the cases cover every mask kind; "
        "'head' takes the first rows, which for a sorted protocol are all full_generation",
    )
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--baseline", type=Path, default=None, help="token_baselines JSON to compare per kind")
    parser.add_argument("--label", default="", help="row label for the experiment matrix")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motions-dir", type=Path, default=None)
    return parser


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_protocol(path: Path) -> ValidationProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = tuple(ValidationSample(**item) for item in payload["items"])
    return ValidationProtocol(
        samples=samples,
        kinds=tuple(payload["kinds"]),
        weights={str(k): float(v) for k, v in payload["weights"].items()},
        version=int(payload["version"]),
        protocol_id=str(payload.get("protocol_id") or ""),
        selection=dict(payload.get("selection") or {}),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    tokenizer_path = resolve(args.tokenizer_checkpoint)
    checkpoint_path = resolve(args.checkpoint)
    protocol_path = resolve(args.protocol)
    device = choose_device(args.device)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    _, tokenizer = load_representation_checkpoint(tokenizer_path, device)
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    spec = adapter.token_spec(representation_id=tokenizer.representation_id)

    payload, model = load_mts_checkpoint(
        checkpoint_path,
        kind="transport",
        build_model=lambda stored: MotionTransportTransformer(adapter, **stored),
        device=device,
        token_spec=spec,
        tokenizer_metadata=tokenizer.representation_metadata(),
        tokenizer_checkpoint=tokenizer_path,
    )
    model.eval()
    model_config = dict(payload["metadata"]["model_config"])
    vocabulary = ContentVocabulary.from_dict(model_config.get("content_vocabulary"))
    provenance = dict(payload.get("provenance") or {})
    store_path = resolve(
        args.token_store
        or (provenance.get("store_identity") or {}).get("store_path")
        or (_ for _ in ()).throw(ValueError("no --token-store and the checkpoint names none"))
    )

    protocol = read_protocol(protocol_path)
    report: dict[str, object] = {
        "kind": "mts_transport_evaluation",
        "label": str(args.label),
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)) if checkpoint_path.is_relative_to(REPO_ROOT) else str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "global_step": int(payload.get("global_step", 0)),
            "epoch": int(payload.get("epoch", 0)),
            "metrics_val_objective": payload.get("metrics", {}).get("val_objective"),
            "tokenizer_checkpoint_sha256": payload.get("tokenizer_checkpoint_sha256"),
            "tokenizer_file_sha256": file_sha256(tokenizer_path),
            "content_kind": None if vocabulary is None else vocabulary.kind,
            "content_classes": 0 if vocabulary is None else len(vocabulary.classes),
        },
        "protocol": {
            "path": str(protocol_path.relative_to(REPO_ROOT)) if protocol_path.is_relative_to(REPO_ROOT) else str(protocol_path),
            "sha256": file_sha256(protocol_path),
            "protocol_id": protocol.protocol_id,
            "fingerprint": protocol.fingerprint(),
            "rows": len(protocol.samples),
            "kinds": list(protocol.kinds),
            "weights": {kind: float(weight) for kind, weight in protocol.weights.items()},
        },
        "device": str(device),
    }
    # The file and the parsed protocol must agree on their own statement of shape:
    # a fingerprint computed from a file that lost rows would otherwise look fine.
    _stated = json.loads(protocol_path.read_text(encoding="utf-8"))
    _described = protocol.describe()
    report["protocol_file_statement"] = {
        "rows_match": int(_stated.get("samples", -1)) == int(_described["samples"]),
        "kinds_match": list(_stated.get("kinds", [])) == list(_described["kinds"]),
        "samples_per_kind_match": {
            str(k): int(v) for k, v in (_stated.get("samples_per_kind") or {}).items()
        }
        == {str(k): int(v) for k, v in _described["samples_per_kind"].items()},
        "splits_match": sorted(_stated.get("splits", [])) == sorted(_described["splits"]),
        "seeds_match": sorted(int(v) for v in _stated.get("seeds", []))
        == sorted(int(v) for v in _described["seeds"]),
    }
    if not all(report["protocol_file_statement"].values()):
        raise ValueError(f"the protocol file does not describe what it parses to: {report['protocol_file_statement']}")

    store = open_any_token_store(store_path)
    try:
        binding = require_token_store_binding(store, tokenizer_checkpoint=tokenizer_path, where="transport eval")
        report["store"] = {"path": str(store_path), **binding}

        # Every row is checked against the store's own split table before it is scored.
        rows_split: dict[str, int] = {}
        for sample in protocol.samples:
            observed = store_split_of_clip(store, int(sample.target_clip))
            if observed != sample.split:
                raise ValueError(
                    f"protocol row {sample.sample_id} claims split {sample.split!r} but clip "
                    f"{sample.target_clip} is {observed!r} in the store"
                )
            rows_split[sample.split] = rows_split.get(sample.split, 0) + 1
        report["rows_verified_against_the_store_split_table"] = rows_split

        frames = int(args.frames)
        windows = windows_by_clip(store, sorted(rows_split)[0], frames=frames)
        needed = {int(sample.target_clip) for sample in protocol.samples}
        missing = sorted(clip for clip in needed if clip not in windows)
        if missing:
            raise ValueError(f"clips without a full {frames}-frame window: {missing[:5]}")
        source = TokenSource(
            store=store,
            windows_by_clip={clip: windows[clip] for clip in needed},
            adapter=adapter,
            frames=frames,
            history=int(tokenizer.history_frames),
            rng=np.random.default_rng(int(args.seed)),
        )
        mask_config = protocol.samples[0].mask_config
        builder = ValidationBatchBuilder(
            token_source=source,
            mask_generator=MaskGenerator(dict(mask_config)),
            adapter=adapter,
            device=device,
            content_vocabulary=vocabulary,
        )
        items = builder.transport_items(protocol, batch_size=int(args.batch_size))

        cas = []
        per_kind: dict[str, dict[str, float]] = {}
        cases_dir = resolve(args.motions_dir) if args.motions_dir else args.output.parent / f"{args.output.stem}_cases"
        cases_written = 0
        # A spread layout spends the decode budget on every kind: the head of a
        # sorted protocol is all full_generation, whose argmax decode is the
        # action-conditional mode and says nothing about context conditioning.
        case_budget = int(args.cases)
        per_kind_budget: dict[str, int] = {}
        if str(args.case_layout) == "spread" and case_budget > 0:
            kinds_present = list(protocol.kinds)
            share = max(1, -(-case_budget // max(len(kinds_present), 1)))  # ceil
            for kind in kinds_present:
                per_kind_budget[kind] = share
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
        for item in items:
            tokens = item["tokens"].to(device)
            visible = item["visible_mask"].to(device)
            valid = None if item["valid_mask"] is None else item["valid_mask"].to(device)
            with torch.no_grad():
                output = model(
                    tokens, visible, content_condition=item.get("content_condition"), valid_mask=valid
                )
            logits = output.logits
            supervision = (~visible) if valid is None else ((~visible) & valid.unsqueeze(-1))
            predicted = logits.argmax(dim=-1)
            sampled = torch.distributions.Categorical(logits=logits / float(args.sample_temperature)).sample()
            for row in range(int(tokens.shape[0])):
                mask_row = supervision[row]
                supervised = int(mask_row.sum())
                if supervised == 0:
                    raise ValueError("a protocol row supervises nothing; it cannot be scored")
                nll = float(
                    masked_cross_entropy(
                        logits[row : row + 1],
                        tokens[row : row + 1],
                        valid_mask=None if valid is None else valid[row : row + 1],
                        coordinate_mask=mask_row.unsqueeze(0),
                        reduction="sum",
                    )
                )
                correct = int(((predicted[row] == tokens[row]) & mask_row).sum())
                # The argmax decode can be degenerate while the distribution is not:
                # the entropy of the predicted distribution at the supervised
                # positions is the diversity measure that does not depend on decoding.
                row_probs = logits[row].softmax(-1)
                row_entropy = float(
                    (-(row_probs * (row_probs + 1e-12).log()).sum(-1))[mask_row].mean()
                )
                cas.append(
                    {
                        "kind": str(item["kind"]),
                        "nll_sum": nll,
                        "supervised_tokens": supervised,
                        "correct_tokens": correct,
                        "probabilities_entropy_mean": row_entropy,
                    }
                )
                entry = per_kind.setdefault(
                    str(item["kind"]),
                    {"nll_sum": 0.0, "supervised_tokens": 0, "correct_tokens": 0, "rows": 0, "entropy_sum": 0.0},
                )
                entry["nll_sum"] += nll
                entry["supervised_tokens"] += supervised
                entry["correct_tokens"] += correct
                entry["entropy_sum"] += row_entropy
                entry["rows"] += 1
                kind_name = str(item["kind"])
                kind_used = sum(1 for case in report.get("cases", []) if case["kind"] == kind_name)
                within_budget = (
                    cases_written < case_budget
                    if str(args.case_layout) == "head"
                    else kind_used < per_kind_budget.get(kind_name, case_budget)
                )
                if within_budget and cases_written < case_budget:
                    case = {
                        # ``case_index`` names the file; ``row`` is the row's position
                        # in the protocol, which is not the same number once a spread
                        # layout skips rows of other kinds.
                        "case_index": cases_written,
                        "row": len(cas) - 1,
                        "kind": str(item["kind"]),
                        "probabilities_entropy_mean": row_entropy,
                        "sample_metadata": item["sample_metadata"][row] if item.get("sample_metadata") else {},
                        "nll": nll / supervised,
                        "accuracy": correct / supervised,
                    }
                    cases_dir.mkdir(parents=True, exist_ok=True)
                    np.save(cases_dir / f"case{cases_written:02d}_target_tokens.npy", tokens[row].cpu().numpy())
                    np.save(cases_dir / f"case{cases_written:02d}_argmax_tokens.npy", predicted[row].cpu().numpy())
                    np.save(cases_dir / f"case{cases_written:02d}_sample_tokens.npy", sampled[row].cpu().numpy())
                    with torch.no_grad():
                        motion_target = (
                            tokenizer.decode_indices(tokens[row][None].to(device).long())[0].cpu().numpy()
                        )
                        motion_argmax = (
                            tokenizer.decode_indices(predicted[row][None].to(device).long())[0].cpu().numpy()
                        )
                        motion_sample = (
                            tokenizer.decode_indices(sampled[row][None].to(device).long())[0].cpu().numpy()
                        )
                    np.save(cases_dir / f"case{cases_written:02d}_target_motion.npy", motion_target)
                    np.save(cases_dir / f"case{cases_written:02d}_argmax_motion.npy", motion_argmax)
                    np.save(cases_dir / f"case{cases_written:02d}_sample_motion.npy", motion_sample)
                    case["files"] = sorted(path.name for path in cases_dir.glob(f"case{cases_written:02d}_*"))
                    report.setdefault("cases", []).append(case)
                    cases_written += 1

        objectives = {
            kind: {
                "nll": entry["nll_sum"] / entry["supervised_tokens"],
                "accuracy": entry["correct_tokens"] / entry["supervised_tokens"],
                "supervised_tokens": entry["supervised_tokens"],
                "rows": entry["rows"],
                "probabilities_entropy_mean": entry["entropy_sum"] / max(entry["rows"], 1),
            }
            for kind, entry in sorted(per_kind.items())
        }
        weighted = 0.0
        used_weight = 0.0
        for kind, entry in objectives.items():
            weight = float(protocol.weights.get(kind, 0.0))
            weighted += weight * entry["nll"]
            used_weight += weight
        report["per_kind"] = {k: v["nll"] for k, v in objectives.items()}
        report["per_kind_detail"] = objectives
        report["weighted_nll"] = weighted / used_weight if used_weight > 0 else None
        report["supervised_tokens_total"] = sum(entry["supervised_tokens"] for entry in objectives.values())
        report["cases_dir"] = str(cases_dir)
        if args.baseline is not None:
            baseline = json.load(open(resolve(args.baseline), encoding="utf-8"))["protocol_baseline"]
            comparison = {}
            for kind, entry in objectives.items():
                reference = baseline["per_kind"].get(kind, {})
                unigram = reference.get("unigram_nll")
                conditioned = reference.get("action_conditioned_unigram_nll")
                comparison[kind] = {
                    "model_nll": entry["nll"],
                    "unigram_nll": unigram,
                    "action_conditioned_unigram_nll": conditioned,
                    "delta_vs_unigram": None if unigram is None else unigram - entry["nll"],
                    "delta_vs_action_conditioned": None if conditioned is None else conditioned - entry["nll"],
                }
            report["baseline_comparison"] = comparison
            report["baseline_source"] = str(resolve(args.baseline))
            report["delta_weighted_vs_unigram"] = (
                None
                if baseline.get("weighted_unigram_nll") is None or report["weighted_nll"] is None
                else baseline["weighted_unigram_nll"] - report["weighted_nll"]
            )
        report["seconds"] = time.perf_counter() - started
    finally:
        store.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"]["path"],
                "global_step": report["checkpoint"]["global_step"],
                "protocol_id": report["protocol"]["protocol_id"],
                "weighted_nll": report["weighted_nll"],
                "per_kind": report["per_kind"],
                "delta_weighted_vs_unigram": report.get("delta_weighted_vs_unigram"),
                "cases": cases_written,
                "seconds": round(float(report["seconds"]), 2),
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
