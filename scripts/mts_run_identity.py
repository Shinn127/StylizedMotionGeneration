#!/usr/bin/env python
"""Write the identity manifest of one training stage (E00.2).

One JSON file names everything a later run must be comparable with: the code
digest of the working tree, the recipes actually used, the tokenizer file, the
token store's own binding hashes and the environment.  Nothing here is inferred
from a previous audit: the digests are recomputed from the files on disk, so a
recipe edited after the last audit shows up as a different hash.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/mts_run_identity.py \
      --config data/configs/mts_revision2_transport_profile.yaml \
      --output outputs/.../run_identity.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    code_identity,
    file_sha256,
    require_token_store_binding,
)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_recipe(path: Path) -> dict[str, object]:
    """Token store, tokenizer and the identity fields a recipe declares."""
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a mapping")
    data = dict(document.get("data") or {})
    tokenizer = dict(document.get("tokenizer") or {})
    transport = dict(document.get("transport") or {})
    return {
        "token_store": data.get("token_store"),
        "frames": data.get("frames"),
        "content_kind": dict(data.get("content") or {}).get("kind"),
        "tokenizer_checkpoint": tokenizer.get("checkpoint"),
        "transport_checkpoint": transport.get("checkpoint"),
        "protocol_id": dict(document.get("evaluation") or {}).get("protocol_id"),
        "output_dir": dict(document.get("training") or {}).get("output_dir"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-store", type=Path, default=None, help="default: from the first recipe")
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None, help="default: from the first recipe")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    configs = [resolve(path) for path in args.config]
    missing = [str(path) for path in configs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"recipes do not exist: {missing}")
    recipes = {str(path.relative_to(REPO_ROOT)): read_recipe(path) for path in configs}
    first = next(iter(recipes.values()))

    tokenizer_checkpoint = resolve(args.tokenizer_checkpoint or first["tokenizer_checkpoint"])
    token_store = resolve(args.token_store or first["token_store"])

    import torch

    report: dict[str, object] = {
        "kind": "mts_run_identity",
        "recipes": recipes,
        "recipe_sha256": {
            name: file_sha256(resolve(name)) for name in sorted(recipes)
        },
    }
    report.update(code_identity(REPO_ROOT))

    report["tokenizer_checkpoint"] = {
        "path": str(tokenizer_checkpoint.relative_to(REPO_ROOT)),
        "sha256": file_sha256(tokenizer_checkpoint),
    }

    store = open_any_token_store(token_store)
    try:
        binding = require_token_store_binding(
            store, tokenizer_checkpoint=tokenizer_checkpoint, where="run identity"
        )
        report["token_store"] = {
            "path": str(token_store.relative_to(REPO_ROOT)),
            "checkpoint_sha256": str(store.manifest.get("checkpoint_sha256")),
            "feature_schema_hash": str(store.manifest.get("feature_schema_hash")),
            "normalization_hash": str(store.manifest.get("normalization_hash")),
            "split_manifest_hash": str(store.manifest.get("split_manifest_hash")),
            "representation_id": str(store.manifest.get("representation_id")),
            "num_coordinates": int(store.num_coordinates),
            "num_levels": int(store.manifest.get("num_levels")),
            "num_clips": int(store.manifest.get("num_clips")),
            "shard_sha256": str(store.manifest.get("shard_sha256")),
            "binding": binding,
        }
    finally:
        store.close()

    from stylized_motion.learning.representation import (
        NEF_FSQ_FAMILY,
        load_representation_checkpoint,
    )

    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    _, tokenizer = load_representation_checkpoint(tokenizer_checkpoint, resolved_device)
    if tokenizer.family != NEF_FSQ_FAMILY:
        raise ValueError(f"expected {NEF_FSQ_FAMILY}, got {tokenizer.family!r}")
    report["tokenizer"] = {
        "representation_id": tokenizer.representation_id,
        "variant": tokenizer.representation_metadata().get("variant"),
        "num_coordinates": int(tokenizer.num_coordinates),
        "num_levels": int(tokenizer.num_levels),
        "num_streams": int(len(tokenizer.token_layout().stream_slices)),
        "history_frames": int(tokenizer.history_frames),
        "receptive_field": int(tokenizer.receptive_field),
    }

    report["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "resolved_device": str(resolved_device),
        "threads": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "code_commit": report["code_commit"],
                "working_tree_dirty": report["working_tree_dirty"],
                "source_digest_combined": report["source_digest"]["combined"],
                "source_files": report["source_digest"]["count"],
                "tokenizer_sha256": report["tokenizer_checkpoint"]["sha256"],
                "token_store_checkpoint_sha256": report["token_store"]["checkpoint_sha256"],
                "recipes": len(recipes),
                "device": report["environment"]["resolved_device"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
