import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets.feature_dataset import build_feature_store
from Genoview import GenoView, GenoViewCompare, build_database_from_feature_array
from models.fsq import FSQMotionAutoencoder
from models.vqvae import CausalMotionVQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="View a continuous database motion segment through an FSQ or VQ-VAE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to FSQ or VQ-VAE checkpoint.")
    parser.add_argument("--feature-database", type=Path, default=None, help="Path to feature_database. Defaults to checkpoint args.")
    parser.add_argument("--range-idx", type=int, required=True, help="Motion shard / range index in feature_database metadata.")
    parser.add_argument("--start", type=int, required=True, help="Target segment start frame.")
    parser.add_argument("--length", type=int, required=True, help="Target segment length in frames.")
    parser.add_argument(
        "--context-left",
        type=int,
        default=None,
        help="Left context frames used for reconstruction warmup. Defaults to the model requirement.",
    )
    parser.add_argument("--view", choices=["source", "recon", "compare"], default="compare")
    parser.add_argument("--compare-spacing", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--resources-root", type=Path, default=Path(__file__).resolve().parent / "resources")
    parser.add_argument("--range-name", type=str, default=None, help="Override displayed range name.")
    parser.add_argument("--save-debug", action="store_true", help="Optionally save source/recon/tokens/meta for inspection.")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/sequence_debug"))
    parser.add_argument("--dry-run", action="store_true", help="Run loading/reconstruction checks without opening Genoview.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def build_model_from_checkpoint(ckpt: dict):
    family = ckpt["model_family"]
    if family not in {"fsq", "vqvae"}:
        raise ValueError(f"Unsupported model_family: {family}")
    model_class = FSQMotionAutoencoder if family == "fsq" else CausalMotionVQVAE
    model = model_class(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    return model, family


def resolve_context_left(cli_context_left: int | None, model) -> int:
    required = int(model.context_left)
    if cli_context_left is None:
        return required
    if cli_context_left < required:
        raise ValueError(
            f"--context-left={cli_context_left} is smaller than the model requirement {required}"
        )
    return cli_context_left


def resolve_feature_database(cli_feature_database: Path | None, ckpt_args: dict) -> Path:
    feature_database = cli_feature_database or ckpt_args.get("feature_database")
    if feature_database is None:
        raise ValueError("--feature-database is required because the checkpoint args do not contain feature_database")
    return Path(feature_database)


def validate_slice(range_idx: int, start: int, length: int, context_left: int, motion: np.ndarray, motion_dim: int) -> None:
    if range_idx < 0:
        raise ValueError(f"--range-idx must be non-negative, got {range_idx}")
    if start < 0:
        raise ValueError(f"--start must be non-negative, got {start}")
    if length <= 0:
        raise ValueError(f"--length must be positive, got {length}")
    if context_left < 0:
        raise ValueError(f"--context-left must be non-negative, got {context_left}")
    if motion.ndim != 2 or motion.shape[1] != motion_dim:
        raise ValueError(f"Expected motion shape [T, {motion_dim}], got {motion.shape}")
    if start + length > motion.shape[0]:
        raise ValueError(f"Requested [{start}, {start + length}) exceeds motion length {motion.shape[0]}")


def inference_factor(model_config: dict, family: str) -> int:
    if family == "vqvae" and model_config["model_type"] == "causal_cnn":
        return 4
    return 1


def slice_with_edge_pad(motion: np.ndarray, start: int, end: int) -> np.ndarray:
    read_end = min(end, motion.shape[0])
    sliced = np.asarray(motion[start:read_end], dtype=np.float32).copy()
    if read_end < end:
        pad_count = end - read_end
        if sliced.shape[0] == 0:
            raise ValueError("Cannot pad an empty motion slice")
        pad = np.repeat(sliced[-1:], pad_count, axis=0)
        sliced = np.concatenate([sliced, pad], axis=0)
    return sliced


def reconstruct_segment(model, family: str, motion: np.ndarray, start: int, length: int, context_left: int, device):
    target_end = start + length
    infer_start = max(0, start - context_left)
    alignment_factor = inference_factor(model.config, family)
    infer_start -= infer_start % alignment_factor
    infer_end = target_end
    infer_len = infer_end - infer_start
    if family == "vqvae" and model.model_type == "causal_transformer" and infer_len > model.config["max_seq_len"]:
        raise ValueError(f"causal_transformer infer length {infer_len} exceeds max_seq_len {model.config['max_seq_len']}")
    pad_right = (-infer_len) % alignment_factor
    infer_features = slice_with_edge_pad(motion, infer_start, infer_end + pad_right)

    x = torch.from_numpy(np.asarray(infer_features, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
    recon_full = output["recon_state"][0].detach().cpu().numpy().astype(np.float32)
    offset = start - infer_start
    recon_features = recon_full[offset : offset + length]
    if recon_features.shape != (length, motion.shape[1]):
        raise ValueError(f"Expected recon shape {(length, motion.shape[1])}, got {recon_features.shape}")

    indices = output.get("indices")
    if indices is not None:
        indices = indices[0].detach().cpu().numpy()
        if indices.shape[0] == recon_full.shape[0]:
            token_end = min(indices.shape[0], offset + length)
            indices = indices[offset:token_end]

    return recon_features, indices, {
        "infer_start": infer_start,
        "infer_end": infer_end,
        "alignment_factor": alignment_factor,
        "pad_right": pad_right,
        "infer_frames": int(infer_features.shape[0]),
    }


def save_debug(args, source_features, recon_features, indices, meta):
    debug_dir = args.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    np.save(debug_dir / "source_features.npy", source_features.astype(np.float32))
    np.save(debug_dir / "recon_features.npy", recon_features.astype(np.float32))
    if indices is not None:
        np.savez(debug_dir / "tokens.npz", indices=indices)
    with (debug_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved debug outputs to {debug_dir}")


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, family = build_model_from_checkpoint(ckpt)
    args.context_left = resolve_context_left(args.context_left, model)
    feature_database = resolve_feature_database(args.feature_database, ckpt["args"])
    store = build_feature_store(feature_database)
    if model.motion_dim != store.motion_dim:
        raise ValueError(f"Model motion_dim={model.motion_dim} does not match feature database motion_dim={store.motion_dim}")
    if args.range_idx >= len(store.motion_files):
        raise ValueError(f"--range-idx {args.range_idx} exceeds range count {len(store.motion_files)}")

    motion_path = store.motion_files[args.range_idx]
    motion = np.load(motion_path, mmap_mode="r")
    validate_slice(args.range_idx, args.start, args.length, args.context_left, motion, model.motion_dim)
    source_features = np.asarray(motion[args.start : args.start + args.length], dtype=np.float32).copy()

    device = choose_device(args.device)
    model = model.to(device)
    model.eval()
    recon_features, indices, infer_meta = reconstruct_segment(
        model=model,
        family=family,
        motion=motion,
        start=args.start,
        length=args.length,
        context_left=args.context_left,
        device=device,
    )

    range_name = args.range_name or str(store.range_names[args.range_idx])
    mirror = bool(store.range_mirror[args.range_idx])
    display_name = f"{range_name}_{'mirror' if mirror else 'orig'}_{args.start:06d}_{args.start + args.length:06d}"
    meta = {
        "checkpoint": str(args.checkpoint),
        "model_family": family,
        "feature_database": str(feature_database),
        "motion_path": str(motion_path),
        "range_idx": args.range_idx,
        "range_name": range_name,
        "mirror": mirror,
        "start": args.start,
        "length": args.length,
        "context_left": args.context_left,
        "view": args.view,
        "device": str(device),
        **infer_meta,
    }
    print(json.dumps(meta, indent=2))

    if args.save_debug:
        save_debug(args, source_features, recon_features, indices, meta)

    if args.view == "source":
        database = build_database_from_feature_array(source_features, args.checkpoint, True, f"{display_name}_source")
        viewer = GenoView(database=database, trajectory_path=None, resources_root=args.resources_root, fps=args.fps)
    elif args.view == "recon":
        database = build_database_from_feature_array(recon_features, args.checkpoint, True, f"{display_name}_{family}_recon")
        viewer = GenoView(database=database, trajectory_path=None, resources_root=args.resources_root, fps=args.fps)
    else:
        source_database = build_database_from_feature_array(source_features, args.checkpoint, True, f"{display_name}_source")
        recon_database = build_database_from_feature_array(recon_features, args.checkpoint, True, f"{display_name}_{family}_recon")
        viewer = GenoViewCompare(
            left_database=source_database,
            right_database=recon_database,
            resources_root=args.resources_root,
            fps=args.fps,
            left_label="Source",
            right_label=f"{family.upper()} Recon",
            compare_spacing=args.compare_spacing,
        )
    if args.dry_run:
        print("dry_run_viewer_ready=true")
        return
    viewer.run()


if __name__ == "__main__":
    main()
