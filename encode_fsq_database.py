from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.feature_dataset import build_feature_store
from models.fsq import FSQMotionAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode complete motion shards into an FSQ token database.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--save-codes", action="store_true", help="Also save quantized float coordinates as float16.")
    parser.add_argument("--overwrite", action="store_true")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_range_name(range_name: str) -> tuple[str, str]:
    style, separator, action = range_name.rpartition("_")
    if not separator or not style or not action:
        raise ValueError(f"Expected 100STYLE range name '<style>_<action>', got {range_name!r}")
    return style, action


def encode_shard(
    model: FSQMotionAutoencoder,
    motion: np.ndarray,
    source_offset: torch.Tensor,
    source_scale: torch.Tensor,
    checkpoint_offset: torch.Tensor,
    checkpoint_scale: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(motion.shape[0])
    num_coordinates = int(model.quantizer.num_coordinates)
    indices_out = np.empty((num_frames, num_coordinates), dtype=np.uint8)
    codes_out = np.empty((num_frames, num_coordinates), dtype=np.float16)
    context_left = int(model.context_left)

    with torch.inference_mode():
        for start in range(0, num_frames, chunk_size):
            end = min(num_frames, start + chunk_size)
            read_start = max(0, start - context_left)
            source_motion = torch.from_numpy(
                np.asarray(motion[read_start:end], dtype=np.float32).copy()
            ).unsqueeze(0).to(device)
            raw_motion = source_motion * source_scale.view(1, 1, -1) + source_offset.view(1, 1, -1)
            model_motion = (raw_motion - checkpoint_offset.view(1, 1, -1)) / checkpoint_scale.view(1, 1, -1)
            codes, indices = model.encode_to_codes(model_motion)
            offset = start - read_start
            length = end - start
            indices_out[start:end] = indices[0, offset : offset + length].cpu().numpy().astype(np.uint8)
            codes_out[start:end] = codes[0, offset : offset + length].cpu().numpy().astype(np.float16)
    return indices_out, codes_out


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    metadata_path = args.output / "metadata.npz"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite to replace the token database")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq":
        raise ValueError(f"{args.checkpoint} is not an FSQ checkpoint")
    model = FSQMotionAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    store = build_feature_store(args.feature_database)
    checkpoint_stats = checkpoint["stats"]
    checkpoint_names = [str(name) for name in np.asarray(checkpoint_stats["names"], dtype=object).tolist()]
    if checkpoint_names != store.names:
        raise ValueError("Checkpoint and feature database use different skeleton joint ordering")
    if int(checkpoint["model_config"]["motion_dim"]) != store.motion_dim:
        raise ValueError("Checkpoint and feature database have different motion dimensions")

    args.output.mkdir(parents=True, exist_ok=True)
    token_dir = args.output / "indices"
    code_dir = args.output / "codes"
    token_dir.mkdir(parents=True, exist_ok=True)
    if args.save_codes:
        code_dir.mkdir(parents=True, exist_ok=True)

    source_offset = torch.from_numpy(store.stats.offset.astype(np.float32)).to(device)
    source_scale = torch.from_numpy(store.stats.scale.astype(np.float32)).to(device)
    checkpoint_offset = torch.as_tensor(checkpoint_stats["offset"], dtype=torch.float32, device=device)
    checkpoint_scale = torch.as_tensor(checkpoint_stats["scale"], dtype=torch.float32, device=device)

    if len(store.motion_files) != len(store.range_names):
        raise ValueError("Expected one range metadata entry per motion shard")
    token_files = []
    code_files = []
    num_frames = []
    styles = []
    actions = []
    style_names = sorted({parse_range_name(str(name))[0] for name in store.range_names})
    action_names = sorted({parse_range_name(str(name))[1] for name in store.range_names})
    style_to_id = {name: index for index, name in enumerate(style_names)}
    action_to_id = {name: index for index, name in enumerate(action_names)}

    for shard_idx, motion_path in enumerate(tqdm(store.motion_files, desc="Encoding FSQ shards")):
        motion = np.load(motion_path, mmap_mode="r")
        indices, codes = encode_shard(
            model,
            motion,
            source_offset,
            source_scale,
            checkpoint_offset,
            checkpoint_scale,
            args.chunk_size,
            device,
        )
        token_rel = Path("indices") / f"indices_{shard_idx:05d}.npy"
        np.save(args.output / token_rel, indices)
        token_files.append(token_rel.as_posix())
        if args.save_codes:
            code_rel = Path("codes") / f"codes_{shard_idx:05d}.npy"
            np.save(args.output / code_rel, codes)
            code_files.append(code_rel.as_posix())
        else:
            code_files.append("")
        num_frames.append(len(indices))
        style, action = parse_range_name(str(store.range_names[shard_idx]))
        styles.append(style_to_id[style])
        actions.append(action_to_id[action])

    def windows_array(split: str) -> np.ndarray:
        return np.asarray(
            [[window.shard_idx, window.start_idx, window.end_idx, window.range_idx] for window in store.split_windows[split]],
            dtype=np.int32,
        )

    metadata = {
        "token_files": np.asarray(token_files, dtype=object),
        "code_files": np.asarray(code_files, dtype=object),
        "num_frames": np.asarray(num_frames, dtype=np.int32),
        "range_names": store.range_names,
        "range_mirror": store.range_mirror,
        "style_names": np.asarray(style_names, dtype=object),
        "action_names": np.asarray(action_names, dtype=object),
        "style_ids": np.asarray(styles, dtype=np.int32),
        "action_ids": np.asarray(actions, dtype=np.int32),
        "train_windows": windows_array("train"),
        "val_windows": windows_array("val"),
        "test_windows": windows_array("test"),
        "window_size": np.asarray(store.window_size, dtype=np.int32),
        "num_coordinates": np.asarray(model.quantizer.num_coordinates, dtype=np.int32),
        "num_levels": np.asarray(model.quantizer.num_levels, dtype=np.int32),
        "checkpoint_path": np.asarray(str(args.checkpoint), dtype=object),
        "checkpoint_sha256": np.asarray(sha256_file(args.checkpoint), dtype=object),
        "feature_database": np.asarray(str(args.feature_database), dtype=object),
        "model_config_json": np.asarray(json.dumps(checkpoint["model_config"], sort_keys=True), dtype=object),
    }
    np.savez(metadata_path, **metadata)
    print(f"saved={args.output}")
    print(f"shards={len(token_files)} frames={sum(num_frames)}")
    print(f"styles={len(style_names)} actions={len(action_names)}")
    print(f"num_coordinates={model.quantizer.num_coordinates} num_levels={model.quantizer.num_levels}")


if __name__ == "__main__":
    main()

