from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from stylized_motion.data.feature_dataset import build_feature_store
from stylized_motion.learning.representation import RepresentationProtocol, load_representation_checkpoint


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


def representation_dimensions(model: RepresentationProtocol) -> tuple[int, int]:
    return int(model.num_coordinates), int(model.num_levels)


def encode_shard(
    model: RepresentationProtocol,
    motion: np.ndarray,
    source_offset: torch.Tensor,
    source_scale: torch.Tensor,
    checkpoint_offset: torch.Tensor,
    checkpoint_scale: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(motion.shape[0])
    num_coordinates, _ = representation_dimensions(model)
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
    store = build_feature_store(args.feature_database)
    feature_schema = store.feature_schema()
    checkpoint, model = load_representation_checkpoint(
        args.checkpoint,
        device,
        feature_schema=feature_schema,
    )
    checkpoint_stats = checkpoint["feature_stats"]
    if int(checkpoint_stats["motion_dim"]) != store.motion_dim:
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
        "motion_dim": np.asarray(store.motion_dim, dtype=np.int32),
        "schema_version": np.asarray(2, dtype=np.int32),
        "representation_family": np.asarray(model.family, dtype=object),
        "representation_variant": np.asarray(model.variant, dtype=object),
        "representation_id": np.asarray(model.representation_id, dtype=object),
        "model_family_legacy": np.asarray(checkpoint["model_family"], dtype=object),
        "coordinate_order": np.asarray(model.representation_metadata()["coordinate_order"], dtype=object),
        "coordinate_counts": np.asarray(json.dumps(model.representation_metadata()["coordinate_counts"], sort_keys=True), dtype=object),
        "num_coordinates": np.asarray(model.num_coordinates, dtype=np.int32),
        "num_levels": np.asarray(model.num_levels, dtype=np.int32),
        "checkpoint_path": np.asarray(str(args.checkpoint), dtype=object),
        "checkpoint_sha256": np.asarray(sha256_file(args.checkpoint), dtype=object),
        "feature_database": np.asarray(str(args.feature_database), dtype=object),
        "feature_schema_hash": np.asarray(store.feature_schema_hash, dtype=object),
        "names_sha256": np.asarray(store.names_sha256, dtype=object),
        "stats_sha256": np.asarray(store.stats_sha256, dtype=object),
        "joint_subset": np.asarray(store.joint_subset, dtype=object),
        "frame_rate": np.asarray(model.representation_metadata()["frame_rate"], dtype=np.int32),
        "temporal_downsample": np.asarray(model.representation_metadata()["temporal_downsample"], dtype=np.int32),
        "receptive_field": np.asarray(model.receptive_field, dtype=np.int32),
        "context_left": np.asarray(model.context_left, dtype=np.int32),
        "lookahead_frames": np.asarray(model.lookahead_frames, dtype=np.int32),
        "decoder_passes_inference": np.asarray(model.representation_metadata()["decoder_passes_inference"], dtype=np.int32),
    }
    np.savez(metadata_path, **metadata)
    print(f"saved={args.output}")
    print(f"shards={len(token_files)} frames={sum(num_frames)}")
    print(f"styles={len(style_names)} actions={len(action_names)}")
    print(f"representation={model.representation_id} family={model.family}")
    print(f"num_coordinates={model.num_coordinates} num_levels={model.num_levels}")


if __name__ == "__main__":
    main()
