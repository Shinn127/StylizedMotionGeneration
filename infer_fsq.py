import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.feature_dataset import FeatureDataset, build_feature_store
from models.fsq import FSQMotionAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Run FSQ motion reconstruction inference and export reconstructed feature clips.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--feature-database", type=Path, default=None, help="Path to feature_database containing normalized 230D motion shards.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=torch.cuda.is_available())
    parser.add_argument("--outdir", type=Path, default=Path("outputs/fsq/infer"))
    parser.add_argument("--tag", type=str, default="recon")
    return parser.parse_args()


def resolve_feature_database(args, ckpt_args):
    feature_database = args.feature_database or ckpt_args.get("feature_database")
    if feature_database is None:
        raise ValueError("--feature-database is required because the checkpoint args do not contain feature_database")
    return Path(feature_database)


def build_dataset(args, feature_database):
    store = build_feature_store(feature_database)
    return FeatureDataset(split=args.split, store=store)


def build_loader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def build_model_from_checkpoint(ckpt):
    if ckpt.get("model_family") != "fsq":
        raise ValueError(f"Expected an FSQ checkpoint, got model_family={ckpt.get('model_family')}")
    args = ckpt["args"]
    model = FSQMotionAutoencoder(
        motion_dim=ckpt["motion_dim"],
        code_dim=args["code_dim"],
        width=args["width"],
        depth=args["depth"],
        dilation_growth_rate=args["dilation_growth_rate"],
        num_latent_tokens=args["fsq_num_latent_tokens"],
        num_levels=args["fsq_num_levels"],
        fsq_scale=args.get("fsq_scale"),
        fsq_preserve_symmetry=args.get("fsq_preserve_symmetry", False),
        fsq_noise_dropout=args.get("fsq_noise_dropout", 0.0),
    )
    model.load_state_dict(ckpt["model"])
    return model, args


def validate_export_inputs(dataset, ckpt):
    if int(ckpt["motion_dim"]) != int(dataset.motion_dim):
        raise ValueError(f"Checkpoint motion_dim={ckpt['motion_dim']} does not match feature database motion_dim={dataset.motion_dim}")


def safe_path_part(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value or "clip"


def export_clip_features(range_to_window, dataset, outdir):
    feature_root = outdir / "clip_features"
    feature_root.mkdir(parents=True, exist_ok=True)

    rel_paths = []
    range_indices = []
    range_names = []
    range_mirror = []
    starts = []
    stops = []
    source_window_counts = []
    covered_frame_counts = []
    coverage_ratios = []

    def flush_segment(range_idx, clip_dir, segment_windows):
        segment_start = min(int(start) for start, _end, _motion in segment_windows)
        segment_stop = max(int(end) for _start, end, _motion in segment_windows)
        if segment_stop <= segment_start:
            raise ValueError(f"Invalid feature segment [{segment_start}, {segment_stop}) for range {range_idx}")

        merged = np.zeros((segment_stop - segment_start, dataset.motion_dim), dtype=np.float32)
        counts = np.zeros((segment_stop - segment_start, 1), dtype=np.float32)
        for local_start, local_end, recon_motion in segment_windows:
            local_start = int(local_start)
            local_end = int(local_end)
            if local_end <= local_start:
                raise ValueError(f"Invalid window [{local_start}, {local_end}) for range {range_idx}")
            start = local_start - segment_start
            end = local_end - segment_start
            merged[start:end] += np.asarray(recon_motion, dtype=np.float32)
            counts[start:end] += 1.0

        covered = counts[:, 0] > 0.0
        if not np.all(covered):
            missing = np.nonzero(~covered)[0]
            raise ValueError(
                f"Range {range_idx} has gaps inside merged feature segment [{segment_start}, {segment_stop}); "
                f"first missing local frame {int(missing[0]) + segment_start}"
            )
        merged = merged / counts

        segment_dir = clip_dir / f"segment_{segment_start:06d}_{segment_stop:06d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        feature_path = segment_dir / "features.npy"
        np.save(feature_path, merged.astype(np.float32))

        rel_paths.append(feature_path.relative_to(outdir).as_posix())
        range_indices.append(range_idx)
        range_names.append(str(dataset.range_names[range_idx]))
        range_mirror.append(bool(dataset.range_mirror[range_idx]))
        starts.append(segment_start)
        stops.append(segment_stop)
        source_window_counts.append(len(segment_windows))
        covered_frame_counts.append(int(np.sum(covered)))
        coverage_ratios.append(float(np.sum(covered)) / float(segment_stop - segment_start))

    for range_idx in range(len(dataset.range_names)):
        windows = sorted(range_to_window.get(range_idx, []), key=lambda x: (x[0], x[1]))
        if not windows:
            continue

        clip_name = safe_path_part(dataset.range_names[range_idx])
        mirror_suffix = "mirror" if bool(dataset.range_mirror[range_idx]) else "orig"
        clip_dir = feature_root / f"{range_idx:05d}_{clip_name}_{mirror_suffix}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        segment_windows = []
        segment_stop = None
        for window in windows:
            local_start = int(window[0])
            local_end = int(window[1])
            if segment_stop is None or local_start > segment_stop:
                if segment_windows:
                    flush_segment(range_idx, clip_dir, segment_windows)
                segment_windows = [window]
                segment_stop = local_end
            else:
                segment_windows.append(window)
                segment_stop = max(segment_stop, local_end)
        if segment_windows:
            flush_segment(range_idx, clip_dir, segment_windows)

    index_path = feature_root / "index.npz"
    np.savez(
        index_path,
        feature_files=np.asarray(rel_paths, dtype=object),
        range_indices=np.asarray(range_indices, dtype=np.int32),
        range_names=np.asarray(range_names, dtype=object),
        range_mirror=np.asarray(range_mirror, dtype=bool),
        window_starts=np.asarray(starts, dtype=np.int32),
        window_stops=np.asarray(stops, dtype=np.int32),
        source_window_counts=np.asarray(source_window_counts, dtype=np.int32),
        covered_frame_counts=np.asarray(covered_frame_counts, dtype=np.int32),
        coverage_ratios=np.asarray(coverage_ratios, dtype=np.float32),
        normalized=np.asarray(True, dtype=bool),
    )
    return feature_root, index_path


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, ckpt_args = build_model_from_checkpoint(ckpt)
    feature_database = resolve_feature_database(args, ckpt_args)
    dataset = build_dataset(args, feature_database)
    validate_export_inputs(dataset, ckpt)
    loader = build_loader(args, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    outdir = args.outdir / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    range_to_window = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            output = model(motion)
            recon = output["recon_state"].cpu().numpy().astype(np.float32)
            for i in range(motion.shape[0]):
                range_idx = int(batch["range_idx"][i])
                range_to_window[range_idx].append((int(batch["start_idx"][i]), int(batch["end_idx"][i]), recon[i]))

    clip_features_dir, clip_features_index = export_clip_features(range_to_window, dataset, outdir)
    np.savez(
        outdir / "recon_windows.npz",
        split=np.array(args.split, dtype=object),
        feature_database=np.array(str(feature_database), dtype=object),
        checkpoint=np.array(str(args.checkpoint), dtype=object),
        clip_features_dir=np.array(str(clip_features_dir), dtype=object),
        clip_features_index=np.array(str(clip_features_index), dtype=object),
    )

    print(f"Exported clip features to {clip_features_dir}")
    print(f"Exported clip feature index to {clip_features_index}")
    print(f"Exported recon metadata to {outdir / 'recon_windows.npz'}")


if __name__ == "__main__":
    main()
