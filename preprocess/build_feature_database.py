import argparse
from pathlib import Path

import numpy as np

from datasets.motion_dataset import MotionWindow, build_motion_store
from motion_features import serialize_motion_feature_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--root-cond-dim", type=int, default=6)
    return parser.parse_args()


def windows_to_array(windows: list[MotionWindow]) -> np.ndarray:
    return np.asarray(
        [[window.start_idx, window.end_idx, window.range_idx] for window in windows],
        dtype=np.int32,
    )


def main():
    args = parse_args()
    store = build_motion_store(
        window_size=args.window_size,
        database_path=args.database,
        use_root_cond=True,
        root_cond_dim=args.root_cond_dim,
        seed=args.seed,
        normalize=True,
    )

    full_motion = store.motion_features.astype(np.float32)
    root_cond = full_motion[:, : store.root_cond_dim].astype(np.float32)
    motion = full_motion[:, store.root_cond_dim :].astype(np.float32)
    stats_payload = serialize_motion_feature_stats(
        store.stats,
        names=store.names,
        parents=store.parents,
        joint_subset=store.joint_subset,
    )

    payload = {
        "motion": motion,
        "root_cond": root_cond,
        "train_windows": windows_to_array(store.split_windows["train"]),
        "val_windows": windows_to_array(store.split_windows["val"]),
        "test_windows": windows_to_array(store.split_windows["test"]),
        "range_names": np.asarray(store.range_names, dtype=object),
        "range_mirror": store.range_mirror.astype(bool),
        "window_size": np.asarray(store.window_size, dtype=np.int32),
        "root_cond_dim": np.asarray(store.root_cond_dim, dtype=np.int32),
        "motion_dim": np.asarray(store.motion_dim, dtype=np.int32),
        "full_motion_dim": np.asarray(store.full_motion_dim, dtype=np.int32),
    }
    payload.update(stats_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)

    print(f"saved={args.output}")
    print(f"database={store.database_path}")
    print(f"motion_shape={motion.shape}")
    print(f"root_cond_shape={root_cond.shape}")
    print(f"train_windows={len(store.split_windows['train'])}")
    print(f"val_windows={len(store.split_windows['val'])}")
    print(f"test_windows={len(store.split_windows['test'])}")


if __name__ == "__main__":
    main()
