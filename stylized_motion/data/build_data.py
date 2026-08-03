import argparse
import os
from pathlib import Path

from stylized_motion.data.build_feature_database import build_processed_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build database.npz and the normalized feature database in one BVH preprocessing pass."
    )
    parser.add_argument("--dataset", choices=["lafan", "100style"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="Root output directory.")
    parser.add_argument("--styles", type=str, default=None)
    parser.add_argument("--max-styles", type=int, default=None)
    parser.add_argument("--prune-ends-and-fingers", action="store_true")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    return parser.parse_args()


def main():
    args = parse_args()
    build_processed_data(
        dataset_name=args.dataset,
        output_dir=args.output,
        styles_arg=args.styles,
        max_styles=args.max_styles,
        prune_ends_and_fingers=args.prune_ends_and_fingers,
        window_size=args.window_size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
