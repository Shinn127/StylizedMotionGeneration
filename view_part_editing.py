from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Genoview import GenoView, GenoViewCompare, build_database_from_feature_array


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a saved Part-FSQ source/donor/edit triplet in GenoView."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--debug-file", type=Path, required=True, help="One .npz file from evaluate_part_editing.py.")
    parser.add_argument(
        "--view",
        choices=["source-edited", "donor-edited", "source-donor", "source", "donor", "edited"],
        default="source-edited",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--compare-spacing", type=float, default=2.0)
    parser.add_argument("--resources-root", type=Path, default=Path(__file__).resolve().parent / "resources")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_features(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing editing debug file: {path}")
    data = np.load(path)
    required = {"source_features", "donor_features", "edited_features"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Editing debug file is missing: {sorted(missing)}")
    return {name: np.asarray(data[name], dtype=np.float32) for name in data.files}


def main(argv=None) -> None:
    args = parse_args(argv)
    data = load_features(args.debug_file)
    names = {
        "source": "Source",
        "donor": "Donor",
        "edited": "Edited",
    }
    databases = {
        name: build_database_from_feature_array(
            data[f"{name}_features"],
            args.checkpoint,
            True,
            f"part_edit_{name}",
        )
        for name in names
    }

    if args.view in {"source", "donor", "edited"}:
        viewer = GenoView(
            database=databases[args.view],
            trajectory_path=None,
            resources_root=args.resources_root,
            fps=args.fps,
        )
    else:
        left_name, right_name = args.view.split("-")
        viewer = GenoViewCompare(
            left_database=databases[left_name],
            right_database=databases[right_name],
            resources_root=args.resources_root,
            fps=args.fps,
            left_label=names[left_name],
            right_label=names[right_name],
            compare_spacing=args.compare_spacing,
        )
    print(f"debug_file={args.debug_file}")
    print(f"view={args.view}")
    print(f"frames={data['source_features'].shape[0]}")
    if args.dry_run:
        print("dry_run_viewer_ready=true")
        return
    viewer.run()


if __name__ == "__main__":
    main()
