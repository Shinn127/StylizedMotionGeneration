"""Unified application entry point for Stylized Motion Generation.

The command registry follows MimicKit's ``run.py`` pattern: domain code stays
inside the package, while one small dispatcher selects the requested workflow.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    # Support both ``python -m stylized_motion.run`` and the MimicKit-style
    # ``python stylized_motion/run.py`` invocation.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from stylized_motion.util.arg_parser import load_arg_file


COMMANDS: dict[tuple[str, str], str] = {
    ("preprocess", "motion-database"): "stylized_motion.data.preprocess",
    ("preprocess", "feature-database"): "stylized_motion.data.preprocess",
    ("preprocess", "feature-cache"): "stylized_motion.data.preprocess",
    ("preprocess", "fsq-window-index"): "stylized_motion.data.preprocess",
    ("preprocess", "token-database"): "stylized_motion.data.preprocess",
    ("preprocess", "trajectory-inputs"): "stylized_motion.data.preprocess",
    ("preprocess", "trajectory-database"): "stylized_motion.data.preprocess",
    ("preprocess", "validate-data"): "stylized_motion.data.preprocess",
    ("train", "representation"): "stylized_motion.learning.runner",
    ("validate", "representation"): "stylized_motion.learning.runner",
    ("test", "representation"): "stylized_motion.learning.runner",
    ("train", "generator"): "stylized_motion.learning.generate",
    ("generate", "motion"): "stylized_motion.learning.generate",
    ("visualize", "motion"): "stylized_motion.anim.view_motion_sequence",
    ("visualize", "part-edit"): "stylized_motion.anim.view_part_edit",
    ("visualize", "genoview"): "stylized_motion.anim.genoview",
    ("visualize", "somaview"): "stylized_motion.anim.somaview",
    ("visualize", "realtime"): "stylized_motion.anim.realtime_fsq_controller",
    ("visualize", "plot"): "stylized_motion.anim.visualization",
    ("preprocess", "soma-assets"): "stylized_motion.anim.soma_assets",
}


def _run_module(module_name: str, forwarded_args: list[str]) -> None:
    previous_argv = sys.argv
    sys.argv = [module_name, *forwarded_args]
    try:
        # Import the workflow normally so multiprocessing workers can resolve
        # functions by their real module path instead of ``__main__``.
        module = importlib.import_module(module_name)
        module.main()
    finally:
        sys.argv = previous_argv


def _option_value(argv: list[str], name: str) -> str | None:
    value: str | None = None
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            value = argv[index + 1]
        elif item.startswith(f"{name}="):
            value = item.split("=", 1)[1]
    return value


def _without_dispatch_options(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in {"--mode", "--pipeline"}:
            skip = True
            continue
        if item.startswith("--mode=") or item.startswith("--pipeline="):
            continue
        filtered.append(item)
    return filtered


def _run_token_database(forwarded_args: list[str]) -> None:
    """Load the checkpoint in the composition root and inject its encoder."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-database", "--feature-store", dest="feature_database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--save-codes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(forwarded_args)
    from stylized_motion.data import open_feature_store
    from stylized_motion.data.preprocess import build_token_database
    from stylized_motion.data.feature_data import sha256_file
    from stylized_motion.learning.representation import load_representation_checkpoint
    from stylized_motion.learning.runner import choose_device

    device = choose_device(args.device)
    feature_store = open_feature_store(args.feature_database)
    try:
        checkpoint, encoder = load_representation_checkpoint(
            args.checkpoint,
            device,
            feature_schema=feature_store.feature_schema(),
        )
        checkpoint_stats = checkpoint.get("feature_stats")
        if not isinstance(checkpoint_stats, dict):
            raise ValueError("Representation checkpoint is missing feature_stats")
        source_offset = torch.from_numpy(feature_store.stats.offset.astype("float32")).to(device)
        source_scale = torch.from_numpy(feature_store.stats.scale.astype("float32")).to(device)
        checkpoint_offset = torch.as_tensor(checkpoint_stats["offset"], dtype=torch.float32, device=device)
        checkpoint_scale = torch.as_tensor(checkpoint_stats["scale"], dtype=torch.float32, device=device)

        def input_adapter(values: torch.Tensor) -> torch.Tensor:
            raw = values * source_scale.view(1, 1, -1) + source_offset.view(1, 1, -1)
            return (raw - checkpoint_offset.view(1, 1, -1)) / checkpoint_scale.view(1, 1, -1)

        build_token_database(
            args.feature_database,
            args.output,
            encoder=encoder,
            checkpoint_sha256=sha256_file(args.checkpoint),
            model_family_legacy=str(checkpoint.get("model_family", "")),
            device=device,
            chunk_size=args.chunk_size,
            save_codes=args.save_codes,
            input_adapter=input_adapter,
            overwrite=args.overwrite,
        )
    finally:
        feature_store.close()


def _validate_representation_dispatch(raw_args: list[str], parser: argparse.ArgumentParser) -> None:
    representation_cli = _option_value(raw_args, "--representation")
    config_path = _option_value(raw_args, "--config")
    if representation_cli is None or config_path is None:
        parser.error("representation workflows require --representation and --config")
    expected = {
        "flat-fsq": "flat_fsq",
        "part-fsq": "part_fsq",
        "residual-part-fsq": "residual_part_fsq",
        "latent-residual-fsq": "latent_residual_fsq",
        "latent-residual-fsq-v2": "latent_residual_fsq_v2",
    }.get(representation_cli)
    if expected is None:
        parser.error(f"Unsupported canonical representation {representation_cli!r}")
    from stylized_motion.learning.representation import representation_spec
    from stylized_motion.learning.runner import load_experiment_config

    config = load_experiment_config(config_path)
    spec = representation_spec(config)
    if spec.family != expected:
        parser.error(
            f"--representation {representation_cli!r} does not match config family {spec.family!r}"
        )


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    modes = sorted({mode for mode, _ in COMMANDS})
    pipelines = sorted({pipeline for _, pipeline in COMMANDS})
    parser = argparse.ArgumentParser(
        description="Run Stylized Motion Generation workflows through one package entry point.",
        add_help=add_help,
    )
    parser.add_argument("--mode", choices=modes, required=True, help="Workflow family to run.")
    parser.add_argument("--pipeline", choices=pipelines, required=True, help="Workflow implementation.")
    parser.add_argument("--arg-file", "--arg_file", type=Path, help="Load additional arguments from a text preset.")
    return parser


def expand_arg_file(argv: list[str]) -> list[str]:
    """Expand one MimicKit-style argument preset before dispatch parsing."""
    expanded: list[str] = []
    arg_file: Path | None = None
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"--arg-file", "--arg_file"}:
            if arg_file is not None or index + 1 >= len(argv):
                raise ValueError("Exactly one --arg-file PATH is required")
            arg_file = Path(argv[index + 1])
            index += 2
            continue
        expanded.append(value)
        index += 1
    if arg_file is None:
        return expanded
    return [*load_arg_file(arg_file), *expanded]


def main(argv: list[str] | None = None) -> None:
    raw_args = expand_arg_file(list(sys.argv[1:] if argv is None else argv))
    has_dispatch = "--mode" in raw_args and "--pipeline" in raw_args
    if ("--help" in raw_args or "-h" in raw_args) and not has_dispatch:
        build_parser().print_help()
        return

    parser = build_parser(add_help=False)
    args, forwarded_args = parser.parse_known_args(raw_args)
    module_name = COMMANDS.get((args.mode, args.pipeline))
    if module_name is None:
        parser.error(f"pipeline {args.pipeline!r} is not available in mode {args.mode!r}")
    forwarded_args = _without_dispatch_options(forwarded_args)
    if args.mode == "preprocess":
        forwarded_args = [args.pipeline, *forwarded_args]
    if module_name == "stylized_motion.learning.runner":
        _validate_representation_dispatch(raw_args, parser)
        forwarded_args = ["--workflow-mode", args.mode, *forwarded_args]
    elif module_name == "stylized_motion.learning.generate":
        forwarded_args = [
            "--workflow-mode",
            "train" if args.mode == "train" else "generate",
            *forwarded_args,
        ]
    if args.mode == "preprocess" and args.pipeline == "token-database":
        _run_token_database(forwarded_args[1:])
    else:
        _run_module(module_name, forwarded_args)


if __name__ == "__main__":
    main()
