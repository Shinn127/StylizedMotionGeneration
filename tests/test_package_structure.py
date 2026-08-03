import ast
from pathlib import Path

import yaml

from stylized_motion import run
from stylized_motion.run import COMMANDS, expand_arg_file
from stylized_motion.util.paths import CONFIG_DIR, DATA_DIR, PROJECT_ROOT, RESOURCE_DIR


def test_canonical_workflows_dispatch_to_single_runner():
    assert COMMANDS[("train", "representation")] == "stylized_motion.learning.runner"
    assert COMMANDS[("validate", "representation")] == "stylized_motion.learning.runner"
    assert COMMANDS[("test", "representation")] == "stylized_motion.learning.runner"
    assert COMMANDS[("preprocess", "token-database")] == "stylized_motion.data.preprocess"


def test_module_dispatch_preserves_importable_module_identity(monkeypatch):
    captured = {}

    class FakeModule:
        def main(self):
            captured["argv"] = list(__import__("sys").argv)

    monkeypatch.setattr(run.importlib, "import_module", lambda name: FakeModule())
    run._run_module("example.workflow", ["--workers", "8"])

    assert captured["argv"] == ["example.workflow", "--workers", "8"]


def test_preprocess_worker_has_stable_importable_module_identity():
    import pickle

    from stylized_motion.data.preprocess import _process_motion_pair

    assert _process_motion_pair.__module__ == "stylized_motion.data.preprocess_worker"
    pickle.dumps(_process_motion_pair)


def test_representation_dispatch_validates_spec_and_strips_outer_options(monkeypatch):
    captured = {}

    def fake_run(module_name, forwarded_args):
        captured["module"] = module_name
        captured["args"] = forwarded_args

    monkeypatch.setattr(run, "_run_module", fake_run)
    run.main(
        [
            "--mode", "validate",
            "--pipeline", "representation",
            "--representation", "part-fsq",
            "--config", "data/configs/part_fsq_40x9.yaml",
            "--checkpoint", "outputs/part_fsq_40x9/best.pt",
        ]
    )
    assert captured["module"] == "stylized_motion.learning.runner"
    assert captured["args"][:3] == ["--workflow-mode", "validate", "--representation"]
    assert "--mode" not in captured["args"]
    assert "--pipeline" not in captured["args"]


def test_preprocess_dispatch_strips_outer_options(monkeypatch):
    captured = {}

    def fake_run(forwarded_args):
        captured["args"] = forwarded_args

    monkeypatch.setattr(run, "_run_token_database", fake_run)
    run.main(
        [
            "--mode", "preprocess",
            "--pipeline", "token-database",
            "--checkpoint", "checkpoint.pt",
            "--feature-database", "features",
            "--output", "tokens",
        ]
    )
    assert captured["args"] == [
        "--checkpoint", "checkpoint.pt", "--feature-database", "features", "--output", "tokens",
    ]


def test_canonical_tree_is_present_and_old_workflow_modules_are_absent():
    root = PROJECT_ROOT / "stylized_motion"
    for path in (
        PROJECT_ROOT / "args" / "flat_fsq_args.txt",
        PROJECT_ROOT / "args" / "part_fsq_args.txt",
        PROJECT_ROOT / "args" / "residual_part_fsq_args.txt",
        PROJECT_ROOT / "args" / "latent_residual_fsq_args.txt",
        CONFIG_DIR / "flat_fsq_40x9.yaml",
        CONFIG_DIR / "part_fsq_40x9.yaml",
        CONFIG_DIR / "residual_part_fsq_40x9.yaml",
        CONFIG_DIR / "latent_residual_fsq_40x9.yaml",
        root / "data" / "__init__.py",
        root / "data" / "feature_data.py",
        root / "data" / "token_data.py",
        root / "data" / "trajectory_data.py",
        root / "data" / "sampling.py",
        root / "data" / "loader.py",
        root / "data" / "preprocess.py",
        root / "learning" / "nets" / "causal_cnn.py",
        root / "learning" / "nets" / "causal_transformer.py",
        root / "learning" / "nets" / "resnet.py",
        root / "learning" / "nets" / "quantizer.py",
    ):
        assert path.exists(), path
    for path in (
        root / "learning" / "model_builder.py",
        root / "learning" / "train_fsq.py",
        root / "learning" / "evaluate_fsq.py",
        root / "data" / "fsq_token_dataset.py",
        root / "data" / "feature_dataset.py",
        root / "data" / "token_store.py",
        root / "data" / "trajectory_store.py",
        root / "data" / "build_data.py",
        root / "data" / "build_database.py",
        root / "data" / "build_feature_database.py",
        root / "data" / "build_trajectory_database.py",
        root / "data" / "build_trajectory_inputs.py",
        root / "data" / "encode_token_database.py",
        root / "data" / "features_to_database.py",
    ):
        assert not path.exists(), path


def test_all_four_configs_use_nested_representation_contract():
    for family in ("flat_fsq", "part_fsq", "residual_part_fsq", "latent_residual_fsq"):
        config = yaml.safe_load((CONFIG_DIR / f"{family}_40x9.yaml").read_text(encoding="utf-8"))
        assert config["representation"]["family"] == family
        assert config["data"]["required_data_schema_version"] == 3
        assert config["sampling"]["target_frames"] == 64
        assert int(config["loader"]["batch_size"]) > 0
        assert "window_size" not in config["data"]
        assert config["training"]["precision"] == "fp32"
        assert "evaluation" in config


def test_mimickit_style_presets_expand_to_canonical_workflow():
    expanded = expand_arg_file(["--arg-file", "args/part_fsq_args.txt", "--epochs", "1"])
    assert expanded[:8] == [
        "--mode", "train", "--pipeline", "representation",
        "--representation", "part-fsq", "--config", "data/configs/part_fsq_40x9.yaml",
    ]
    assert expanded[-2:] == ["--epochs", "1"]


def test_repository_paths_are_rooted_at_the_checkout():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert CONFIG_DIR == PROJECT_ROOT / "data" / "configs"
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RESOURCE_DIR == PROJECT_ROOT / "data" / "assets" / "genoview"


def test_runner_counts_training_frames_once():
    source = (PROJECT_ROOT / "stylized_motion" / "learning" / "runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_epoch = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_epoch"
    )
    increments = [
        node for node in ast.walk(train_epoch)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "count"
    ]
    assert len(increments) == 1
