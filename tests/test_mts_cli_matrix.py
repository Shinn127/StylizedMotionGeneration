"""C09: the real CLI integration matrix.

The R00 counterexamples (F01-F05) were all of one shape: a helper worked, and
the entry point never reached it.  This file therefore drives the *scripts*
(``main(argv)`` or a subprocess) against a fixture that is tiny but genuine: a
real NEF-FSQ tokenizer checkpoint (its own SHA recorded in the token store), a
store whose style ids and action ids are deliberately different numbers, and a
label table with several styles so a pair exists at all.

Cases (closure plan C09):

* A  ``--build-manifest-only`` with no MTS checkpoint at all (in test_mts_cli.py)
* B  transport: 2 steps -> frozen validation -> save -> reload
* C  three operators x {reference, style-ID}: 2 steps -> val -> save/load -> eval -> generate
* D  feature online reader vs token store on the same window (real store, read only)
* E  shuffled CTMC: level order, generator and probabilities survive save + load
* F  wrong hash / empty validation / illegal style / unknown config -> loud failure

The matrices reuse one frozen evaluation manifest and replay it at two batch
sizes, because "the manifest is the experiment" is only true if the numbers do
not move with the batch size.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    file_sha256,
    load_mts_checkpoint,
    load_operator_bundle,
    require_tokenizer_checkpoint,
)
from stylized_motion.learning.mts_operator.style_encoder import StyleIDEncoder  # noqa: E402
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer  # noqa: E402
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402

TINY_STYLES = [f"Style{index}" for index in range(6)]
TINY_ACTIONS = ["walk", "run"]
#: 16 clips: the first 12 are the train split, two clips per style with different
#: actions in each pair.  Style ids and action ids deliberately disagree
#: (Style5 -> action 1, Style2 -> action 0), so no code path can pass by mixing the
#: two tables up.
TINY_CLIP_STYLES = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 0, 1, 1]
TINY_CLIP_ACTIONS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
TINY_CLIPS = len(TINY_CLIP_STYLES)
#: 66 frames per clip, so a 64-frame window also has a tail window at offset 2.
TINY_CLIP_FRAMES = 66
WINDOW_FRAMES = 64
SEED = 7


def _load_test_module(name: str):
    """Loads a sibling test module by path (no package, no sys.path games)."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CLI_FIXTURES = None


def cli_fixtures():
    global _CLI_FIXTURES
    if _CLI_FIXTURES is None:
        _CLI_FIXTURES = _load_test_module("test_mts_cli")
    return _CLI_FIXTURES


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture: a real NEF tokenizer plus a store that matches it


def write_tiny_tokenizer(tmp_path: Path, *, name: str = "nef.pt", kinematics: bool = True):
    """A real NEF-FSQ checkpoint (production code path), small but not fake.

    ``kinematics`` adds the names/parents/ref_pos block the physics comparison
    needs; the real checkpoints carry it, the bare test helper does not.
    """
    persistence = _load_test_module("test_nef_persistence")
    path, representation, motion = persistence._save_checkpoint(tmp_path, name=name)
    if kinematics:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        names = [str(value) for value in payload["representation"]["nef_layout"]["names"]]
        parents = [int(value) for value in payload["representation"]["nef_layout"]["parents"]]
        stats = dict(payload["feature_stats"])
        stats.update(
            {
                "names": names,
                "parents": parents,
                "ref_pos": np.zeros((len(names), 3), dtype=np.float32),
            }
        )
        payload["feature_stats"] = stats
        torch.save(payload, path)
    return path, representation, motion


def expected_style_index(*, seed: int = SEED) -> dict[str, int]:
    """The style-id map this store must produce, from public helpers only.

    Re-derived here instead of read back from the run's own output: a test that
    asks the code under test what the answer is cannot catch a wrong answer.
    """
    from stylized_motion.learning.mts_operator.pairs import (
        ClipRecord,
        StylePairSampler,
        split_styles_by_performer,
    )

    records = [
        ClipRecord(
            clip_id=clip,
            style=TINY_STYLES[TINY_CLIP_STYLES[clip]],
            content=TINY_ACTIONS[TINY_CLIP_ACTIONS[clip]],
            source_group=clip,
            split="train" if clip < TINY_CLIPS - 4 else ("val" if clip < TINY_CLIPS - 2 else "test"),
            frames=TINY_CLIP_FRAMES,
        )
        for clip in range(TINY_CLIPS)
    ]
    split = split_styles_by_performer(records, val_fraction=0.2, unseen_fraction=0.2, seed=seed)
    sampler = StylePairSampler(records, style_split=split, seed=seed, window_frames=WINDOW_FRAMES)
    styles = {
        str(target.style)
        for target in sampler.eligible_targets(stage="train")
        if sampler.pairs_for(target, mode="same_style", count=1, stage="train")
    }
    return {style: index for index, style in enumerate(sorted(styles))}


def write_matching_token_store(
    tmp_path: Path,
    tokenizer_path: Path,
    *,
    clip_lengths: "list[int] | None" = None,
    split_ids: "list[int] | None" = None,
) -> Path:
    layout = cli_fixtures().write_tiny_token_store
    _, tokenizer = load_representation_checkpoint(tokenizer_path, torch.device("cpu"))
    return layout(
        tmp_path,
        clips=TINY_CLIPS,
        frames=TINY_CLIP_FRAMES,
        checkpoint_sha256=file_sha256(tokenizer_path),
        motion_dim=int(tokenizer.motion_dim),
        style_ids=TINY_CLIP_STYLES,
        action_ids=TINY_CLIP_ACTIONS,
        clip_lengths=list(clip_lengths or [TINY_CLIP_FRAMES] * TINY_CLIPS),
        styles=TINY_STYLES,
        actions=TINY_ACTIONS,
        split_ids=split_ids,
    )


#: The tiny store's own split table: clips 0-11 train, 12-13 val, 14-15 test.
TINY_TRAIN_CLIPS = [clip for clip in range(TINY_CLIPS) if clip < TINY_CLIPS - 4]
TINY_VAL_CLIPS = [clip for clip in range(TINY_CLIPS) if TINY_CLIPS - 4 <= clip < TINY_CLIPS - 2]


def build_env(tmp_path: Path) -> dict:
    """Tokenizer + store + configs, all inside ``tmp_path``."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path, _representation, _motion = write_tiny_tokenizer(tokenizer_dir)
    store = write_matching_token_store(tmp_path / "tokens", tokenizer_path)
    outputs = tmp_path / "runs"
    outputs.mkdir(parents=True, exist_ok=True)
    return {
        "root": tmp_path,
        "tokenizer": tokenizer_path,
        "store": store,
        "outputs": outputs,
        "transport_output": outputs / "transport",
        "style_index": expected_style_index(),
    }


def _masking() -> dict:
    return {
        "random_coordinate": 0.30,
        "stream": 0.20,
        "temporal_span": 0.20,
        "spatiotemporal_block": 0.15,
        "full_generation": 0.15,
        "coordinate_ratio": 0.5,
        "span_ratio": 0.5,
        "block_frames": 4,
        "block_coordinates": 4,
    }


def write_transport_config(env: dict, *, output: Path | None = None, content: str = "action_id") -> Path:
    config = {
        "tokenizer": {"checkpoint": str(env["tokenizer"]), "freeze": True},
        "data": {
            "token_store": str(env["store"]),
            "frames": WINDOW_FRAMES,
            "content": {"kind": content},
        },
        "transport": {
            "dim": 16,
            "token_embed_dim": 8,
            "position_encoding": "sinusoidal",
            "depth": 1,
            "heads": 2,
            "dropout": 0.0,
            "graph_mode": "local_relational",
            "graph_depth": 0,
            "temporal_mode": "bidirectional",
            "content_dim": None,
            "content_classes": None,
        },
        "masking": _masking(),
        "training": {
            "epochs": 2,
            "lr": 0.01,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "precision": "fp32",
            "seed": 7,
            "log_every_steps": 1,
            "steps_per_epoch": 1,
            "max_steps": 2,
            "output_dir": str(output or env["transport_output"]),
        },
        "sampling": {
            "strategy": "clip_uniform",
            "target_frames": WINDOW_FRAMES,
            "samples_per_epoch": 64,
            "seed": 7,
        },
        "loader": {"batch_size": 4, "num_workers": 0, "return_metadata": True},
        "evaluation": {
            "protocol_id": "tiny-transport-val-v1",
            "validation_kinds": ["full_generation"],
            "validation_rows_per_kind": 1,
        },
    }
    path = env["root"] / (
        "transport.yaml" if output is None else f"transport_{Path(output).name}.yaml"
    )
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def write_operator_config(
    env: dict,
    *,
    transport: Path,
    output: Path,
    operator: str = "logit_field",
    encoder_kind: str = "reference",
    num_styles: int | None = None,
) -> Path:
    config = {
        "tokenizer": {"checkpoint": str(env["tokenizer"]), "freeze": True},
        "transport": {
            "checkpoint": str(transport),
            "freeze": True,
            "dim": 16,
            "depth": 1,
            "heads": 2,
            "dropout": 0.0,
            "graph_mode": "local_relational",
            "graph_depth": 0,
            "temporal_mode": "bidirectional",
            "position_encoding": "sinusoidal",
            "token_embed_dim": 8,
        },
        "style_encoder": {
            "kind": encoder_kind,
            "dim": 16,
            "depth": 1,
            "heads": 2,
            "graph_depth": 0,
            "temporal_mode": "bidirectional",
            "position_encoding": "sinusoidal",
            "output_dim": 16,
            "num_styles": num_styles,
        },
        "operator": {
            "name": operator,
            "hidden_dim": 16,
            "coordinate_dim": 8,
            "strength": 1.0,
            # Only the chosen family's own options: the entry point refuses keys a
            # family does not implement instead of quietly ignoring them.
            **{
                "logit_field": {},
                "arbitrary_kernel": {"identity_mix": 0.1},
                "birth_death": {
                    "max_rate": 2.0,
                    "uniformization_tolerance": 1.0e-10,
                    "max_terms": 256,
                },
            }[operator],
        },
        "data": {
            "token_store": str(env["store"]),
            "frames": WINDOW_FRAMES,
            "reference_frames": WINDOW_FRAMES,
            "content": {"kind": "action_id"},
            "pairs": {
                "mode": "same_style",
                "stage": "train",
                "target_sampling": "style_uniform",
                "held_out_styles": [],
            },
            "style_split": {"val_fraction": 0.2, "unseen_fraction": 0.2},
        },
        "masking": _masking(),
        "evaluation": {
            "protocol_id": "tiny-operator-val-v1",
            "validation_batches_per_kind": 1,
            "validation_kinds": ["full_generation", "random_coordinate"],
        },
        "training": {
            "epochs": 2,
            "lr": 0.01,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "precision": "fp32",
            "seed": 7,
            "log_every_steps": 1,
            "steps_per_epoch": 1,
            "max_steps": 2,
            "output_dir": str(output),
            "content_weight": 0.0,
            "strength_range": [0.2, 1.0],
        },
        "sampling": {
            "strategy": "clip_uniform",
            "target_frames": WINDOW_FRAMES,
            "samples_per_epoch": 64,
            "seed": 7,
            "mirror_probability": 0.0,
        },
        "loader": {"batch_size": 2, "num_workers": 0},
    }
    path = env["root"] / f"operator_{operator}_{encoder_kind}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def matrix(tmp_path_factory) -> dict:
    """One tokenizer/store/config set, shared by the whole matrix.

    Sharing is deliberate: every operator arm must be trained and evaluated
    against the *same* frozen manifest, not against a per-test fixture that
    happens to look similar.
    """
    root = tmp_path_factory.mktemp("mts_matrix")
    env = build_env(root)
    env["transport_config"] = write_transport_config(env)
    return env


def train_transport(env: dict, monkeypatch=None, extra: list[str] | None = None) -> None:
    module = load_script("train_mts_transport")
    if extra is None and (env["transport_output"] / "best.pt").exists():
        # The module shares one trained tiny transport on purpose (every arm must
        # face the same frozen upstream), and a run no longer overwrites one.
        return
    argv = ["--config", str(env["transport_config"]), "--device", "cpu"] + list(extra or [])
    module.main(argv)
    assert (env["transport_output"] / "best.pt").exists()


def _write_training_config(
    env: dict,
    *,
    name: str,
    output: Path,
    training: dict,
    sampling: dict | None = None,
    loader: dict | None = None,
) -> Path:
    """A transport recipe with the given training/sampling/loader blocks."""
    config = write_transport_config(env, output=output)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["training"].update(training)
    if sampling:
        document["sampling"].update(sampling)
    if loader:
        document["loader"].update(loader)
    path = env["root"] / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# T02: the stated budget, one validation per epoch and the checkpoint evidence


def test_overfit_runs_exactly_the_stated_steps_and_claims_no_best(matrix):
    """T02: the step budget is met by repeating the frozen windows, and an overfit
    run writes no validation-best.

    The tiny loader serves one batch per epoch here, so a run that simply walked
    the loader would take far fewer than the five steps it was told to take.
    """
    env = matrix
    output = env["root"] / "runs" / "t02_overfit"
    config = _write_training_config(
        env,
        name="t02_overfit",
        output=output,
        training={"epochs": 1, "steps_per_epoch": 5, "max_steps": 5, "log_every_steps": 1},
        sampling={"samples_per_epoch": 4},
        loader={"batch_size": 4},
    )
    module = load_script("train_mts_transport")
    module.main(["--config", str(config), "--device", "cpu", "--overfit-clips", "2"])
    assert (output / "overfit_last.pt").exists()
    assert not (output / "best.pt").exists(), "an overfit run must not claim a validation-best"
    assert not (output / "last.pt").exists(), "an overfit run writes overfit_last.pt, not last.pt"
    checkpoint = torch.load(output / "overfit_last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 5
    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["global_step"] == 5
    assert summary["steps_shortfall"] == 0
    assert summary["best_val_loss"] is None
    epochs = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines() if line.strip()]
    assert len(epochs) == 1
    assert epochs[0]["steps"] == 5.0
    assert any(key.startswith("monitor_") for key in epochs[0])
    assert not any(key.startswith("val_") for key in epochs[0]), epochs[0].keys()
    frozen = json.loads((output / "overfit_frozen.json").read_text(encoding="utf-8"))
    assert frozen["windows"] == 2
    assert frozen["repeated_to_steps"] == 5
    # E02.3: the per-window numbers, not only the aggregate, are in the artifacts --
    # the manifest that names each window and mask, and the epoch's per-window rows.
    assert len(frozen["monitor"]) == 2
    for row in frozen["monitor"]:
        assert row["mask_kind"] in module.MONITOR_KINDS
        assert row["hidden_tokens"] > 0
        assert row["mask_seed"] is not None
    per_window = epochs[0]["monitor_per_window"]
    assert [row["window"] for row in per_window] == [0, 1]
    assert [row["mask_kind"] for row in per_window] == [row["mask_kind"] for row in frozen["monitor"]]
    assert all(row["nll"] is not None for row in per_window)
    assert sum(row["supervised_tokens"] for row in per_window) == epochs[0]["monitor_supervised_tokens"]


def test_a_short_run_is_not_reported_as_the_stated_budget(matrix):
    """T02: an epoch that runs out of data must not silently shrink the budget."""
    env = matrix
    output = env["root"] / "runs" / "t02_short"
    config = _write_training_config(
        env,
        name="t02_short",
        output=output,
        training={"epochs": 2, "steps_per_epoch": 3, "max_steps": 6, "log_every_steps": 5},
        sampling={"samples_per_epoch": 4},
        loader={"batch_size": 4},
    )
    module = load_script("train_mts_transport")
    with pytest.raises(SystemExit) as caught:
        module.main(["--config", str(config), "--device", "cpu"])
    assert caught.value.code == 1
    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["global_step"] == 2
    assert summary["planned_steps"] == 6
    assert summary["steps_shortfall"] == 4
    assert summary["completed"] is False


def test_validation_runs_once_per_epoch_and_is_the_number_that_is_saved(matrix):
    """T02: one validation pass per epoch, and log/history/payload/best share it."""
    env = matrix
    module = load_script("train_mts_transport")
    output = env["root"] / "runs" / "t02_single_validation"
    config = _write_training_config(
        env,
        name="t02_single_validation",
        output=output,
        training={"epochs": 2, "steps_per_epoch": 1, "max_steps": 2, "log_every_steps": 1},
    )
    calls = {"count": 0}
    original = module.ValidationBatchBuilder.evaluate

    def counted(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    module.ValidationBatchBuilder.evaluate = counted
    try:
        module.main(["--config", str(config), "--device", "cpu"])
    finally:
        module.ValidationBatchBuilder.evaluate = original
    assert calls["count"] == 2, "one validation pass per epoch"
    epochs = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines() if line.strip()]
    assert len(epochs) == 2
    for entry in epochs:
        assert "val_objective" in entry
        assert "val_seconds" in entry and "train_seconds" in entry and "checkpoint_seconds" in entry
        assert entry["optimizer_steps"] == entry["global_step"]
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert last["metrics"]["val_objective"] == pytest.approx(epochs[-1]["val_objective"])
    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    # The best metric the run reports is the objective it selected on, and the
    # protocol that produced it is recorded with its full fingerprint.
    assert summary["best_val_loss"] == pytest.approx(last["metrics"]["val_objective"])
    assert last["metrics"]["protocol_id"] == "tiny-transport-val-v1"
    assert len(last["metrics"]["protocol_hash"]) == 64
    assert set(last["metrics"]["val_counts"]) == set(last["metrics"]["val_per_kind"])
    assert last["metrics"]["val_supervised_tokens"] > 0


def test_a_wall_time_cap_stops_the_run_without_claiming_completion(matrix):
    """T02: an interrupted run says so; it never reports its budget as met."""
    env = matrix
    output = env["root"] / "runs" / "t02_wall_cap"
    config = _write_training_config(
        env,
        name="t02_wall_cap",
        output=output,
        training={"epochs": 1, "steps_per_epoch": 50, "max_steps": 50, "log_every_steps": 50},
        sampling={"samples_per_epoch": 256},
        loader={"batch_size": 8},
    )
    module = load_script("train_mts_transport")
    with pytest.raises(SystemExit) as caught:
        module.main(
            ["--config", str(config), "--device", "cpu", "--max-wall-seconds", "0.0001"]
        )
    assert caught.value.code == 1
    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["interrupted"] is True
    assert summary["interrupt_reason"] == "wall_time_cap"
    assert summary["completed"] is False
    assert summary["global_step"] < summary["planned_steps"]


def test_an_existing_run_directory_is_refused_by_default(matrix):
    """T02: a second run must not overwrite the first one's artifacts."""
    env = matrix
    output = env["root"] / "runs" / "t02_no_overwrite"
    config = _write_training_config(
        env,
        name="t02_no_overwrite",
        output=output,
        training={"epochs": 1, "steps_per_epoch": 1, "max_steps": 1},
    )
    module = load_script("train_mts_transport")
    module.main(["--config", str(config), "--device", "cpu"])
    assert (output / "train_summary.json").exists()
    with pytest.raises(SystemExit) as caught:
        module.main(["--config", str(config), "--device", "cpu"])
    assert caught.value.code == 1
    # The explicit override exists for deliberate scratch reruns.
    module.main(
        ["--config", str(config), "--device", "cpu", "--allow-existing-output"]
    )


def test_operator_overfit_claims_no_best_and_meets_its_budget(trained_transport):
    """T02: the operator's debug entry obeys the same rules as the transport's.

    The old overfit branch fell back to the *training* loss and wrote it as
    ``best.pt``; the frozen pairs are a training monitor and can never be a
    validation-best.
    """
    env = trained_transport
    output = env["outputs"] / "t02_operator_overfit"
    config = write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=output,
        operator="logit_field",
        encoder_kind="reference",
    )
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["training"].update(
        {"epochs": 1, "steps_per_epoch": 4, "max_steps": 4, "log_every_steps": 1}
    )
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module = load_script("train_mts_operator")
    module.main(
        [
            "--config", str(config),
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
            "--device", "cpu",
            "--overfit-pairs", "2",
        ]
    )
    assert (output / "overfit_last.pt").exists()
    assert not (output / "best.pt").exists(), "an overfit run must not claim a best checkpoint"
    assert not (output / "last.pt").exists()
    checkpoint = torch.load(output / "overfit_last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 4
    assert checkpoint["metrics"]["val_objective"] is None
    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["steps_shortfall"] == 0
    assert summary["best_val_nll"] is None
    epochs = [
        json.loads(line)
        for line in (output / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert epochs[-1]["steps"] == 4.0
    assert any(key.startswith("monitor_") for key in epochs[-1])
    assert not any(key.startswith("val_") for key in epochs[-1])


# ---------------------------------------------------------------------------
# B: transport 2 steps -> frozen validation -> save -> reload


def test_transport_cli_runs_the_frozen_protocol_and_reloads(matrix):
    """B: the real entry point, with a validation set that is built once."""
    env = matrix
    train_transport(env)
    output = env["transport_output"]
    assert (output / "best.pt").exists() and (output / "last.pt").exists()
    protocol = json.loads((output / "validation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["version"] == 2 and protocol["samples"] > 0
    # The protocol points at real windows of the store, not at clip 0 / start 0.
    starts = [item["target_start"] for item in protocol["items"]]
    assert any(int(start) > 0 for start in starts), starts
    # Every row carries its own seed and the run's mask configuration.
    for item in protocol["items"]:
        assert isinstance(item["seed"], int)
        assert item["mask_config"]["coordinate_ratio"] == 0.5

    checkpoint, model = load_mts_checkpoint(
        output / "best.pt",
        kind="transport",
        build_model=lambda stored: MotionTransportTransformer(_adapter(env), **stored),
        device="cpu",
        token_spec=_adapter(env).token_spec(representation_id=_representation_id(env)),
        tokenizer_metadata=_tokenizer_metadata(env),
        # The real tokenizer file: the transport cannot be loaded without proving
        # which tokenizer trained it.
        tokenizer_checkpoint=env["tokenizer"],
    )
    metrics = checkpoint["metrics"]
    assert metrics["val_objective"] is not None and np.isfinite(metrics["val_objective"])
    # The record must be the one the run printed: a checkpoint that hides its
    # validation number is not a validation-best.
    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["best_val_loss"] == pytest.approx(float(metrics["val_objective"]))
    assert summary["content_condition"]["classes"] == sorted(TINY_ACTIONS)
    # The optimizer state travels with the checkpoint: 2 steps were really taken.
    assert checkpoint["global_step"] == 2
    assert model.token_embed_dim == 8


# ---------------------------------------------------------------------------
# T00: the frozen validation protocol is held-out, fixed and batch-size free


def _dry_run_transport(
    module,
    env: dict,
    *,
    name: str,
    batch_size: int,
    evaluation: dict,
) -> dict:
    """One dry run that resolves the config and writes the frozen protocol."""
    output = env["root"] / "runs" / name
    config = write_transport_config(env, output=output)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["evaluation"] = dict(evaluation)
    document["loader"]["batch_size"] = int(batch_size)
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module.main(["--config", str(config), "--device", "cpu", "--dry-run"])
    return json.loads((output / "validation_protocol.json").read_text(encoding="utf-8"))


def test_transport_validation_rows_come_from_the_val_split_only(matrix):
    """T00: "the protocol exists" is not the acceptance; its rows are checked.

    Every row is looked up in the store's own split table (not in the run's own
    string), the train and val clip *and take* groups must be disjoint, the
    per-kind counts must equal the configured ones, and the row set must not
    move when the training batch size changes.
    """
    env = matrix
    module = load_script("train_mts_transport")
    evaluation = {
        "protocol_id": "tiny-val-v1",
        "validation_kinds": ["full_generation", "random_coordinate"],
        "validation_rows_per_kind": 2,
    }
    protocol = _dry_run_transport(module, env, name="t00_protocol_a", batch_size=4, evaluation=evaluation)
    other = _dry_run_transport(module, env, name="t00_protocol_b", batch_size=3, evaluation=evaluation)

    assert protocol["items"] == other["items"]
    assert protocol["protocol_id"] == "tiny-val-v1"
    assert protocol["splits"] == ["val"]
    assert protocol["samples_per_kind"] == {"full_generation": 2, "random_coordinate": 2}
    assert all(item["split"] == "val" for item in protocol["items"])

    store = open_any_token_store(env["store"])
    try:
        split_ids = np.asarray(store.split_ids)
        source_ids = np.asarray(store.source_clip_ids)
    finally:
        store.close()
    used = sorted({int(item["target_clip"]) for item in protocol["items"]})
    assert used and set(used) <= set(TINY_VAL_CLIPS)
    for clip in used:
        assert int(split_ids[clip]) == 1
    # Take-group isolation: no take appears on both sides of the boundary.
    train_groups = {int(source_ids[clip]) for clip in TINY_TRAIN_CLIPS}
    val_groups = {int(source_ids[clip]) for clip in used}
    assert train_groups & val_groups == set()


def test_transport_refuses_a_val_split_without_a_full_window(tmp_path):
    """T00: an empty or too-short val split fails; it never falls back to train."""
    module = load_script("train_mts_transport")
    env = build_env(tmp_path / "empty_val")
    store = write_matching_token_store(
        env["root"] / "tokens_all_train", env["tokenizer"], split_ids=[0] * TINY_CLIPS
    )
    config = write_transport_config({**env, "store": store}, output=env["root"] / "runs" / "empty")
    with pytest.raises(ValueError, match="val split has no clip with a full"):
        module.main(["--config", str(config), "--device", "cpu", "--dry-run"])

    short = build_env(tmp_path / "short_val")
    lengths = [TINY_CLIP_FRAMES] * (TINY_CLIPS - 4) + [10] * 4
    store = write_matching_token_store(
        short["root"] / "tokens_short_val", short["tokenizer"], clip_lengths=lengths
    )
    config = write_transport_config({**short, "store": store}, output=short["root"] / "runs" / "short")
    with pytest.raises(ValueError, match="val split has no clip with a full"):
        module.main(["--config", str(config), "--device", "cpu", "--dry-run"])


def test_transport_never_shrinks_the_protocol_to_the_available_clips(matrix):
    """T00: asking for more rows than the split can supply is an error.

    Silently scoring fewer rows would make two runs with different "budgets"
    look comparable when they are not.
    """
    env = matrix
    module = load_script("train_mts_transport")
    evaluation = {
        "protocol_id": "tiny-too-many",
        "validation_kinds": ["full_generation"],
        "validation_rows_per_kind": len(TINY_VAL_CLIPS) + 1,
    }
    with pytest.raises(ValueError, match="usable clips"):
        _dry_run_transport(module, env, name="t00_protocol_too_many", batch_size=4, evaluation=evaluation)


def test_transport_migrates_the_legacy_validation_field_explicitly(matrix):
    """T00: the old field is refused with instructions, never accepted-and-ignored."""
    env = matrix
    module = load_script("train_mts_transport")
    output = env["root"] / "runs" / "t00_legacy"
    config = write_transport_config(env, output=output)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["evaluation"] = {
        "validation_batches_per_kind": 4,
        "validation_kinds": ["full_generation"],
    }
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_rows_per_kind"):
        module.main(["--config", str(config), "--device", "cpu", "--dry-run"])
    with pytest.raises(SystemExit, match="validation_rows_per_kind"):
        module.main(["--config", str(config), "--device", "cpu", "--val-rows", "2", "--dry-run"])


def _adapter(env: dict):
    from stylized_motion.learning.mts_operator import LayoutAdapter

    _, tokenizer = load_representation_checkpoint(env["tokenizer"], torch.device("cpu"))
    return LayoutAdapter(tokenizer.token_layout(), num_levels=int(tokenizer.num_levels))


def _representation_id(env: dict) -> str:
    _, tokenizer = load_representation_checkpoint(env["tokenizer"], torch.device("cpu"))
    return str(tokenizer.representation_id)


def _tokenizer_metadata(env: dict) -> dict:
    _, tokenizer = load_representation_checkpoint(env["tokenizer"], torch.device("cpu"))
    return dict(tokenizer.representation_metadata())


def test_transport_dry_run_resolves_everything_and_trains_nothing(matrix):
    """C04: a dry run binds the artifacts and builds the protocol, then stops."""
    env = matrix
    output = env["root"] / "runs" / "dry_run"
    config = write_transport_config(env, output=output)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["loader"]["batch_size"] = 3
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module = load_script("train_mts_transport")
    # The dry run prints its report; catching the JSON keeps the test independent of
    # the log format around it.
    module.main(["--config", str(config), "--device", "cpu", "--dry-run"])
    assert not output.exists() or not (output / "best.pt").exists()
    assert not (output / "last.pt").exists()


def test_transport_reload_binds_the_tokenizer_file_not_just_its_shape(matrix):
    """B (negative half): same structure, different weights must not pass.

    The metadata block only describes the alphabet; two NEF tokenizers with the
    same layout agree there.  The recorded SHA is what separates them, and it is
    compared against a *file* at load time (F05).
    """
    env = matrix
    other_dir = env["root"] / "other_tokenizer"
    other_dir.mkdir(exist_ok=True)
    other_path, _representation, _motion = write_tiny_tokenizer(other_dir, name="other.pt")
    assert file_sha256(other_path) != file_sha256(env["tokenizer"])
    checkpoint, model = load_mts_checkpoint(
        env["transport_output"] / "best.pt",
        kind="transport",
        build_model=lambda stored: MotionTransportTransformer(_adapter(env), **stored),
        device="cpu",
        token_spec=_adapter(env).token_spec(representation_id=_representation_id(env)),
        tokenizer_metadata=_tokenizer_metadata(env),
        tokenizer_checkpoint=env["tokenizer"],
    )
    assert checkpoint["tokenizer_checkpoint_sha256"] == file_sha256(env["tokenizer"])
    with pytest.raises(ValueError, match="(?i)same structure is not the same weights"):
        require_tokenizer_checkpoint(checkpoint, tokenizer_checkpoint=other_path)
    # The file it was trained with passes, and the model is the trained one.
    require_tokenizer_checkpoint(checkpoint, tokenizer_checkpoint=env["tokenizer"])
    assert model.token_embed_dim == 8


def test_transport_refuses_a_store_that_records_another_tokenizer(matrix):
    """F: the store's tokenizer SHA is checked by the entry point, not just recorded."""
    env = matrix
    swapped = env["root"] / "swapped_store"
    _, tokenizer = load_representation_checkpoint(env["tokenizer"], torch.device("cpu"))
    cli_fixtures().write_tiny_token_store(
        swapped,
        clips=TINY_CLIPS,
        frames=TINY_CLIP_FRAMES,
        checkpoint_sha256="not-the-tokenizer-sha",
        motion_dim=int(tokenizer.motion_dim),
        style_ids=TINY_CLIP_STYLES,
        action_ids=TINY_CLIP_ACTIONS,
        clip_lengths=[TINY_CLIP_FRAMES] * TINY_CLIPS,
        styles=TINY_STYLES,
        actions=TINY_ACTIONS,
    )
    config = write_transport_config(env, output=env["root"] / "runs" / "binding", content="none")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["data"]["token_store"] = str(swapped)
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module = load_script("train_mts_transport")
    with pytest.raises(ValueError, match="same structure is not the same weights"):
        module.main(["--config", str(config), "--device", "cpu"])
    output = env["root"] / "runs" / "binding"
    assert not (output / "best.pt").exists() and not (output / "last.pt").exists()


# ---------------------------------------------------------------------------
# C: three operators x {reference, style-ID} through the real entry points

OPERATORS = ("logit_field", "arbitrary_kernel", "birth_death")
ENCODERS = ("reference", "style_id")


def operator_outputs(env: dict, operator: str, encoder: str) -> dict:
    output = env["outputs"] / f"{operator}_{encoder}"
    return {
        "output": output,
        "config": env["root"] / f"operator_{operator}_{encoder}.yaml",
        "manifest": env["root"] / "manifest",
        "eval": output / "eval",
        "generate": output / "generate",
    }


def run_operator_training(env: dict, operator: str, encoder: str, *, extra: list[str] | None = None) -> dict:
    paths = operator_outputs(env, operator, encoder)
    config = write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=paths["output"],
        operator=operator,
        encoder_kind=encoder,
        num_styles=len(env["style_index"]) if encoder == "style_id" else None,
    )
    module = load_script("train_mts_operator")
    module.main(
        [
            "--config", str(config),
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
            "--device", "cpu",
            # Several tests retrain the same tiny arm on purpose; the override is
            # what makes that an explicit scratch rerun instead of an overwrite.
            "--allow-existing-output",
        ]
        + list(extra or [])
    )
    return paths


def build_shared_manifest(env: dict) -> Path:
    """One frozen manifest, built from the data identity alone, reused by every arm."""
    paths = operator_outputs(env, "logit_field", "reference")
    module = load_script("evaluate_mts_operator")
    module.main(
        [
            "--build-manifest-only",
            "--token-store", str(env["store"]),
            "--split", "train",
            "--batches", "1",
            "--batch-size", "3",
            "--frames", str(WINDOW_FRAMES),
            "--mask-kind", "full_generation",
            "--seed", str(SEED),
            "--output", str(paths["manifest"]),
        ]
    )
    return paths["manifest"] / "eval_manifest.json"


@pytest.fixture(scope="module")
def trained_transport(matrix) -> dict:
    env = matrix
    if not (env["transport_output"] / "best.pt").exists():
        train_transport(env)
    return env


@pytest.fixture(scope="module")
def manifest(trained_transport) -> Path:
    return build_shared_manifest(trained_transport)


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("encoder", ENCODERS)
def test_operator_cli_trains_validates_and_reloads(trained_transport, operator, encoder):
    """C1-C6: 2 steps, a frozen validation, a saved best and a strict reload."""
    env = trained_transport
    paths = run_operator_training(env, operator, encoder)
    assert (paths["output"] / "best.pt").exists(), f"{operator}/{encoder} wrote no best.pt"
    protocol = json.loads((paths["output"] / "validation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["version"] == 2 and protocol["samples"] > 0
    assert protocol["splits"] == ["val"]

    adapter = _adapter(env)
    checkpoint, model = load_operator_bundle(
        paths["output"] / "best.pt",
        adapter=adapter,
        tokenizer_identity=_tokenizer_metadata(env),
        tokenizer_checkpoint=env["tokenizer"],
        device="cpu",
    )
    model_config = checkpoint["metadata"]["model_config"]
    assert model_config["operator"]["name"] == operator
    assert model_config["style_encoder"]["kind"] == encoder
    # Two training steps were taken (``last.pt`` is the final state); ``best.pt``
    # is the epoch whose validation objective improved, so its step is earlier.
    _, last = load_operator_bundle(
        paths["output"] / "last.pt",
        adapter=adapter,
        tokenizer_identity=_tokenizer_metadata(env),
        tokenizer_checkpoint=env["tokenizer"],
        device="cpu",
    )
    assert last is not None
    last_checkpoint = torch.load(paths["output"] / "last.pt", map_location="cpu", weights_only=False)
    assert last_checkpoint["global_step"] == 2
    assert checkpoint["global_step"] >= 1
    # The validation number the run selected on travels with the checkpoint.
    assert checkpoint["metrics"]["operator"] == operator
    assert checkpoint["metrics"]["style_encoder_kind"] == encoder
    # T02: the payload records the objective the run selected on, under that name,
    # with the full protocol fingerprint -- not a different number under "val_nll".
    assert checkpoint["metrics"]["val_objective"] == pytest.approx(checkpoint["metrics"]["val_nll"])
    assert checkpoint["metrics"]["protocol_id"] == "tiny-operator-val-v1"
    assert len(checkpoint["metrics"]["protocol_hash"]) == 64
    assert set(checkpoint["metrics"]["val_counts"]) == set(checkpoint["metrics"]["val_per_kind"])
    budget = json.loads((paths["output"] / "train_summary.json").read_text(encoding="utf-8"))
    assert budget["best_val_nll"] == pytest.approx(checkpoint["metrics"]["val_objective"])
    assert budget["steps_shortfall"] == 0 and budget["completed"] is True
    epochs = [
        json.loads(line)
        for line in (paths["output"] / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(epochs) == 2
    assert epochs[-1]["optimizer_steps"] == 2
    assert epochs[-1]["val_objective"] == pytest.approx(last_checkpoint["metrics"]["val_objective"])
    assert {"train_seconds", "val_seconds", "checkpoint_seconds", "total_seconds"} <= set(epochs[-1])
    exposure = checkpoint["provenance"]["training_exposure"]
    assert exposure["exposure_unknown"] is False
    assert exposure["actions"] == sorted(TINY_ACTIONS)
    if encoder == "style_id":
        stored = checkpoint["provenance"]["style_to_id"]
        assert stored == env["style_index"], (stored, env["style_index"])
        assert model_config["style_encoder"]["num_styles"] == len(env["style_index"])
    else:
        assert checkpoint["provenance"]["style_to_id"] is None
    assert checkpoint["tokenizer_checkpoint_sha256"] == file_sha256(env["tokenizer"])


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("encoder", ENCODERS)
def test_operator_cli_evaluates_its_own_frozen_manifest(trained_transport, manifest, operator, encoder):
    """C1-C6: the evaluator consumes the shared manifest and writes numbers."""
    env = trained_transport
    paths = operator_outputs(env, operator, encoder)
    if not (paths["output"] / "best.pt").exists():
        paths = run_operator_training(env, operator, encoder)
    module = load_script("evaluate_mts_operator")
    argv = [
        "--checkpoint", str(paths["output"] / "best.pt"),
        "--tokenizer-checkpoint", str(env["tokenizer"]),
        "--token-store", str(env["store"]),
        "--eval-manifest", str(manifest),
        "--split", "train",
        "--batch-size", "2",
        # A mixture kernel (arbitrary_kernel) requires 0 <= strength <= 1, so the
        # sweep stays inside the family's own validity range.
        "--strengths", "0.0", "0.5", "1.0",
        "--strength", "1.0",
        "--seed", str(SEED),
        "--label", f"{operator}:{encoder}",
        "--device", "cpu",
        "--output", str(paths["eval"]),
    ]
    if encoder == "style_id":
        # The *last* style id is deliberately outside the transport's action range:
        # feeding the style label into the action condition used to crash or, worse,
        # silently condition the transport on the wrong action.
        argv += ["--style-label", sorted(env["style_index"])[-1]]
        assert len(env["style_index"]) - 1 >= len(TINY_ACTIONS)
    module.main(argv)

    summary = json.loads((paths["eval"] / "operator_metrics.json").read_text(encoding="utf-8"))
    assert summary["operator"] == operator and summary["style_encoder"] == encoder
    assert summary["batches"] >= 1
    # Per-sample numeric rows, not just ids and reasons.
    rows = [json.loads(line) for line in (paths["eval"] / "eval_rows.jsonl").read_text().splitlines()]
    assert rows, "eval_rows.jsonl must carry one row per sample"
    for row in rows:
        assert row["supervised_tokens"] > 0
        assert row["target_token_nll_styled"] is not None
        assert row["target_token_nll_base"] is not None
        if encoder == "reference":
            assert row["retrieval_rank"] is not None
            assert row["retrieval_top1_hit"] in (True, False)
        else:
            # A style-ID model has no reference encoder: the row says so, and the
            # summary marks retrieval N/A instead of relabelling a number.
            assert row["retrieval_rank"] is None
            assert summary["style_retrieval"]["not_applicable"]
    # The three physics comparisons are reported separately, never averaged.
    physics = summary["physics_per_comparison"]
    for name in ("source_to_base", "base_to_styled", "source_to_styled"):
        assert physics[name] is not None, name
    assert summary["aggregate"]["nll_correct"] is not None


def test_evaluating_the_same_manifest_at_another_batch_size_keeps_the_rows(
    trained_transport, manifest
):
    """C09: the manifest is the experiment; the batch size is not."""
    env = trained_transport
    paths = operator_outputs(env, "logit_field", "reference")
    if not (paths["output"] / "best.pt").exists():
        paths = run_operator_training(env, "logit_field", "reference")
    module = load_script("evaluate_mts_operator")
    second = paths["output"] / "eval_b1"
    module.main(
        [
            "--checkpoint", str(paths["output"] / "best.pt"),
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--token-store", str(env["store"]),
            "--eval-manifest", str(manifest),
            "--split", "train",
            "--batch-size", "1",
            "--seed", str(SEED),
            "--device", "cpu",
            "--output", str(second),
        ]
    )
    first_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (paths["eval"] / "eval_rows.jsonl").read_text().splitlines()
        )
    }
    second_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line) for line in (second / "eval_rows.jsonl").read_text().splitlines()
        )
    }
    assert set(first_rows) == set(second_rows)
    for sample_id, row in first_rows.items():
        other = second_rows[sample_id]
        assert row["mask_seed"] == other["mask_seed"]
        assert row["target"]["clip_id"] == other["target"]["clip_id"]
        assert row["supervised_tokens"] == other["supervised_tokens"]
        # Same window, same mask, same candidate set: the NLL may differ only
        # through floating point, not through a different protocol.
        assert row["target_token_nll_styled"] == pytest.approx(
            other["target_token_nll_styled"], rel=1e-4
        )


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("encoder", ENCODERS)
def test_operator_cli_generates_from_the_saved_bundle(trained_transport, operator, encoder):
    """C1-C6: the generation entry point consumes the same bundle, with steps>1."""
    env = trained_transport
    paths = operator_outputs(env, operator, encoder)
    if not (paths["output"] / "best.pt").exists():
        paths = run_operator_training(env, operator, encoder)
    module = load_script("generate_mts_operator")
    argv = [
        "--checkpoint", str(paths["output"] / "best.pt"),
        "--tokenizer-checkpoint", str(env["tokenizer"]),
        "--token-store", str(env["store"]),
        "--split", "train",
        "--content-clip", "0",
        "--regions", "left_arm",
        "--graph-radius", "1",
        "--frames", str(WINDOW_FRAMES),
        "--samples", "2",
        "--steps", "3",
        "--seed", str(SEED),
        "--device", "cpu",
        "--output", str(paths["generate"]),
    ]
    if encoder == "style_id":
        argv += ["--style-label", sorted(env["style_index"])[0]]
    else:
        argv += ["--style-clip", "6"]
    module.main(argv)

    summary = json.loads((paths["generate"] / "generation.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 3
    assert summary["samples"] >= 1
    trace = np.load(paths["generate"] / "commit_trace.npy").astype(bool)
    region = np.load(paths["generate"] / "edit_mask.npy").astype(bool)
    # One commit mask per step; every edited position is committed exactly once,
    # and nothing outside the region is ever committed.
    assert trace.shape[0] == 3
    per_step = trace.sum(axis=tuple(range(1, trace.ndim)))
    assert all(int(count) > 0 for count in per_step), per_step
    assert np.array_equal(trace.sum(axis=0) > 0, np.broadcast_to(region, trace.shape[1:]))
    assert np.array_equal(trace.sum(axis=0), (trace.sum(axis=0) > 0).astype(int))
    assert np.load(paths["generate"] / "tokens.npy").shape[0] == summary["samples"]
    # Outside the region the draw reproduces the input tokens exactly.
    assert summary["outside_region_unchanged"] is True
    assert summary["lock_region_outside_unchanged"] is not False if "lock_region_outside_unchanged" in summary else True
    # A style-ID run needs no reference clip; giving one is a configuration error,
    # not a flag that is quietly ignored.
    if encoder == "style_id":
        with pytest.raises(SystemExit, match="style-clip"):
            module.main(
                [
                    "--checkpoint", str(paths["output"] / "best.pt"),
                    "--tokenizer-checkpoint", str(env["tokenizer"]),
                    "--token-store", str(env["store"]),
                    "--split", "train",
                    "--content-clip", "0",
                    "--style-clip", "6",
                    "--style-label", sorted(env["style_index"])[0],
                    "--samples", "1",
                    "--output", str(paths["generate"] / "conflict"),
                ]
            )
        assert not (paths["generate"] / "conflict" / "tokens.npy").exists()


# ---------------------------------------------------------------------------
# E: a shuffled CTMC must survive save and reload


def test_shuffled_ctmc_round_trip_keeps_its_layout_and_kernels(trained_transport):
    """E: level order, generator and probabilities are the same after a reload."""
    env = trained_transport
    output = env["outputs"] / "shuffled_birth_death"
    config = write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=output,
        operator="birth_death",
        encoder_kind="reference",
    )
    module = load_script("train_mts_operator")
    module.main(
        [
            "--config", str(config),
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
            "--shuffled-adjacency", "5",
            "--device", "cpu",
        ]
    )
    adapter = _adapter(env)
    first, model = load_operator_bundle(
        output / "best.pt",
        adapter=adapter,
        tokenizer_identity=_tokenizer_metadata(env),
        tokenizer_checkpoint=env["tokenizer"],
        device="cpu",
    )
    second, reloaded = load_operator_bundle(
        output / "best.pt",
        adapter=adapter,
        tokenizer_identity=_tokenizer_metadata(env),
        tokenizer_checkpoint=env["tokenizer"],
        device="cpu",
    )
    order = list(model.operator.level_order)
    assert sorted(order) == list(range(adapter.num_levels)), order
    assert order != list(range(adapter.num_levels)), "the fixture must really be shuffled"
    assert list(reloaded.operator.level_order) == order

    batch = _small_batch(env, adapter)
    with torch.no_grad():
        left = model(batch)
        right = reloaded(batch)
    # Same weights, same batch, same numbers: bitwise, not "close enough".
    assert torch.equal(left.probabilities, right.probabilities)
    assert torch.equal(left.base_probabilities, right.base_probabilities)
    # The generator is rebuilt from the model's own rates, and it is still a
    # generator (rows sum to zero) after the shuffled adjacency is applied.
    from stylized_motion.learning.mts_operator.operators import birth_death_generator

    order = torch.as_tensor(model.operator.level_order)
    inverse = torch.argsort(order)
    up_rate, down_rate, support = model.operator.rates(model.operator_inputs(batch))
    generator = birth_death_generator(up_rate, down_rate)[..., inverse][..., inverse, :]
    assert torch.allclose(
        generator.sum(dim=-1), torch.zeros_like(generator.sum(dim=-1)), atol=1e-6
    ), "a CTMC generator must have rows that sum to zero"
    # The edit region is the whole body here, so the rates must actually be alive:
    # an all-zero generator would make the round-trip check vacuous.
    assert float(up_rate.detach().abs().max()) > 0.0
    assert first["metrics"]["operator"] == "birth_death"


def _small_batch(env: dict, adapter):
    """One fixed two-sample batch read from the store, for model-level checks."""
    from stylized_motion.data.packed_token import open_any_token_store
    from stylized_motion.learning.mts_operator.model import OperatorBatch
    from stylized_motion.learning.mts_operator.windows import TokenSource, windows_by_clip

    store = open_any_token_store(env["store"])
    try:
        source = TokenSource(
            store=store,
            windows_by_clip=windows_by_clip(store, "train", frames=WINDOW_FRAMES),
            adapter=adapter,
            frames=WINDOW_FRAMES,
            history=0,
        )
        targets, references = [], []
        for clip in (0, 1):
            sample = source.window_at(clip, source.windows_by_clip[clip][0].target_start)
            targets.append(sample.tokens)
            references.append(sample.tokens)
        tokens = torch.stack(targets)
        return OperatorBatch(
            target_tokens=tokens,
            reference_tokens=torch.stack(references),
            visible_mask=torch.zeros_like(tokens, dtype=torch.bool),
            hard_mask=torch.ones(tokens.shape[1:], dtype=torch.bool),
            content_condition=torch.tensor([0, 1]),
            strength=1.0,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# F: loud failures, no artifacts


def test_unknown_style_label_is_refused_by_the_evaluator(trained_transport, manifest):
    """F: a style label the checkpoint never trained on is an error, not id 0."""
    env = trained_transport
    paths = operator_outputs(env, "logit_field", "style_id")
    if not (paths["output"] / "best.pt").exists():
        paths = run_operator_training(env, "logit_field", "style_id")
    module = load_script("evaluate_mts_operator")
    with pytest.raises(ValueError, match="Unknown style label"):
        module.main(
            [
                "--checkpoint", str(paths["output"] / "best.pt"),
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--token-store", str(env["store"]),
                "--eval-manifest", str(manifest),
                "--split", "train",
                "--style-label", "NotAStyle",
                "--batch-size", "2",
                "--device", "cpu",
                "--output", str(paths["output"] / "eval_unknown_style"),
            ]
        )
    assert not (paths["output"] / "eval_unknown_style" / "operator_metrics.json").exists()


def test_unknown_config_key_is_refused_before_any_training(trained_transport):
    """F: a typo in the recipe stops the run; it is not silently dropped."""
    env = trained_transport
    output = env["outputs"] / "unknown_config"
    config = write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=output,
        operator="logit_field",
        encoder_kind="reference",
    )
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["training"]["learning_rate"] = 1e-3
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module = load_script("train_mts_operator")
    with pytest.raises(ValueError, match="Unknown training fields|learning_rate"):
        module.main(
            ["--config", str(config), "--tokenizer-checkpoint", str(env["tokenizer"]),
             "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
             "--device", "cpu"]
        )
    assert not (output / "best.pt").exists() and not (output / "last.pt").exists()


def test_a_validation_split_without_pairs_stops_the_run(trained_transport):
    """F: an empty validation set is an error, never a run with no best."""
    env = trained_transport
    thin = env["root"] / "thin_store"
    _, tokenizer = load_representation_checkpoint(env["tokenizer"], torch.device("cpu"))
    # The val split holds two clips of one style with the *same* content label, so
    # same_style cannot form a single valid pair: the frozen protocol is empty.
    splits = [0] * 12 + [1] + [1] + [2, 2]
    thin_actions = list(TINY_CLIP_ACTIONS)
    thin_actions[13] = thin_actions[12]
    cli_fixtures().write_tiny_token_store(
        thin,
        clips=TINY_CLIPS,
        frames=TINY_CLIP_FRAMES,
        checkpoint_sha256=file_sha256(env["tokenizer"]),
        motion_dim=int(tokenizer.motion_dim),
        style_ids=TINY_CLIP_STYLES,
        action_ids=thin_actions,
        clip_lengths=[TINY_CLIP_FRAMES] * TINY_CLIPS,
        styles=TINY_STYLES,
        actions=TINY_ACTIONS,
        split_ids=splits,
    )
    output = env["outputs"] / "thin_val"
    config = write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=output,
        operator="logit_field",
        encoder_kind="reference",
    )
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["data"]["token_store"] = str(thin)
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module = load_script("train_mts_operator")
    with pytest.raises(ValueError, match="validation split produced no|no .* batches"):
        module.main(
            ["--config", str(config), "--tokenizer-checkpoint", str(env["tokenizer"]),
             "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
             "--device", "cpu"]
        )
    assert not (output / "best.pt").exists() and not (output / "validation_protocol.json").exists()


def test_content_kind_disagreement_is_refused(trained_transport, manifest):
    """F: the checkpoint is the authority on the condition; the flag only agrees."""
    env = trained_transport
    paths = operator_outputs(env, "logit_field", "reference")
    module = load_script("evaluate_mts_operator")
    with pytest.raises(ValueError, match="content-kind"):
        module.main(
            [
                "--checkpoint", str(paths["output"] / "best.pt"),
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--token-store", str(env["store"]),
                "--eval-manifest", str(manifest),
                "--split", "train",
                "--content-kind", "none",
                "--batch-size", "2",
                "--device", "cpu",
                "--output", str(paths["output"] / "eval_kind_conflict"),
            ]
        )


# ---------------------------------------------------------------------------
# D: the feature online path and the token store must agree on the same window


REAL_TOKENIZER = REPO_ROOT / "outputs" / "nef_fsq_soma_packed_40x9_ah" / "best.pt"
REAL_FEATURES = REPO_ROOT / "data" / "processed" / "seed_soma_pruned_v4_ah"
REAL_TOKENS = REPO_ROOT / "data" / "processed" / "seed_soma_pruned_v4_ah_tokens"
STALE_TOKENIZER = REPO_ROOT / "outputs" / "nef_fsq_soma_packed_40x9_1h" / "best.pt"


def test_real_data_identity_binds_the_holdout_tokenizer():
    """D/C02 on real artifacts: the store names the tokenizer that made it."""
    from stylized_motion.data.packed_token import open_any_token_store
    from stylized_motion.learning.mts_operator.checkpoint import require_token_store_binding

    for path in (REAL_TOKENIZER, REAL_FEATURES, REAL_TOKENS):
        if not path.exists():
            pytest.skip(f"real artifact missing: {path}")
    store = open_any_token_store(REAL_TOKENS)
    try:
        observed = require_token_store_binding(store, tokenizer_checkpoint=REAL_TOKENIZER)
        assert observed["checkpoint_sha256"] == file_sha256(REAL_TOKENIZER)
    finally:
        store.close()
    # The older 1h tokenizer has the same layout and different weights: it must be
    # refused for this store rather than silently re-encoding with other weights.
    stale = STALE_TOKENIZER
    if stale.exists() and file_sha256(stale) != file_sha256(REAL_TOKENIZER):
        store = open_any_token_store(REAL_TOKENS)
        try:
            with pytest.raises(ValueError, match="(?i)same structure is not the same weights"):
                require_token_store_binding(store, tokenizer_checkpoint=stale)
        finally:
            store.close()


@pytest.mark.parametrize("clip_position", [0, 1])
def test_online_feature_reader_matches_the_token_store_window(clip_position):
    """D: reading a window online must reproduce the stored tokens.

    The two readers must agree on the *window* (history included) and on the
    encoder's input space.  Reading a packed store as if its raw frames were
    already normalized changed ~71% of a window's tokens (measured), so this
    compares against the stored tokens rather than against another copy of the
    same reading code.  The stored tokens were produced on CUDA, so on a CPU-only
    host a handful of borderline levels may flip; the count is asserted, not
    waved away.
    """
    from stylized_motion.data import open_any_feature_store
    from stylized_motion.data.packed_token import open_any_token_store
    from stylized_motion.data.sampling import SampleRequest
    from stylized_motion.learning.mts_operator.windows import read_window_tokens

    for path in (REAL_TOKENIZER, REAL_FEATURES, REAL_TOKENS):
        if not path.exists():
            pytest.skip(f"real artifact missing: {path}")
    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(
        REAL_TOKENIZER, torch.device("cpu")
    )
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    frames = 64
    history = int(tokenizer.history_frames)
    tokens = open_any_token_store(REAL_TOKENS)
    features = open_any_feature_store(REAL_FEATURES)
    try:
        offsets = np.asarray(tokens.clip_offset, dtype=np.int64)
        lengths = np.asarray(tokens.clip_length, dtype=np.int64)
        chosen = np.flatnonzero(lengths >= frames + 2)[:4]
        assert len(chosen) >= 2, "the store must have windows with real tail context"
        compared = 0
        total_mismatch = 0
        total_entries = 0
        for clip in chosen:
            # Two starts per clip: the clip's own start (no history available) and a
            # start two frames in (`history/valid` must not be dropped).
            for delta in (0, 2):
                start = int(offsets[clip]) + delta + (clip_position % 2)
                if start + frames > int(offsets[clip]) + int(lengths[clip]):
                    continue
                request = SampleRequest(
                    shard_idx=int(tokens.clip_shard[clip]),
                    target_start=start,
                    target_frames=frames,
                    variant_idx=int(clip),
                )
                stored = read_window_tokens(tokens, request, frames=frames, history=0)
                online = read_window_tokens(
                    features,
                    request,
                    frames=frames,
                    history=history,
                    tokenizer=tokenizer,
                    feature_stats=feature_stats,
                )
                assert stored.shape == online.shape == (frames, 40)
                mismatch = int((stored != online).sum())
                total_mismatch += mismatch
                total_entries += int(stored.numel())
                compared += 1
        assert compared >= 2, compared
        # CPU vs CUDA float32 lands on different sides of a quantization boundary for
        # a few thousandths of the entries; anything larger is a reader/window bug
        # (the raw-space bug sat at 0.7).
        ratio = total_mismatch / max(total_entries, 1)
        assert ratio < 0.005, (
            f"{total_mismatch}/{total_entries} token entries disagree with the stored "
            "tokens: the online reader and the token store are not reading the same input"
        )
    finally:
        tokens.close()
        features.close()


def test_online_feature_reader_is_exact_on_the_device_that_built_the_store():
    """D: with CUDA the two readers agree bit for bit, not only statistically."""
    from stylized_motion.data import open_any_feature_store
    from stylized_motion.data.packed_token import open_any_token_store
    from stylized_motion.data.sampling import SampleRequest
    from stylized_motion.learning.mts_operator.windows import read_window_tokens

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the stored tokens were built on one")
    for path in (REAL_TOKENIZER, REAL_FEATURES, REAL_TOKENS):
        if not path.exists():
            pytest.skip(f"real artifact missing: {path}")
    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(
        REAL_TOKENIZER, torch.device("cuda")
    )
    frames = 64
    tokens = open_any_token_store(REAL_TOKENS)
    features = open_any_feature_store(REAL_FEATURES)
    try:
        start = int(np.asarray(tokens.clip_offset, dtype=np.int64)[0])
        request = SampleRequest(shard_idx=0, target_start=start, target_frames=frames, variant_idx=0)
        stored = read_window_tokens(tokens, request, frames=frames, history=0)
        online = read_window_tokens(
            features,
            request,
            frames=frames,
            history=int(tokenizer.history_frames),
            tokenizer=tokenizer,
            feature_stats=tokenizer_checkpoint.get("feature_stats"),
        )
        assert torch.equal(stored, online), (
            "on the device that produced the store the online reader must be exact"
        )
    finally:
        tokens.close()
        features.close()


def test_a_window_that_leaves_its_clip_is_refused_not_padded(trained_transport):
    """C09 fixture: nobody fabricates padding; the readers refuse an overrun.

    The sampler never requests an overflowing window, and the stores refuse to
    serve one (`IndexError`), so a batch cannot silently contain invented frames.
    A partial valid mask is therefore carried, not manufactured: the builder takes
    whatever the window reader reports (the C03 case injects one), and this test
    pins the refusal half of that contract.
    """
    from stylized_motion.data.packed_token import open_any_token_store
    from stylized_motion.learning.mts_operator import MaskGenerator
    from stylized_motion.learning.mts_operator.eval_protocol import (
        ValidationBatchBuilder,
        ValidationProtocol,
        ValidationSample,
    )
    from stylized_motion.learning.mts_operator.masking import MaskConfig
    from stylized_motion.learning.mts_operator.windows import TokenSource, WindowSample, windows_by_clip

    env = trained_transport
    adapter = _adapter(env)
    store = open_any_token_store(env["store"])
    try:
        grouped = windows_by_clip(store, "train", frames=WINDOW_FRAMES)
        offset = int(store.range_starts[0])
        length = int(store.range_stops[0] - store.range_starts[0])
        start = offset + length - WINDOW_FRAMES // 2
        overflow = type(grouped[0][0])(
            shard_idx=0, target_start=start, target_frames=WINDOW_FRAMES, variant_idx=0
        )
        source = TokenSource(
            store=store, windows_by_clip={0: [overflow]}, adapter=adapter,
            frames=WINDOW_FRAMES, history=0,
        )
        with pytest.raises(IndexError, match="range|leaves clip"):
            source.window_at(0, start)

        # A reader that does report a partial window has its mask carried verbatim.
        class PartialSource(TokenSource):
            def window_at(self, clip_id, start):  # noqa: D102 - fixture override
                sample = TokenSource.window_at(self, clip_id, start)
                assert sample is not None
                return WindowSample(
                    tokens=sample.tokens,
                    valid_mask=torch.zeros(WINDOW_FRAMES, dtype=torch.bool).index_fill_(
                        0, torch.arange(WINDOW_FRAMES - 3), True
                    ),
                    metadata=sample.metadata,
                )

        partial = PartialSource(
            store=store,
            windows_by_clip=grouped,
            adapter=adapter,
            frames=WINDOW_FRAMES,
            history=0,
        )
        builder = ValidationBatchBuilder(
            token_source=partial,
            mask_generator=MaskGenerator(MaskConfig(mixture={"full_generation": 1.0})),
            adapter=adapter,
            device="cpu",
        )
        protocol = ValidationProtocol(
            samples=(
                ValidationSample(
                    sample_id=0, kind="full_generation", split="train", target_clip=0,
                    target_start=int(grouped[0][0].target_start), reference_clip=0,
                    reference_start=int(grouped[0][0].target_start), seed=11,
                ),
            ),
            kinds=("full_generation",),
        )
        batch = builder.build_batch(protocol.samples)
        assert int(batch.target_valid_mask.sum()) == WINDOW_FRAMES - 3
        assert int(batch.reference_valid_mask.sum()) == WINDOW_FRAMES - 3
    finally:
        store.close()


def test_an_anchor_inside_the_edit_region_survives_generation(trained_transport):
    """C09 fixture: a batch with an explicit anchor keeps it observed."""
    from stylized_motion.learning.mts_operator.model import OperatorBatch

    env = trained_transport
    paths = operator_outputs(env, "logit_field", "reference")
    if not (paths["output"] / "best.pt").exists():
        paths = run_operator_training(env, "logit_field", "reference")
    _, model = load_operator_bundle(
        paths["output"] / "best.pt",
        adapter=_adapter(env),
        tokenizer_identity=_tokenizer_metadata(env),
        tokenizer_checkpoint=env["tokenizer"],
        device="cpu",
    )
    batch = _small_batch(env, _adapter(env))
    anchor = torch.zeros_like(batch.target_tokens, dtype=torch.bool)
    anchor[:, :8] = True
    anchored = OperatorBatch(
        target_tokens=batch.target_tokens,
        reference_tokens=batch.reference_tokens,
        visible_mask=torch.zeros_like(batch.target_tokens, dtype=torch.bool),
        hard_mask=torch.ones_like(batch.target_tokens, dtype=torch.bool),
        anchor_mask=anchor,
        content_condition=batch.content_condition,
        strength=1.0,
    )
    edit = anchored.effective_edit_mask(model.spec)
    assert not bool((edit & anchor).any()), "an anchor must stay evidence, not a target"
    assert bool(edit.any())
    drawn = model.generate_edit(anchored, generator=torch.Generator().manual_seed(3))
    assert torch.equal(drawn[anchor], anchored.target_tokens[anchor])


# ---------------------------------------------------------------------------
# T03: the experiment recipes are executable, and the no-reference control is real

RECIPE_DIR = REPO_ROOT / "data" / "configs"
OPERATOR_RECIPES = {
    "mts_revision2_style_id_logit.yaml": ("logit_field", "style_id"),
    "mts_revision2_style_id_ctmc.yaml": ("birth_death", "style_id"),
    "mts_revision2_reference_logit.yaml": ("logit_field", "reference"),
    "mts_revision2_noref_logit.yaml": ("logit_field", "constant"),
}
#: Fields each family owns; a recipe may not carry another family's knobs.
FAMILY_FIELDS = {
    "logit_field": {"hidden_dim", "coordinate_dim", "strength", "name"},
    "arbitrary_kernel": {"hidden_dim", "coordinate_dim", "strength", "identity_mix", "name"},
    "birth_death": {
        "hidden_dim", "coordinate_dim", "strength", "max_rate", "uniformization_tolerance",
        "max_terms", "level_order", "name",
    },
}


def test_every_operator_recipe_carries_only_its_own_family_fields():
    """T03: a recipe per arm, not one CTMC config switching every operator.

    Each file lists exactly the fields its family implements, states an integer
    step budget, and writes to its own run directory: nothing here can be a
    "which operator is this?" flag away from a different experiment.
    """
    seen_outputs = {}
    for name, (family, encoder_kind) in OPERATOR_RECIPES.items():
        path = RECIPE_DIR / name
        assert path.exists(), name
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["operator"]["name"] == family, name
        extras = sorted(set(document["operator"]) - FAMILY_FIELDS[family])
        assert extras == [], f"{name} carries foreign operator fields {extras}"
        assert document["style_encoder"]["kind"] == encoder_kind, name
        training = document["training"]
        assert isinstance(training["max_steps"], int) and training["max_steps"] > 0, name
        assert isinstance(training["steps_per_epoch"], int) and training["steps_per_epoch"] > 0, name
        assert training["epochs"] * training["steps_per_epoch"] >= training["max_steps"], name
        output = str(training["output_dir"])
        assert output.startswith("outputs/mts_revision2/"), name
        assert output not in seen_outputs, f"{name} overwrites {seen_outputs.get(output)}"
        seen_outputs[output] = name
        # The no-reference control owns no encoder configuration beyond its width.
        if encoder_kind == "constant":
            assert sorted(document["style_encoder"]) == ["kind", "output_dim"], name
        elif encoder_kind == "style_id":
            assert isinstance(document["style_encoder"]["num_styles"], int), name
        # One shared validation protocol for every arm; the transport recipes have
        # their own ids (different row counts must not look like the same thing).
        assert document["evaluation"]["protocol_id"], name
    # Every arm starts from the same base checkpoint path.
    bases = {
        yaml.safe_load((RECIPE_DIR / name).read_text(encoding="utf-8"))["transport"]["checkpoint"]
        for name in OPERATOR_RECIPES
    }
    assert len(bases) == 1, bases
    # And the transport recipes pin their own budgets, so training can start.
    for name in ("mts_revision2_transport_profile.yaml", "mts_revision2_transport_pilot.yaml"):
        document = yaml.safe_load((RECIPE_DIR / name).read_text(encoding="utf-8"))
        assert isinstance(document["training"]["max_steps"], int)
        assert document["data"]["content"]["kind"] == "action_id", name
        assert document["evaluation"]["protocol_id"], name


def test_operator_cli_dry_runs_the_control_arm_and_refuses_stray_encoder_keys(trained_transport, tmp_path):
    """T03: the no-reference arm goes through the real entry point.

    It must resolve without a style index at all, and a config that gives the
    control reference-encoder knobs must be refused: the control has no encoder to
    configure, and a config that looked like it did would misreport its parameters.
    """
    env = trained_transport
    output = env["outputs"] / "t03_noref_dry"
    config = write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=output,
        operator="logit_field",
        encoder_kind="constant",
        num_styles=None,
    )
    module = load_script("train_mts_operator")
    # The shared fixture writes a full reference-style encoder block; the control
    # owns only a width, and saying otherwise is what the second half refuses.
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["style_encoder"] = {"kind": "constant", "output_dim": 16}
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module.main(
        [
            "--config", str(config),
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
            "--device", "cpu",
            "--dry-run",
        ]
    )
    report = json.loads((output / "dry_run.json").read_text(encoding="utf-8"))
    assert report["style_encoder"] == "constant"
    assert report["style_index"] is None
    assert report["validation"]["items"] > 0
    assert report["validation"]["splits"] == ["val"]

    document["style_encoder"] = {"kind": "constant", "output_dim": 16, "depth": 2}
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="constant must not carry"):
        module.main(
            [
                "--config", str(config),
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
                "--device", "cpu",
                "--dry-run",
            ]
        )


def test_every_revision2_recipe_pins_an_executable_budget():
    """T04/train_val_disjoint: no recipe may leave its step budget to the loader.

    A recipe whose ``steps_per_epoch`` and ``max_steps`` are both empty describes a
    run whose length is "however many batches the data happens to serve"; training
    now refuses it, and this test makes the refusal visible as a config defect
    instead of a surprise at start-up.
    """
    recipes = sorted(
        path for path in (REPO_ROOT / "data" / "configs").glob("mts_revision2_*.yaml")
    )
    assert len(recipes) >= 9
    for path in recipes:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("training"), dict):
            continue  # the manifest is not a recipe
        training = document["training"]
        max_steps = training.get("max_steps")
        steps_per_epoch = training.get("steps_per_epoch")
        assert isinstance(max_steps, int) and max_steps > 0, path.name
        assert isinstance(steps_per_epoch, int) and steps_per_epoch > 0, path.name
        assert int(training["epochs"]) * int(steps_per_epoch) >= int(max_steps), path.name
        assert str(training["output_dir"]).startswith("outputs/"), path.name


def test_the_overfit_diagnostic_recipe_differs_from_the_profile_only_in_monitoring():
    """E02: the diagnostic must exercise the *profile's* training path, not a variant.

    Same tokenizer, data, architecture, masking and content conditioning as the
    profile recipe; a different budget and a monitor protocol named apart.  A
    diagnostic that silently changed the model or the masks would not say anything
    about the run the pilot will perform, and a recipe whose monitor scored
    full_generation would be judging a task the plan explicitly excludes.
    """
    configs = REPO_ROOT / "data" / "configs"
    profile = yaml.safe_load((configs / "mts_revision2_transport_profile.yaml").read_text(encoding="utf-8"))
    overfit = yaml.safe_load((configs / "mts_revision2_transport_overfit.yaml").read_text(encoding="utf-8"))
    for section in ("tokenizer", "data", "transport", "masking", "sampling"):
        assert overfit[section] == profile[section], section
    assert overfit["data"]["content"]["kind"] == "action_id"
    assert overfit["loader"]["batch_size"] == 8

    training = overfit["training"]
    assert training["max_steps"] == 200
    assert int(training["epochs"]) * int(training["steps_per_epoch"]) == 200
    assert training["steps_per_epoch"] == 20  # fixed-mask monitor points at 20/100/200
    assert training["log_every_steps"] == 10
    assert training["output_dir"] != profile["training"]["output_dir"]
    assert "overfit" in str(training["output_dir"])

    evaluation = overfit["evaluation"]
    assert evaluation["protocol_id"] == "mts-transport-r2-overfit-monitor-v1"
    assert evaluation["protocol_id"] != profile["evaluation"]["protocol_id"]
    assert "full_generation" not in evaluation["validation_kinds"]
    assert evaluation["validation_rows_per_kind"] == 2


def test_the_overfit_recipe_is_started_with_frozen_windows_not_a_validation_set(tmp_path):
    """The diagnostic entry point: `--overfit-clips 8` freezes windows, scores no val.

    The dry-run must not claim a validation protocol (there is none in this mode)
    and must report the frozen window count, so the two things an overfit run can
    be confused about -- "it validated" and "it trained on the whole split" -- are
    both visible before a single step is spent.
    """
    module = load_script("train_mts_transport")
    for path in (REAL_TOKENIZER, REAL_TOKENS):
        if not path.exists():
            pytest.skip(f"real artifact missing: {path}")
    config = tmp_path / "overfit.yaml"
    document = yaml.safe_load(
        (REPO_ROOT / "data" / "configs" / "mts_revision2_transport_overfit.yaml").read_text(encoding="utf-8")
    )
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / "overfit_run"
    module.main(
        [
            "--config", str(config),
            "--output", str(output),
            "--overfit-clips", "8",
            "--device", "cpu",
            "--dry-run",
        ]
    )
    report = json.loads((output / "dry_run.json").read_text(encoding="utf-8"))
    assert report["overfit_clips"] == 8
    assert report["validation"] is None or report["validation"]["rows"] == 0
    assert not (output / "validation_protocol.json").exists()
    assert not (output / "best.pt").exists()
    assert report["budget"]["planned_steps"] == 200
    assert report["content_condition"]["kind"] == "action_id"


def test_the_overfit_monitor_states_one_mask_per_window_and_records_it(matrix):
    """E02.3: the diagnostic curve must move because of learning, not new masks.

    The monitor items carry one explicit mask per frozen window, the kind is
    assigned round-robin over the four context kinds, the manifest names each
    window's clip/start/action/mask kind/mask seed/hidden fraction, and the same
    seed reproduces the masks bit for bit while another seed does not.  Without
    this the "same mask at step 20/100/200" comparison the plan asks for would be
    comparing different supervision sets.
    """
    from stylized_motion.learning.mts_operator.masking import MaskGenerator, MaskConfig
    from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer

    module = load_script("train_mts_transport")
    adapter = _adapter(matrix)
    spec = adapter.token_spec()
    generator = MaskGenerator(MaskConfig())

    torch.manual_seed(0)
    tokens = torch.randint(0, int(spec.num_levels), (2, 8, int(spec.num_coordinates)), dtype=torch.long)
    batch = {
        "tokens": tokens,
        "valid_mask": torch.ones((2, 8), dtype=torch.bool),
        "content_condition": None,
        "sample_metadata": [
            {"variant_idx": 11, "target_start": 128, "action": "Dancing"},
            {"variant_idx": 12, "target_start": 256, "action": "Stretching"},
        ],
    }
    items, manifest = module.fix_monitor_masks(
        [batch], mask_generator=generator, adapter=adapter, spec=spec, seed=3407
    )
    assert len(items) == len(manifest) == 2
    assert [row["mask_kind"] for row in manifest] == ["random_coordinate", "stream"]
    assert [row["clip_id"] for row in manifest] == [11, 12]
    assert [row["target_start"] for row in manifest] == [128, 256]
    assert [row["action"] for row in manifest] == ["Dancing", "Stretching"]
    assert [row["mask_seed"] for row in manifest] == [3407, 3408]
    for item, row in zip(items, manifest):
        hidden = ~item["visible_mask"]
        assert int(hidden.sum()) == row["hidden_tokens"] > 0
        assert int(hidden.sum()) == int((~item["visible_mask"]).sum())
        assert item["kind"] == row["mask_kind"]
        assert item["visible_mask"].shape == tokens[:1].shape
        # The window itself is untouched: the mask is the only addition.
        assert torch.equal(item["tokens"], batch["tokens"][row["window"] : row["window"] + 1])
    # No-kind-overload: a monitor item is a valid trainer input with a stated mask.
    model = MotionTransportTransformer(adapter, **{"dim": 16, "depth": 1, "heads": 2, "token_embed_dim": 4})
    from stylized_motion.learning.mts_operator.training import TrainerConfig, TransportTrainer

    trainer = TransportTrainer(
        model, adapter=adapter, device="cpu", config=TrainerConfig(seed=0), mask_generator=generator
    )
    first = trainer.evaluate(items)
    second = trainer.evaluate(items)
    assert first["loss"] == second["loss"]  # a stated mask does not move
    assert first["supervised_tokens"] == sum(row["hidden_tokens"] for row in manifest)

    # Same seed -> identical masks; another seed -> different masks (so the seed is used).
    repeat_items, _ = module.fix_monitor_masks(
        [batch], mask_generator=generator, adapter=adapter, spec=spec, seed=3407
    )
    other_items, _ = module.fix_monitor_masks(
        [batch], mask_generator=generator, adapter=adapter, spec=spec, seed=99
    )
    for left, right in zip(items, repeat_items):
        assert torch.equal(left["visible_mask"], right["visible_mask"])
    assert any(
        not torch.equal(left["visible_mask"], right["visible_mask"])
        for left, right in zip(items, other_items)
    )
    # Round-robin really covers the four context kinds once the run has eight windows.
    eight = {**batch, "tokens": tokens.repeat(4, 1, 1), "sample_metadata": batch["sample_metadata"] * 4}
    _, wide = module.fix_monitor_masks(
        [eight], mask_generator=generator, adapter=adapter, spec=spec, seed=3407, kinds=module.MONITOR_KINDS
    )
    assert sorted({row["mask_kind"] for row in wide}) == sorted(module.MONITOR_KINDS)
    assert len(wide) == 8


def test_the_pilot_saves_its_step_zero_weights_and_the_evaluator_scores_them(matrix):
    """E03.1: step 0 comes from the run's own weights, and the evaluator scores them.

    The transport run saves ``init.pt`` before its first step (a reference model,
    no optimizer state) and the base evaluator reproduces the trainer's own
    protocol numbers from a checkpoint and a protocol file alone -- which is what
    makes a step-0/step-N comparison meaningful.  A checkpoint that does not match
    the tokenizer file is refused, and the evaluator refuses protocol rows whose
    split the store disagrees with.
    """
    env = matrix
    module = load_script("train_mts_transport")
    output = env["root"] / "runs" / "e03_step_zero"
    config = _write_training_config(
        env,
        name="e03_step_zero",
        output=output,
        # The tiny loader serves one batch per epoch, so two steps means two epochs.
        training={"epochs": 2, "steps_per_epoch": 1, "max_steps": 2, "log_every_steps": 1},
        sampling={"samples_per_epoch": 4},
        loader={"batch_size": 4},
    )
    module.main(["--config", str(config), "--device", "cpu"])
    assert (output / "init.pt").exists(), "the pilot must save its step-0 weights"
    init_payload = torch.load(output / "init.pt", map_location="cpu", weights_only=False)
    assert init_payload["global_step"] == 0
    assert "optimizer" not in init_payload, "step 0 is a reference model, not a resume point"
    protocol = json.loads((output / "validation_protocol.json").read_text(encoding="utf-8"))

    evaluator = load_script("evaluate_mts_transport")
    evaluation_path = env["root"] / "runs" / "e03_step_zero" / "eval_init.json"
    evaluator.main(
        [
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--checkpoint", str(output / "init.pt"),
            "--protocol", str(output / "validation_protocol.json"),
            "--cases", "2",
            "--device", "cpu",
            "--output", str(evaluation_path),
        ]
    )
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert report["checkpoint"]["global_step"] == 0
    assert report["protocol"]["rows"] == protocol["samples"]
    assert all(report["protocol_file_statement"].values()), report["protocol_file_statement"]
    assert report["rows_verified_against_the_store_split_table"] == {"val": protocol["samples"]}
    assert report["supervised_tokens_total"] == sum(
        entry["supervised_tokens"] for entry in report["per_kind_detail"].values()
    )
    assert report["supervised_tokens_total"] > 0
    assert set(report["per_kind"]) == set(protocol["kinds"])
    assert all(value > 0 for value in report["per_kind"].values())
    assert len(report["cases"]) == min(2, protocol["samples"])
    cases_dir = Path(report["cases_dir"])
    for case in report["cases"]:
        assert any(name.endswith("argmax_motion.npy") for name in case["files"]), case
        assert (cases_dir / f"case{case['case_index']:02d}_target_motion.npy").exists()
        motion = np.load(cases_dir / f"case{case['case_index']:02d}_argmax_motion.npy")
        assert motion.ndim == 2 and motion.shape[0] > 0
    # A protocol row whose split the store disagrees with must be refused.
    tampered = json.loads((output / "validation_protocol.json").read_text(encoding="utf-8"))
    tampered["items"][0]["split"] = "train"
    # The file's own statement is updated too, so the *row-level* store check is
    # what has to catch it -- otherwise the cheaper file-consistency guard fires
    # and the store check would go untested.
    tampered["splits"] = ["train"]
    tampered_path = env["root"] / "runs" / "e03_step_zero_tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="claims split"):
        evaluator.main(
            [
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--checkpoint", str(output / "init.pt"),
                "--protocol", str(tampered_path),
                "--cases", "0",
                "--device", "cpu",
                "--output", str(env["root"] / "runs" / "e03_tampered_eval.json"),
            ]
        )


def test_operator_protocol_excludes_rows_the_frozen_vocabulary_cannot_condition_on():
    """E04.1: a val row whose action is outside the transport's vocabulary is dropped.

    The sampler draws from the split; the model can only be conditioned on the
    train split's actions, and borrowing an id for an unknown label would silently
    condition on the wrong action.  Such rows are excluded and counted (the same
    rule the transport's own protocol follows), and a mask kind that loses every
    row is an error rather than a quietly smaller protocol.
    """
    from stylized_motion.learning.mts_operator.eval_protocol import (
        ValidationSample,
        content_vocabulary_filter,
    )
    from stylized_motion.learning.mts_operator.windows import ContentVocabulary

    vocabulary = ContentVocabulary(kind="action_id", classes=("walk", "run"))
    samples = [
        ValidationSample(
            sample_id=index,
            kind="full_generation",
            split="val",
            target_clip=index,
            target_start=0,
            reference_clip=index + 100,
            reference_start=0,
            seed=1,
            content=content,
        )
        for index, content in enumerate(("walk", "sports", "run", "sports"))
    ]
    kept, report = content_vocabulary_filter(samples, vocabulary)
    assert [sample.content for sample in kept] == ["walk", "run"]
    assert report["excluded"] == 2
    assert report["by_action"] == {"sports": 2}
    assert report["kept_per_kind"] == {"full_generation": 2}
    # A kind that loses every row is refused: an empty protocol kind is not a result.
    only_unknown = [samples[1]]
    with pytest.raises(ValueError, match="left no action"):
        content_vocabulary_filter(only_unknown, vocabulary)
    # No vocabulary (unconditional content) keeps every row.
    kept_all, report_all = content_vocabulary_filter(samples, None)
    assert len(kept_all) == 4 and report_all["excluded"] == 0



def test_the_validation_rows_do_not_follow_the_training_seed(trained_transport):
    """N04: two seeds of one recipe must score the *same* frozen rows.

    The rows were drawn with the training seed, so seed 3408 silently produced a
    different 201-row protocol than seed 3407's 213 rows -- the "seed" axis changed
    the measurement as well as the model.  ``evaluation.seed`` pins the rows to the
    experiment; a recipe without it keeps the historical behaviour.
    """
    env = trained_transport
    records = {}
    for seed, name in ((3407, "seed_pinned_3407"), (3408, "seed_pinned_3408")):
        output = env["outputs"] / name
        config = write_operator_config(
            env,
            transport=env["transport_output"] / "best.pt",
            output=output,
            operator="logit_field",
            encoder_kind="style_id",
            num_styles=len(env["style_index"]),
        )
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
        document["evaluation"]["seed"] = 3407  # the experiment's rows, not the run's
        document["training"].update({"epochs": 2, "steps_per_epoch": 1, "max_steps": 2})
        config.write_text(yaml.safe_dump(document), encoding="utf-8")
        module = load_script("train_mts_operator")
        module.main(
            [
                "--config", str(config),
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--transport-checkpoint", str(env["transport_output"] / "best.pt"),
                "--seed", str(seed),
                "--output", str(output),
                "--device", "cpu",
            ]
        )
        summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
        protocol = json.loads((output / "validation_protocol.json").read_text(encoding="utf-8"))
        records[seed] = {
            "hash": summary["protocol_hash"],
            "rows": len(protocol["items"]),
            "sample_ids": [item["sample_id"] for item in protocol["items"]],
            "targets": [item["target_clip"] for item in protocol["items"]],
        }
    # Same rows, same order, same windows under both training seeds.
    assert records[3407]["hash"] == records[3408]["hash"], records
    assert records[3407]["sample_ids"] == records[3408]["sample_ids"]
    assert records[3407]["targets"] == records[3408]["targets"]
    # And the resolved protocol seed is the recipe's, not the run's.
    dry = load_script("train_mts_operator")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert document["evaluation"]["seed"] == 3407
