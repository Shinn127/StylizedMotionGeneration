"""C10: the revision-2 preflight answers "can this be trained on".

The preflight is only useful if it *fails* on a combination that looks fine on
disk.  These tests drive the real script: the real holdout tokenizer with the real
holdout token store must pass the data level, the older 1h tokenizer with the same
store must fail, the window cap must hold, and the run must not produce a model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.learning.mts_operator.checkpoint import file_sha256  # noqa: E402
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402

REAL_TOKENIZER = REPO_ROOT / "outputs" / "nef_fsq_soma_packed_40x9_ah" / "best.pt"
REAL_FEATURES = REPO_ROOT / "data" / "processed" / "seed_soma_pruned_v4_ah"
REAL_TOKENS = REPO_ROOT / "data" / "processed" / "seed_soma_pruned_v4_ah_tokens"
STALE_TOKENIZER = REPO_ROOT / "outputs" / "nef_fsq_soma_packed_40x9_1h" / "best.pt"
REAL_CONFIG = REPO_ROOT / "data" / "configs" / "mts_revision2_style.yaml"
TINY_CONFIG = REPO_ROOT / "data" / "configs" / "mts_revision2_style_smoke.yaml"


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_cli_matrix_module():
    path = Path(__file__).with_name("test_mts_cli_matrix.py")
    spec = importlib.util.spec_from_file_location("test_mts_cli_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["test_mts_cli_matrix"] = module
    spec.loader.exec_module(module)
    return module


def run_preflight(argv: list[str]) -> dict:
    module = load_script("preflight_mts_revision2")
    module.main(argv)
    return json.loads((Path(argv[argv.index("--output") + 1]) / "preflight.json").read_text())


def require_real_artifacts() -> None:
    for path in (REAL_TOKENIZER, REAL_FEATURES, REAL_TOKENS, REAL_CONFIG):
        if not path.exists():
            pytest.skip(f"real artifact missing: {path}")


def test_real_holdout_data_passes_the_data_level(tmp_path):
    require_real_artifacts()
    output = tmp_path / "preflight"
    payload = run_preflight(
        [
            "--config", str(REAL_CONFIG),
            "--output", str(output),
        ]
    )
    assert payload["stage"] == "data"
    assert payload["ready"] is True, payload["required_not_passed"]
    assert payload["ok"] is True, payload["failed"]
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    assert statuses["tokenizer"] == "pass"
    assert statuses["store_identity"] == "pass"
    assert statuses["split_isolation"] == "pass"
    assert statuses["windows"] == "pass"
    # Checks that belong to later stages are reported as not_applicable here, not
    # as failures and not as fabricated passes.
    assert statuses["validation_protocol"] == "not_applicable"
    assert statuses["transport"] == "not_applicable"
    # No model was created anywhere by a preflight.
    assert not list(tmp_path.rglob("best.pt"))


def test_a_tiny_fixture_passes_the_data_level(tmp_path):
    """A small but genuine fixture: real tokenizer, store bound by SHA, labels."""
    matrix = load_cli_matrix_module()
    env = matrix.build_env(tmp_path / "env")
    config = matrix.write_operator_config(
        env,
        transport=tmp_path / "unused.pt",
        output=tmp_path / "runs" / "unused",
        operator="logit_field",
        encoder_kind="reference",
    )
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["data"]["required_data_schema_version"] = 3
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    payload = run_preflight(
        [
            "--config", str(config),
            "--tokenizer-checkpoint", str(env["tokenizer"]),
            "--token-store", str(env["store"]),
            "--max-windows", "2",
            "--output", str(tmp_path / "preflight"),
        ]
    )
    assert payload["ok"] is True, payload["failed"]
    windows = next(item for item in payload["checks"] if item["id"] == "windows")
    assert windows["evidence"]["clips_read"] == 2
    assert windows["evidence"]["cap_respected"] is True
    assert windows["evidence"]["non_zero_starts"] >= 1
    # The store carries no actor column, so the report says so rather than
    # claiming a performer-disjoint split.
    splits = next(item for item in payload["checks"] if item["id"] == "split_isolation")
    assert splits["status"] == "pass"
    assert splits["evidence"]["test_actor_exposure"] == "unknown"


def test_the_window_cap_is_enforced(tmp_path):
    """C10: the preflight reads a bounded number of windows, and says how many."""
    matrix = load_cli_matrix_module()
    env = matrix.build_env(tmp_path / "env")
    config = matrix.write_operator_config(
        env,
        transport=tmp_path / "unused.pt",
        output=tmp_path / "runs" / "unused",
        operator="logit_field",
        encoder_kind="reference",
    )
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["data"]["required_data_schema_version"] = 3
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    module = load_script("preflight_mts_revision2")
    output = tmp_path / "preflight"
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--config", str(config),
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--token-store", str(env["store"]),
                "--max-windows", "0",
                "--output", str(output),
            ]
        )
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    windows = next(item for item in payload["checks"] if item["id"] == "windows")
    assert windows["status"] == "fail"
    assert "max_windows" in windows["reason"]


def test_a_single_style_store_fails_the_label_check(tmp_path):
    """A store with one style cannot support a style axis, and the check says why."""
    matrix = load_cli_matrix_module()
    env = matrix.build_env(tmp_path / "env")
    thin = tmp_path / "single_style"
    _, tokenizer = load_representation_checkpoint(env["tokenizer"], torch.device("cpu"))
    matrix.cli_fixtures().write_tiny_token_store(
        thin,
        clips=matrix.TINY_CLIPS,
        frames=matrix.TINY_CLIP_FRAMES,
        checkpoint_sha256=file_sha256(env["tokenizer"]),
        motion_dim=int(tokenizer.motion_dim),
        style_ids=[0] * matrix.TINY_CLIPS,
        action_ids=matrix.TINY_CLIP_ACTIONS,
        clip_lengths=[matrix.TINY_CLIP_FRAMES] * matrix.TINY_CLIPS,
        styles=["OnlyStyle"],
        actions=matrix.TINY_ACTIONS,
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "tokenizer": {"checkpoint": str(env["tokenizer"]), "freeze": True},
                "data": {
                    "token_store": str(thin),
                    "frames": matrix.WINDOW_FRAMES,
                    "required_data_schema_version": 3,
                    "content": {"kind": "action_id"},
                    "pairs": {"mode": "same_style", "target_sampling": "style_uniform"},
                    "style_split": {"val_fraction": 0.2, "unseen_fraction": 0.2},
                },
                "masking": {
                    "random_coordinate": 1.0,
                    "coordinate_ratio": 0.5,
                },
                "loader": {"batch_size": 2, "num_workers": 0},
                "training": {"output_dir": "outputs/mts_revision2/preflight_thin"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "preflight"
    module = load_script("preflight_mts_revision2")
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--config", str(config),
                "--tokenizer-checkpoint", str(env["tokenizer"]),
                "--token-store", str(thin),
                "--output", str(output),
            ]
        )
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert "label_tables" in payload["failed"]
    reason = next(
        item["reason"] for item in payload["checks"] if item["id"] == "label_tables"
    )
    assert "one style" in reason


def test_the_older_tokenizer_is_rejected_for_the_holdout_store(tmp_path):
    """The negative acceptance case: same layout, different weights."""
    require_real_artifacts()
    if not STALE_TOKENIZER.exists():
        pytest.skip("the older 1h tokenizer is not present")
    assert file_sha256(STALE_TOKENIZER) != file_sha256(REAL_TOKENIZER)
    output = tmp_path / "preflight"
    module = load_script("preflight_mts_revision2")
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--config", str(REAL_CONFIG),
                "--tokenizer-checkpoint", str(STALE_TOKENIZER),
                "--output", str(output),
            ]
        )
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ok"] is False
    assert "store_identity" in payload["failed"]
    reason = next(
        item["reason"] for item in payload["checks"] if item["id"] == "store_identity"
    )
    assert "same structure is not the same weights" in reason
    assert not list(tmp_path.rglob("best.pt"))


# ---------------------------------------------------------------------------
# T01: the stage table, the real positive path and the two negative branches


@pytest.fixture(scope="module")
def tiny_chain(tmp_path_factory):
    """A trained tiny transport plus the protocols the dry runs froze.

    This is the positive path: a real (if tiny) revision-2 transport, a real NEF
    tokenizer file and real frozen protocols, all produced by the real entry
    points with no training beyond the two-step transport.
    """
    matrix = load_cli_matrix_module()
    root = tmp_path_factory.mktemp("t01_chain")
    env = matrix.build_env(root)
    env["transport_config"] = matrix.write_transport_config(env)
    matrix.train_transport(env)

    # The transport's own protocol, frozen by its dry run (no training).
    transport_run = root / "runs" / "transport_dry"
    transport_config = matrix.write_transport_config(env, output=transport_run)
    env["transport_recipe"] = transport_config
    env["transport_protocol"] = transport_run / "validation_protocol.json"
    load_script("train_mts_transport").main(
        ["--config", str(transport_config), "--device", "cpu", "--dry-run"]
    )

    # The operator's protocol, frozen by *its* dry run against the trained
    # transport.  The output stays in the test's own directory.
    operator_run = root / "runs" / "operator_dry"
    env["operator_config"] = matrix.write_operator_config(
        env,
        transport=env["transport_output"] / "best.pt",
        output=operator_run,
        operator="logit_field",
        encoder_kind="reference",
    )
    env["operator_protocol"] = operator_run / "validation_protocol.json"
    load_script("train_mts_operator").main(
        ["--config", str(env["operator_config"]), "--device", "cpu", "--dry-run"]
    )

    # A fixed evaluation manifest, built by the real evaluator entry point.
    manifest_dir = root / "runs" / "eval_manifest"
    load_script("evaluate_mts_operator").main(
        [
            "--build-manifest-only",
            "--token-store", str(env["store"]),
            "--split", "test",
            "--batches", "1",
            "--batch-size", "2",
            "--output", str(manifest_dir),
        ]
    )
    env["eval_manifest"] = manifest_dir / "eval_manifest.json"
    return env


def _preflight_argv(tiny: dict, stage: str, output: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "--config", str(tiny["operator_config"]),
        "--stage", stage,
        "--tokenizer-checkpoint", str(tiny["tokenizer"]),
        "--token-store", str(tiny["store"]),
        "--transport-checkpoint", str(tiny["transport_output"] / "best.pt"),
        "--validation-protocol", str(tiny["operator_protocol"]),
        "--max-windows", "2",
        "--output", str(output),
    ] + list(extra or [])


def test_operator_stage_passes_with_a_correct_tiny_transport(tiny_chain, tmp_path):
    """T01 positive branch: the preflight succeeds when the upstream is the right one."""
    module = load_script("preflight_mts_revision2")
    output = tmp_path / "preflight"
    module.main(_preflight_argv(tiny_chain, "operator_train", output))
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is True, payload["required_not_passed"]
    assert payload["failed"] == []
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    assert statuses["transport"] == "pass"
    assert statuses["validation_protocol"] == "pass"
    protocol = next(item for item in payload["checks"] if item["id"] == "validation_protocol")
    # The rows were looked up in the store's own split table, not taken on faith.
    assert protocol["evidence"]["rows_with_wrong_store_split"] == 0
    assert protocol["evidence"]["rows_verified_against_store"] == protocol["evidence"]["samples"]


def test_operator_stage_fails_when_the_transport_uses_another_tokenizer(tiny_chain, tmp_path):
    """T01 negative branch 1: same structure, different weights must not pass.

    The transport checkpoint is rebuilt here with a different tokenizer file's SHA
    recorded, which is what "the upstream was trained with another tokenizer"
    looks like on disk.
    """
    module = load_script("preflight_mts_revision2")
    other_dir = tmp_path / "other_tokenizer"
    other_dir.mkdir()
    matrix = load_cli_matrix_module()
    other_path, _representation, _motion = matrix.write_tiny_tokenizer(other_dir, name="other.pt")
    assert file_sha256(other_path) != file_sha256(tiny_chain["tokenizer"])
    output = tmp_path / "preflight"
    argv = _preflight_argv(tiny_chain, "operator_train", output)
    argv[argv.index("--tokenizer-checkpoint") + 1] = str(other_path)
    # The store was built by the *original* tokenizer, so the binding fails before
    # the transport is even reached: two independent checks agree.
    with pytest.raises(SystemExit) as caught:
        module.main(argv)
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is False
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    assert statuses["store_identity"] == "fail"
    reasons = " ".join(item["reason"] for item in payload["checks"])
    assert "same structure is not the same weights" in reasons


def test_operator_stage_is_not_ready_when_the_transport_is_missing(tiny_chain, tmp_path):
    """T01 negative branch 2: a missing dependency is blocked, and ready is false."""
    module = load_script("preflight_mts_revision2")
    output = tmp_path / "preflight"
    argv = _preflight_argv(tiny_chain, "operator_train", output)
    argv[argv.index("--transport-checkpoint") + 1] = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(SystemExit) as caught:
        module.main(argv)
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is False
    assert "transport" in payload["blocked"]
    assert "transport" in payload["required_not_passed"]


def test_transport_stage_is_ready_without_any_transport_checkpoint(tiny_chain, tmp_path):
    """T01: preparing the base run must not require the base run to exist.

    The earlier report could not distinguish "ready to train the transport" from
    "ready to train the operators"; this is that distinction, in the stage table.
    """
    module = load_script("preflight_mts_revision2")
    output = tmp_path / "preflight"
    module.main(
        [
            "--config", str(tiny_chain["transport_recipe"]),
            "--stage", "transport_train",
            "--tokenizer-checkpoint", str(tiny_chain["tokenizer"]),
            "--token-store", str(tiny_chain["store"]),
            "--validation-protocol", str(tiny_chain["transport_protocol"]),
            "--max-windows", "2",
            "--output", str(output),
        ]
    )
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is True, payload["required_not_passed"]
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    # The transport checkpoint is not required at this stage...
    assert statuses["transport"] == "not_applicable"
    assert "transport" not in payload["required_not_passed"]
    assert statuses["transport_recipe"] == "pass"
    assert statuses["validation_protocol"] == "pass"
    assert statuses["budget"] == "pass"


def test_a_protocol_file_that_lies_about_its_split_is_refused(tiny_chain, tmp_path):
    """T01: "the file exists" is not evidence; the rows are checked against the store.

    The frozen protocol is rewritten so its rows name training clips while still
    claiming split=val.  The preflight must notice, because a protocol that
    silently scores training windows is exactly the defect this stage exists for.
    """
    protocol = json.loads(tiny_chain["operator_protocol"].read_text())
    for item in protocol["items"]:
        item["target_clip"] = 0  # clip 0 is a train clip in the tiny store
    tampered = tmp_path / "tampered_protocol.json"
    tampered.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    module = load_script("preflight_mts_revision2")
    output = tmp_path / "preflight"
    argv = _preflight_argv(tiny_chain, "operator_train", output)
    argv[argv.index("--validation-protocol") + 1] = str(tampered)
    with pytest.raises(SystemExit) as caught:
        module.main(argv)
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is False
    check = next(item for item in payload["checks"] if item["id"] == "validation_protocol")
    assert check["status"] == "fail"
    assert check["evidence"]["rows_with_wrong_store_split"] == len(protocol["items"])


def test_the_evaluate_stage_needs_an_artifact_and_a_manifest(tiny_chain, tmp_path):
    """T01: the evaluate stage requires a checkpoint and a fixed manifest.

    It must not require training to be pending, and it must not accept "no
    checkpoint yet" as ready: the stage table says which artifacts this stage is
    about.
    """
    module = load_script("preflight_mts_revision2")
    # A copy of the recipe whose upstream does not exist: nothing to evaluate.
    document = yaml.safe_load(tiny_chain["operator_config"].read_text(encoding="utf-8"))
    document["transport"]["checkpoint"] = str(tmp_path / "missing_transport.pt")
    config = tmp_path / "operator_config.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / "preflight"
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--config", str(config),
                "--stage", "evaluate",
                "--tokenizer-checkpoint", str(tiny_chain["tokenizer"]),
                "--token-store", str(tiny_chain["store"]),
                "--output", str(output),
            ]
        )
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is False
    assert "evaluate_model" in payload["blocked"]
    assert "checkpoint_bindings" in payload["blocked"]
    assert "manifest_leakage" in payload["blocked"]


def test_evaluate_stage_passes_with_a_bound_transport_and_manifest(tiny_chain, tmp_path):
    """T01 positive branch for the last stage: a bound checkpoint plus a manifest."""
    module = load_script("preflight_mts_revision2")
    output = tmp_path / "preflight"
    module.main(
        [
            "--config", str(tiny_chain["operator_config"]),
            "--stage", "evaluate",
            "--tokenizer-checkpoint", str(tiny_chain["tokenizer"]),
            "--token-store", str(tiny_chain["store"]),
            "--transport-checkpoint", str(tiny_chain["transport_output"] / "best.pt"),
            "--eval-manifest", str(tiny_chain["eval_manifest"]),
            "--output", str(output),
        ]
    )
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ready"] is True, payload["required_not_passed"]
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    assert statuses["evaluate_model"] == "pass"
    assert statuses["manifest_leakage"] == "pass"
    # The stage must not have started a training run or written a model.
    assert not list(tmp_path.rglob("best.pt"))


def test_the_operator_stage_is_blocked_without_a_revision2_model(tmp_path):
    """No trained transport means the operator stage is blocked, not "passed".

    The important half is ``ready=false`` with a non-zero exit: an automation that
    keys on ``ok``/exit 0 would otherwise start training against a missing upstream.
    """
    require_real_artifacts()
    output = tmp_path / "preflight"
    module = load_script("preflight_mts_revision2")
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--config", str(REAL_CONFIG),
                "--stage", "operator_train",
                "--output", str(output),
            ]
        )
    assert caught.value.code == 1
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["stage"] == "operator_train"
    assert payload["ready"] is False
    assert "transport" in payload["blocked"]
    assert "validation_protocol" in payload["blocked"]
    assert set(payload["required_not_passed"]) >= {"transport", "validation_protocol"}
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    assert statuses["budget"] == "pass"
    budget = next(item for item in payload["checks"] if item["id"] == "budget")
    # The budget check records what a real run would need, and where it would go.
    assert budget["evidence"]["source_digest"]["count"] >= 10
    assert budget["evidence"]["output_dir"].startswith("outputs/mts_revision2/")
    assert budget["evidence"]["masking"]["coordinate_ratio"] == pytest.approx(0.80)
    # The deprecated level still maps, and says which stage it mapped to.
    with pytest.raises(SystemExit) as legacy:
        module.main(
            [
                "--config", str(REAL_CONFIG),
                "--level", "experiment",
                "--output", str(tmp_path / "legacy"),
            ]
        )
    assert legacy.value.code == 1
    legacy_payload = json.loads((tmp_path / "legacy" / "preflight.json").read_text())
    assert legacy_payload["stage"] == "operator_train"
    assert legacy_payload["level"] == "experiment"
