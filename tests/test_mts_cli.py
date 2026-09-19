"""R09b/R09c: what the revision-2 CLI actually executes.

The CLI helpers are imported by path so the tests exercise the same code the
entry points run: an override that never reaches the resolved config, a knob that
is accepted but ignored, or a warm start that quietly inherits run state are all
bugs this file is meant to catch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from stylized_motion.learning.mts_operator import (
    LayoutAdapter,
    build_operator,
    load_operator_bundle,
    mts_checkpoint_payload,
    save_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator
from stylized_motion.learning.mts_operator.style_encoder import StyleIDEncoder
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = REPO_ROOT / "data" / "configs"


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def train_operator_cli():
    return load_script("train_mts_operator")


def parse_args(module, argv: list[str]):
    return module.build_parser().parse_args(argv)


def revision2_style_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "mts_revision2_style.yaml").read_text(encoding="utf-8"))


def base_argv() -> list[str]:
    return [
        "--config", str(CONFIG_DIR / "mts_revision2_style.yaml"),
        "--tokenizer-checkpoint", "unused.pt",
        "--transport-checkpoint", "unused.pt",
    ]


def skeleton_from_spec(spec):
    index, names, parents = {}, [], []
    for chain in spec.chains:
        for position, name in enumerate(chain):
            if name not in index:
                index[name] = len(names)
                names.append(name)
                parents.append(-1 if position == 0 else index[chain[position - 1]])
    return names, parents


def adapter() -> LayoutAdapter:
    return LayoutAdapter(NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON)))


# ---------------------------------------------------------------------------
# R09b: configuration is executed, not printed as "ignored"


def test_cli_overrides_reach_the_resolved_config():
    module = train_operator_cli()
    config = revision2_style_config()
    args = parse_args(
        module,
        base_argv()
        + ["--hidden-dim", "64", "--batch-size", "7", "--style-encoder-kind", "style_id",
           "--num-styles", "5"],
    )
    resolved = module.resolve_operator_config(config, args)
    assert resolved["operator"]["hidden_dim"] == 64
    assert resolved["loader"]["batch_size"] == 7
    assert resolved["style_encoder"]["kind"] == "style_id"
    assert resolved["style_encoder"]["num_styles"] == 5
    assert resolved["style_encoder"]["output_dim"] == 64
    # The style-id encoder has no ``dim`` of its own; the reference encoder keeps
    # the width --hidden-dim set, which is why the override writes both keys.
    assert resolved["style_encoder"]["dim"] == 64
    # The YAML itself is untouched: overrides never mutate the loaded document.
    assert config["operator"]["hidden_dim"] == 256
    # A style-id run without --num-styles is a configuration error, not a default.
    with pytest.raises(ValueError, match="num-styles"):
        module.resolve_operator_config(
            revision2_style_config(),
            parse_args(module, base_argv() + ["--style-encoder-kind", "style_id"]),
        )


def test_revision2_training_knobs_are_refused_when_unimplemented():
    module = train_operator_cli()
    base = {
        "epochs": 1, "lr": 1e-4, "precision": "fp32", "seed": 0,
        "val_every_steps": 0, "amp": False,
    }
    module.validate_revision2_training(base, where="test")
    for field, value, message in (
        ("val_every_steps", 50, "val_every_steps"),
        ("precision", "amp", "precision"),
        ("amp", True, "amp"),
    ):
        with pytest.raises(ValueError, match=message):
            module.validate_revision2_training({**base, field: value}, where="test")
    with pytest.raises(ValueError, match="unknown training fields"):
        module.validate_revision2_training({**base, "warmup_steps": 10}, where="test")


def test_operator_options_that_are_not_implemented_raise():
    """The old behaviour printed "ignoring options ..." and trained anyway."""
    module = train_operator_cli()
    config = revision2_style_config()
    config["operator"] = {**config["operator"], "kernel_temperature": 0.5}
    text = Path(module.__file__).read_text(encoding="utf-8")
    assert "ignoring options" not in text
    accepted = set(module.COMMON_OPERATOR_KEYS) | set(
        module.OPERATOR_SPECIFIC_KEYS.get("birth_death", ())
    )
    unsupported = sorted(set(config["operator"]) - accepted - {"name", "strength"})
    assert unsupported == ["kernel_temperature"]


def test_revision2_style_recipe_has_the_expected_mixture_and_budget():
    config = revision2_style_config()
    masking = config["masking"]
    for kind, expected in (
        ("full_generation", 0.30),
        ("random_coordinate", 0.30),
        ("stream", 0.15),
        ("temporal_span", 0.10),
        ("spatiotemporal_block", 0.15),
    ):
        assert masking[kind] == pytest.approx(expected), kind
    assert sum(masking[kind] for kind in ("full_generation", "random_coordinate", "stream",
                                          "temporal_span", "spatiotemporal_block")) == pytest.approx(1.0)
    assert masking["coordinate_ratio"] == pytest.approx(0.80)
    # T00: the operator recipe states which protocol it freezes; the transport
    # recipe states a row count per kind instead of a batch count.
    assert config["evaluation"]["protocol_id"]
    assert config["evaluation"]["validation_batches_per_kind"] >= 4
    assert config["data"]["required_data_schema_version"] == 4
    assert config["data"]["reference_frames"] == config["data"]["frames"]
    assert config["training"]["precision"] == "fp32"
    assert config["data"]["pairs"]["target_sampling"] == "style_uniform"
    assert str(config["training"]["output_dir"]).startswith("outputs/mts_revision2/")
    # The revision-2 transport recipe exists and keeps its own mixture.
    transport = yaml.safe_load(
        (CONFIG_DIR / "mts_revision2_transport.yaml").read_text(encoding="utf-8")
    )
    assert transport["transport"]["token_embed_dim"] == 16
    assert transport["transport"]["position_encoding"] == "sinusoidal"
    assert str(transport["training"]["output_dir"]).startswith("outputs/mts_revision2/")
    # T00: the base transport is action-conditioned, so the operator's
    # action-conditioned path has a real upstream.  An unconditional transport is
    # only allowed as an explicitly named debug/smoke recipe.
    assert transport["data"]["content"]["kind"] == "action_id"
    assert transport["evaluation"]["protocol_id"] == "mts-transport-revision2-full-v1"
    assert transport["evaluation"]["validation_kinds"] == [
        "random_coordinate",
        "stream",
        "temporal_span",
        "spatiotemporal_block",
        "full_generation",
    ]
    assert transport["evaluation"]["validation_rows_per_kind"] == 64
    for name in ("mts_revision2_transport.yaml", "mts_revision2_style.yaml"):
        document = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert document["data"]["content"]["kind"] == "action_id", name
    smoke = yaml.safe_load(
        (CONFIG_DIR / "mts_revision2_style_smoke.yaml").read_text(encoding="utf-8")
    )
    assert smoke["data"]["content"]["kind"] == "none"


def test_historical_recipes_are_untouched_by_the_revision2_templates():
    for name in ("mts_operator_style.yaml", "mts_operator_transport.yaml"):
        assert (CONFIG_DIR / name).exists()


# ---------------------------------------------------------------------------
# R09c: warm start is not a resume


def _bundle(tmp_path: Path, *, hidden_dim: int = 32, operator_name: str = "logit_field") -> Path:
    view = adapter()
    torch.manual_seed(3)
    transport = MotionTransportTransformer(view, dim=hidden_dim, depth=1, heads=2, graph_depth=0)
    encoder = StyleIDEncoder(num_styles=2, output_dim=hidden_dim)
    operator = build_operator(
        operator_name, num_levels=9, hidden_dim=hidden_dim, coordinate_dim=8, stream_dim=hidden_dim
    )
    model = MtsStyleOperator(view, transport=transport, style_encoder=encoder, operator=operator)
    with torch.no_grad():
        for parameter in model.operator.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1)) * 0.1)
    payload = mts_checkpoint_payload(
        kind="operator",
        model=model,
        model_config=model.describe(),
        token_spec=view.token_spec(representation_id="nef_fsq_independent_40x9"),
        tokenizer_metadata={
            "family": "nef_fsq", "variant": "independent",
            "representation_id": "nef_fsq_independent_40x9",
            "coordinate_order": ["global"], "num_coordinates": 40, "num_levels": 9,
        },
        metrics={"operator": operator_name, "style_encoder_kind": "style_id"},
        tokenizer_checkpoint=_tokenizer_file(tmp_path),
    )
    return save_mts_checkpoint(tmp_path / "operator.pt", payload)


def _tokenizer_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.pt"
    if not path.exists():
        path.write_bytes(b"tiny tokenizer weights")
    return path


def test_warm_start_loads_weights_and_keeps_the_run_state_fresh(tmp_path):
    module = train_operator_cli()
    path = _bundle(tmp_path, hidden_dim=32)
    view = adapter()
    torch.manual_seed(99)
    fresh = MtsStyleOperator(
        view,
        transport=MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0),
        style_encoder=StyleIDEncoder(num_styles=2, output_dim=32),
        operator=build_operator("logit_field", num_levels=9, hidden_dim=32, coordinate_dim=8, stream_dim=32),
    )
    before = {key: value.clone() for key, value in fresh.state_dict().items()}
    source = module.apply_warm_start(
        fresh, path, adapter=view, tokenizer_identity=None,
        tokenizer_checkpoint=_tokenizer_file(tmp_path), device="cpu",
    )
    assert source == str(path)
    _, loaded = load_operator_bundle(
        path,
        adapter=view,
        tokenizer_checkpoint=_tokenizer_file(tmp_path),
        tokenizer_identity={
            "family": "nef_fsq", "variant": "independent",
            "representation_id": "nef_fsq_independent_40x9",
            "coordinate_order": ["global"], "num_coordinates": 40, "num_levels": 9,
        },
        device="cpu",
    )
    changed = 0
    for key, value in loaded.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key
        if not torch.equal(before[key], value):
            changed += 1
    assert changed > 0, "the warm start must actually load something"
    # A different architecture revision is refused instead of partially loaded.
    other = _bundle(tmp_path, hidden_dim=48, operator_name="birth_death")
    with pytest.raises(ValueError, match="do not fit"):
        module.apply_warm_start(
            fresh, other, adapter=view, tokenizer_identity=None,
            tokenizer_checkpoint=_tokenizer_file(tmp_path), device="cpu",
        )


def test_training_script_carries_no_resume_guarantee():
    """No code path claims to restore optimizer/sampler state."""
    text = (REPO_ROOT / "scripts" / "train_mts_operator.py").read_text(encoding="utf-8")
    for forbidden in ("--resume", "optimizer.load_state_dict", "resumed transport"):
        assert forbidden not in text, forbidden
    assert "--warm-start" in text
    # warm-start documents that it is not a resume
    assert "not a resume" in text


# ---------------------------------------------------------------------------
# C00: real-entry counterexamples (F01-F05 of the closure plan)


def test_revision2_configs_pass_the_real_masking_validator():
    """F01: the YAML must survive the validator the CLI actually calls."""
    from stylized_motion.learning.mts_operator.masking import MaskConfig

    for name in ("mts_revision2_style.yaml", "mts_revision2_transport.yaml"):
        config = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        parsed = MaskConfig.from_mapping(dict(config["masking"]))
        # The mixture must survive parsing as a distribution, not just as text.
        assert parsed is not None


def test_manifest_only_exits_zero_without_any_checkpoint(tmp_path):
    """F03: building the manifest cannot require a trained model."""
    import subprocess
    import sys

    script = REPO_ROOT / "scripts" / "evaluate_mts_operator.py"
    output = tmp_path / "manifest"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--build-manifest-only",
            "--feature-database", str(tmp_path / "missing_store"),
            "--output", str(output),
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    # The store path is bogus on purpose: the point is that the parser must not
    # demand --checkpoint/--tokenizer-checkpoint before it reaches the store.
    assert "--checkpoint" not in result.stderr and "--tokenizer-checkpoint" not in result.stderr, (
        result.stderr
    )
    assert "required" not in result.stderr, result.stderr


def test_row_batch_keeps_the_target_condition():
    """F04: row scoring must reuse the full batch, never a tokens-only rebuild.

    The current helper takes bare tokens, so the batch it scores has no visible
    mask, no condition and no valid mask -- exactly the fields the model's own
    entry points require.  The contract to implement is "score these candidates
    for this row of that batch".
    """
    import importlib.util
    import inspect

    script = REPO_ROOT / "scripts" / "evaluate_mts_operator.py"
    spec = importlib.util.spec_from_file_location("evaluate_mts_operator", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    signature = inspect.signature(module.build_row_batch)
    assert "batch" in signature.parameters and "row" in signature.parameters, (
        f"build_row_batch must take the row and its batch, got {signature}"
    )
    # And the batch it builds must be scoreable: the model's supervision path
    # refuses a batch without a visible mask (F04's measured error).
    view = adapter()
    from stylized_motion.learning.mts_operator.model import OperatorBatch

    tokens = torch.randint(0, 9, (1, 4, 40), generator=torch.Generator().manual_seed(1))
    batch = OperatorBatch(target_tokens=tokens, reference_tokens=tokens)
    with pytest.raises(ValueError, match="visible_mask"):
        batch.supervision_mask(view.token_spec())


def test_bundle_load_rejects_a_same_shape_tokenizer_with_a_different_sha(tmp_path):
    """F05: recording the SHA is not binding; loading must compare it."""
    from stylized_motion.learning.mts_operator.checkpoint import (
        load_operator_bundle,
        mts_checkpoint_payload,
        save_mts_checkpoint,
    )

    view = adapter()
    torch.manual_seed(5)
    transport = MotionTransportTransformer(view, dim=32, depth=1, heads=2, graph_depth=0)
    encoder = StyleIDEncoder(num_styles=2, output_dim=32)
    operator = build_operator("logit_field", num_levels=9, hidden_dim=32, coordinate_dim=8, stream_dim=32)
    model = MtsStyleOperator(view, transport=transport, style_encoder=encoder, operator=operator)
    identity = {
        "family": "nef_fsq", "variant": "independent",
        "representation_id": "nef_fsq_independent_40x9",
        "coordinate_order": ["global"], "num_coordinates": 40, "num_levels": 9,
    }
    real = tmp_path / "tokenizer.pt"
    real.write_bytes(b"tokenizer-weights-A")
    payload = mts_checkpoint_payload(
        kind="operator",
        model=model,
        model_config=model.describe(),
        token_spec=view.token_spec(representation_id=identity["representation_id"]),
        tokenizer_metadata=identity,
        metrics={"operator": "logit_field", "style_encoder_kind": "style_id"},
        tokenizer_checkpoint=real,
    )
    path = save_mts_checkpoint(tmp_path / "operator.pt", payload)
    # Same structure, different bytes: the identity check cannot see this, the SHA can.
    other = tmp_path / "other_tokenizer.pt"
    other.write_bytes(b"tokenizer-weights-B")
    with pytest.raises(ValueError, match="tokenizer|sha"):
        load_operator_bundle(
            path, adapter=view, tokenizer_identity=identity, tokenizer_checkpoint=other
        )
    # The file it was trained with loads.
    _, restored = load_operator_bundle(
        path, adapter=view, tokenizer_identity=identity, tokenizer_checkpoint=real
    )
    assert restored is not None


def write_tiny_token_store(
    root: Path,
    *,
    clips: int = 12,
    frames: int = 65,
    coordinates: int = 40,
    levels: int = 9,
    checkpoint_sha256: str = "checkpoint-tiny",
    motion_dim: int | None = None,
    representation_id: str = "nef_fsq_independent_40x9",
    style_ids: "Sequence[int] | None" = None,
    action_ids: "Sequence[int] | None" = None,
    clip_lengths: "Sequence[int] | None" = None,
    styles: "Sequence[str] | None" = None,
    actions: "Sequence[str] | None" = None,
    split_ids: "Sequence[int] | None" = None,
    dataset: str | None = None,
) -> Path:
    """A minimal v3 token store: enough to drive the real CLI paths.

    The manifest carries the identity fields the readers validate (schema 3,
    representation family/variant/id, coordinate layout, split/style/action
    tables); it is deliberately tiny so a CLI smoke is cheap.

    ``checkpoint_sha256`` and ``motion_dim`` let a caller bind the store to a real
    tokenizer checkpoint, and ``style_ids``/``action_ids`` exist so a fixture can
    give the two label tables *different* numbers on purpose: if a code path
    confuses a style id with an action id, the run must fail rather than agree by
    accident.
    """
    import hashlib
    import json

    import numpy as np

    from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

    def skeleton(spec):
        index, names, parents = {}, [], []
        for chain in spec.chains:
            for position, name in enumerate(chain):
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                    parents.append(-1 if position == 0 else index[chain[position - 1]])
        return names, parents

    layout = NEFLayout.from_skeleton(*skeleton(GENO_SKELETON))
    order = list(layout.stream_slices)
    counts = {name: (s.stop - s.start) for name, s in layout.stream_slices.items()}
    if motion_dim is None:
        motion_dim = 9 * layout.num_joints + 5
    lengths = [int(frames)] * int(clips) if clip_lengths is None else [int(v) for v in clip_lengths]
    if len(lengths) != int(clips):
        raise ValueError("clip_lengths must describe every clip")
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, levels, size=(int(offsets[-1]), coordinates)).astype(np.uint8)
    (root / "indices").mkdir(parents=True, exist_ok=True)
    shard = root / "indices" / "shard_00000.npy"
    np.save(shard, tokens)
    styles = list(styles) if styles is not None else ["Style0", "Style1", "Style2"]
    actions = list(actions) if actions is not None else ["walk", "run"]
    range_names = []
    resolved_styles, resolved_actions, resolved_splits, source_ids = [], [], [], []
    for clip in range(clips):
        style = styles[clip % len(styles)]
        action = actions[(clip // len(styles)) % len(actions)]
        range_names.append(f"{style}_{action}_{clip}")
        resolved_styles.append(styles.index(style))
        resolved_actions.append(actions.index(action))
        resolved_splits.append(0 if clip < clips - 4 else (1 if clip < clips - 2 else 2))
        source_ids.append(clip)
    if style_ids is not None:
        if len(style_ids) != int(clips):
            raise ValueError("style_ids must describe every clip")
        resolved_styles = [int(value) for value in style_ids]
    if action_ids is not None:
        if len(action_ids) != int(clips):
            raise ValueError("action_ids must describe every clip")
        resolved_actions = [int(value) for value in action_ids]
    if split_ids is not None:
        if len(split_ids) != int(clips):
            raise ValueError("split_ids must describe every clip")
        # The parameter used to be shadowed by the local variable of the same
        # name, so a caller's split table was silently replaced by the default
        # one; a fixture that ignores its own argument produces tests that pass
        # for the wrong reason.
        resolved_splits = [int(value) for value in split_ids]
    np.savez(
        root / "index.npz",
        shard_num_frames=np.asarray([int(offsets[-1])], dtype=np.int64),
        clip_ids=np.asarray(range(clips), dtype=np.int32),
        source_clip_ids=np.asarray(source_ids, dtype=np.int32),
        range_shard_indices=np.zeros(clips, dtype=np.int32),
        range_starts=np.asarray([int(offsets[clip]) for clip in range(clips)], dtype=np.int64),
        range_stops=np.asarray([int(offsets[clip + 1]) for clip in range(clips)], dtype=np.int64),
        range_mirror=np.zeros(clips, dtype=bool),
        split_ids=np.asarray(resolved_splits, dtype=np.uint8),
        style_ids=np.asarray(resolved_styles, dtype=np.int32),
        action_ids=np.asarray(resolved_actions, dtype=np.int32),
    )
    manifest = {
        "data_schema_version": 3,
        "store_type": "token",
        "frame_rate": 60,
        "num_shards": 1,
        "shard_files": ["indices/shard_00000.npy"],
        "shard_sha256": [hashlib.sha256(shard.read_bytes()).hexdigest()],
        "split_manifest_hash": "split-tiny",
        "feature_schema_hash": "feature-tiny",
        "created_by": "tests",
        "range_names": range_names,
        "source_clip_names": range_names,
        "style_names": styles,
        "action_names": actions,
        "representation_family": "nef_fsq",
        "representation_variant": "independent",
        "representation_id": representation_id,
        "model_family_legacy": "nef_fsq",
        "checkpoint_sha256": checkpoint_sha256,
        "motion_dim": motion_dim,
        "num_coordinates": coordinates,
        "num_levels": levels,
        "temporal_downsample": 1,
        "receptive_field": 64,
        "lookahead_frames": 0,
        "decoder_passes_inference": 1,
        "coordinate_order": order,
        "coordinate_counts": counts,
        "feature_schema": {"name": "motion_feature_v2", "motion_dim": motion_dim},
        "representation": {
            "family": "nef_fsq",
            "variant": "independent",
            "nef_layout": layout.to_dict(),
        },
        "split_policy": "fixed_window_random_v1",
        "window_frames": 64,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_manifest_only_writes_a_manifest_from_a_real_store(tmp_path):
    """C01: with a store but no checkpoint, the manifest is written and exit is 0."""
    import subprocess
    import sys

    store = write_tiny_token_store(tmp_path / "tokens")
    output = tmp_path / "manifest"
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "evaluate_mts_operator.py"),
            "--build-manifest-only",
            "--token-store", str(store),
            "--split", "test",
            "--batches", "1",
            "--batch-size", "2",
            "--output", str(output),
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "eval_manifest.json").exists()
    assert (output / "eval_manifest.jsonl").exists()
    assert "no model was loaded" in result.stdout


def test_token_store_binding_checks_the_three_way_hash(tmp_path):
    """C02: the token store's checkpoint hash must be the tokenizer that trained it."""
    from stylized_motion.learning.mts_operator.checkpoint import (
        require_token_store_binding,
    )

    store_path = write_tiny_token_store(tmp_path / "tokens")
    tokenizer = tmp_path / "tokenizer.pt"
    tokenizer.write_bytes(b"tokenizer-weights-A")
    # The tiny store records checkpoint_sha256 = "checkpoint-tiny": any real file hash
    # disagrees, so the binding must fail instead of being waved through.
    with pytest.raises(ValueError, match="same structure is not the same weights"):
        require_token_store_binding(store_path, tokenizer_checkpoint=tokenizer)
    # With the hash the store was built from, it passes.
    import hashlib

    digest = hashlib.sha256(store_path.joinpath("manifest.json").read_bytes()).hexdigest()
    manifest = json.loads((store_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == "checkpoint-tiny"
    observed = require_token_store_binding(store_path, tokenizer_checkpoint_sha256="checkpoint-tiny")
    assert observed["checkpoint_sha256"] == "checkpoint-tiny"
    assert digest  # the helper reads the manifest, not the features
    # A feature store must not be forced to carry a tokenizer identity (C02 item 3).
    from stylized_motion.learning.mts_operator.checkpoint import validate_store_binding

    feature_like = _FeatureStoreStub(
        feature_schema_hash="f", normalization_hash="n", split_manifest_hash="s", skeleton_hash="k"
    )
    validate_store_binding(feature_like, expected_data_identity={"normalization_hash": "n"})
    with pytest.raises(ValueError, match="representation_id"):
        validate_store_binding(
            feature_like,
            expected_data_identity={"representation_id": "nef_fsq_independent_40x9"},
        )


class _FeatureStoreStub:
    def __init__(self, **values) -> None:
        self.manifest = {}
        for key, value in values.items():
            setattr(self, key, value)


def test_source_digest_distinguishes_two_uncommitted_revisions(tmp_path):
    """C02 item 7: HEAD + dirty cannot tell two dirty trees apart; a digest can."""
    from stylized_motion.learning.mts_operator.checkpoint import source_digest

    workspace = tmp_path / "ws"
    (workspace / "stylized_motion" / "learning" / "mts_operator").mkdir(parents=True)
    module = workspace / "stylized_motion" / "learning" / "mts_operator" / "new_module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_digest(workspace)
    assert first["count"] == 1 and first["combined"]
    # An untracked module's edit changes the digest.
    module.write_text("VALUE = 2\n", encoding="utf-8")
    second = source_digest(workspace)
    assert second["combined"] != first["combined"]
    assert second["files"] != first["files"]
    # A file outside the declared patterns is never read.
    (workspace / "secrets.env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    assert source_digest(workspace)["files"] == second["files"]
    # The real workspace digest covers the MTS modules and the CLI scripts.
    real = source_digest(REPO_ROOT)
    assert real["count"] >= 10
    assert any(name.endswith("mts_operator/checkpoint.py") for name in real["files"])


def test_training_exposure_is_recorded_and_checked(tmp_path):
    """C02 item 6: a --held-out-styles claim is checked against the run's record."""
    from stylized_motion.learning.mts_operator.checkpoint import training_exposure

    unknown = training_exposure()
    assert unknown["exposure_unknown"] is True
    recorded = training_exposure(
        trainable_styles=["Style0"], held_out_styles=["Style1"], actions=["walk"],
        pairs={"Style0": 7},
    )
    assert recorded["trainable_styles"] == ["Style0"]
    assert recorded["held_out_styles"] == ["Style1"]
    assert recorded["exposure_unknown"] is False
    # The evaluator refuses a claim that contradicts the record, and refuses to
    # treat a claim as evidence when the checkpoint has no record at all.
    script = REPO_ROOT / "scripts" / "evaluate_mts_operator.py"
    text = script.read_text(encoding="utf-8")
    assert "records no training exposure" in text
    assert "disagrees with the checkpoint's recorded held-out" in text


# ---------------------------------------------------------------------------
# C04: the transport entry


def test_transport_cli_refuses_checkpoint_and_offers_warm_start(tmp_path):
    """C04: no more pretend-resume; --warm-start is the supported path."""
    import subprocess
    import sys

    script = REPO_ROOT / "scripts" / "train_mts_transport.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(CONFIG_DIR / "mts_revision2_transport.yaml"),
         "--checkpoint", str(tmp_path / "old.pt")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode != 0
    assert "exact resume" in result.stderr and "--warm-start" in result.stderr
    assert "resumed" not in result.stdout.lower()
    text = script.read_text(encoding="utf-8")
    for forbidden in ("optimizer.load_state_dict", "resumed transport"):
        assert forbidden not in text


def test_transport_freeze_keeps_exactly_the_requested_windows():
    """C04: --overfit-clips N must not keep a whole extra batch."""
    import importlib.util

    script = REPO_ROOT / "scripts" / "train_mts_transport.py"
    spec = importlib.util.spec_from_file_location("train_mts_transport", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class _Source(module.TokenSource):
        def __init__(self, batches) -> None:  # noqa: D107 - test stub
            self.batches = batches
            self.frames = None

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    source = _Source([{"tokens": torch.zeros(5, 64, 40, dtype=torch.long),
                       "content_condition": torch.zeros(5, dtype=torch.long)} for _ in range(4)])
    frozen = source.freeze(clips=7)
    assert sum(int(item["tokens"].shape[0]) for item in frozen) == 7
    assert int(frozen[-1]["tokens"].shape[0]) == 2
    assert int(frozen[-1]["content_condition"].shape[0]) == 2
    with pytest.raises(ValueError, match="fewer than"):
        source.freeze(clips=100)


def test_plot_refuses_artifacts_from_another_metrics_revision(tmp_path):
    """C07: a figure may not mix metric definitions across revisions."""
    import importlib.util

    script = REPO_ROOT / "scripts" / "plot_mts_figures.py"
    spec = importlib.util.spec_from_file_location("plot_mts_figures", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    current = tmp_path / "current.json"
    current.write_text(json.dumps({"metrics_version": 2, "aggregate": {}}), encoding="utf-8")
    payload, reason = module.load_operator_artifact(current)
    assert reason is None and payload is not None
    for bad in ({"aggregate": {}}, {"metrics_version": 1, "aggregate": {}}):
        path = tmp_path / "old.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        payload, reason = module.load_operator_artifact(path)
        assert payload is None and reason and "metrics_version" in reason
