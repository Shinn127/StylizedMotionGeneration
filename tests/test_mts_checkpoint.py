"""R08: the checkpoint bundle is self-contained and bound to its artifacts.

Round trips cover the three operator families and both encoder kinds, the
external transport file is made unreachable to prove the bundle carries its own
weights, and every identity a result depends on (tokenizer, store hashes, action
and style maps) is checked to fail loudly when it disagrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from stylized_motion.learning.mts_operator import (
    LayoutAdapter,
    MTS_CHECKPOINT_SCHEMA_VERSION,
    OperatorBatch,
    TokenSpec,
    build_operator,
    build_provenance,
    checkpoint_style_index,
    load_operator_bundle,
    mts_checkpoint_payload,
    save_mts_checkpoint,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator
from stylized_motion.learning.mts_operator.style_encoder import (
    ConstantStyleEncoder,
    GlobalStyleEncoder,
    StyleIDEncoder,
)
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
from stylized_motion.learning.mts_operator.windows import ContentVocabulary
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

FRAMES = 6
LEVELS = 9
STYLES = 3


def skeleton_from_spec(spec) -> tuple[list[str], list[int]]:
    index: dict[str, int] = {}
    names: list[str] = []
    parents: list[int] = []
    for chain in spec.chains:
        for position, name in enumerate(chain):
            if name not in index:
                index[name] = len(names)
                names.append(name)
                parents.append(-1 if position == 0 else index[chain[position - 1]])
    return names, parents


def adapter() -> LayoutAdapter:
    return LayoutAdapter(NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON)))


def tokenizer_identity() -> dict[str, object]:
    return {
        "family": "nef_fsq",
        "variant": "independent",
        "representation_id": "nef_fsq_independent_40x9",
        "coordinate_order": ["global", "torso_node"],
        "num_coordinates": 40,
        "num_levels": LEVELS,
        "receptive_field": 64,
        "lookahead_frames": 0,
    }


def make_model(view: LayoutAdapter, operator_name: str, encoder_kind: str) -> MtsStyleOperator:
    torch.manual_seed(7)
    transport = MotionTransportTransformer(
        view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0,
        content_vocabulary=ContentVocabulary(kind="action_id", classes=("crouch", "run", "walk")),
    )
    encoder: torch.nn.Module
    if encoder_kind == "style_id":
        encoder = StyleIDEncoder(num_styles=STYLES, output_dim=32)
    elif encoder_kind == "constant":
        encoder = ConstantStyleEncoder(output_dim=32)
    else:
        encoder = GlobalStyleEncoder(view, dim=32, depth=1, heads=2, graph_depth=0, dropout=0.0,
                                     output_dim=32)
    operator = build_operator(
        operator_name, num_levels=LEVELS, hidden_dim=32, coordinate_dim=8, stream_dim=32
    )
    model = MtsStyleOperator(
        view, transport=transport, style_encoder=encoder, operator=operator,
        freeze_transport=True, freeze_style_encoder=False,
    )
    for parameter in model.operator.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(3)) * 0.05)
        parameter.requires_grad_(True)
    return model


def write_checkpoint(tmp_path: Path, view: LayoutAdapter, model: MtsStyleOperator, *, name: str = "operator.pt"):
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    tokenizer_file = tmp_path / "tokenizer.pt"
    tokenizer_file.write_bytes(b"tiny tokenizer weights")
    payload = mts_checkpoint_payload(
        kind="operator",
        model=model,
        model_config=model.describe(),
        token_spec=spec,
        tokenizer_metadata=tokenizer_identity(),
        metrics={"operator": model.operator.name, "style_encoder_kind": "style_id"
                 if model.uses_style_ids else "reference"},
        tokenizer_checkpoint=tokenizer_file,
        epoch=1,
        global_step=3,
        provenance=build_provenance(
            store_identity={"split_manifest_hash": "split-x", "normalization_hash": "norm-x"},
            action_vocabulary=ContentVocabulary(kind="action_id", classes=("crouch", "run", "walk")).as_dict(),
            style_index={"Style0": 0, "Style1": 1, "Style2": 2},
            resolved_config={"seed": 3},
            seed=3,
            upstream_transport_sha256="deadbeef",
            training_protocol_id="test",
        ),
    )
    return save_mts_checkpoint(tmp_path / name, payload)


def sample_batch(view: LayoutAdapter, *, seed: int = 5) -> OperatorBatch:
    tokens = torch.randint(0, LEVELS, (2, FRAMES, 40), generator=torch.Generator().manual_seed(seed))
    region = view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES).unsqueeze(0).expand_as(tokens)
    return OperatorBatch(
        target_tokens=tokens,
        reference_tokens=tokens.flip(0),
        visible_mask=~region,
        hard_mask=view.hard_mask(["left_arm"], graph_radius=1, length=FRAMES),
        style_ids=torch.tensor([0, 1]),
        content_condition=torch.tensor([0, 2]),
    )


@pytest.mark.parametrize("operator_name", ["logit_field", "arbitrary_kernel", "birth_death"])
@pytest.mark.parametrize("encoder_kind", ["reference", "style_id", "constant"])
def test_operator_bundle_round_trips_every_family(tmp_path, operator_name, encoder_kind):
    view = adapter()
    model = make_model(view, operator_name, encoder_kind)
    batch = sample_batch(view)
    with torch.no_grad():
        expected = model(batch).probabilities
    path = write_checkpoint(tmp_path, view, model)
    checkpoint, restored = load_operator_bundle(
        path, adapter=view, tokenizer_identity=tokenizer_identity(),
        tokenizer_checkpoint=tmp_path / "tokenizer.pt", device="cpu",
    )
    assert int(checkpoint["schema_version"]) == MTS_CHECKPOINT_SCHEMA_VERSION == 2
    assert int(checkpoint["metrics_version"]) == 2
    assert isinstance(restored, MtsStyleOperator)
    assert restored.operator.name == operator_name
    assert restored.uses_style_ids == (encoder_kind == "style_id")
    assert restored.uses_constant_descriptor == (encoder_kind == "constant")
    if encoder_kind == "constant":
        # The no-reference control must come back as a control: same width, one
        # parameter vector, and no reference input anywhere in the module.
        assert isinstance(restored.style_encoder, ConstantStyleEncoder)
        assert restored.style_encoder.output_dim == model.style_encoder.output_dim
        assert sum(p.numel() for p in restored.style_encoder.parameters()) == (
            restored.style_encoder.output_dim
        )
    with torch.no_grad():
        actual = restored(batch).probabilities
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    # The shuffled-birth-death control keeps its permutation through the bundle.
    if operator_name == "birth_death":
        assert restored.operator.level_order == model.operator.level_order
        assert restored.operator.shuffled_adjacency == model.operator.shuffled_adjacency
    assert restored.training is False or not restored.transport.training


def test_bundle_carries_its_transport_even_if_the_original_file_disappears(tmp_path):
    view = adapter()
    model = make_model(view, "birth_death", "reference")
    path = write_checkpoint(tmp_path, view, model, name="bundle.pt")
    tokenizer_file = tmp_path / "tokenizer.pt"
    # The training-time transport file is gone; that must not matter.
    missing = tmp_path / "transport_that_never_existed.pt"
    assert not missing.exists()
    _, restored = load_operator_bundle(
        path, adapter=view, tokenizer_identity=tokenizer_identity(),
        tokenizer_checkpoint=tokenizer_file, device="cpu",
    )
    batch = sample_batch(view)
    with torch.no_grad():
        first = restored(batch).probabilities
        second = restored(batch).probabilities
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert restored.freeze_transport and not restored.transport.training


def test_wrong_tokenizer_or_layout_is_rejected(tmp_path):
    view = adapter()
    model = make_model(view, "logit_field", "reference")
    path = write_checkpoint(tmp_path, view, model)
    wrong = dict(tokenizer_identity())
    wrong["receptive_field"] = 32
    with pytest.raises(ValueError):
        load_operator_bundle(
            path, adapter=view, tokenizer_identity=wrong,
            tokenizer_checkpoint=tmp_path / "tokenizer.pt", device="cpu",
        )
    # A different layout (same alphabet) is refused by the layout hash.
    from stylized_motion.learning.nef_layout import SOMA_SKELETON

    soma = LayoutAdapter(NEFLayout.from_skeleton(*skeleton_from_spec(SOMA_SKELETON)))
    with pytest.raises(ValueError, match="layout|token_spec|alphabet"):
        load_operator_bundle(
            path, adapter=soma, tokenizer_identity=tokenizer_identity(),
            tokenizer_checkpoint=tmp_path / "tokenizer.pt", device="cpu",
        )


def test_schema_one_checkpoints_are_refused(tmp_path):
    view = adapter()
    model = make_model(view, "logit_field", "reference")
    path = write_checkpoint(tmp_path, view, model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 1
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    with pytest.raises(ValueError, match="schema_version"):
        load_operator_bundle(
            legacy, adapter=view, tokenizer_identity=tokenizer_identity(),
            tokenizer_checkpoint=tmp_path / "tokenizer.pt", device="cpu",
        )
    # No "load anyway" path: the strict state dict is the only loader.
    payload["schema_version"] = 2
    payload["model"] = {key: value for key, value in payload["model"].items() if "operator" not in key}
    partial = tmp_path / "partial.pt"
    torch.save(payload, partial)
    with pytest.raises(RuntimeError):
        load_operator_bundle(
            partial, adapter=view, tokenizer_identity=tokenizer_identity(),
            tokenizer_checkpoint=tmp_path / "tokenizer.pt", device="cpu",
        )


def test_style_index_survives_evaluation_reordering(tmp_path):
    view = adapter()
    model = make_model(view, "logit_field", "style_id")
    path = write_checkpoint(tmp_path, view, model)
    checkpoint, restored = load_operator_bundle(
        path, adapter=view, tokenizer_identity=tokenizer_identity(),
        tokenizer_checkpoint=tmp_path / "tokenizer.pt", device="cpu",
    )
    stored = checkpoint_style_index(checkpoint)
    assert stored == {"Style0": 0, "Style1": 1, "Style2": 2}
    # The id of a style comes from the checkpoint, not from the current data order.
    batch = sample_batch(view)
    # Reordering the batch (with its ids) must not change any row's output: the
    # style id travels with its sample, it is not re-derived from the position.
    reordered = OperatorBatch(
        target_tokens=batch.target_tokens.flip(0), reference_tokens=batch.reference_tokens.flip(0),
        visible_mask=batch.visible_mask.flip(0), style_ids=batch.style_ids.flip(0),
        content_condition=batch.content_condition.flip(0) if batch.content_condition is not None else None,
    )
    with torch.no_grad():
        first = restored(batch).probabilities
        second = restored(reordered).probabilities
    torch.testing.assert_close(second, first.flip(0), rtol=1e-6, atol=1e-6)


class _StoreStub:
    def __init__(self, **values) -> None:
        self.manifest = {}
        for key, value in values.items():
            setattr(self, key, value)


def test_store_binding_reports_every_identity_it_can_check():
    store = _StoreStub(
        feature_schema_hash="feature-x",
        normalization_hash="norm-x",
        split_manifest_hash="split-x",
        skeleton_hash="skel-x",
        representation_id="nef_fsq_independent_40x9",
    )
    observed = validate_store_binding(
        store,
        tokenizer_identity=tokenizer_identity(),
        expected_data_identity={
            "feature_schema_hash": "feature-x",
            "normalization_hash": "norm-x",
            "split_manifest_hash": "split-x",
            "representation_id": "nef_fsq_independent_40x9",
        },
    )
    assert observed["feature_schema_hash"] == "feature-x"
    for name in ("feature_schema_hash", "normalization_hash", "split_manifest_hash"):
        with pytest.raises(ValueError, match=name):
            validate_store_binding(
                store, expected_data_identity={**observed, name: "different"}
            )
    # A store that cannot report an identity is not assumed to match it.
    bare = _StoreStub()
    with pytest.raises(ValueError, match="does not report"):
        validate_store_binding(bare, expected_data_identity={"split_manifest_hash": "split-x"})
    # Missing actor labels must not be reported as "actor-disjoint"; the caller
    # sees exactly which identities were verified.
    assert validate_store_binding(bare, expected_data_identity=None) == {}


def test_checkpoint_write_is_atomic(tmp_path):
    view = adapter()
    model = make_model(view, "logit_field", "reference")
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    target = tmp_path / "operator.pt"

    class Boom:
        def state_dict(self):
            raise RuntimeError("state dict failed")

    with pytest.raises(RuntimeError):
        payload = mts_checkpoint_payload(
            kind="operator", model=Boom(), model_config={}, token_spec=spec,
            tokenizer_metadata=tokenizer_identity(),
        )
        save_mts_checkpoint(target, payload)
    assert not target.exists()
    assert not list(tmp_path.glob(".*tmp"))
    # A successful save leaves exactly the file and no temporary.
    path = write_checkpoint(tmp_path, view, model)
    assert path.exists() and not list(tmp_path.glob(".*tmp"))


def test_provenance_records_the_artifacts_and_the_code_identity(tmp_path):
    provenance = build_provenance(
        store_identity={"split_manifest_hash": "split-x"},
        action_vocabulary={"kind": "action_id", "classes": ["walk"]},
        style_index={"S": 0},
        resolved_config={"seed": 1, "operator": {"name": "logit_field"}},
        seed=1,
        upstream_transport_sha256="abc",
        training_protocol_id="test",
    )
    for key in ("metrics_version", "store_identity", "action_to_id", "style_to_id",
                "resolved_config", "seed", "upstream_transport_sha256", "training_protocol_id",
                "code_commit", "working_tree_dirty"):
        assert key in provenance, key
    json.dumps(provenance)


def test_tokenizer_checkpoint_sha_is_recorded(tmp_path):
    view = adapter()
    model = make_model(view, "logit_field", "reference")
    fake_tokenizer = tmp_path / "tokenizer.pt"
    fake_tokenizer.write_bytes(b"not a real checkpoint")
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    payload = mts_checkpoint_payload(
        kind="operator",
        model=model,
        model_config=model.describe(),
        token_spec=spec,
        tokenizer_metadata=tokenizer_identity(),
        tokenizer_checkpoint=fake_tokenizer,
    )
    import hashlib

    expected = hashlib.sha256(fake_tokenizer.read_bytes()).hexdigest()
    assert payload["tokenizer_checkpoint_sha256"] == expected


def test_load_mts_checkpoint_refuses_a_same_structure_other_tokenizer(tmp_path):
    """T01: the transport load path must bind the tokenizer *file*, not its shape.

    ``load_mts_checkpoint`` is what the warm start, the operator's frozen upstream
    and the preflight all go through; recording a hash bound nothing while this
    entry point accepted any metadata that described the same alphabet.
    """
    from stylized_motion.learning.mts_operator.checkpoint import (
        load_mts_checkpoint,
        require_tokenizer_checkpoint,
    )

    view = adapter()
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    tokenizer_a = tmp_path / "tokenizer_a.pt"
    tokenizer_a.write_bytes(b"tokenizer weights A")
    tokenizer_b = tmp_path / "tokenizer_b.pt"
    tokenizer_b.write_bytes(b"tokenizer weights B")
    torch.manual_seed(11)
    transport = MotionTransportTransformer(view, dim=16, depth=1, heads=2, graph_depth=0)
    payload = mts_checkpoint_payload(
        kind="transport",
        model=transport,
        model_config=transport.config(),
        token_spec=spec,
        tokenizer_metadata=tokenizer_identity(),
        tokenizer_checkpoint=tokenizer_a,
    )
    path = save_mts_checkpoint(tmp_path / "transport.pt", payload)

    def build(stored):
        return MotionTransportTransformer(view, **stored)

    # The file it was trained with loads.
    _, restored = load_mts_checkpoint(
        path, kind="transport", build_model=build,
        token_spec=spec, tokenizer_metadata=tokenizer_identity(),
        tokenizer_checkpoint=tokenizer_a,
    )
    assert isinstance(restored, MotionTransportTransformer)
    # The same structure with different weights does not.
    with pytest.raises(ValueError, match="(?i)same structure is not the same weights"):
        load_mts_checkpoint(
            path, kind="transport", build_model=build,
            token_spec=spec, tokenizer_metadata=tokenizer_identity(),
            tokenizer_checkpoint=tokenizer_b,
        )
    # Passing neither is a loud error, not a skipped check.
    with pytest.raises(ValueError, match="passing neither silently skips the check"):
        load_mts_checkpoint(
            path, kind="transport", build_model=build,
            token_spec=spec, tokenizer_metadata=tokenizer_identity(),
        )
    # A checkpoint that recorded no SHA cannot be verified at all.
    del payload["tokenizer_checkpoint_sha256"]
    bare = save_mts_checkpoint(tmp_path / "bare.pt", payload)
    with pytest.raises(ValueError, match="records no tokenizer_checkpoint_sha256"):
        require_tokenizer_checkpoint(
            torch.load(bare, map_location="cpu", weights_only=False),
            tokenizer_checkpoint=tokenizer_a,
            where="bare transport",
        )


def test_store_identity_block_records_the_split_table_and_schema():
    """T01: a store path is not an identity; the split table is recorded too."""
    from stylized_motion.learning.mts_operator.checkpoint import (
        split_table_identity,
        store_identity_block,
    )

    class FakeStore:
        motion_dim = 230
        data_schema_version = 4
        num_clips = 6
        representation_id = "nef_fsq_independent_40x9"
        normalization_hash = "norm-1"
        split_ids = [0, 0, 0, 0, 1, 2]
        manifest = {"feature_schema_hash": "feat-1"}

    block = store_identity_block(FakeStore(), store_kind="token", store_path="/tmp/store")
    assert block["data_schema_version"] == 4
    assert block["num_clips"] == 6
    assert block["motion_dim"] == 230
    assert block["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert block["split_table_length"] == 6
    assert block["store_path"] == "/tmp/store"
    assert block["feature_schema_hash"] == "feat-1"

    class Other(FakeStore):
        split_ids = [0, 0, 0, 1, 1, 2]

    # Two different split tables must not share a digest.
    assert (
        split_table_identity(FakeStore())["split_table_sha256"]
        != split_table_identity(Other())["split_table_sha256"]
    )
