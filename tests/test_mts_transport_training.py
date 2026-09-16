"""MTS transport training: masked objective, small-batch overfit, generation smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from stylized_motion.learning.mts_operator import LayoutAdapter
from stylized_motion.learning.mts_operator.checkpoint import (
    checkpoint_token_spec,
    load_mts_checkpoint,
    mts_checkpoint_payload,
    save_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.contract import masked_cross_entropy
from stylized_motion.learning.mts_operator.masking import MaskGenerator
from stylized_motion.learning.mts_operator.training import TrainerConfig, TransportTrainer
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer
from stylized_motion.learning.nef_layout import GENO_SKELETON, NEFLayout

CONFIG_PATH = Path(__file__).parents[1] / "data" / "configs" / "mts_operator_transport.yaml"


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
            elif position > 0:
                assert parents[index[name]] == index[chain[position - 1]]
    return names, parents


def adapter() -> LayoutAdapter:
    return LayoutAdapter(NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON)))


def model(view: LayoutAdapter, **kwargs) -> MotionTransportTransformer:
    options = dict(dim=32, depth=1, heads=2, graph_depth=1, dropout=0.0)
    options.update(kwargs)
    return MotionTransportTransformer(view, **options)


def fixed_batch(batch: int = 4, frames: int = 16, *, seed: int = 3) -> torch.Tensor:
    return torch.randint(0, 9, (batch, frames, 40), generator=torch.Generator().manual_seed(seed))


def test_transport_config_declares_dim_depth_heads_and_mask_mixture():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["transport"]["dim"] == 256
    assert config["transport"]["depth"] == 8
    assert config["transport"]["heads"] == 8
    assert config["transport"]["graph_mode"] == "local_relational"
    assert config["tokenizer"]["freeze"] is True
    mixture = {kind: config["masking"][kind] for kind in (
        "random_coordinate", "stream", "temporal_span", "spatiotemporal_block", "full_generation"
    )}
    assert sum(mixture.values()) == pytest.approx(1.0)
    assert mixture["full_generation"] >= 0.15
    trainer = TrainerConfig.from_mapping(config["training"])
    assert trainer.epochs == 100 and trainer.lr == pytest.approx(2e-4)
    # The masking block must be accepted by the generator as written.
    generator = MaskGenerator(config["masking"])
    assert generator.config.block_frames == 16
    assert config["training"]["steps_per_epoch"] is None
    with pytest.raises(ValueError, match="Unknown training options"):
        TrainerConfig.from_mapping({"learning_rate": 1e-3})
    assert TrainerConfig.from_mapping({"epochs": 3, "steps_per_epoch": 5}).steps_per_epoch == 5
    with pytest.raises(ValueError, match="must be positive"):
        TrainerConfig.from_mapping({"steps_per_epoch": 0})


def test_masked_objective_trains_a_small_batch_until_it_overfits():
    view = adapter()
    net = model(view)
    # The hardest mask: nothing is visible, so the model must fit the batch from
    # the token identity alone.  A frozen batch that keeps improving is the
    # Phase 2 "small-batch overfit" check.
    trainer = TransportTrainer(
        net,
        adapter=view,
        mask_generator=MaskGenerator({"full_generation": 1.0}),
        device="cpu",
        config=TrainerConfig(epochs=1, lr=2e-2, log_every_steps=0),
    )
    batch = fixed_batch(batch=2, frames=4)
    first = trainer.train_step(batch)
    for _ in range(119):
        last = trainer.train_step(batch)
    assert last.loss < first.loss * 0.85
    assert last.accuracy > first.accuracy * 1.5
    assert last.supervised_tokens == batch.numel()
    assert last.kind == "full_generation"
    assert trainer.global_step == 120


def test_fit_reports_history_and_stops_at_max_steps():
    view = adapter()
    net = model(view)
    trainer = TransportTrainer(
        net,
        adapter=view,
        mask_generator=MaskGenerator({"random_coordinate": 1.0}),
        device="cpu",
        config=TrainerConfig(epochs=5, lr=1e-3, max_steps=4, log_every_steps=0),
    )
    batch = fixed_batch()

    def train_batches(epoch):
        for _ in range(3):
            yield batch

    history = trainer.fit(train_batches, val_batches=train_batches, log=None)
    assert trainer.global_step == 4
    assert len(history["history"]) == 2  # the third epoch never starts
    assert "val_loss" in history["history"][0]
    for name in ("loss", "accuracy", "steps", "seconds"):
        assert name in history["history"][0]


def test_evaluate_averages_over_masks_and_handles_empty_input():
    view = adapter()
    trainer = TransportTrainer(
        model(view), adapter=view, mask_generator=MaskGenerator({"stream": 1.0}), device="cpu"
    )
    batch = fixed_batch()
    metrics = trainer.evaluate([batch, batch])
    assert metrics["supervised_tokens"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    empty = trainer.evaluate([])
    assert empty["supervised_tokens"] == 0 and empty["loss"] != empty["loss"]  # NaN


def test_full_mask_generation_produces_legal_tokens_and_respects_support():
    view = adapter()
    torch.manual_seed(7)
    net = model(view, graph_depth=1)
    support = view.hard_mask(["left_arm"], graph_radius=1, length=12)
    generated = net.generate(
        frames=12,
        batch=2,
        steps=4,
        support=support,
        generator=torch.Generator().manual_seed(11),
    )
    assert generated.shape == (2, 12, 40)
    assert int(generated.min()) >= 0 and int(generated.max()) <= 8
    outside = (~support).unsqueeze(0).expand_as(generated)
    assert bool((generated[outside] == 0).all())  # untouched outside the support

    # locked_edit: given tokens outside the support survive generation exactly.
    locked = torch.full((1, 12, 40), 5, dtype=torch.long)
    edited = net.generate(
        frames=12,
        batch=1,
        visible_tokens=locked,
        support=support,
        steps=3,
        generator=torch.Generator().manual_seed(13),
    )
    locked_outside = (~support).unsqueeze(0).expand_as(edited)
    locked_inside = support.unsqueeze(0).expand_as(edited)
    assert bool((edited[locked_outside] == 5).all())
    assert bool((edited[locked_inside] != 5).any())
    with pytest.raises(ValueError, match="support must be"):
        net.generate(frames=12, batch=1, support=torch.ones(8, 40, dtype=torch.bool))
    with pytest.raises(ValueError, match="steps must be positive"):
        net.generate(frames=4, batch=1, steps=0)
    with pytest.raises(ValueError, match="temperature"):
        net.generate(frames=4, batch=1, temperature=0.0)
    with pytest.raises(ValueError, match="visible_tokens"):
        net.generate(frames=4, batch=2, visible_tokens=torch.zeros(2, 3, 40, dtype=torch.long))


def test_generation_uses_the_sampler_hook_and_beats_chance_after_training():
    view = adapter()
    net = model(view)
    trainer = TransportTrainer(
        net,
        adapter=view,
        mask_generator=MaskGenerator({"full_generation": 1.0}),
        device="cpu",
        config=TrainerConfig(epochs=1, lr=2e-2, log_every_steps=0),
    )
    batch = fixed_batch(batch=2, frames=4)
    for _ in range(60):
        metrics = trainer.train_step(batch)
    assert metrics.accuracy > 1.0 / 9.0  # better than guessing a level

    # A deterministic sampler makes one step of generation exactly the model's
    # own prediction; further steps re-condition on the tokens just filled in.
    def argmax(probs):
        return probs.argmax(dim=-1)

    drawn = net.generate(frames=4, batch=2, steps=1, support=None, sampler=argmax)
    with torch.no_grad():
        expected = net(batch, torch.zeros_like(batch, dtype=torch.bool)).logits.argmax(dim=-1)
    torch.testing.assert_close(drawn[:, 0], expected[:, 0])
    assert int(drawn.min()) >= 0 and int(drawn.max()) <= 8


def test_checkpoint_round_trip_binds_the_tokenizer_fingerprint(tmp_path: Path):
    view = adapter()
    spec = view.token_spec(representation_id="nef_fsq_independent_40x9")
    tokenizer_metadata = {
        "family": "nef_fsq",
        "variant": "independent",
        "representation_id": "nef_fsq_independent_40x9",
        "num_coordinates": 40,
        "num_levels": 9,
        "nef_layout_hash": view.layout_hash,
        "feature_schema": {"motion_dim": 230},
    }
    net = model(view, content_classes=3)
    payload = mts_checkpoint_payload(
        kind="transport",
        model=net,
        model_config=net.config(),
        token_spec=spec,
        tokenizer_metadata=tokenizer_metadata,
        metrics={"val_loss": 1.0},
        epoch=2,
        global_step=9,
    )
    path = save_mts_checkpoint(tmp_path / "transport.pt", payload)

    def build(stored):
        return MotionTransportTransformer(view, **stored)

    checkpoint, restored = load_mts_checkpoint(
        path, kind="transport", build_model=build, token_spec=spec,
        tokenizer_metadata=tokenizer_metadata,
    )
    assert checkpoint_token_spec(checkpoint) == spec
    assert checkpoint["metrics"]["val_loss"] == 1.0
    tokens = fixed_batch()
    visible = torch.ones_like(tokens, dtype=torch.bool)
    net.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            restored(tokens, visible).logits, net(tokens, visible).logits, rtol=0.0, atol=0.0
        )
    with pytest.raises(ValueError, match="Expected a 'operator' checkpoint"):
        load_mts_checkpoint(path, kind="operator", build_model=build)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_mts_checkpoint(
            path,
            kind="transport",
            build_model=build,
            tokenizer_metadata={**tokenizer_metadata, "nef_layout_hash": "0" * 64},
        )
    with pytest.raises(ValueError, match="token_spec"):
        load_mts_checkpoint(
            path, kind="transport", build_model=build,
            token_spec=view.token_spec(representation_id="other"),
        )


def test_checkpoint_helpers_are_json_safe_and_reject_a_bad_schema(tmp_path: Path):
    view = adapter()
    spec = view.token_spec()
    payload = mts_checkpoint_payload(
        kind="transport",
        model=model(view),
        model_config={},
        token_spec=spec,
        tokenizer_metadata=None,
    )
    json.dumps(payload["metadata"], default=str)
    path = save_mts_checkpoint(tmp_path / "broken.pt", {**payload, "schema_version": 99})
    with pytest.raises(ValueError, match="schema_version"):
        load_mts_checkpoint(
            path, kind="transport", build_model=lambda stored: model(view)
        )
    with pytest.raises(ValueError, match="kind must be one of"):
        mts_checkpoint_payload(
            kind="style", model=model(view), model_config={}, token_spec=spec,
            tokenizer_metadata=None,
        )


def test_masked_cross_entropy_only_scores_supervised_tokens():
    view = adapter()
    net = model(view)
    tokens = fixed_batch()
    masks = MaskGenerator({"random_coordinate": 1.0})
    batch = masks.sample_kind("random_coordinate", 4, 16, adapter=view, generator=torch.Generator().manual_seed(5))
    output = net(tokens, batch.visible_mask)
    supervised = masked_cross_entropy(output.logits, tokens, coordinate_mask=batch.supervision_mask)
    everything = masked_cross_entropy(output.logits, tokens)
    assert float(supervised.detach()) != pytest.approx(float(everything.detach()))
    # Frames marked invalid never contribute, whatever the coordinate mask says.
    valid = torch.ones(4, 16, dtype=torch.bool)
    valid[:, 8:] = False
    trimmed = masked_cross_entropy(output.logits, tokens, valid_mask=valid)
    expected = torch.nn.functional.cross_entropy(
        output.logits[:, :8].reshape(-1, 9), tokens[:, :8].reshape(-1)
    )
    assert float(trimmed.detach()) == pytest.approx(float(expected.detach()), rel=1e-5)
