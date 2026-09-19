"""Read-only, multi-arm summary over one frozen validation protocol.

E04, E05 and E06 each grew a scoring script inside their own output directory and
the three drifted apart: one read the action vocabulary from the wrong config
level, so every score silently lost its condition; one averaged rows while the
table said it averaged tokens; one wrote a NaN because the writer and the reader
used different key names.  This module is the single read-only entry those
scripts converge to.

Three aggregations, never mixed (``AGGREGATIONS``):

* ``micro`` -- every supervised token has the same weight;
* ``macro_row`` -- every protocol row has the same weight;
* ``protocol_weighted`` -- token-weighted inside each mask kind, then combined
  with the protocol's fixed kind weights.  This is the trainer's own objective,
  i.e. exactly what a checkpoint's recorded ``val_objective`` is.

Every number is written *under the name of its aggregation*, so a table cannot
silently stitch a micro base to a macro arm.  Differences are reported as
``nll_improvement_*`` and are positive when the model is better; the old
``correct_minus_wrong`` fields held ``wrong - correct``, which read as a loss.

The content condition is read from the loaded transport (never from a config
lookup that can miss): a model trained with an action conditioner is refused when
the condition is missing.  ``omit_action`` exists only to reproduce the historical
buggy number, and it relabels the protocol as ``action_omitted`` so the ablated
score can never be compared with a conditioned one by accident.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import file_sha256, load_operator_bundle
from .eval_protocol import ValidationBatchBuilder
from .masking import MaskGenerator
from .windows import ContentVocabulary

#: The aggregation rules a number can be reported under.  A summary that reports
#: an unlabelled mean is the bug this list exists to prevent.
AGGREGATIONS: tuple[str, ...] = ("micro", "macro_row", "protocol_weighted")
SUMMARY_SCHEMA_VERSION = 1
#: Default tolerance for "the rescored objective reproduces the checkpoint's own
#: validation number".  Both sides are fp32 sums over the same frozen rows, so the
#: difference is rounding, not agreement to the printed digits.
OBJECTIVE_REPRODUCTION_TOLERANCE = 1e-5


def require_finite(payload: Any, *, where: str = "summary") -> None:
    """Rejects NaN/inf: an uncomputed metric is ``None``, never a NaN in a JSON.

    E06 wrote ``tv`` keys its own reader never found, and the missing values came
    back as NaN through a mean; a NaN is not a number and must stop the writer.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            require_finite(value, where=f"{where}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            require_finite(value, where=f"{where}[{index}]")
        return
    if isinstance(payload, (float, np.floating)) and not np.isfinite(float(payload)):
        raise ValueError(
            f"{where} is not finite ({payload!r}); an uncomputed metric must be null, "
            "not NaN"
        )


#: The six labels the plan requires a stage to report separately.  A summary that
#: collapses them into one "passed" is what let a run claim 7/7 while the physics
#: and the visual check had never been executed.
STAGE_STATUS_LABELS: tuple[str, ...] = (
    "implementation_complete",
    "measurement_valid",
    "likelihood_signal",
    "motion_quality",
    "style_fidelity",
    "generalization",
)


def stage_status(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The six status labels, every unmeasured one written ``not_evaluated``.

    A label only leaves ``not_evaluated`` when a caller states its value with the
    evidence behind it; a missing label means "nobody measured this", never "fine".
    """
    given = dict(values or {})
    unknown = sorted(set(given) - set(STAGE_STATUS_LABELS))
    if unknown:
        raise ValueError(
            f"Unknown stage status labels {unknown}; the labels are fixed so one 'passed' "
            "cannot stand in for a requirement that was never executed"
        )
    report: dict[str, Any] = {}
    for label in STAGE_STATUS_LABELS:
        if label not in given:
            report[label] = {"value": "not_evaluated", "reason": "not measured in this stage"}
            continue
        value = given[label]
        if isinstance(value, str):
            report[label] = {"value": value, "evidence": None}
        elif isinstance(value, Mapping) and "value" in value:
            report[label] = dict(value)
        else:
            raise ValueError(
                f"Stage status {label!r} must be a value string or a mapping with a 'value' key, "
                f"got {value!r}"
            )
    return report


def check_status(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarizes checks without letting an unrun check count as a pass."""
    passed = sorted(name for name, entry in checks.items() if entry.get("passed") is True)
    failed = sorted(name for name, entry in checks.items() if entry.get("passed") is False)
    not_run = sorted(name for name, entry in checks.items() if entry.get("passed") is None)
    return {
        "checks": len(checks),
        "passed": passed,
        "failed": failed,
        "not_run": not_run,
        "all_passed": bool(checks) and not failed and not not_run,
    }


def objective_reproduction(
    recorded: float | None,
    recomputed: float | None,
    *,
    tolerance: float = OBJECTIVE_REPRODUCTION_TOLERANCE,
) -> dict[str, Any]:
    """A checkpoint's recorded ``val_objective`` against the recomputed one.

    This is the regression the E04–E06 path failed: those scripts scored every arm
    without its action condition and still reported the checkpoint's recorded
    number beside their own.  Two fp32 sums over the same frozen rows differ only
    by rounding, so a difference above ``tolerance`` means the evaluation is not
    scoring the run that selected the checkpoint.
    """
    if recorded is None or recomputed is None:
        return {
            "recorded_val_objective": recorded,
            "recomputed_protocol_weighted": recomputed,
            "absolute_difference": None,
            "tolerance": float(tolerance),
            "passed": None,
            "reason": "nothing recorded to compare against",
        }
    difference = abs(float(recomputed) - float(recorded))
    return {
        "recorded_val_objective": float(recorded),
        "recomputed_protocol_weighted": float(recomputed),
        "absolute_difference": float(difference),
        "tolerance": float(tolerance),
        "passed": bool(difference <= float(tolerance)),
        "reason": None
        if difference <= float(tolerance)
        else "the scoring path does not reproduce the run that selected this checkpoint",
    }


def write_summary(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Writes the summary atomically after the finiteness check."""
    require_finite(payload, where=str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------------------
# arms


@dataclass(frozen=True)
class ArmSpec:
    """One arm of a comparison: a name and the operator bundle it reads."""

    name: str
    path: Path

    def __post_init__(self) -> None:
        if not self.name or "=" in self.name:
            raise ValueError(f"Arm name {self.name!r} must be non-empty and free of '='")

    @classmethod
    def parse(cls, text: str) -> "ArmSpec":
        if "=" not in text:
            raise ValueError(
                f"An arm is written name=checkpoint.pt, got {text!r}; a bare path has no "
                "name to report and two arms would be indistinguishable in the summary"
            )
        name, _, path = text.partition("=")
        return cls(name=name.strip(), path=Path(path.strip()))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path)}


@dataclass
class LoadedArm:
    """One evaluated checkpoint plus the identities a claim about it needs."""

    spec: ArmSpec
    sha256: str
    checkpoint: Mapping[str, Any]
    model: Any
    operator_kind: str
    encoder_kind: str
    style_index: dict[str, int] | None
    content_vocabulary: ContentVocabulary | None
    seed: int | None
    optimizer_steps: int
    recorded_val_objective: float | None
    recorded_protocol_hash: str | None
    transport_digest: str
    trainable_parameters: int

    def describe(self) -> dict[str, Any]:
        vocabulary = self.content_vocabulary
        return {
            "name": self.spec.name,
            "path": str(self.spec.path),
            "sha256": self.sha256,
            "operator": self.operator_kind,
            "style_encoder": self.encoder_kind,
            "style_index": None if self.style_index is None else dict(self.style_index),
            "content_condition": None
            if vocabulary is None
            else {"kind": vocabulary.kind, "classes": len(vocabulary.classes)},
            "seed": self.seed,
            "optimizer_steps": int(self.optimizer_steps),
            "recorded_val_objective": self.recorded_val_objective,
            "recorded_protocol_hash": self.recorded_protocol_hash,
            "transport_digest": self.transport_digest,
            "trainable_parameters": int(self.trainable_parameters),
        }


def transport_state_digest(model: Any) -> str:
    """A digest of the frozen transport's weights, tensor by tensor.

    "Both arms share the upstream" was previously argued from a recorded SHA, which
    says what was *loaded*; this says what the loaded model actually contains, so
    two arms whose upstreams differ cannot be compared even if both recorded the
    same file.
    """
    digest = hashlib.sha256()
    for name, value in sorted(model.transport.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().to(torch.float64).cpu().contiguous()
        digest.update(np.asarray(tensor).tobytes())
    return digest.hexdigest()


def transport_state_mismatches(left: Any, right: Any) -> list[str]:
    """Names of the transport tensors that differ (empty when identical)."""
    left_state = left.transport.state_dict()
    right_state = right.transport.state_dict()
    if set(left_state) != set(right_state):
        return sorted(set(left_state) ^ set(right_state))
    return sorted(
        name
        for name, value in left_state.items()
        if not torch.equal(value.detach().cpu(), right_state[name].detach().cpu())
    )


def load_arm(
    spec: ArmSpec,
    *,
    adapter: Any,
    tokenizer_identity: Mapping[str, Any] | None,
    tokenizer_checkpoint: str | Path | None,
    device: torch.device | str = "cpu",
    require_tokenizer: bool = True,
) -> LoadedArm:
    """Loads one operator bundle and reads its own identities.

    Seed and step budget are read from the checkpoint, not from a caller's flag:
    a hardcoded ``same_seed=True`` was the previous round's way of asserting an
    equality nobody had checked.
    """
    checkpoint, model = load_operator_bundle(
        spec.path,
        adapter=adapter,
        tokenizer_identity=tokenizer_identity,
        tokenizer_checkpoint=tokenizer_checkpoint,
        require_tokenizer=require_tokenizer,
        device=device,
    )
    model_config = dict(checkpoint.get("metadata", {}).get("model_config") or {})
    provenance = dict(checkpoint.get("provenance") or {})
    metrics = dict(checkpoint.get("metrics") or {})
    encoder_config = dict(model_config.get("style_encoder") or {})
    encoder_kind = str(encoder_config.get("kind", "reference"))
    vocabulary = getattr(model.transport, "content_vocabulary", None)
    recorded_objective = metrics.get("val_objective")
    return LoadedArm(
        spec=spec,
        sha256=file_sha256(spec.path),
        checkpoint=checkpoint,
        model=model,
        operator_kind=str(dict(model_config.get("operator") or {}).get("name", "unknown")),
        encoder_kind=encoder_kind,
        style_index=None
        if encoder_kind != "style_id"
        else {str(k): int(v) for k, v in (provenance.get("style_to_id") or {}).items()},
        content_vocabulary=vocabulary,
        seed=None if provenance.get("seed") is None else int(provenance["seed"]),
        optimizer_steps=int(metrics.get("optimizer_steps", checkpoint.get("global_step", 0)) or 0),
        recorded_val_objective=None if recorded_objective is None else float(recorded_objective),
        recorded_protocol_hash=None
        if metrics.get("protocol_hash") is None
        else str(metrics["protocol_hash"]),
        transport_digest=transport_state_digest(model),
        trainable_parameters=int(
            sum(parameter.numel() for parameter in model.trainable_parameters())
        ),
    )


# ---------------------------------------------------------------------------
# the content condition


def condition_label(arm: LoadedArm, *, omit_action: bool) -> str:
    """What condition this evaluation actually applies, stated in one word.

    The label travels with every number: ``action``, ``action_omitted`` (the
    reproduction of the old buggy path) or ``unconditional``.
    """
    vocabulary = arm.content_vocabulary
    if vocabulary is None or vocabulary.unconditional:
        return "unconditional"
    return "action_omitted" if omit_action else "action"


def condition_vector(
    vocabulary: ContentVocabulary | None,
    rows: Sequence[Mapping[str, Any]],
    *,
    omit_action: bool,
    device: torch.device | str,
) -> torch.Tensor | None:
    """The transport's own action ids for these rows, or ``None`` when ablated.

    The vocabulary comes from the loaded transport, so a missing condition is a
    deliberate ablation, not a config lookup that returned nothing.
    """
    if vocabulary is None or vocabulary.unconditional:
        return None
    if omit_action:
        return None
    return vocabulary.vector([str(row["content"]) for row in rows]).to(device)


def assert_condition_present(arm: LoadedArm, batch: Any, *, omit_action: bool) -> None:
    """Refuses "the conditioner exists but saw no condition".

    This is the exact failure the E04–E06 scripts shipped: the vocabulary was read
    from ``model_config["content_vocabulary"]`` (a key that does not exist) and the
    builder quietly built unconditional batches, so every arm was scored without
    the condition it was trained with.  Silence is the bug; this raises.
    """
    vocabulary = arm.content_vocabulary
    if vocabulary is None or vocabulary.unconditional or omit_action:
        return
    if getattr(batch, "content_condition", None) is None:
        raise ValueError(
            f"Arm {arm.spec.name!r} was trained with content.kind={vocabulary.kind!r} but the "
            "batch carries no content condition; a conditional arm scored without its condition "
            "is a different experiment (pass omit_action=True to run the ablated protocol "
            "explicitly, which is labelled as such)"
        )


# ---------------------------------------------------------------------------
# scoring


def batch_digest(batch: Any) -> str:
    """A digest of everything two arms must agree on before their scores compare.

    The condition, the mask, the target and the reference windows -- not the
    style ids, which are arm-specific by construction (a style-ID arm carries ids
    where a reference arm carries none).
    """
    digest = hashlib.sha256()
    for name in (
        "target_tokens",
        "visible_mask",
        "reference_tokens",
        "target_valid_mask",
        "reference_valid_mask",
        "content_condition",
    ):
        value = getattr(batch, name, None)
        if value is None:
            digest.update(f"{name}:none;".encode("utf-8"))
            continue
        tensor = value.detach().to(torch.int64).cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:".encode("utf-8"))
        digest.update(np.asarray(tensor).tobytes())
    return digest.hexdigest()


def _per_row_nll(
    probabilities: torch.Tensor,
    batch: Any,
    spec: Any,
    *,
    clamp_min: float = 1e-12,
) -> tuple[list[float], list[int]]:
    """``(nll_sum, supervised_tokens)`` per row, exactly as the trainer counts them."""
    mask = batch.supervision_mask(spec)
    if batch.target_valid_mask is not None:
        mask = mask & batch.target_valid_mask.to(mask.device).bool().unsqueeze(-1)
    picked = probabilities.float().clamp_min(float(clamp_min)).log().gather(
        -1, batch.target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    weights = mask.to(picked.dtype)
    sums = (-(picked * weights).sum(dim=(1, 2))).double().tolist()
    counts = weights.sum(dim=(1, 2)).to(torch.int64).tolist()
    return sums, counts


@dataclass
class ReferencePool:
    """Real val clips a *reference* arm can be re-scored against.

    Wrong-style and alternative references are drawn from this pool with the
    leakage rules the data identity states (a different take, a different source
    clip, the same mirror flag), so "the reference changed the output" cannot be
    an artefact of swapping in the target's own mirror or neighbour crop.  When no
    legal candidate exists the row says so and stays unscored; nothing is
    fabricated from a roll or a random tensor.
    """

    labels: dict[int, dict[str, Any]] = field(default_factory=dict)
    by_action: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_store(cls, store: Any, windows: Mapping[int, Any]) -> "ReferencePool":
        labels: dict[int, dict[str, Any]] = {}
        by_action: dict[str, list[int]] = {}
        for clip in sorted(windows):
            label = store.clip_label(int(clip))
            labels[int(clip)] = label
            by_action.setdefault(str(label["action"]), []).append(int(clip))
        return cls(labels=labels, by_action=by_action)

    def pick(
        self,
        target_clip: int,
        *,
        exclude: Iterable[int],
        style: str | None = None,
        other_style: bool = True,
    ) -> dict[str, Any] | None:
        target = self.labels.get(int(target_clip))
        if target is None:
            return None
        excluded = {int(clip) for clip in exclude}
        forbidden_groups = {int(target["source_group"])}
        forbidden_sources = {int(target["source_id"])}
        for clip in excluded:
            label = self.labels.get(int(clip))
            if label is not None:
                forbidden_groups.add(int(label["source_group"]))
                forbidden_sources.add(int(label["source_id"]))
        candidates = [
            clip
            for clip in self.by_action.get(str(target["action"]), [])
            if clip not in excluded
            and int(self.labels[clip]["source_group"]) not in forbidden_groups
            and int(self.labels[clip]["source_id"]) not in forbidden_sources
            and bool(self.labels[clip]["mirror"]) == bool(target["mirror"])
            and (style is None or str(self.labels[clip]["style"]) == str(style))
            and (not other_style or str(self.labels[clip]["style"]) != str(target["style"]))
        ]
        if not candidates:
            return None
        return self.labels[sorted(candidates)[0]]


def score_arms_on_rows(
    arms: Sequence[LoadedArm],
    samples: Sequence[Any],
    *,
    token_source: Any,
    adapter: Any,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
    omit_action: bool = False,
    wrong_style_ids: bool = True,
    reference_pool: ReferencePool | None = None,
    wrong_reference: bool = False,
    strength: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scores every arm on the *same* frozen rows under the stated condition.

    Returns the per-row records and a comparability report.  Every arm builds its
    own batch (a style-ID arm needs its ids, a reference arm its tokens) but the
    batches are digested and required to agree on targets, masks and condition, so
    "same protocol" is checked per batch instead of asserted.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("No rows to score")
    device = torch.device(device)
    vocabulary = arms[0].content_vocabulary
    for arm in arms[1:]:
        if getattr(arm.content_vocabulary, "classes", ()) != getattr(vocabulary, "classes", ()):
            raise ValueError(
                f"Arm {arm.spec.name!r} was trained with a different action vocabulary than "
                f"{arms[0].spec.name!r}; the two cannot be scored on one protocol"
            )
        if arm.model.spec.as_dict() != arms[0].model.spec.as_dict():
            raise ValueError(f"Arm {arm.spec.name!r} has a different token alphabet")
    builders = {}
    for arm in arms:
        builder = ValidationBatchBuilder(
            token_source=token_source,
            mask_generator=MaskGenerator(dict(samples[0].mask_config)),
            adapter=adapter,
            device=device,
            content_vocabulary=arm.content_vocabulary,
            style_index=arm.style_index,
            encoder_kind=arm.encoder_kind,
            strength=float(strength),
        )
        builders[arm.spec.name] = builder

    # One kind at a time, in fixed chunks: the mask kind is a property of the row,
    # and a batch that mixes kinds is refused by the builder -- so grouping here is
    # what makes "the same rows, in the same order, at any batch size" true.
    chunks: list[list[Any]] = []
    for kind in sorted({str(sample.kind) for sample in samples}):
        of_kind = [sample for sample in samples if str(sample.kind) == kind]
        for start in range(0, len(of_kind), int(batch_size)):
            chunks.append(of_kind[start : start + int(batch_size)])
    rows: list[dict[str, Any]] = []
    digest_mismatches: list[dict[str, Any]] = []
    condition_batches: dict[str, int] = {arm.spec.name: 0 for arm in arms}
    for chunk in chunks:
        payloads: dict[str, Any] = {}
        for arm in arms:
            batch = builders[arm.spec.name].build_batch(chunk)
            assert_condition_present(arm, batch, omit_action=omit_action)
            if getattr(batch, "content_condition", None) is not None:
                condition_batches[arm.spec.name] += 1
            payloads[arm.spec.name] = batch
        reference = arms[0].spec.name
        for arm in arms[1:]:
            if batch_digest(payloads[arm.spec.name]) != batch_digest(payloads[reference]):
                digest_mismatches.append(
                    {
                        "rows": [int(sample.sample_id) for sample in chunk],
                        "arms": [reference, arm.spec.name],
                    }
                )
        for index, sample in enumerate(chunk):
            row: dict[str, Any] = {
                "sample_id": int(sample.sample_id),
                "kind": str(sample.kind),
                "split": str(sample.split),
                "target_clip": int(sample.target_clip),
                "target_start": int(sample.target_start),
                "reference_clip": int(sample.reference_clip),
                "reference_start": int(sample.reference_start),
                "mask_seed": int(sample.seed),
                "mask_config": dict(sample.mask_config),
                "strength": float(strength),
                "content": str(sample.content),
                "style": str(sample.style),
                "actor": str(sample.actor),
                "supervised_tokens": 0,
            }
            label = token_source.store.clip_label(int(sample.target_clip))
            row.update(
                {
                    "take": int(label["source_group"]),
                    "source_id": int(label["source_id"]),
                    "mirror": bool(label["mirror"]),
                    "target_style": str(label["style"]),
                    "target_action": str(label["action"]),
                }
            )
            reference_label = token_source.store.clip_label(int(sample.reference_clip))
            row["reference_style"] = str(reference_label["style"])
            row["reference_action"] = str(reference_label["action"])
            row["reference_source_id"] = int(reference_label["source_id"])
            rows.append(row)

        with torch.inference_mode():
            for arm in arms:
                name = arm.spec.name
                batch = payloads[name]
                result = arm.model(batch)
                styled, counts = _per_row_nll(result.probabilities, batch, arm.model.spec)
                base, _ = _per_row_nll(result.base_probabilities, batch, arm.model.spec)
                offset = len(rows) - len(chunk)
                for index in range(len(chunk)):
                    rows[offset + index][f"nll_{name}_styled_sum"] = float(styled[index])
                    rows[offset + index][f"nll_{name}_base_sum"] = float(base[index])
                    rows[offset + index]["supervised_tokens"] = int(counts[index])
                if not (wrong_style_ids and arm.encoder_kind == "style_id"):
                    continue
                # One forward per candidate id for the whole chunk, not one per row:
                # the row's own id is excluded from its average, everything else is
                # the same distribution it would see alone.
                candidate_ids = sorted({int(value) for value in (arm.style_index or {}).values()})
                totals = {index: [] for index in range(len(chunk))}
                for candidate in candidate_ids:
                    swapped = _with_style_id(batch, candidate)
                    values, _ = _per_row_nll(
                        arm.model(swapped).probabilities, swapped, arm.model.spec
                    )
                    for index in range(len(chunk)):
                        if candidate != int(batch.style_ids[index]):
                            totals[index].append(float(values[index]))
                for index in range(len(chunk)):
                    if totals[index]:
                        rows[offset + index][f"nll_{name}_wrong_sum"] = float(np.mean(totals[index]))
                        rows[offset + index][f"nll_{name}_wrong_variants"] = len(totals[index])
            if wrong_reference:
                for arm in arms:
                    if arm.encoder_kind != "reference":
                        continue
                    name = arm.spec.name
                    variants = []
                    positions = []
                    offset = len(rows) - len(chunk)
                    for index, sample in enumerate(chunk):
                        pick = (
                            None
                            if reference_pool is None
                            else reference_pool.pick(
                                int(sample.target_clip), exclude=(int(sample.reference_clip),)
                            )
                        )
                        if pick is None:
                            rows[offset + index][f"nll_{name}_wrong_sum"] = None
                            rows[offset + index][f"nll_{name}_wrong_style"] = None
                            rows[offset + index][f"nll_{name}_wrong_reason"] = (
                                "no_legal_different_style_reference"
                            )
                            continue
                        variants.append(_with_reference(sample, int(pick["clip_id"]), token_source))
                        positions.append(index)
                        rows[offset + index][f"nll_{name}_wrong_style"] = str(pick["style"])
                        rows[offset + index][f"nll_{name}_wrong_clip"] = int(pick["clip_id"])
                    if not variants:
                        continue
                    variant_batch = builders[name].build_batch(variants)
                    assert_condition_present(arm, variant_batch, omit_action=omit_action)
                    values, _ = _per_row_nll(
                        arm.model(variant_batch).probabilities, variant_batch, arm.model.spec
                    )
                    for position, value in zip(positions, values):
                        rows[offset + position][f"nll_{name}_wrong_sum"] = float(value)
    comparability = {
        "batch_digest_mismatches": digest_mismatches,
        "batches_with_a_condition": condition_batches,
        "condition": condition_label(arms[0], omit_action=omit_action),
        "rows": len(rows),
    }
    return rows, comparability


def _with_style_id(batch: Any, style_id: int) -> Any:
    """The same batch with every row's style id set to ``style_id``.

    Per-row averages over the *other* ids are then taken from whole-chunk
    forwards: which id a row is scored under is the only thing that changes, so
    the row's own number is the same one a single-row batch would produce.
    """
    import dataclasses

    ids = torch.full_like(batch.style_ids, int(style_id))
    return dataclasses.replace(batch, style_ids=ids)


def _with_reference(sample: Any, clip_id: int, token_source: Any) -> Any:
    """The same protocol row with another reference window."""
    import dataclasses

    window = token_source.window_at(int(clip_id), _first_start(token_source, int(clip_id)))
    if window is None:
        raise ValueError(f"Reference clip {clip_id} has no window in the evaluation source")
    return dataclasses.replace(
        sample,
        reference_clip=int(clip_id),
        reference_start=int(window.metadata["target_start"]),
    )


def _first_start(token_source: Any, clip_id: int) -> int:
    candidates = token_source.windows_by_clip.get(int(clip_id))
    if not candidates:
        raise ValueError(f"Clip {clip_id} has no window in the evaluation source")
    return int(min(int(getattr(request, "target_start", 0)) for request in candidates))


# ---------------------------------------------------------------------------
# aggregation


def summarize_metric(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    weights: Mapping[str, float],
    max_rows: int | None = None,
) -> dict[str, Any]:
    """One metric in three aggregations, each under its own name.

    ``key`` names a per-row *sum* (``nll_<arm>_styled_sum``); the counts come from
    the row's ``supervised_tokens``.  Rows with zero supervised tokens, or without
    the key at all (an arm that could not score a row), are excluded and counted --
    a mean that silently drops them is how a wrong-reference study ended up
    reporting 143 rows as if they were 213.
    """
    usable = [
        row
        for row in rows
        if row.get(key) is not None and int(row.get("supervised_tokens", 0) or 0) > 0
    ]
    missing = len(rows) - len(usable)
    result: dict[str, Any] = {
        "key": str(key),
        "rows": len(rows),
        "rows_scored": len(usable),
        "rows_unscored": missing,
    }
    if not usable:
        for name in AGGREGATIONS:
            result[name] = {"nll": None, "supervised_tokens": 0, "rows": 0}
        return result
    counts = np.asarray([float(row["supervised_tokens"]) for row in usable], dtype=np.float64)
    values = np.asarray([float(row[key]) for row in usable], dtype=np.float64)
    token_total = float(counts.sum())
    result["micro"] = {
        "aggregation": "micro",
        "nll": float(values.sum() / token_total),
        "supervised_tokens": int(token_total),
        "rows": len(usable),
    }
    result["macro_row"] = {
        "aggregation": "macro_row",
        "nll": float(np.mean(values / counts)),
        "supervised_tokens": int(token_total),
        "rows": len(usable),
    }
    by_kind: dict[str, dict[str, Any]] = {}
    objective = 0.0
    missing_weights: list[str] = []
    for kind in sorted({str(row["kind"]) for row in usable}):
        subset = [row for row in usable if str(row["kind"]) == kind]
        kind_counts = float(sum(float(row["supervised_tokens"]) for row in subset))
        kind_sum = float(sum(float(row[key]) for row in subset))
        by_kind[kind] = {
            "aggregation": "token_weighted_within_kind",
            "nll": kind_sum / kind_counts,
            "supervised_tokens": int(kind_counts),
            "rows": len(subset),
        }
        weight = weights.get(kind)
        if weight is None:
            missing_weights.append(kind)
        else:
            objective += float(weight) * (kind_sum / kind_counts)
    result["by_kind"] = by_kind
    result["protocol_weighted"] = {
        "aggregation": "protocol_weighted",
        "nll": None if missing_weights else objective,
        "missing_weights_for_kinds": missing_weights,
        "supervised_tokens": int(token_total),
        "rows": len(usable),
    }
    result["by_style"] = {}
    for style in sorted({str(row.get("style", "")) for row in usable}):
        subset = [row for row in usable if str(row.get("style", "")) == style]
        style_counts = float(sum(float(row["supervised_tokens"]) for row in subset))
        style_sum = float(sum(float(row[key]) for row in subset))
        result["by_style"][style] = {
            "aggregation": "token_weighted_within_style",
            "nll": style_sum / style_counts if style_counts > 0 else None,
            "supervised_tokens": int(style_counts),
            "rows": len(subset),
        }
    if max_rows is not None and len(rows) > int(max_rows):
        raise ValueError(
            f"The rows exceed the stated sample cap ({len(rows)} > {int(max_rows)}); the cap is "
            "part of the protocol, not a suggestion"
        )
    return result


def paired_metric_comparison(
    rows: Sequence[Mapping[str, Any]],
    key_left: str,
    key_right: str,
    *,
    weights: Mapping[str, float],
    label_left: str,
    label_right: str,
) -> dict[str, Any]:
    """Two per-row metrics compared only on the rows where *both* exist.

    The unpaired aggregates answer "what does this arm score on its own rows";
    this answers "does replacing one input with the other change the score of the
    same row".  The reference study needed the second: its wrong-reference number
    covered 143 rows and its styled number 213, and the difference between those
    two aggregates read as a 0.027 nat improvement that does not exist row by row
    (there the change is below 1e-5 of a nat).
    """
    paired = [
        row
        for row in rows
        if row.get(key_left) is not None
        and row.get(key_right) is not None
        and int(row.get("supervised_tokens", 0) or 0) > 0
    ]
    report: dict[str, Any] = {
        "left": str(label_left),
        "right": str(label_right),
        "rows": len(rows),
        "rows_paired": len(paired),
        "rows_left_only": sum(
            1 for row in rows if row.get(key_left) is not None and row.get(key_right) is None
        ),
        "rows_right_only": sum(
            1 for row in rows if row.get(key_right) is not None and row.get(key_left) is None
        ),
    }
    if not paired:
        report["improvement"] = None
        return report
    report["left_aggregate"] = summarize_metric(paired, key_left, weights=weights)
    report["right_aggregate"] = summarize_metric(paired, key_right, weights=weights)
    report["improvement"] = improvement(
        report["left_aggregate"], report["right_aggregate"], label=label_right
    )
    counts = np.asarray([float(row["supervised_tokens"]) for row in paired], dtype=np.float64)
    deltas = np.asarray(
        [float(row[key_right]) - float(row[key_left]) for row in paired], dtype=np.float64
    ) / counts
    report["per_row_delta"] = {
        "definition": f"per-token ({label_right} - {label_left}); positive means {label_right} is worse",
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "max": float(deltas.max()),
        "min": float(deltas.min()),
        "absolute_mean": float(np.abs(deltas).mean()),
        "rows_where_left_better": int(np.count_nonzero(deltas > 0)),
        "rows_where_right_better": int(np.count_nonzero(deltas < 0)),
        "rows_with_no_change_below_1e-6": int(np.count_nonzero(np.abs(deltas) < 1e-6)),
    }
    return report


def improvement(better: Mapping[str, Any], worse: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """``worse - better`` per aggregation, keyed ``nll_improvement_vs_<label>``.

    The sign convention is stated once, here: a positive number always means the
    first metric's model has the lower (better) NLL.  The old reports named
    ``correct_minus_wrong`` fields that actually held ``wrong - correct``.
    """
    report: dict[str, Any] = {
        "versus": str(label),
        "note": "positive means the first model has the lower (better) NLL",
    }
    for aggregation in AGGREGATIONS:
        left = (better.get(aggregation) or {}).get("nll")
        right = (worse.get(aggregation) or {}).get("nll")
        report[aggregation] = {
            "aggregation": aggregation,
            f"nll_improvement_vs_{label}": None if left is None or right is None else float(right) - float(left),
        }
    return report


__all__ = [
    "AGGREGATIONS",
    "OBJECTIVE_REPRODUCTION_TOLERANCE",
    "STAGE_STATUS_LABELS",
    "SUMMARY_SCHEMA_VERSION",
    "ArmSpec",
    "LoadedArm",
    "ReferencePool",
    "assert_condition_present",
    "batch_digest",
    "check_status",
    "condition_label",
    "condition_vector",
    "improvement",
    "load_arm",
    "objective_reproduction",
    "paired_metric_comparison",
    "require_finite",
    "score_arms_on_rows",
    "stage_status",
    "summarize_metric",
    "transport_state_digest",
    "transport_state_mismatches",
    "write_summary",
]
