from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from raylib import IsKeyPressed, KEY_R

from datasets.feature_dataset import build_feature_store
from datasets.fsq_token_dataset import build_fsq_token_store
from datasets.fsq_trajectory_dataset import (
    FSQTrajectoryStore,
    TrajectoryNormalization,
    build_fsq_trajectory_store,
)
from Genoview import GenoView, PlaybackController, load_feature_stats
from models.fsq import FSQMotionAutoencoder
from models.fsq_generator import (
    FSQCausalTransformerGenerator,
    FSQConditionalTransformerGenerator,
    FSQGeneratorCache,
)
from motion_features import reconstruct_motion_state_from_features
from train_fsq_generator import choose_device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a realtime FSQ token rollout inside the existing GenoView character viewer."
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=Path("outputs/fsq_generator/best.pt"),
    )
    parser.add_argument(
        "--token-database",
        type=Path,
        default=Path("data/processed/100style_test5_pruned/fsq_20x9_full_loss"),
        help="Range metadata; token files are only used with --seed-source token-db.",
    )
    parser.add_argument(
        "--fsq-checkpoint",
        type=Path,
        default=Path("outputs/fsq_pruned_frame_causal_cnn_20x9_full_loss/best.pt"),
    )
    parser.add_argument("--range-idx", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed-frames", type=int, default=64)
    parser.add_argument(
        "--seed-source",
        choices=("reencode", "token-db"),
        default="reencode",
        help=(
            "reencode (default) encodes the selected raw motion prefix with the loaded FSQ checkpoint; "
            "token-db reuses stored token indices and requires tokenizer compatibility."
        ),
    )
    parser.add_argument(
        "--feature-database",
        type=Path,
        default=None,
        help="Feature database for --seed-source reencode; defaults to token metadata's feature_database.",
    )
    parser.add_argument(
        "--initial-capacity",
        type=int,
        default=256,
        help="Initial pose-buffer capacity; GUI mode grows this buffer without a frame limit.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Deprecated alias for a larger initial capacity; it no longer limits GUI generation.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Sample levels instead of greedy decoding.")
    parser.add_argument(
        "--style-id",
        type=int,
        default=0,
        help="Style class for an fsq_conditional_generator checkpoint.",
    )
    parser.add_argument(
        "--style-name",
        type=str,
        default=None,
        help="Style name for an fsq_conditional_generator checkpoint; overrides --style-id.",
    )
    parser.add_argument(
        "--trajectory-database",
        type=Path,
        default=None,
        help=(
            "Optional aligned FSQ trajectory database for reference controls. "
            "Without it, a conditional generator receives an explicit invalid/zero trajectory until an external caller "
            "sets one through RealtimeFSQController.set_trajectory_control()."
        ),
    )
    parser.add_argument(
        "--allow-tokenizer-mismatch",
        action="store_true",
        help="Only for --seed-source token-db: allow an incompatible stored-token seed for debugging.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--resources-root",
        type=Path,
        default=Path(__file__).resolve().parent / "resources",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate a finite smoke-test rollout without opening a window.")
    parser.add_argument("--dry-run-frames", type=int, default=120, help="Generated frames for --dry-run.")
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="Optional .npz path for generated features, poses, rotations, and tokens.",
    )
    return parser.parse_args()


def load_generator(path: Path, store, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    family = checkpoint.get("model_family")
    if family == "fsq_generator":
        model_type = FSQCausalTransformerGenerator
    elif family == "fsq_conditional_generator":
        model_type = FSQConditionalTransformerGenerator
    else:
        raise ValueError(f"Unsupported generator checkpoint family: {checkpoint.get('model_family')}")
    if "tokenizer_checkpoint_sha256" not in checkpoint:
        raise KeyError(f"Generator checkpoint {path} does not record tokenizer_checkpoint_sha256")
    model = model_type(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if (model.num_coordinates, model.num_levels) != (store.num_coordinates, store.num_levels):
        raise ValueError("Generator dimensions do not match token database dimensions")
    return checkpoint, model


def load_fsq(path: Path, store, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Missing FSQ checkpoint: {path}")
    checkpoint_sha = sha256_file(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq":
        raise ValueError(f"Unsupported FSQ checkpoint family: {checkpoint.get('model_family')}")
    model = FSQMotionAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if model.quantizer.num_coordinates != store.num_coordinates or model.quantizer.num_levels != store.num_levels:
        raise ValueError("FSQ checkpoint dimensions do not match token database dimensions")
    return checkpoint, model, checkpoint_sha


def validate_tokenizer_compatibility(
    generator_checkpoint: dict,
    fsq_checkpoint_sha256: str,
    token_store,
    seed_source: str,
    allow_tokenizer_mismatch: bool,
) -> None:
    generator_tokenizer_sha256 = str(generator_checkpoint["tokenizer_checkpoint_sha256"])
    if generator_tokenizer_sha256 != fsq_checkpoint_sha256:
        raise ValueError(
            "Generator checkpoint was trained with a different FSQ tokenizer than --fsq-checkpoint: "
            f"generator={generator_tokenizer_sha256}, fsq={fsq_checkpoint_sha256}"
        )
    if seed_source != "token-db" or token_store.checkpoint_sha256 == generator_tokenizer_sha256:
        return

    message = (
        "Stored token seed and the generator tokenizer have different SHA256 values: "
        f"stored={token_store.checkpoint_sha256}, generator={generator_tokenizer_sha256}"
    )
    if not allow_tokenizer_mismatch:
        raise ValueError(message + ". Use --seed-source reencode, or pass --allow-tokenizer-mismatch only for debugging.")
    print(f"WARNING: {message}")


def resolve_style_id(
    generator: FSQCausalTransformerGenerator,
    generator_checkpoint: dict,
    token_store,
    style_id: int,
    style_name: str | None,
) -> int | None:
    if not isinstance(generator, FSQConditionalTransformerGenerator):
        if style_name is not None or style_id != 0:
            raise ValueError("--style-id/--style-name require an fsq_conditional_generator checkpoint")
        return None
    names = [str(name) for name in generator_checkpoint.get("style_names", token_store.style_names)]
    if len(names) != generator.num_styles:
        raise ValueError("Conditional checkpoint style metadata does not match its style embedding")
    if style_name is not None:
        if style_name not in names:
            raise ValueError(f"Unknown style {style_name!r}; available examples: {names[:10]}")
        style_id = names.index(style_name)
    if style_id < 0 or style_id >= generator.num_styles:
        raise ValueError(f"style_id must be in [0,{generator.num_styles - 1}], got {style_id}")
    return int(style_id)


def resolve_trajectory_conditioning(
    generator: FSQCausalTransformerGenerator,
    generator_checkpoint: dict,
    token_store,
    trajectory_database: Path | None,
) -> tuple[FSQTrajectoryStore | None, TrajectoryNormalization | None]:
    if not isinstance(generator, FSQConditionalTransformerGenerator):
        if trajectory_database is not None:
            raise ValueError("--trajectory-database requires an fsq_conditional_generator checkpoint")
        return None, None
    if "trajectory_normalization" not in generator_checkpoint:
        raise KeyError("Conditional generator checkpoint does not contain trajectory normalization statistics")
    normalization = TrajectoryNormalization.from_checkpoint(generator_checkpoint["trajectory_normalization"])
    if normalization.trajectory_dim != generator.trajectory_dim:
        raise ValueError("Conditional checkpoint trajectory normalization has the wrong dimension")
    if trajectory_database is None:
        # The normalization is still needed by the programmatic external-control
        # API even when no reference trajectory is supplied on the CLI.
        return None, normalization
    store = build_fsq_trajectory_store(trajectory_database, token_store)
    if store.trajectory_dim != generator.trajectory_dim:
        raise ValueError(
            f"Trajectory database dim={store.trajectory_dim} does not match generator dim={generator.trajectory_dim}"
        )
    return store, normalization


def resolve_feature_database(override: Path | None, token_store) -> Path:
    if override is not None:
        candidates = [override]
    else:
        configured = Path(token_store.feature_database)
        candidates = [configured]
        if not configured.is_absolute():
            candidates.append(token_store.database.parent / configured)
    for candidate in candidates:
        if (candidate / "metadata.npz").exists():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find feature database metadata.npz. Tried: {attempted}")


def encode_seed_from_feature_database(
    fsq: FSQMotionAutoencoder,
    fsq_checkpoint: dict,
    feature_database: Path,
    token_store,
    range_idx: int,
    start: int,
    seed_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode only the seed prefix with the active FSQ tokenizer.

    Source feature shards are normalized with their own dataset statistics, so
    convert through raw feature space before applying the checkpoint statistics.
    The left receptive-field context makes this numerically consistent with
    encode_fsq_database.py for a seed that starts in the middle of a clip.
    """
    feature_store = build_feature_store(feature_database)
    if len(feature_store.motion_files) != len(token_store.token_files):
        raise ValueError("Feature database and token metadata have different numbers of motion ranges")
    if str(feature_store.range_names[range_idx]) != str(token_store.range_names[range_idx]):
        raise ValueError(
            "Feature database range ordering does not match token metadata at "
            f"range_idx={range_idx}: {feature_store.range_names[range_idx]!r} vs "
            f"{token_store.range_names[range_idx]!r}"
        )
    if feature_store.motion_dim != fsq.motion_dim:
        raise ValueError(
            f"Feature motion_dim={feature_store.motion_dim} does not match FSQ motion_dim={fsq.motion_dim}"
        )

    checkpoint_stats = fsq_checkpoint.get("stats")
    if checkpoint_stats is None:
        raise KeyError("FSQ checkpoint does not contain feature statistics")
    checkpoint_names = [str(name) for name in np.asarray(checkpoint_stats["names"], dtype=object).tolist()]
    if checkpoint_names != feature_store.names:
        raise ValueError("Feature database and FSQ checkpoint have different skeleton joint ordering")

    motion = np.load(feature_store.motion_files[range_idx], mmap_mode="r")
    if motion.ndim != 2 or motion.shape[1] != feature_store.motion_dim:
        raise ValueError(f"Unexpected feature shard shape at {feature_store.motion_files[range_idx]}: {motion.shape}")
    if start < 0 or start + seed_frames > motion.shape[0]:
        raise ValueError(
            f"Seed [{start}, {start + seed_frames}) exceeds feature shard length {motion.shape[0]}"
        )

    read_start = max(0, start - int(fsq.context_left))
    source_motion = np.asarray(motion[read_start : start + seed_frames], dtype=np.float32).copy()
    source_offset = torch.from_numpy(feature_store.stats.offset.astype(np.float32)).to(device)
    source_scale = torch.from_numpy(feature_store.stats.scale.astype(np.float32)).to(device)
    checkpoint_offset = torch.as_tensor(checkpoint_stats["offset"], dtype=torch.float32, device=device)
    checkpoint_scale = torch.as_tensor(checkpoint_stats["scale"], dtype=torch.float32, device=device)
    if source_offset.shape != checkpoint_offset.shape or source_scale.shape != checkpoint_scale.shape:
        raise ValueError("Feature database and FSQ checkpoint statistics have incompatible dimensions")

    source_motion_tensor = torch.from_numpy(source_motion).unsqueeze(0).to(device)
    raw_motion = source_motion_tensor * source_scale.view(1, 1, -1) + source_offset.view(1, 1, -1)
    model_motion = (raw_motion - checkpoint_offset.view(1, 1, -1)) / checkpoint_scale.view(1, 1, -1)
    with torch.inference_mode():
        encoded = fsq.encode_to_indices(model_motion)
    offset = start - read_start
    seed_indices = encoded[:, offset : offset + seed_frames]
    if seed_indices.shape[1] != seed_frames:
        raise RuntimeError(f"Expected {seed_frames} encoded seed frames, got {seed_indices.shape[1]}")
    return seed_indices


class RealtimeFSQController:
    def __init__(
        self,
        generator: FSQCausalTransformerGenerator,
        fsq: FSQMotionAutoencoder,
        fsq_checkpoint: dict,
        seed_indices: torch.Tensor,
        initial_capacity: int,
        device: torch.device,
        temperature: float,
        sample: bool,
        stats_source: Path,
        source_range_idx: int = 0,
        source_start: int = 0,
        style_id: int | None = None,
        trajectory_store: FSQTrajectoryStore | None = None,
        trajectory_normalization: TrajectoryNormalization | None = None,
    ) -> None:
        if seed_indices.ndim != 3 or seed_indices.shape[0] != 1:
            raise ValueError(f"seed_indices must have shape [1,T,K], got {tuple(seed_indices.shape)}")
        if seed_indices.shape[1] <= 0 or seed_indices.shape[1] > generator.context_frames:
            raise ValueError(
                f"seed length must be in [1,{generator.context_frames}], got {seed_indices.shape[1]}"
            )
        if initial_capacity < seed_indices.shape[1]:
            raise ValueError("initial_capacity must be at least seed length")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        conditional = isinstance(generator, FSQConditionalTransformerGenerator)
        if conditional and style_id is None:
            raise ValueError("A conditional generator requires a style_id")
        if not conditional and (trajectory_store is not None or trajectory_normalization is not None):
            raise ValueError("Trajectory controls require an FSQConditionalTransformerGenerator")
        if trajectory_store is not None and trajectory_normalization is None:
            raise ValueError("A trajectory_store requires trajectory normalization statistics")
        if conditional and trajectory_store is not None:
            if trajectory_store.trajectory_dim != generator.trajectory_dim:
                raise ValueError("Trajectory store and conditional generator dimensions differ")
            if trajectory_normalization.trajectory_dim != generator.trajectory_dim:
                raise ValueError("Trajectory normalization and conditional generator dimensions differ")

        self.generator = generator
        self.fsq = fsq
        self.fsq_checkpoint = fsq_checkpoint
        self.device = device
        self.temperature = float(temperature)
        self.sample = bool(sample)
        self.conditional = conditional
        self.style_id = None if style_id is None else int(style_id)
        self.source_range_idx = int(source_range_idx)
        self.source_start = int(source_start)
        self.trajectory_store = trajectory_store
        self.trajectory_normalization = trajectory_normalization
        self._trajectory_override: tuple[torch.Tensor, torch.Tensor] | None = None
        self.seed_frames = int(seed_indices.shape[1])
        self.initial_capacity = int(initial_capacity)
        self.available_frames = self.seed_frames
        self.seed_indices = seed_indices.clone()
        # The generator cache is already a fixed-size rolling context. Keep the
        # decoder input rolling too, rather than retaining every generated token
        # on the accelerator during an open-ended session.
        self.token_history = seed_indices.clone()
        if self.conditional:
            self.seed_control_history, self.seed_control_valid_history = self._controls_for_input_tokens(
                self.source_start,
                self.seed_frames,
            )
            self.control_history = self.seed_control_history.clone()
            self.control_valid_history = self.seed_control_valid_history.clone()
        with torch.inference_mode():
            self.next_logits, self.cache = self._prefill_generator()
        if not isinstance(self.cache, FSQGeneratorCache):
            raise RuntimeError("Generator prefill did not return a cache")

        stats, metadata = load_feature_stats(stats_source)
        self.stats = stats
        self.names = [str(name) for name in metadata["names"]]
        self.parents = np.asarray(metadata["parents"], dtype=np.int32)
        self.joint_subset = str(metadata["joint_subset"])

        with torch.inference_mode():
            seed_features = self.fsq.decode_from_indices(self.token_history)
        self.features = np.zeros((self.initial_capacity, seed_features.shape[-1]), dtype=np.float32)
        seed_features_np = seed_features[0].detach().cpu().numpy().astype(np.float32)
        self.features[: self.seed_frames] = seed_features_np
        seed_state = reconstruct_motion_state_from_features(
            seed_features_np,
            stats=self.stats,
            parents=self.parents,
            normalized=True,
        )
        self.last_feature = seed_features_np[-1].copy()
        self.last_root_position = seed_state.root_positions[-1].copy()
        self.last_root_rotation = seed_state.root_rotations[-1].copy()
        self.positions = np.zeros((self.initial_capacity, len(self.names), 3), dtype=np.float32)
        self.rotations = np.zeros((self.initial_capacity, len(self.names), 4), dtype=np.float32)
        self.indices = np.zeros((self.initial_capacity, seed_indices.shape[-1]), dtype=np.uint8)
        self.indices[: self.seed_frames] = seed_indices[0].detach().cpu().numpy().astype(np.uint8)
        self.positions[: self.seed_frames] = seed_state.local_positions
        self.rotations[: self.seed_frames] = seed_state.local_rotations
        self.positions[self.seed_frames :] = self.positions[self.seed_frames - 1]
        self.rotations[self.seed_frames :] = self.rotations[self.seed_frames - 1]
        self.seed_features = seed_features_np.copy()
        self.seed_positions = seed_state.local_positions.copy()
        self.seed_rotations = seed_state.local_rotations.copy()
        self.seed_last_feature = self.last_feature.copy()
        self.seed_last_root_position = self.last_root_position.copy()
        self.seed_last_root_rotation = self.last_root_rotation.copy()

        self.generated_count = 0
        self.last_step_ms = 0.0
        self.total_generation_ms = 0.0

    def _zero_control(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.conditional:
            raise RuntimeError("Unconditional generators do not have trajectory controls")
        assert isinstance(self.generator, FSQConditionalTransformerGenerator)
        values = torch.zeros(
            (1, 1, self.generator.trajectory_dim),
            dtype=torch.float32,
            device=self.device,
        )
        valid = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        return values, valid

    def _control_for_target(self, target_local_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the normalized command that controls the prediction of one token."""
        if not self.conditional:
            raise RuntimeError("Unconditional generators do not have trajectory controls")
        if self._trajectory_override is not None:
            values, valid = self._trajectory_override
            return values.clone(), valid.clone()
        if self.trajectory_store is None or self.trajectory_normalization is None:
            return self._zero_control()
        try:
            values, valid = self.trajectory_store.normalized_window(
                self.source_range_idx,
                int(target_local_frame),
                int(target_local_frame) + 1,
                self.trajectory_normalization,
            )
        except IndexError:
            # A reference clip may end before an open-ended rollout.  The model
            # still runs with the explicit invalid-condition embedding.
            return self._zero_control()
        return (
            torch.from_numpy(values).unsqueeze(0).to(self.device, dtype=torch.float32),
            torch.from_numpy(valid).unsqueeze(0).to(self.device, dtype=torch.bool),
        )

    def _controls_for_input_tokens(
        self,
        input_start_local_frame: int,
        input_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        values, valid = zip(
            *[
                self._control_for_target(input_start_local_frame + offset + 1)
                for offset in range(input_length)
            ]
        )
        return torch.cat(values, dim=1), torch.cat(valid, dim=1)

    def _prefill_generator(self) -> tuple[torch.Tensor, FSQGeneratorCache]:
        if not self.conditional:
            return self.generator.prefill(self.token_history)
        assert isinstance(self.generator, FSQConditionalTransformerGenerator)
        assert self.style_id is not None
        return self.generator.prefill(
            self.token_history,
            style_ids=torch.tensor([self.style_id], dtype=torch.long, device=self.device),
            seed_trajectory=self.control_history,
            seed_trajectory_valid=self.control_valid_history,
        )

    def _rebuild_cache(self) -> None:
        with torch.inference_mode():
            self.next_logits, self.cache = self._prefill_generator()
        if not isinstance(self.cache, FSQGeneratorCache):
            raise RuntimeError("Generator prefill did not return a cache")

    def _replace_latest_control(self, target_local_frame: int) -> None:
        if not self.conditional:
            return
        values, valid = self._control_for_target(target_local_frame)
        # Rebuild rather than write in place: after a rollout the history may
        # originate from torch.inference_mode(), whose tensors are immutable
        # outside that context.
        self.control_history = torch.cat((self.control_history[:, :-1], values), dim=1).clone()
        self.control_valid_history = torch.cat(
            (self.control_valid_history[:, :-1], valid), dim=1
        ).clone()

    def set_style(self, style_id: int) -> None:
        """Switch the style prefix and make it effective for the next token."""
        if not self.conditional:
            raise RuntimeError("Style control requires an FSQConditionalTransformerGenerator")
        assert isinstance(self.generator, FSQConditionalTransformerGenerator)
        style_id = int(style_id)
        if style_id < 0 or style_id >= self.generator.num_styles:
            raise ValueError(f"style_id must be in [0,{self.generator.num_styles - 1}], got {style_id}")
        self.style_id = style_id
        self._rebuild_cache()

    def set_trajectory_control(self, trajectory: np.ndarray | list[float] | None, valid: bool = True) -> None:
        """Set an external raw root-local 18-D command for the next prediction.

        The input must use the control database layout
        ``[pos(+20,+40,+60), dir(+20,+40,+60)]``.  Rebuilding only the rolling
        cache makes this operation safe during an open-ended controller session.
        Pass ``None`` to resume an optional reference-trajectory provider.
        """
        if not self.conditional:
            raise RuntimeError("Trajectory control requires an FSQConditionalTransformerGenerator")
        assert isinstance(self.generator, FSQConditionalTransformerGenerator)
        if trajectory is None:
            self._trajectory_override = None
            self._replace_latest_control(self.source_start + self.available_frames)
            self._rebuild_cache()
            return
        raw = np.asarray(trajectory, dtype=np.float32).reshape(-1)
        if raw.shape != (self.generator.trajectory_dim,):
            raise ValueError(
                f"trajectory must have {self.generator.trajectory_dim} values, got {tuple(raw.shape)}"
            )
        if self.trajectory_normalization is None:
            raise RuntimeError(
                "An external trajectory command requires checkpoint normalization statistics; "
                "load a conditional checkpoint that records them."
            )
        normalized = (raw - self.trajectory_normalization.mean) / self.trajectory_normalization.std
        values = torch.from_numpy(normalized.astype(np.float32))[None, None].to(self.device)
        valid_tensor = torch.full((1, 1), bool(valid), dtype=torch.bool, device=self.device)
        if not valid:
            values.zero_()
        self._trajectory_override = (values, valid_tensor)
        self._replace_latest_control(self.source_start + self.available_frames)
        self._rebuild_cache()

    @property
    def generated_until(self) -> int:
        return self.available_frames - 1

    def _ensure_capacity(self, required_frames: int) -> None:
        if required_frames <= self.positions.shape[0]:
            return
        capacity = self.positions.shape[0]
        while capacity < required_frames:
            capacity *= 2
        positions = np.zeros((capacity, *self.positions.shape[1:]), dtype=np.float32)
        rotations = np.zeros((capacity, *self.rotations.shape[1:]), dtype=np.float32)
        features = np.zeros((capacity, self.features.shape[1]), dtype=np.float32)
        indices = np.zeros((capacity, self.indices.shape[1]), dtype=np.uint8)
        positions[: self.available_frames] = self.positions[: self.available_frames]
        rotations[: self.available_frames] = self.rotations[: self.available_frames]
        features[: self.available_frames] = self.features[: self.available_frames]
        indices[: self.available_frames] = self.indices[: self.available_frames]
        positions[self.available_frames :] = positions[self.available_frames - 1]
        rotations[self.available_frames :] = rotations[self.available_frames - 1]
        self.positions = positions
        self.rotations = rotations
        self.features = features
        self.indices = indices

    def _generate_one(self, frame_index: int) -> None:
        if frame_index != self.seed_frames + self.generated_count:
            raise ValueError(
                f"Generation must be sequential: expected frame {self.seed_frames + self.generated_count}, "
                f"got {frame_index}"
            )
        self._ensure_capacity(frame_index + 1)
        started = time.perf_counter()
        with torch.inference_mode():
            current = self.generator.sample_next(
                self.next_logits,
                temperature=self.temperature,
                greedy=not self.sample,
            )
            if self.conditional:
                assert isinstance(self.generator, FSQConditionalTransformerGenerator)
                # ``current`` is source frame source_start + frame_index. Its
                # decoder input gets the command for the next generated token.
                next_control, next_control_valid = self._control_for_target(
                    self.source_start + frame_index + 1
                )
                self.next_logits, self.cache = self.generator.decode_step(
                    current,
                    self.cache,
                    trajectory=next_control,
                    trajectory_valid=next_control_valid,
                )
            else:
                self.next_logits, self.cache = self.generator.decode_step(current, self.cache)
            next_token_history = torch.cat((self.token_history, current[:, None]), dim=1)
            next_token_history = next_token_history[:, -self.generator.context_frames :]
            if self.conditional:
                next_control_history = torch.cat((self.control_history, next_control), dim=1)
                next_control_valid_history = torch.cat(
                    (self.control_valid_history, next_control_valid), dim=1
                )
                next_control_history = next_control_history[:, -self.generator.context_frames :]
                next_control_valid_history = next_control_valid_history[:, -self.generator.context_frames :]
            current_feature = self.fsq.decode_from_indices(next_token_history)[:, -1]
        # Keep controller-owned histories as ordinary tensors so the public
        # style/trajectory setters can update them between realtime frames.
        self.token_history = next_token_history.clone()
        if self.conditional:
            self.control_history = next_control_history.clone()
            self.control_valid_history = next_control_valid_history.clone()
        current_feature_np = current_feature[0].detach().cpu().numpy().astype(np.float32)

        state = reconstruct_motion_state_from_features(
            np.stack((self.last_feature, current_feature_np), axis=0),
            stats=self.stats,
            parents=self.parents,
            dt=1.0 / 60.0,
            root_position0=self.last_root_position,
            root_rotation0=self.last_root_rotation,
            normalized=True,
        )
        self.features[frame_index] = current_feature_np
        self.positions[frame_index] = state.local_positions[-1]
        self.rotations[frame_index] = state.local_rotations[-1]
        self.indices[frame_index] = current[0].detach().cpu().numpy().astype(np.uint8)
        self.last_feature = current_feature_np
        self.last_root_position = state.root_positions[-1].copy()
        self.last_root_rotation = state.root_rotations[-1].copy()
        self.generated_count += 1
        self.available_frames = frame_index + 1
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        self.total_generation_ms += self.last_step_ms

    def ensure_frame(self, frame_index: int) -> None:
        frame_index = int(frame_index)
        if frame_index < self.seed_frames or frame_index <= self.generated_until:
            return
        while self.generated_until < frame_index:
            self._generate_one(self.seed_frames + self.generated_count)

    def materialize(self, generated_frames: int) -> None:
        """Generate a finite prefix for headless validation only."""
        if generated_frames <= 0:
            return
        self.ensure_frame(self.seed_frames + int(generated_frames) - 1)

    def reset(self) -> None:
        self.token_history = self.seed_indices.clone()
        if self.conditional:
            self.control_history = self.seed_control_history.clone()
            self.control_valid_history = self.seed_control_valid_history.clone()
            if self._trajectory_override is not None:
                self._replace_latest_control(self.source_start + self.seed_frames)
        self._rebuild_cache()
        self.features[: self.seed_frames] = self.seed_features
        self.positions[: self.seed_frames] = self.seed_positions
        self.rotations[: self.seed_frames] = self.seed_rotations
        self.indices[: self.seed_frames] = self.seed_indices[0].detach().cpu().numpy().astype(np.uint8)
        self.positions[self.seed_frames :] = self.positions[self.seed_frames - 1]
        self.rotations[self.seed_frames :] = self.rotations[self.seed_frames - 1]
        self.features[self.seed_frames :] = 0.0
        self.last_feature = self.seed_last_feature.copy()
        self.last_root_position = self.seed_last_root_position.copy()
        self.last_root_rotation = self.seed_last_root_rotation.copy()
        self.generated_count = 0
        self.available_frames = self.seed_frames
        self.last_step_ms = 0.0
        self.total_generation_ms = 0.0

    def database(self) -> dict[str, np.ndarray]:
        return {
            # Supply the allocated buffer to GenoView. RealtimeGenoView swaps
            # these references if the controller needs to grow it.
            "positions": self.positions,
            "rotations": self.rotations,
            "velocities": np.zeros_like(self.positions),
            "angular_velocities": np.zeros_like(self.positions),
            "contacts": np.zeros((self.positions.shape[0], 2), dtype=np.uint8),
            "parents": self.parents,
            "names": np.asarray(self.names, dtype=object),
            "range_starts": np.asarray([0], dtype=np.int32),
            "range_stops": np.asarray([self.positions.shape[0]], dtype=np.int32),
            "range_names": np.asarray(["Realtime FSQ"], dtype=object),
            "range_mirror": np.asarray([False], dtype=bool),
            "joint_subset": np.asarray(self.joint_subset, dtype=object),
        }


class RealtimeGenoView(GenoView):
    def __init__(self, database: dict[str, np.ndarray], controller: RealtimeFSQController, resources_root: Path, fps: int):
        super().__init__(database=database, trajectory_path=None, resources_root=resources_root, fps=fps)
        self.controller = controller

        # GenoView copies arrays in its constructor. Rebind its pose source to
        # the controller-owned buffers so generated frames become visible.
        self._sync_controller_arrays()
        self.playback = RealtimePlaybackController(frame_time=1.0 / float(fps), playing=True)

    def _sync_controller_arrays(self) -> None:
        if self.positions is not self.controller.positions:
            self.positions = self.controller.positions
            self.rotations = self.controller.rotations
            self.database["positions"] = self.positions
            self.database["rotations"] = self.rotations
            self.range_stops[0] = self.positions.shape[0]
            self.database["range_stops"] = self.range_stops

    def _sync_playback_frame(self):
        if IsKeyPressed(KEY_R):
            self.controller.reset()
            self.playback.set_current_frame(0)
            self.playback.playing = True

        target_frame = self.playback.current_frame
        if self.playback.playing:
            self.controller.ensure_frame(target_frame + 1)
        elif target_frame > self.controller.generated_until:
            self.playback.set_current_frame(self.controller.generated_until)
            target_frame = self.playback.current_frame

        self._sync_controller_arrays()
        self.playback.frame_count = max(1, self.controller.available_frames)
        self.frame_index = target_frame

    def _frame_range_name(self):
        return (
            f"Realtime FSQ | {self.controller.generated_count} generated "
            f"| {self.controller.last_step_ms:.1f} ms | Space: pause | R: reset"
        )


class RealtimePlaybackController(PlaybackController):
    """Playback clock that never wraps or clamps to a finite clip length."""

    def __init__(self, frame_time: float, playing: bool = True):
        super().__init__(frame_count=1, frame_time=frame_time, playing=playing)

    @property
    def current_frame(self) -> int:
        return max(0, int(self.frame))

    def _clamp_frame(self, frame: int) -> int:
        return max(0, int(frame))

    def update(self, dt):
        if self.playing and not self.scrubbing:
            self.frame += self.current_speed * dt / self.frame_time
        return self.current_frame


def save_output(path: Path, controller: RealtimeFSQController) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = controller.available_frames
    np.savez_compressed(
        path,
        features=controller.features[:frame_count],
        positions=controller.positions[:frame_count],
        rotations=controller.rotations[:frame_count],
        indices=controller.indices[:frame_count],
    )
    print(f"saved={path}")


def main() -> None:
    args = parse_args()
    initial_capacity = args.max_frames if args.max_frames is not None else args.initial_capacity
    if args.fps <= 0 or initial_capacity <= 0 or args.dry_run_frames <= 0:
        raise ValueError("fps, initial capacity, and dry_run_frames must be positive")
    device = choose_device(args.device)
    store = build_fsq_token_store(args.token_database)
    generator_checkpoint, generator = load_generator(args.generator_checkpoint, store, device)
    style_id = resolve_style_id(
        generator,
        generator_checkpoint,
        store,
        args.style_id,
        args.style_name,
    )
    trajectory_store, trajectory_normalization = resolve_trajectory_conditioning(
        generator,
        generator_checkpoint,
        store,
        args.trajectory_database,
    )
    fsq_checkpoint, fsq, fsq_checkpoint_sha256 = load_fsq(args.fsq_checkpoint, store, device)
    validate_tokenizer_compatibility(
        generator_checkpoint,
        fsq_checkpoint_sha256,
        store,
        args.seed_source,
        args.allow_tokenizer_mismatch,
    )
    if args.range_idx < 0 or args.range_idx >= len(store.token_files):
        raise ValueError(f"range_idx must be in [0, {len(store.token_files) - 1}], got {args.range_idx}")
    if args.seed_frames <= 0 or args.seed_frames > generator.context_frames:
        raise ValueError(
            f"seed_frames must be in [1,{generator.context_frames}], got {args.seed_frames}"
        )
    feature_database = None
    if args.seed_source == "reencode":
        feature_database = resolve_feature_database(args.feature_database, store)
        seed_indices = encode_seed_from_feature_database(
            fsq=fsq,
            fsq_checkpoint=fsq_checkpoint,
            feature_database=feature_database,
            token_store=store,
            range_idx=args.range_idx,
            start=args.start,
            seed_frames=args.seed_frames,
            device=device,
        )
    else:
        token_shard = np.load(store.token_files[args.range_idx], mmap_mode="r")
        if args.start < 0 or args.start + args.seed_frames > token_shard.shape[0]:
            raise ValueError(
                f"Seed [{args.start}, {args.start + args.seed_frames}) exceeds token shard length {token_shard.shape[0]}"
            )
        if token_shard.shape[1] != generator.num_coordinates:
            raise ValueError("Token shard coordinate count does not match generator")
        seed_np = np.asarray(
            token_shard[args.start : args.start + args.seed_frames],
            dtype=np.int64,
        ).copy()
        seed_indices = torch.from_numpy(seed_np).unsqueeze(0).to(device)
    controller = RealtimeFSQController(
        generator=generator,
        fsq=fsq,
        fsq_checkpoint=fsq_checkpoint,
        seed_indices=seed_indices,
        initial_capacity=initial_capacity,
        device=device,
        temperature=args.temperature,
        sample=args.sample,
        stats_source=args.fsq_checkpoint,
        source_range_idx=args.range_idx,
        source_start=args.start,
        style_id=style_id,
        trajectory_store=trajectory_store,
        trajectory_normalization=trajectory_normalization,
    )

    if args.dry_run:
        controller.materialize(args.dry_run_frames)
        if args.save_output is not None:
            save_output(args.save_output, controller)
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "device": str(device),
                    "range_idx": args.range_idx,
                    "start": args.start,
                    "seed_frames": args.seed_frames,
                    "seed_source": args.seed_source,
                    "feature_database": None if feature_database is None else str(feature_database),
                    "generated_frames": controller.generated_count,
                    "mean_generation_ms": controller.total_generation_ms / max(controller.generated_count, 1),
                    "generator_epoch": int(generator_checkpoint.get("epoch", 0)),
                    "generator_model_family": generator_checkpoint.get("model_family"),
                    "tokenizer_checkpoint_sha256": str(generator_checkpoint["tokenizer_checkpoint_sha256"]),
                    "fsq_model_family": fsq_checkpoint.get("model_family"),
                    "style_id": style_id,
                    "style_name": (
                        None
                        if style_id is None
                        else str(generator_checkpoint.get("style_names", store.style_names)[style_id])
                    ),
                    "trajectory_database": (
                        None if trajectory_store is None else str(trajectory_store.database)
                    ),
                },
                indent=2,
            )
        )
        return

    database = controller.database()
    viewer = RealtimeGenoView(database, controller, args.resources_root, args.fps)
    viewer.run()
    if args.save_output is not None:
        save_output(args.save_output, controller)


if __name__ == "__main__":
    main()
