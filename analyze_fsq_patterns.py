from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from datasets.fsq_token_dataset import (
    FSQTokenDataset,
    build_fsq_token_store,
    build_full_clip_windows,
)


REPRESENTATIONS = (
    "histogram",
    "transition",
    "histogram_transition",
    "shuffled_transition",
    "reversed_transition",
    "block_shuffled_transition",
    "ngram3",
    "ngram4",
    "run_length",
    "position_histogram",
    "spectrum",
    "coordinate_covariance",
    "coordinate_cooccurrence",
    "lagged_coordination",
    "raw_sequence",
    "histogram_ngram3",
    "histogram_run_length",
    "histogram_position",
    "histogram_spectrum",
    "histogram_cooccurrence",
    "histogram_coordination",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether FSQ style information comes from level occupancy or temporal transition patterns."
    )
    parser.add_argument("--token-database", type=Path, required=True)
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--max-windows-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--probe-model", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--hash-dim", type=int, default=4096)
    parser.add_argument("--position-bins", type=int, default=4)
    parser.add_argument("--spectrum-bins", type=int, default=16)
    parser.add_argument("--coordination-lags", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of representations evaluated concurrently; -1 uses all CPUs.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("threading", "loky"),
        default="threading",
        help="threading shares large token arrays; loky uses processes and more memory.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def selected_indices(length: int, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or maximum >= length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=maximum, replace=False))


def load_windows(dataset: FSQTokenDataset, maximum: int | None, seed: int) -> dict[str, np.ndarray]:
    selection = selected_indices(len(dataset), maximum, seed)
    indices = []
    style_ids = []
    action_ids = []
    for index in selection.tolist():
        item = dataset[index]
        indices.append(item["indices"].numpy().astype(np.uint8))
        style_ids.append(int(item["style_id"]))
        action_ids.append(int(item["action_id"]))
    if not indices:
        raise ValueError(f"Split {dataset.split} contains no token windows")
    return {
        "indices": np.stack(indices),
        "style_ids": np.asarray(style_ids, dtype=np.int64),
        "action_ids": np.asarray(action_ids, dtype=np.int64),
    }


def histogram_features(indices: np.ndarray, num_levels: int) -> np.ndarray:
    num_samples, seq_len, num_coordinates = indices.shape
    features = np.zeros((num_samples, num_coordinates, num_levels), dtype=np.float32)
    sample_offsets = np.arange(num_samples, dtype=np.int64)[:, None] * num_levels
    for coordinate in range(num_coordinates):
        values = indices[:, :, coordinate].astype(np.int64) + sample_offsets
        counts = np.bincount(values.reshape(-1), minlength=num_samples * num_levels)
        features[:, coordinate] = counts.reshape(num_samples, num_levels) / float(seq_len)
    return features.reshape(num_samples, -1)


def transition_features(indices: np.ndarray, num_levels: int) -> np.ndarray:
    num_samples, seq_len, num_coordinates = indices.shape
    features = np.zeros((num_samples, num_coordinates, num_levels * num_levels), dtype=np.float32)
    if seq_len < 2:
        return features.reshape(num_samples, -1)
    num_transitions = num_levels * num_levels
    sample_offsets = np.arange(num_samples, dtype=np.int64)[:, None] * num_transitions
    for coordinate in range(num_coordinates):
        transitions = (
            indices[:, :-1, coordinate].astype(np.int64) * num_levels
            + indices[:, 1:, coordinate].astype(np.int64)
        )
        counts = np.bincount(
            (transitions + sample_offsets).reshape(-1),
            minlength=num_samples * num_transitions,
        )
        features[:, coordinate] = counts.reshape(num_samples, num_transitions) / float(seq_len - 1)
    return features.reshape(num_samples, -1)


def shuffled_indices(indices: np.ndarray, seed: int) -> np.ndarray:
    shuffled = indices.copy()
    for sample in range(len(shuffled)):
        rng = np.random.default_rng(seed + sample)
        shuffled[sample] = shuffled[sample, rng.permutation(shuffled.shape[1])]
    return shuffled


def reversed_indices(indices: np.ndarray) -> np.ndarray:
    return indices[:, ::-1].copy()


def block_shuffled_indices(indices: np.ndarray, seed: int, block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    shuffled = indices.copy()
    for sample in range(len(shuffled)):
        blocks = [shuffled[sample, start : start + block_size].copy() for start in range(0, shuffled.shape[1], block_size)]
        order = np.random.default_rng(seed + sample).permutation(len(blocks))
        shuffled[sample] = np.concatenate([blocks[index] for index in order], axis=0)
    return shuffled


def hashed_ngram_features(indices: np.ndarray, num_levels: int, order: int, hash_dim: int) -> np.ndarray:
    num_samples, seq_len, num_coordinates = indices.shape
    if order < 2 or hash_dim <= 0:
        raise ValueError("ngram order must be >= 2 and hash_dim must be positive")
    features = np.zeros((num_samples, hash_dim), dtype=np.float32)
    if seq_len < order:
        return features
    sample_offsets = np.arange(num_samples, dtype=np.int64)[:, None] * hash_dim
    for coordinate in range(num_coordinates):
        keys = np.zeros((num_samples, seq_len - order + 1), dtype=np.int64)
        for offset in range(order):
            keys = keys * num_levels + indices[:, offset : offset + keys.shape[1], coordinate]
        buckets = (keys * 2654435761 + coordinate * 2246822519 + order * 3266489917) % hash_dim
        counts = np.bincount((buckets + sample_offsets).reshape(-1), minlength=num_samples * hash_dim)
        features += counts.reshape(num_samples, hash_dim)
    return features / float(num_coordinates * (seq_len - order + 1))


def run_length_features(indices: np.ndarray, num_levels: int) -> np.ndarray:
    num_samples, _, num_coordinates = indices.shape
    duration_edges = np.asarray([1, 2, 3, 4, 8, 16], dtype=np.int64)
    num_bins = len(duration_edges) + 1
    features = np.zeros((num_samples, num_coordinates, num_levels, num_bins), dtype=np.float32)
    for sample in range(num_samples):
        for coordinate in range(num_coordinates):
            sequence = indices[sample, :, coordinate]
            starts = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1]) + 1]
            ends = np.r_[starts[1:], len(sequence)]
            levels = sequence[starts]
            bins = np.searchsorted(duration_edges, ends - starts, side="left")
            np.add.at(features[sample, coordinate], (levels, bins), 1.0)
            total = features[sample, coordinate].sum()
            if total:
                features[sample, coordinate] /= total
    return features.reshape(num_samples, -1)


def position_histogram_features(indices: np.ndarray, num_levels: int, position_bins: int) -> np.ndarray:
    if position_bins <= 0 or position_bins > indices.shape[1]:
        raise ValueError("position_bins must be between 1 and sequence length")
    return np.concatenate(
        [histogram_features(indices[:, segment], num_levels) for segment in np.array_split(np.arange(indices.shape[1]), position_bins)],
        axis=1,
    )


def spectrum_features(indices: np.ndarray, spectrum_bins: int) -> np.ndarray:
    values = indices.astype(np.float32)
    values -= values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(values, axis=1))[:, 1:] ** 2
    num_bins = min(spectrum_bins, power.shape[1])
    power = power[:, :num_bins]
    normalized = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    frequencies = np.arange(1, num_bins + 1, dtype=np.float32)[None, :, None] / indices.shape[1]
    centroid = (normalized * frequencies).sum(axis=1)
    entropy = -(normalized * np.log(normalized + 1e-12)).sum(axis=1) / np.log(max(num_bins, 2))
    return np.concatenate([normalized.transpose(0, 2, 1).reshape(len(indices), -1), centroid, entropy], axis=1).astype(np.float32)


def coordinate_covariance_features(indices: np.ndarray, num_levels: int) -> np.ndarray:
    values = indices.astype(np.float32) / max(num_levels - 1, 1)
    values -= values.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntk,ntl->nkl", values, values) / max(indices.shape[1] - 1, 1)
    rows, cols = np.triu_indices(indices.shape[2])
    return covariance[:, rows, cols]


def coordinate_cooccurrence_features(indices: np.ndarray, num_levels: int, hash_dim: int) -> np.ndarray:
    """Hashed categorical same-frame co-occurrence for every coordinate pair."""
    num_samples, seq_len, num_coordinates = indices.shape
    features = np.zeros((num_samples, hash_dim), dtype=np.float32)
    sample_offsets = np.arange(num_samples, dtype=np.int64)[:, None] * hash_dim
    num_pairs = 0
    for left in range(num_coordinates):
        for right in range(left + 1, num_coordinates):
            keys = indices[:, :, left].astype(np.int64) * num_levels + indices[:, :, right]
            buckets = (keys * 2654435761 + left * 2246822519 + right * 3266489917) % hash_dim
            counts = np.bincount((buckets + sample_offsets).reshape(-1), minlength=num_samples * hash_dim)
            features += counts.reshape(num_samples, hash_dim)
            num_pairs += 1
    return features / float(max(num_pairs * seq_len, 1))


def lagged_coordination_features(indices: np.ndarray, lags: list[int]) -> np.ndarray:
    changes = (indices[:, 1:] != indices[:, :-1]).astype(np.float32)
    outputs = []
    for lag in lags:
        if lag < 0 or lag >= changes.shape[1]:
            continue
        left = changes[:, : changes.shape[1] - lag or None]
        right = changes[:, lag:]
        joint = np.einsum("ntk,ntl->nkl", left, right) / left.shape[1]
        expected = left.mean(axis=1)[:, :, None] * right.mean(axis=1)[:, None, :]
        outputs.append((joint - expected).reshape(len(indices), -1))
    if not outputs:
        raise ValueError("No coordination lag is shorter than the sequence")
    return np.concatenate(outputs, axis=1)


def raw_sequence_features(indices: np.ndarray, num_levels: int) -> np.ndarray:
    return np.eye(num_levels, dtype=np.float32)[indices].reshape(len(indices), -1)


def build_features(indices: np.ndarray, representation: str, num_levels: int, seed: int, args: argparse.Namespace) -> np.ndarray:
    if representation == "histogram":
        return histogram_features(indices, num_levels)
    if representation == "transition":
        return transition_features(indices, num_levels)
    if representation == "histogram_transition":
        return np.concatenate(
            [histogram_features(indices, num_levels), transition_features(indices, num_levels)],
            axis=1,
        )
    if representation == "shuffled_transition":
        return transition_features(shuffled_indices(indices, seed), num_levels)
    if representation == "reversed_transition":
        return transition_features(reversed_indices(indices), num_levels)
    if representation == "block_shuffled_transition":
        return transition_features(block_shuffled_indices(indices, seed, args.block_size), num_levels)
    if representation.startswith("ngram"):
        return hashed_ngram_features(indices, num_levels, int(representation[-1]), args.hash_dim)
    if representation == "run_length":
        return run_length_features(indices, num_levels)
    if representation == "position_histogram":
        return position_histogram_features(indices, num_levels, args.position_bins)
    if representation == "spectrum":
        return spectrum_features(indices, args.spectrum_bins)
    if representation == "coordinate_covariance":
        return coordinate_covariance_features(indices, num_levels)
    if representation == "coordinate_cooccurrence":
        return coordinate_cooccurrence_features(indices, num_levels, args.hash_dim)
    if representation == "lagged_coordination":
        return lagged_coordination_features(indices, args.coordination_lags)
    if representation == "raw_sequence":
        return raw_sequence_features(indices, num_levels)
    combined = {
        "histogram_ngram3": "ngram3",
        "histogram_run_length": "run_length",
        "histogram_position": "position_histogram",
        "histogram_spectrum": "spectrum",
        "histogram_cooccurrence": "coordinate_cooccurrence",
        "histogram_coordination": "lagged_coordination",
    }
    if representation in combined:
        return np.concatenate(
            [histogram_features(indices, num_levels), build_features(indices, combined[representation], num_levels, seed, args)],
            axis=1,
        )
    raise ValueError(f"Unsupported representation: {representation}")


def fit_probe(features: np.ndarray, labels: np.ndarray, args: argparse.Namespace):
    if args.probe_model == "linear":
        classifier = LogisticRegression(
            C=args.c,
            max_iter=args.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=args.seed,
        )
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(args.mlp_hidden_dim,),
            max_iter=args.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=args.seed,
        )
    return classifier.fit(features, labels)


def standard_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> float:
    classifier = fit_probe(train_features, train_labels, args)
    return float(balanced_accuracy_score(test_labels, classifier.predict(test_features)))


def held_out_action_probe(
    features: np.ndarray,
    style_ids: np.ndarray,
    action_ids: np.ndarray,
    action_names: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    scores = {}
    for action_id, action_name in enumerate(action_names):
        train_mask = action_ids != action_id
        test_mask = action_ids == action_id
        if not np.any(test_mask):
            continue
        train_styles = np.unique(style_ids[train_mask])
        test_styles = np.unique(style_ids[test_mask])
        if not np.all(np.isin(test_styles, train_styles)):
            scores[action_name] = None
            continue
        classifier = fit_probe(features[train_mask], style_ids[train_mask], args)
        scores[action_name] = float(
            balanced_accuracy_score(style_ids[test_mask], classifier.predict(features[test_mask]))
        )
    valid_scores = [score for score in scores.values() if score is not None]
    return {
        "per_action": scores,
        "mean": float(np.mean(valid_scores)) if valid_scores else None,
        "num_actions": len(valid_scores),
    }


def summarize_pattern_hypothesis(reports: list[dict[str, object]]) -> dict[str, float | str | None]:
    by_name = {report["representation"]: report for report in reports}

    def held_mean(name: str) -> float | None:
        report = by_name.get(name)
        if report is None:
            return None
        return report["style_held_out_action"]["mean"]

    histogram = held_mean("histogram")
    transition = held_mean("transition")
    combined = held_mean("histogram_transition")
    shuffled = held_mean("shuffled_transition")
    transition_gain = None if histogram is None or transition is None else transition - histogram
    combined_gain = None if histogram is None or combined is None else combined - histogram
    temporal_order_gain = None if transition is None or shuffled is None else transition - shuffled

    available = [value for value in (combined_gain, temporal_order_gain) if value is not None]
    if not available:
        interpretation = "insufficient_representations"
    elif max(available) < 0.01:
        interpretation = "marginal_occupancy_dominant"
    elif temporal_order_gain is not None and temporal_order_gain >= 0.03:
        interpretation = "strong_temporal_order_evidence"
    else:
        interpretation = "mixed_or_weak_temporal_evidence"
    return {
        "histogram_held_out_action_accuracy": histogram,
        "transition_minus_histogram": transition_gain,
        "combined_minus_histogram": combined_gain,
        "transition_minus_shuffled_transition": temporal_order_gain,
        "interpretation": interpretation,
    }


def analyze_representation(
    representation: str,
    split_data: dict[str, dict[str, np.ndarray]] | None,
    all_data: dict[str, np.ndarray],
    num_levels: int,
    action_names: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    all_features = build_features(all_data["indices"], representation, num_levels, args.seed + 2, args)
    if split_data is not None:
        train_features = build_features(
            split_data["train"]["indices"], representation, num_levels, args.seed, args
        )
        test_features = build_features(
            split_data["test"]["indices"], representation, num_levels, args.seed + 1, args
        )
        style_standard = standard_probe(
            train_features,
            split_data["train"]["style_ids"],
            test_features,
            split_data["test"]["style_ids"],
            args,
        )
        action_standard = standard_probe(
            train_features,
            split_data["train"]["action_ids"],
            test_features,
            split_data["test"]["action_ids"],
            args,
        )
    else:
        train_features = all_features
        style_standard = None
        action_standard = None
    return {
        "representation": representation,
        "feature_dim": int(train_features.shape[1]),
        "style_standard_split_balanced_accuracy": style_standard,
        "action_standard_split_balanced_accuracy": action_standard,
        "style_held_out_action": held_out_action_probe(
            all_features,
            all_data["style_ids"],
            all_data["action_ids"],
            action_names,
            args,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.max_windows_per_split is not None and args.max_windows_per_split <= 0:
        raise ValueError("--max-windows-per-split must be positive when provided")
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError("--n-jobs must be -1 or a positive integer")
    store = build_fsq_token_store(args.token_database)
    custom_windows = args.window_size is not None
    if custom_windows:
        stride = args.window_stride or max(args.window_size // 2, 1)
        windows = build_full_clip_windows(store, args.window_size, stride)
        full_data = load_windows(FSQTokenDataset("all", store, windows=windows), args.max_windows_per_split, args.seed)
        split_data = None
        all_data = full_data
    else:
        split_data = {
        split: load_windows(
            FSQTokenDataset(split, store),
            args.max_windows_per_split,
            args.seed + split_index * 100000,
        )
        for split_index, split in enumerate(("train", "val", "test"))
        }
        all_data = {
        key: np.concatenate([split_data[split][key] for split in ("train", "val", "test")], axis=0)
        for key in ("indices", "style_ids", "action_ids")
        }

    print(
        f"parallel_backend={args.parallel_backend} n_jobs={args.n_jobs} "
        f"num_representations={len(args.representations)}"
    )
    with threadpool_limits(limits=1):
        reports = Parallel(n_jobs=args.n_jobs, backend=args.parallel_backend)(
            delayed(analyze_representation)(
                representation,
                split_data,
                all_data,
                store.num_levels,
                store.action_names,
                args,
            )
            for representation in args.representations
        )
    for report in reports:
        print(json.dumps(report, indent=2))

    result = {
        "token_database": str(args.token_database),
        "checkpoint_path": store.checkpoint_path,
        "checkpoint_sha256": store.checkpoint_sha256,
        "num_coordinates": store.num_coordinates,
        "num_levels": store.num_levels,
        "style_names": store.style_names,
        "action_names": store.action_names,
        "analysis_window_size": int(all_data["indices"].shape[1]),
        "probe_model": args.probe_model,
        "parallel_backend": args.parallel_backend,
        "n_jobs": args.n_jobs,
        "standard_split_available": split_data is not None,
        "num_windows": ({split: len(split_data[split]["indices"]) for split in ("train", "val", "test")} if split_data is not None else {"all": len(all_data["indices"])}),
        "reports": reports,
        "hypothesis_summary": summarize_pattern_hypothesis(reports),
    }
    print("hypothesis_summary")
    print(json.dumps(result["hypothesis_summary"], indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
