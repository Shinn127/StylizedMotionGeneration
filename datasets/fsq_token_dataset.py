from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FSQTokenWindow:
    shard_idx: int
    start_idx: int
    end_idx: int
    range_idx: int


@dataclass
class FSQTokenStore:
    database: Path
    token_files: list[Path]
    code_files: list[Path | None]
    range_names: np.ndarray
    range_mirror: np.ndarray
    style_names: list[str]
    action_names: list[str]
    style_ids: np.ndarray
    action_ids: np.ndarray
    num_frames: np.ndarray
    window_size: int
    num_coordinates: int
    num_levels: int
    split_windows: dict[str, list[FSQTokenWindow]]
    checkpoint_path: str
    checkpoint_sha256: str
    feature_database: str


def _load_windows(values: np.ndarray) -> list[FSQTokenWindow]:
    array = np.asarray(values, dtype=np.int32)
    return [
        FSQTokenWindow(
            shard_idx=int(row[0]),
            start_idx=int(row[1]),
            end_idx=int(row[2]),
            range_idx=int(row[3]),
        )
        for row in array
    ]


def build_fsq_token_store(database: str | Path) -> FSQTokenStore:
    database = Path(database)
    metadata_path = database / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing FSQ token metadata: {metadata_path}")
    npz = np.load(metadata_path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}

    token_files = [database / str(path) for path in np.asarray(data["token_files"], dtype=object).tolist()]
    raw_code_files = np.asarray(data["code_files"], dtype=object).tolist()
    code_files = [database / str(path) if str(path) else None for path in raw_code_files]
    range_names = np.asarray(data["range_names"], dtype=object)
    style_ids = np.asarray(data["style_ids"], dtype=np.int32)
    action_ids = np.asarray(data["action_ids"], dtype=np.int32)
    if not (len(token_files) == len(range_names) == len(style_ids) == len(action_ids)):
        raise ValueError("Token shard metadata arrays must have matching lengths")

    return FSQTokenStore(
        database=database,
        token_files=token_files,
        code_files=code_files,
        range_names=range_names,
        range_mirror=np.asarray(data["range_mirror"], dtype=bool),
        style_names=[str(name) for name in np.asarray(data["style_names"], dtype=object).tolist()],
        action_names=[str(name) for name in np.asarray(data["action_names"], dtype=object).tolist()],
        style_ids=style_ids,
        action_ids=action_ids,
        num_frames=np.asarray(data["num_frames"], dtype=np.int32),
        window_size=int(np.asarray(data["window_size"]).item()),
        num_coordinates=int(np.asarray(data["num_coordinates"]).item()),
        num_levels=int(np.asarray(data["num_levels"]).item()),
        split_windows={
            split: _load_windows(data[f"{split}_windows"])
            for split in ("train", "val", "test")
        },
        checkpoint_path=str(np.asarray(data["checkpoint_path"], dtype=object).item()),
        checkpoint_sha256=str(np.asarray(data["checkpoint_sha256"], dtype=object).item()),
        feature_database=str(np.asarray(data["feature_database"], dtype=object).item()),
    )


def build_full_clip_windows(
    store: FSQTokenStore,
    window_size: int,
    stride: int,
) -> list[FSQTokenWindow]:
    """Re-window complete token shards for analyses beyond the training window size."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    windows: list[FSQTokenWindow] = []
    for shard_idx, num_frames in enumerate(store.num_frames.tolist()):
        num_frames = int(num_frames)
        if num_frames < window_size:
            continue
        starts = list(range(0, num_frames - window_size + 1, stride))
        tail_start = num_frames - window_size
        if starts[-1] != tail_start:
            starts.append(tail_start)
        windows.extend(
            FSQTokenWindow(shard_idx, start, start + window_size, shard_idx)
            for start in starts
        )
    return windows


class FSQTokenDataset(Dataset):
    def __init__(
        self,
        split: str,
        store: FSQTokenStore,
        include_codes: bool = False,
        windows: list[FSQTokenWindow] | None = None,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported token split: {split}")
        self.store = store
        self.split = split
        self.include_codes = include_codes
        if windows is not None:
            self.windows = windows
        elif split == "all":
            self.windows = [
                window
                for split_name in ("train", "val", "test")
                for window in store.split_windows[split_name]
            ]
        else:
            self.windows = store.split_windows[split]
        if include_codes and any(path is None for path in store.code_files):
            raise ValueError("Token database does not contain quantized float codes")
        self._indices: list[np.ndarray] | None = None
        self._codes: list[np.ndarray | None] | None = None

    def _ensure_open(self) -> None:
        if self._indices is not None:
            return
        self._indices = [np.load(path, mmap_mode="r") for path in self.store.token_files]
        self._codes = [
            np.load(path, mmap_mode="r") if path is not None else None
            for path in self.store.code_files
        ]
        for shard_idx, indices in enumerate(self._indices):
            expected = (int(self.store.num_frames[shard_idx]), self.store.num_coordinates)
            if indices.shape != expected:
                raise ValueError(f"Token shard {shard_idx} has shape {indices.shape}, expected {expected}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        self._ensure_open()
        window = self.windows[index]
        indices = np.asarray(
            self._indices[window.shard_idx][window.start_idx : window.end_idx],
            dtype=np.int64,
        ).copy()
        item: dict[str, torch.Tensor | int | str | bool] = {
            "indices": torch.from_numpy(indices),
            "style_id": int(self.store.style_ids[window.range_idx]),
            "action_id": int(self.store.action_ids[window.range_idx]),
            "style_name": self.store.style_names[int(self.store.style_ids[window.range_idx])],
            "action_name": self.store.action_names[int(self.store.action_ids[window.range_idx])],
            "range_name": str(self.store.range_names[window.range_idx]),
            "mirror": bool(self.store.range_mirror[window.range_idx]),
            "shard_idx": window.shard_idx,
            "start_idx": window.start_idx,
            "end_idx": window.end_idx,
        }
        if self.include_codes:
            codes = np.asarray(
                self._codes[window.shard_idx][window.start_idx : window.end_idx],
                dtype=np.float32,
            ).copy()
            item["codes"] = torch.from_numpy(codes)
        return item
