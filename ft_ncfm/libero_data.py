"""Demo-disjoint access to one official LIBERO demonstration file."""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DemoSplit:
    train: tuple[str, ...]
    influence_reference: tuple[str, ...]
    selection_validation: tuple[str, ...]
    final_test: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "influence_reference": list(self.influence_reference),
            "selection_validation": list(self.selection_validation),
            "final_test": list(self.final_test),
        }


def _demo_number(name: str) -> int:
    try:
        return int(name.removeprefix("demo_"))
    except ValueError as error:
        raise ValueError(f"Invalid LIBERO demonstration key: {name}") from error


def split_demo_keys(
    demo_keys: Sequence[str],
    *,
    train_count: int,
    reference_count: int,
    selection_count: int,
    test_count: int,
    seed: int,
) -> DemoSplit:
    """Create exact, deterministic partitions without ever splitting a trajectory."""
    counts = (train_count, reference_count, selection_count, test_count)
    if any(count <= 0 for count in counts):
        raise ValueError("All demo split counts must be positive")
    ordered = sorted(set(demo_keys), key=_demo_number)
    if len(ordered) != len(demo_keys):
        raise ValueError("Demonstration keys must be unique")
    if sum(counts) != len(ordered):
        raise ValueError("Demo split counts must consume every demonstration exactly once")
    permutation = torch.randperm(
        len(ordered), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    shuffled = [ordered[index] for index in permutation]
    train_end = train_count
    reference_end = train_end + reference_count
    selection_end = reference_end + selection_count
    return DemoSplit(
        train=tuple(shuffled[:train_end]),
        influence_reference=tuple(shuffled[train_end:reference_end]),
        selection_validation=tuple(shuffled[reference_end:selection_end]),
        final_test=tuple(shuffled[selection_end:]),
    )


def tokenize_instruction(instruction: str, *, max_length: int = 80) -> Tensor:
    """Encode UTF-8 bytes with 0 reserved for padding and 257 for BOS."""
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    values = [257, *(byte + 1 for byte in instruction.encode("utf-8"))]
    values = values[:max_length]
    values.extend([0] * (max_length - len(values)))
    return torch.tensor(values, dtype=torch.long)


def process_libero_rgb(image: Any, *, flip_vertical: bool = True) -> Tensor:
    """Match LIBERO/robomimic's HWC uint8 to normalized CHW preprocessing."""
    tensor = torch.as_tensor(image).permute(2, 0, 1).contiguous().float() / 255.0
    if flip_vertical:
        tensor = torch.flip(tensor, dims=(-2,))
    return tensor


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_libero_hdf5(path: str | Path) -> dict[str, Any]:
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demo_keys = sorted(data.keys(), key=_demo_number)
        return {
            "demo_keys": demo_keys,
            "num_demos": len(demo_keys),
            "total_transitions": sum(int(data[key]["actions"].shape[0]) for key in demo_keys),
            "problem_info": str(data.attrs.get("problem_info", "")),
            "env_args": str(data.attrs.get("env_args", "")),
            "tag": str(data.attrs.get("tag", "")),
        }


def _require_h5py():
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "LIBERO HDF5 support requires h5py; install the project 'libero' extra"
        ) from error
    return h5py


class LiberoHDF5Dataset(Dataset):
    """Lazy transition dataset limited to complete, explicitly named demonstrations."""

    def __init__(
        self,
        path: str | Path,
        demo_keys: Sequence[str],
        *,
        instruction: str,
        instruction_length: int = 80,
        transition_stride: int = 1,
    ) -> None:
        if transition_stride <= 0:
            raise ValueError("transition_stride must be positive")
        self.path = str(Path(path).resolve())
        self.demo_keys = tuple(demo_keys)
        if not self.demo_keys:
            raise ValueError("At least one demonstration is required")
        self.instruction_tokens = tokenize_instruction(
            instruction, max_length=instruction_length
        )
        self._handle = None
        self._records: list[tuple[str, int, int]] = []
        h5py = _require_h5py()
        with h5py.File(self.path, "r") as handle:
            data = handle["data"]
            missing = sorted(set(self.demo_keys).difference(data.keys()))
            if missing:
                raise ValueError(f"Unknown demonstration keys: {missing}")
            for demo_index, key in enumerate(self.demo_keys):
                length = int(data[key]["actions"].shape[0])
                self._records.extend(
                    (key, step, demo_index) for step in range(0, length, transition_stride)
                )
        self._offsets = list(range(len(self._records) + 1))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._records)

    def _data(self):
        if self._handle is None:
            self._handle = _require_h5py().File(self.path, "r")
        return self._handle["data"]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Keep an explicit monotone-offset lookup so the sample identity remains
        # stable if this adapter later switches from stored records to ranges.
        record_index = bisect.bisect_right(self._offsets, index) - 1
        key, step, demo_index = self._records[record_index]
        demo = self._data()[key]
        obs = demo["obs"]
        agent = process_libero_rgb(obs["agentview_rgb"][step])
        wrist = process_libero_rgb(obs["eye_in_hand_rgb"][step])
        proprio = torch.cat(
            [
                torch.as_tensor(obs["gripper_states"][step], dtype=torch.float32),
                torch.as_tensor(obs["joint_states"][step], dtype=torch.float32),
            ]
        )
        return {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "demo_index": torch.tensor(demo_index, dtype=torch.long),
            "step_index": torch.tensor(step, dtype=torch.long),
            "image": torch.cat([agent, wrist], dim=0),
            "instruction": self.instruction_tokens.clone(),
            "proprio": proprio,
            "action": torch.as_tensor(demo["actions"][step], dtype=torch.float32),
        }
