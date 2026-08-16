from __future__ import annotations

import numpy as np
import pytest
import torch

from ft_ncfm.libero_data import (
    LiberoHDF5Dataset,
    process_libero_rgb,
    split_demo_keys,
    tokenize_instruction,
)


def test_demo_split_is_exact_reproducible_and_disjoint() -> None:
    keys = [f"demo_{index}" for index in range(50)]
    first = split_demo_keys(
        keys,
        train_count=35,
        reference_count=5,
        selection_count=5,
        test_count=5,
        seed=42,
    )
    second = split_demo_keys(
        list(reversed(keys)),
        train_count=35,
        reference_count=5,
        selection_count=5,
        test_count=5,
        seed=42,
    )
    assert first == second
    partitions = [
        set(first.train),
        set(first.influence_reference),
        set(first.selection_validation),
        set(first.final_test),
    ]
    assert [len(partition) for partition in partitions] == [35, 5, 5, 5]
    assert set.union(*partitions) == set(keys)
    assert sum(len(partition) for partition in partitions) == len(set.union(*partitions))


def test_rgb_processing_flips_vertical_axis() -> None:
    image = np.zeros((2, 1, 3), dtype=np.uint8)
    image[0, 0] = [255, 128, 0]
    tensor = process_libero_rgb(image)
    assert tensor.shape == (3, 2, 1)
    torch.testing.assert_close(tensor[:, 1, 0], torch.tensor([1.0, 128 / 255, 0.0]))


def test_instruction_tokens_are_padded_and_bounded() -> None:
    tokens = tokenize_instruction("pick bowl", max_length=12)
    assert tokens.shape == (12,)
    assert tokens[0].item() == 257
    assert tokens.max().item() <= 257


def test_hdf5_dataset_preserves_demo_and_step_identity(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_3")
        demo.create_dataset("actions", data=np.ones((2, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.ones((2, 4, 4, 3), dtype=np.uint8))
        obs.create_dataset("gripper_states", data=np.zeros((2, 2), dtype=np.float32))
        obs.create_dataset("joint_states", data=np.zeros((2, 7), dtype=np.float32))

    dataset = LiberoHDF5Dataset(path, ["demo_3"], instruction="pick bowl")
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["sample_id"].item() == 1
    assert sample["demo_index"].item() == 0
    assert sample["step_index"].item() == 1
    assert sample["image"].shape == (6, 4, 4)
    assert sample["proprio"].shape == (9,)
    assert sample["action"].shape == (7,)
    dataset.close()
