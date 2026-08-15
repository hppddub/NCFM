"""Public MNIST adapter that exposes vision-language-action-shaped samples."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset


def image_to_action(image: Tensor) -> Tensor:
    """Map a digit image to a continuous 2-D target and a grip-like density value."""
    if image.ndim != 3:
        raise ValueError("Expected a CHW image")
    pixels = image.float().clamp_min(0)
    height, width = pixels.shape[-2:]
    mass = pixels.sum().clamp_min(1e-8)
    y_coordinates = torch.linspace(-1.0, 1.0, height, dtype=pixels.dtype)
    x_coordinates = torch.linspace(-1.0, 1.0, width, dtype=pixels.dtype)
    y_center = (pixels * y_coordinates.view(1, height, 1)).sum() / mass
    x_center = (pixels * x_coordinates.view(1, 1, width)).sum() / mass
    density = pixels.mean()
    return torch.stack([x_center, y_center, density])


class MiniVLADataset(Dataset):
    """Adapt MNIST into `(V, L, A)` records without pretending it is robotics data."""

    def __init__(self, mnist_dataset: Dataset):
        self.dataset = mnist_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, label = self.dataset[index]
        if not isinstance(image, Tensor):
            raise TypeError("MNIST must be configured with transforms.ToTensor()")
        return {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "image": image,
            "instruction": torch.tensor(label, dtype=torch.long),
            "action": image_to_action(image),
        }


def stratified_subset_indices(
    targets: Sequence[int] | Tensor,
    ratio: float,
    *,
    seed: int,
) -> list[int]:
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    desired_total = max(1, round(target_tensor.numel() * ratio))
    labels = torch.unique(target_tensor, sorted=True)
    class_indices = [torch.where(target_tensor == label)[0] for label in labels.tolist()]
    exact_counts = [indices.numel() * ratio for indices in class_indices]
    counts = [math.floor(value) for value in exact_counts]
    remainder = desired_total - sum(counts)
    remainder_order = sorted(
        range(len(counts)),
        key=lambda index: exact_counts[index] - counts[index],
        reverse=True,
    )
    for index in remainder_order[:remainder]:
        counts[index] += 1

    selected: list[int] = []
    for indices, count in zip(class_indices, counts, strict=True):
        count = max(1, count)
        permutation = torch.randperm(indices.numel(), generator=generator)
        selected.extend(indices[permutation[:count]].tolist())
    return sorted(selected)


def load_mnist_minivla(
    cache_dir: str,
    *,
    train_ratio: float,
    validation_size: int,
    seed: int,
    download: bool = True,
) -> tuple[Subset, Subset]:
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_base = datasets.MNIST(cache_dir, train=True, transform=transform, download=download)
    validation_base = datasets.MNIST(cache_dir, train=False, transform=transform, download=download)
    train_dataset = MiniVLADataset(train_base)
    validation_dataset = MiniVLADataset(validation_base)
    train_indices = stratified_subset_indices(train_base.targets, train_ratio, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    validation_count = min(validation_size, len(validation_base))
    validation_indices = torch.randperm(len(validation_base), generator=generator)[
        :validation_count
    ].tolist()
    return Subset(train_dataset, train_indices), Subset(validation_dataset, validation_indices)
