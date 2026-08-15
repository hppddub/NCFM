"""Influence-aware sampling utilities for Algorithm 1 line 24."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Sampler


class InfluenceWeightedSampler(Sampler[int]):
    """Sample indices from a fixed, normalized influence distribution."""

    def __init__(
        self,
        probabilities: Tensor,
        *,
        num_samples: int | None = None,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        probabilities = torch.as_tensor(probabilities, dtype=torch.float64).flatten().cpu()
        if probabilities.numel() == 0:
            raise ValueError("probabilities cannot be empty")
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("probabilities must be finite and non-negative")
        if probabilities.sum() <= 0:
            raise ValueError("at least one probability must be positive")
        self.probabilities = probabilities / probabilities.sum()
        self.num_samples = num_samples or probabilities.numel()
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        indices = torch.multinomial(
            self.probabilities,
            self.num_samples,
            replacement=self.replacement,
            generator=self.generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
