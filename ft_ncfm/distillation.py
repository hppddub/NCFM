"""A small influence-weighted NCFM adversarial game for proof-of-concept runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from NCFM.NCFM import CFLossFunc


class FrequencyNet(nn.Module):
    """Three-layer auxiliary network matching the FT-NCFM appendix description."""

    def __init__(self, feature_dim: int, num_frequencies: int, hidden_dim: int = 32):
        super().__init__()
        width = max(feature_dim, hidden_dim)
        self.register_buffer("noise", torch.randn(num_frequencies, feature_dim))
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.LayerNorm(width),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(width, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def forward(self) -> Tensor:
        return self.layers(self.noise)


@dataclass(frozen=True)
class DistillationStep:
    step: int
    discriminator_loss: float
    generator_loss: float


@dataclass(frozen=True)
class ConvergenceSummary:
    initial_median: float
    final_median: float
    finite: bool
    converged: bool
    window: int


class WeightedNCFMDistiller:
    """Optimize synthetic features against an influence-weighted real CF.

    This feature-level implementation is an algorithmic gate. A benchmark adapter
    must optimize modality tensors `(V, L, A)` before downstream VLA claims can be
    compared with the paper.
    """

    def __init__(
        self,
        real_features: Tensor,
        real_weights: Tensor | None,
        *,
        synthetic_count: int,
        num_frequencies: int = 64,
        hidden_dim: int = 32,
        generator_lr: float = 0.03,
        discriminator_lr: float = 0.001,
        alpha: float = 0.5,
        beta: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        if real_features.ndim != 2 or real_features.shape[0] < 2:
            raise ValueError("real_features must be an N x D tensor with N >= 2")
        if synthetic_count <= 0:
            raise ValueError("synthetic_count must be positive")
        self.device = torch.device(device)
        self.real_features = F.normalize(real_features.detach().to(self.device), dim=1)
        if real_weights is None:
            weights = torch.full(
                (real_features.shape[0],),
                1.0 / real_features.shape[0],
                device=self.device,
            )
        else:
            weights = real_weights.detach().flatten().to(self.device)
            if weights.numel() != real_features.shape[0]:
                raise ValueError("One real weight is required per real feature")
            if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Real weights must be finite, non-negative, and nonzero")
            weights = weights / weights.sum()
        self.real_weights = weights

        feature_dim = real_features.shape[1]
        center = self.real_features.mean(dim=0, keepdim=True)
        scale = self.real_features.std(dim=0, keepdim=True).clamp_min(0.05)
        initial = center + torch.randn(synthetic_count, feature_dim, device=self.device) * scale
        self.synthetic_features = nn.Parameter(initial)
        self.frequency_net = FrequencyNet(feature_dim, num_frequencies, hidden_dim).to(self.device)
        self.cf_loss = CFLossFunc(alpha_for_loss=alpha, beta_for_loss=beta)
        self.generator_optimizer = torch.optim.Adam([self.synthetic_features], lr=generator_lr)
        self.discriminator_optimizer = torch.optim.Adam(
            self.frequency_net.parameters(), lr=discriminator_lr
        )

    def _loss(self, *, detach_synthetic: bool, detach_frequencies: bool) -> Tensor:
        synthetic = (
            self.synthetic_features.detach() if detach_synthetic else self.synthetic_features
        )
        synthetic = F.normalize(synthetic, dim=1)
        frequencies = self.frequency_net()
        if detach_frequencies:
            frequencies = frequencies.detach()
        return self.cf_loss(
            self.real_features,
            synthetic,
            frequencies,
            args=None,
            target_weights=self.real_weights,
        )

    def step(self, step: int, *, discriminator_steps: int = 1) -> DistillationStep:
        if discriminator_steps <= 0:
            raise ValueError("discriminator_steps must be positive")
        discriminator_loss = torch.tensor(float("nan"), device=self.device)
        for _ in range(discriminator_steps):
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss = self._loss(detach_synthetic=True, detach_frequencies=False)
            (-discriminator_loss).backward()
            self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad(set_to_none=True)
        generator_loss = self._loss(detach_synthetic=False, detach_frequencies=True)
        generator_loss.backward()
        self.generator_optimizer.step()

        return DistillationStep(
            step=step,
            discriminator_loss=float(discriminator_loss.detach().cpu()),
            generator_loss=float(generator_loss.detach().cpu()),
        )

    def run(self, steps: int, *, discriminator_steps: int = 1) -> list[DistillationStep]:
        if steps <= 1:
            raise ValueError("At least two distillation steps are required")
        trace = [self.step(step, discriminator_steps=discriminator_steps) for step in range(steps)]
        values = torch.tensor([record.generator_loss for record in trace])
        if not torch.isfinite(values).all():
            raise FloatingPointError("NCFM generator loss became non-finite")
        return trace

    def coreset(self) -> Tensor:
        return F.normalize(self.synthetic_features.detach(), dim=1).cpu()


def summarize_convergence(
    trace: list[DistillationStep], *, window_ratio: float = 0.1
) -> ConvergenceSummary:
    if len(trace) < 2:
        raise ValueError("A convergence trace requires at least two records")
    if not 0 < window_ratio <= 0.5:
        raise ValueError("window_ratio must be in (0, 0.5]")
    values = torch.tensor([record.generator_loss for record in trace])
    window = max(1, math.ceil(len(trace) * window_ratio))
    initial = float(values[:window].median())
    final = float(values[-window:].median())
    finite = bool(torch.isfinite(values).all())
    return ConvergenceSummary(
        initial_median=initial,
        final_median=final,
        finite=finite,
        converged=finite and final < initial,
        window=window,
    )


def trace_as_dicts(trace: list[DistillationStep]) -> list[dict[str, float | int]]:
    return [asdict(record) for record in trace]
