"""Common downstream action-head training for comparable coreset evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ActionRegressor(nn.Module):
    """Small action head trained identically for every real or synthetic coreset."""

    def __init__(self, representation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, representation: Tensor) -> Tensor:
        return self.layers(representation)


@dataclass(frozen=True)
class DownstreamSummary:
    initial_train_loss: float
    final_train_loss: float
    selection_mse: float
    test_mse: float | None
    finite: bool
    steps: int


def _split_joint_features(joint_features: Tensor, representation_dim: int) -> tuple[Tensor, Tensor]:
    if joint_features.ndim != 2:
        raise ValueError("joint_features must be a two-dimensional tensor")
    if not 0 < representation_dim < joint_features.shape[1]:
        raise ValueError("representation_dim must leave at least one action dimension")
    if not torch.isfinite(joint_features).all():
        raise ValueError("joint_features must be finite")
    return joint_features[:, :representation_dim], joint_features[:, representation_dim:]


def evaluate_action_mse(
    model: nn.Module,
    joint_features: Tensor,
    *,
    representation_dim: int,
    device: torch.device | str,
) -> float:
    representations, actions = _split_joint_features(joint_features, representation_dim)
    model.eval()
    with torch.no_grad():
        prediction = model(representations.to(device))
        loss = F.mse_loss(prediction, actions.to(device))
    return float(loss.cpu())


def train_action_regressor(
    train_joint_features: Tensor,
    selection_joint_features: Tensor,
    *,
    representation_dim: int,
    hidden_dim: int = 64,
    steps: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device | str = "cpu",
    test_joint_features: Tensor | None = None,
) -> tuple[ActionRegressor, DownstreamSummary, list[dict[str, float | int]]]:
    """Train one fixed-protocol action head and score isolated held-out features."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    train_representations, train_actions = _split_joint_features(
        train_joint_features, representation_dim
    )
    _, selection_actions = _split_joint_features(selection_joint_features, representation_dim)
    action_dim = train_actions.shape[1]
    if selection_actions.shape[1] != action_dim:
        raise ValueError("Train and selection action dimensions must match")

    run_device = torch.device(device)
    devices = [run_device.index or 0] if run_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = ActionRegressor(representation_dim, action_dim, hidden_dim).to(run_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)
    trace: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        indices = torch.randint(
            train_representations.shape[0],
            (min(batch_size, train_representations.shape[0]),),
            generator=generator,
        )
        prediction = model(train_representations[indices].to(run_device))
        loss = F.mse_loss(prediction, train_actions[indices].to(run_device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.append({"step": step, "train_loss": float(loss.detach().cpu())})

    selection_mse = evaluate_action_mse(
        model,
        selection_joint_features,
        representation_dim=representation_dim,
        device=run_device,
    )
    test_mse = (
        evaluate_action_mse(
            model,
            test_joint_features,
            representation_dim=representation_dim,
            device=run_device,
        )
        if test_joint_features is not None
        else None
    )
    values = torch.tensor([record["train_loss"] for record in trace])
    summary = DownstreamSummary(
        initial_train_loss=float(values[: max(1, steps // 10)].median()),
        final_train_loss=float(values[-max(1, steps // 10) :].median()),
        selection_mse=selection_mse,
        test_mse=test_mse,
        finite=bool(torch.isfinite(values).all())
        and math.isfinite(selection_mse)
        and (test_mse is None or math.isfinite(test_mse)),
        steps=steps,
    )
    return model, summary, trace


def downstream_summary_dict(summary: DownstreamSummary) -> dict[str, float | int | bool | None]:
    return asdict(summary)
