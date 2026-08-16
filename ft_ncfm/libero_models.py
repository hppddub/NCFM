"""Compact two-camera language-conditioned policy for the LIBERO gate."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LiberoVLAPolicy(nn.Module):
    """Fuse two RGB cameras, byte-token language, and robot proprioception."""

    def __init__(
        self,
        *,
        representation_dim: int = 128,
        visual_dim: int = 96,
        language_dim: int = 32,
        proprio_dim: int = 32,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        if representation_dim <= 0:
            raise ValueError("representation_dim must be positive")
        self.representation_dim = representation_dim
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, visual_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # 128x128 -> 16x16 above; a fixed pool avoids CUDA adaptive-pool
            # nondeterminism and leaves one visual token.
            nn.AvgPool2d(kernel_size=16),
            nn.Flatten(),
        )
        self.language_embedding = nn.Embedding(258, language_dim, padding_idx=0)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(9, proprio_dim),
            nn.LayerNorm(proprio_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + language_dim + proprio_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(representation_dim, action_dim)

    def _language_features(self, instruction: Tensor) -> Tensor:
        embedded = self.language_embedding(instruction.long())
        mask = instruction.ne(0).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        return (embedded * mask).sum(dim=1) / count

    def encode(self, image: Tensor, instruction: Tensor, proprio: Tensor) -> Tensor:
        visual = self.visual_encoder(image)
        language = self._language_features(instruction)
        state = self.proprio_encoder(proprio)
        return self.fusion(torch.cat([visual, language, state], dim=-1))

    def forward(self, image: Tensor, instruction: Tensor, proprio: Tensor) -> Tensor:
        return self.action_head(self.encode(image, instruction, proprio))

    def freeze_encoders_for_influence(self) -> None:
        """Bound IHVP cost while retaining the trained fusion and action head."""
        for module in (
            self.visual_encoder,
            self.language_embedding,
            self.proprio_encoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def libero_policy_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    prediction = model(batch["image"], batch["instruction"], batch["proprio"])
    return torch.nn.functional.mse_loss(prediction, batch["action"])
