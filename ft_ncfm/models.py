"""Small guide-policy components for the public-data algorithmic proxy."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MiniVLAPolicy(nn.Module):
    """Fuse a visual observation and instruction token to predict an action."""

    def __init__(self, hidden_dim: int = 32, num_instructions: int = 10, action_dim: int = 3):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # The preceding convolutions map the fixed 28x28 MNIST proxy input
            # to 7x7.  A 4x4/stride-3 average pool is exactly equivalent to
            # adaptive 2x2 pooling for that shape, while retaining a
            # deterministic CUDA backward path on the paper-class A100 runtime.
            nn.AvgPool2d(kernel_size=4, stride=3),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, hidden_dim),
        )
        language_dim = max(4, hidden_dim // 4)
        self.language_encoder = nn.Embedding(num_instructions, language_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + language_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode(self, image: Tensor, instruction: Tensor) -> Tensor:
        visual = self.visual_encoder(image)
        language = self.language_encoder(instruction.long())
        return self.fusion(torch.cat([visual, language], dim=-1))

    def forward(self, image: Tensor, instruction: Tensor) -> Tensor:
        return self.action_head(self.encode(image, instruction))
