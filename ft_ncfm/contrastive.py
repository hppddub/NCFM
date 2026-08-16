"""Contrastive verification utilities corresponding to FT-NCFM equations 3-5."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def refine_influence_weights(
    base_scores: Tensor,
    original_scores: Tensor,
    counterexample_scores: Tensor,
    *,
    beta: float = 1.0,
) -> Tensor:
    """Apply Eq. 5 from the paper without hiding sign or normalization choices."""
    if (
        base_scores.shape != original_scores.shape
        or base_scores.shape != counterexample_scores.shape
    ):
        raise ValueError("Base, original, and counterexample scores must have identical shapes")
    modulation = 1.0 + torch.tanh(beta * (original_scores - counterexample_scores))
    return base_scores * modulation


def normalize_influence_weights(
    raw_weights: Tensor,
    *,
    policy: str = "positive_shift",
    epsilon: float = 1e-8,
) -> Tensor:
    """Convert underspecified raw paper weights into valid sampling probabilities.

    The paper only says to normalize weights to sum to one. Eq. 2 can produce
    negative values, so the transformation is explicit and recorded as an
    experimental policy rather than silently clamping values.
    """
    if raw_weights.ndim != 1:
        raise ValueError("Influence weights must be a one-dimensional tensor")
    if raw_weights.numel() == 0:
        raise ValueError("Influence weights cannot be empty")
    if not torch.isfinite(raw_weights).all():
        raise ValueError("Influence weights must be finite")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if policy == "positive_shift":
        transformed = raw_weights - raw_weights.min() + epsilon
    elif policy == "clamp":
        transformed = raw_weights.clamp_min(0) + epsilon
    elif policy == "softmax":
        transformed = torch.softmax(raw_weights, dim=0)
    else:
        raise ValueError(f"Unknown influence normalization policy: {policy}")

    return transformed / transformed.sum()


def gradient_dot_score(
    reference_gradient: Sequence[Tensor], sample_gradient: Sequence[Tensor]
) -> Tensor:
    """Compute the gradient alignment used by paper equations 3 and 4."""
    if len(reference_gradient) != len(sample_gradient):
        raise ValueError("Gradient tuples must contain the same number of tensors")
    return sum(
        (ref.detach() * sample.detach()).sum()
        for ref, sample in zip(reference_gradient, sample_gradient, strict=True)
    )


class VisionPerturbationLibrary:
    """Deterministic visual-only counterexamples for the MNIST VLA proxy.

    Language and action fields are copied unchanged by :meth:`apply_to_batch`.
    Benchmark adapters should replace these tensor operations with simulator APIs.
    """

    names = ("object_substitution", "size_scaling", "position_change")

    @staticmethod
    def object_substitution(image: Tensor) -> Tensor:
        return 1.0 - image

    @staticmethod
    def size_scaling(image: Tensor) -> Tensor:
        if image.ndim != 3:
            raise ValueError("Expected a single CHW image")
        height, width = image.shape[-2:]
        scaled_height = max(1, int(round(height * 0.55)))
        scaled_width = max(1, int(round(width * 0.55)))
        scaled = F.interpolate(
            image.unsqueeze(0),
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        pad_left = (width - scaled_width) // 2
        pad_right = width - scaled_width - pad_left
        pad_top = (height - scaled_height) // 2
        pad_bottom = height - scaled_height - pad_top
        return F.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom))

    @staticmethod
    def position_change(image: Tensor) -> Tensor:
        if image.ndim != 3:
            raise ValueError("Expected a single CHW image")
        shift_y = max(1, image.shape[-2] // 4)
        shift_x = max(1, image.shape[-1] // 4)
        shifted = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
        shifted[..., :shift_y, :] = 0
        shifted[..., :, :shift_x] = 0
        return shifted

    def apply(self, image: Tensor, template_name: str) -> Tensor:
        if template_name not in self.names:
            raise ValueError(f"Unknown perturbation template: {template_name}")
        return getattr(self, template_name)(image)

    def apply_to_batch(
        self,
        batch: Mapping[str, Tensor],
        template_names: Sequence[str] | None = None,
    ) -> dict[str, Tensor]:
        images = batch["image"]
        if images.ndim != 4:
            raise ValueError("Expected a BCHW image batch")
        if template_names is None:
            template_names = [self.names[i % len(self.names)] for i in range(len(images))]
        if len(template_names) != len(images):
            raise ValueError("One perturbation template is required per image")
        counterexample = {key: value.clone() for key, value in batch.items()}
        counterexample["image"] = torch.stack(
            [self.apply(image, name) for image, name in zip(images, template_names, strict=True)]
        )
        return counterexample


@dataclass(frozen=True)
class ContrastiveResult:
    elite_indices: Tensor
    original_scores: Tensor
    counterexample_scores: Tensor
    raw_refined_weights: Tensor
    normalized_weights: Tensor
    template_names: tuple[str, ...]


class ContrastiveVerifier:
    """Refine only the top-K base-influence samples as specified by the paper."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[nn.Module, Mapping[str, Tensor]], Tensor],
        *,
        beta: float = 1.0,
        elite_ratio: float = 0.05,
        ranking: str = "largest",
        normalization: str = "positive_shift",
        epsilon: float = 1e-8,
        perturbations: VisionPerturbationLibrary | None = None,
    ) -> None:
        if not 0 < elite_ratio <= 1:
            raise ValueError("elite_ratio must be in (0, 1]")
        if ranking not in {"largest", "smallest"}:
            raise ValueError("ranking must be 'largest' or 'smallest'")
        self.model = model
        self.loss_fn = loss_fn
        self.beta = beta
        self.elite_ratio = elite_ratio
        self.ranking = ranking
        self.normalization = normalization
        self.epsilon = epsilon
        self.perturbations = perturbations or VisionPerturbationLibrary()
        self.parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        self.device = self.parameters[0].device

    def _gradient(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, ...]:
        prepared = {key: value.to(self.device) for key, value in batch.items()}
        self.model.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model, prepared)
        gradients = torch.autograd.grad(loss, self.parameters)
        return tuple(gradient.detach() for gradient in gradients)

    def verify(
        self,
        base_scores: Tensor,
        sample_batches: Sequence[Mapping[str, Tensor]],
        reference_gradient: Sequence[Tensor],
    ) -> ContrastiveResult:
        if base_scores.ndim != 1 or len(sample_batches) != base_scores.numel():
            raise ValueError("A base score and sample batch are required for every sample")
        elite_count = max(1, math.ceil(base_scores.numel() * self.elite_ratio))
        elite_indices = torch.topk(
            base_scores,
            k=elite_count,
            largest=self.ranking == "largest",
        ).indices
        original_scores = torch.empty(elite_count, device=base_scores.device)
        counterexample_scores = torch.empty_like(original_scores)
        template_names: list[str] = []

        for output_index, sample_index in enumerate(elite_indices.tolist()):
            sample = sample_batches[sample_index]
            template_name = self.perturbations.names[output_index % len(self.perturbations.names)]
            counterexample = self.perturbations.apply_to_batch(sample, [template_name])
            original_scores[output_index] = gradient_dot_score(
                reference_gradient, self._gradient(sample)
            )
            counterexample_scores[output_index] = gradient_dot_score(
                reference_gradient, self._gradient(counterexample)
            )
            template_names.append(template_name)

        raw_refined = base_scores.clone()
        raw_refined[elite_indices] = refine_influence_weights(
            base_scores[elite_indices],
            original_scores,
            counterexample_scores,
            beta=self.beta,
        )
        normalized = normalize_influence_weights(
            raw_refined, policy=self.normalization, epsilon=self.epsilon
        )
        return ContrastiveResult(
            elite_indices=elite_indices,
            original_scores=original_scores,
            counterexample_scores=counterexample_scores,
            raw_refined_weights=raw_refined,
            normalized_weights=normalized,
            template_names=tuple(template_names),
        )
