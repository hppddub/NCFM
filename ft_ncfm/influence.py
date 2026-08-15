"""Influence-function prescreening with a memory-efficient LiSSA IHVP."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

Batch = Mapping[str, Tensor]
LossFunction = Callable[[nn.Module, Batch], Tensor]


def move_batch_to_device(batch: Batch, device: torch.device | str) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _replace_missing_gradients(
    gradients: Sequence[Tensor | None], parameters: Sequence[nn.Parameter]
) -> tuple[Tensor, ...]:
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(gradients, parameters, strict=True)
    )


def parameter_gradient(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    create_graph: bool = False,
) -> tuple[Tensor, ...]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    return _replace_missing_gradients(gradients, parameters)


def tuple_dot(left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
    if len(left) != len(right):
        raise ValueError("Tensor tuples must have identical lengths")
    return sum(
        (left_item * right_item).sum() for left_item, right_item in zip(left, right, strict=True)
    )


def hessian_vector_product(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    vector: Sequence[Tensor],
) -> tuple[Tensor, ...]:
    first_gradient = parameter_gradient(loss, parameters, create_graph=True)
    directional_derivative = tuple_dot(first_gradient, vector)
    second_gradient = torch.autograd.grad(
        directional_derivative,
        parameters,
        allow_unused=True,
    )
    return _replace_missing_gradients(second_gradient, parameters)


def _infinite_batches(batches: Iterable[Batch]) -> Iterator[Batch]:
    while True:
        observed = False
        for batch in batches:
            observed = True
            yield batch
        if not observed:
            raise ValueError("LiSSA requires at least one Hessian batch")


def lissa_ihvp(
    model: nn.Module,
    reference_gradient: Sequence[Tensor],
    hessian_batches: Iterable[Batch],
    loss_fn: LossFunction,
    *,
    depth: int = 50,
    damping: float = 0.01,
    scale: float = 10.0,
    repeats: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, ...]:
    """Approximate ``H^-1 v`` using the stochastic LiSSA recurrence.

    `scale` and `repeats` are not specified by the FT-NCFM paper and therefore
    remain explicit configuration values.
    """
    if depth <= 0 or repeats <= 0:
        raise ValueError("LiSSA depth and repeats must be positive")
    if not 0 <= damping < 1:
        raise ValueError("LiSSA damping must be in [0, 1)")
    if scale <= 0:
        raise ValueError("LiSSA scale must be positive")

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if len(parameters) != len(reference_gradient):
        raise ValueError("Reference gradient does not match trainable model parameters")
    run_device = device or parameters[0].device
    accumulated = [torch.zeros_like(value) for value in reference_gradient]
    batch_stream = _infinite_batches(hessian_batches)

    for _ in range(repeats):
        estimate = [value.detach().clone() for value in reference_gradient]
        for _ in range(depth):
            batch = move_batch_to_device(next(batch_stream), run_device)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            hessian_estimate = hessian_vector_product(loss, parameters, estimate)
            estimate = [
                (
                    reference.detach()
                    + (1.0 - damping) * current.detach()
                    - hessian_value.detach() / scale
                )
                for reference, current, hessian_value in zip(
                    reference_gradient, estimate, hessian_estimate, strict=True
                )
            ]
        for index, value in enumerate(estimate):
            accumulated[index] += value / scale

    return tuple(value / repeats for value in accumulated)


@dataclass(frozen=True)
class InfluenceResult:
    sample_ids: Tensor
    base_scores: Tensor
    reference_gradient: tuple[Tensor, ...]
    inverse_hessian_vector: tuple[Tensor, ...]


class InfluenceEngine:
    """Compute paper Eq. 2 for a model, reference set, and training samples."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFunction,
        *,
        sign: str = "paper",
        device: torch.device | str | None = None,
    ) -> None:
        if sign not in {"paper", "gradient_alignment"}:
            raise ValueError("sign must be 'paper' or 'gradient_alignment'")
        self.model = model
        self.loss_fn = loss_fn
        self.sign = sign
        self.parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not self.parameters:
            raise ValueError("InfluenceEngine requires trainable model parameters")
        self.device = torch.device(device or self.parameters[0].device)

    def gradient(self, batch: Batch) -> tuple[Tensor, ...]:
        prepared = move_batch_to_device(batch, self.device)
        self.model.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model, prepared)
        gradients = parameter_gradient(loss, self.parameters)
        return tuple(gradient.detach() for gradient in gradients)

    def reference_gradient(self, reference_batches: Iterable[Batch]) -> tuple[Tensor, ...]:
        accumulated = [torch.zeros_like(parameter) for parameter in self.parameters]
        count = 0
        for batch in reference_batches:
            gradients = self.gradient(batch)
            for index, gradient in enumerate(gradients):
                accumulated[index] += gradient
            count += 1
        if count == 0:
            raise ValueError("At least one reference batch is required")
        return tuple(value / count for value in accumulated)

    def score_samples(
        self,
        sample_batches: Iterable[Batch],
        reference_batches: Iterable[Batch],
        hessian_batches: Iterable[Batch],
        *,
        lissa_depth: int = 50,
        lissa_damping: float = 0.01,
        lissa_scale: float = 10.0,
        lissa_repeats: int = 1,
    ) -> InfluenceResult:
        reference_gradient = self.reference_gradient(reference_batches)
        ihvp = lissa_ihvp(
            self.model,
            reference_gradient,
            hessian_batches,
            self.loss_fn,
            depth=lissa_depth,
            damping=lissa_damping,
            scale=lissa_scale,
            repeats=lissa_repeats,
            device=self.device,
        )
        coefficient = -1.0 if self.sign == "paper" else 1.0
        sample_ids: list[int] = []
        scores: list[Tensor] = []
        for batch in sample_batches:
            ids = batch["sample_id"].flatten()
            if ids.numel() != 1:
                raise ValueError("Influence scoring requires batches of exactly one sample")
            gradient = self.gradient(batch)
            sample_ids.append(int(ids.item()))
            scores.append((coefficient * tuple_dot(ihvp, gradient)).detach().cpu())
        if not scores:
            raise ValueError("At least one sample is required for influence scoring")
        return InfluenceResult(
            sample_ids=torch.tensor(sample_ids, dtype=torch.long),
            base_scores=torch.stack(scores),
            reference_gradient=tuple(value.detach() for value in reference_gradient),
            inverse_hessian_vector=tuple(value.detach() for value in ihvp),
        )
