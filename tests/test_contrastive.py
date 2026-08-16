import torch
from torch import nn

from ft_ncfm.contrastive import (
    ContrastiveVerifier,
    VisionPerturbationLibrary,
    normalize_influence_weights,
    refine_influence_weights,
)


def test_refinement_matches_equation_five() -> None:
    base = torch.tensor([2.0, 3.0])
    original = torch.tensor([1.5, -0.25])
    counterexample = torch.tensor([0.5, 0.75])
    expected = base * (1 + torch.tanh(original - counterexample))
    actual = refine_influence_weights(base, original, counterexample, beta=1.0)
    torch.testing.assert_close(actual, expected)


def test_normalization_policies_produce_probabilities() -> None:
    raw = torch.tensor([-2.0, 0.5, 4.0])
    for policy in ("positive_shift", "clamp", "softmax"):
        probabilities = normalize_influence_weights(raw, policy=policy)
        assert torch.all(probabilities >= 0)
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_counterexample_changes_only_vision() -> None:
    batch = {
        "sample_id": torch.tensor([7]),
        "image": torch.zeros(1, 1, 28, 28),
        "instruction": torch.tensor([3]),
        "action": torch.tensor([[0.1, -0.2, 0.3]]),
    }
    counterexample = VisionPerturbationLibrary().apply_to_batch(batch, ["object_substitution"])
    assert not torch.equal(counterexample["image"], batch["image"])
    for key in ("sample_id", "instruction", "action"):
        torch.testing.assert_close(counterexample[key], batch[key])


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        del instruction
        return image.flatten(1).mean(dim=1, keepdim=True) * self.weight


def tiny_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (model(batch["image"], batch["instruction"]) - batch["action"]).square().mean()


def test_contrastive_ranking_selects_declared_direction() -> None:
    batches = [
        {
            "sample_id": torch.tensor([index]),
            "image": torch.full((1, 1, 4, 4), float(index + 1)),
            "instruction": torch.tensor([0]),
            "action": torch.tensor([[0.0]]),
        }
        for index in range(4)
    ]
    scores = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    reference_gradient = (torch.tensor(1.0),)

    largest = ContrastiveVerifier(
        TinyPolicy(), tiny_loss, elite_ratio=0.5, ranking="largest"
    ).verify(scores, batches, reference_gradient)
    smallest = ContrastiveVerifier(
        TinyPolicy(), tiny_loss, elite_ratio=0.5, ranking="smallest"
    ).verify(scores, batches, reference_gradient)

    assert set(largest.elite_indices.tolist()) == {2, 3}
    assert set(smallest.elite_indices.tolist()) == {0, 1}
