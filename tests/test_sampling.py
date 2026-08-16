import torch

from ft_ncfm.data import disjoint_evaluation_indices, stratified_subset_indices
from ft_ncfm.sampling import InfluenceWeightedSampler


def test_sampler_tracks_high_probability_item() -> None:
    generator = torch.Generator().manual_seed(123)
    sampler = InfluenceWeightedSampler(
        torch.tensor([0.9, 0.05, 0.05]),
        num_samples=1000,
        generator=generator,
    )
    draws = torch.tensor(list(sampler))
    assert (draws == 0).float().mean() > 0.85


def test_stratified_subset_has_exact_requested_size() -> None:
    targets = torch.tensor([0] * 13 + [1] * 17 + [2] * 30)
    indices = stratified_subset_indices(targets, 0.25, seed=9)
    assert len(indices) == 15
    assert len(set(indices)) == len(indices)


def test_evaluation_splits_are_exact_reproducible_and_disjoint() -> None:
    first = disjoint_evaluation_indices(
        100, reference_size=10, selection_size=20, test_size=30, seed=7
    )
    second = disjoint_evaluation_indices(
        100, reference_size=10, selection_size=20, test_size=30, seed=7
    )

    assert first == second
    assert [len(indices) for indices in first] == [10, 20, 30]
    assert not (set(first[0]) & set(first[1]))
    assert not (set(first[0]) & set(first[2]))
    assert not (set(first[1]) & set(first[2]))
