import torch

from ft_ncfm.distillation import WeightedNCFMDistiller, summarize_convergence


def test_weighted_adversarial_game_is_finite_and_converges() -> None:
    torch.manual_seed(17)
    high_value = torch.randn(48, 2) * 0.08 + torch.tensor([1.5, 0.0])
    low_value = torch.randn(48, 2) * 0.08 + torch.tensor([-1.5, 0.0])
    real = torch.cat([high_value, low_value])
    weights = torch.cat([torch.full((48,), 0.95 / 48), torch.full((48,), 0.05 / 48)])
    distiller = WeightedNCFMDistiller(
        real,
        weights,
        synthetic_count=6,
        num_frequencies=24,
        hidden_dim=16,
        generator_lr=0.05,
        discriminator_lr=0.0005,
    )
    trace = distiller.run(steps=100, discriminator_steps=1)
    summary = summarize_convergence(trace, window_ratio=0.1)
    assert summary.finite
    assert summary.converged


def test_standardized_distiller_returns_trainable_raw_space_coreset() -> None:
    real = torch.tensor(
        [
            [10.0, -2.0, 0.1],
            [12.0, -1.0, 0.3],
            [14.0, 0.0, 0.5],
            [16.0, 1.0, 0.7],
        ]
    )
    distiller = WeightedNCFMDistiller(
        real,
        None,
        synthetic_count=2,
        num_frequencies=4,
        hidden_dim=4,
        standardize_features=True,
    )

    raw = distiller.coreset(raw_space=True)

    assert raw.shape == (2, 3)
    assert torch.isfinite(raw).all()
