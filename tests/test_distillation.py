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
