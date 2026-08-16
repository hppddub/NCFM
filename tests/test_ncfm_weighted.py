from types import SimpleNamespace

import torch

from NCFM.NCFM import CFLossFunc, calculate_real


def test_uniform_weights_preserve_original_cf_path() -> None:
    torch.manual_seed(4)
    projection = torch.randn(9, 6)
    original = calculate_real(projection)
    weighted = calculate_real(projection, torch.ones(6))
    torch.testing.assert_close(weighted, original)


def test_weighted_cf_loss_is_differentiable_for_synthetic_features() -> None:
    torch.manual_seed(7)
    real = torch.randn(8, 4)
    synthetic = torch.randn(3, 4, requires_grad=True)
    frequencies = torch.randn(12, 4)
    weights = torch.tensor([0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
    loss = CFLossFunc()(real, synthetic, frequencies, SimpleNamespace(num_freqs=12), weights)
    loss.backward()
    assert synthetic.grad is not None
    assert torch.isfinite(synthetic.grad).all()
