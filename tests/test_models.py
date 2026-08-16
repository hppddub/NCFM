import torch

from ft_ncfm.models import MiniVLAPolicy


def test_fixed_pool_matches_adaptive_pool_for_mnist_feature_shape() -> None:
    features = torch.randn(2, 16, 7, 7)

    fixed = torch.nn.AvgPool2d(kernel_size=4, stride=3)(features)
    adaptive = torch.nn.AdaptiveAvgPool2d((2, 2))(features)

    torch.testing.assert_close(fixed, adaptive)


def test_policy_accepts_mnist_proxy_inputs() -> None:
    policy = MiniVLAPolicy()

    actions = policy(torch.randn(3, 1, 28, 28), torch.tensor([0, 1, 2]))

    assert actions.shape == (3, 3)
