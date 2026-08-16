import torch
from torch import nn

from ft_ncfm.influence import lissa_ihvp


class ScalarQuadratic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(1.0))


def quadratic_loss(model: ScalarQuadratic, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    curvature = batch["curvature"].mean()
    target = batch["target"].mean()
    return 0.5 * curvature * (model.theta - target).square()


def test_lissa_matches_scalar_inverse_hessian() -> None:
    model = ScalarQuadratic()
    reference_gradient = (torch.tensor(6.0),)
    batches = [{"curvature": torch.tensor([2.0]), "target": torch.tensor([0.0])}]
    result = lissa_ihvp(
        model,
        reference_gradient,
        batches,
        quadratic_loss,
        depth=80,
        damping=0.0,
        scale=4.0,
        repeats=1,
    )
    torch.testing.assert_close(result[0], torch.tensor(3.0), atol=1e-5, rtol=1e-5)
