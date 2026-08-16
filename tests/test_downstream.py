import torch

from ft_ncfm.downstream import train_action_regressor


def test_downstream_action_head_uses_isolated_selection_and_test_sets() -> None:
    torch.manual_seed(3)
    representations = torch.randn(160, 4)
    actions = torch.stack(
        [representations[:, 0] - representations[:, 1], representations[:, 2] * 0.5],
        dim=1,
    )
    joint = torch.cat([representations, actions], dim=1)

    _, summary, trace = train_action_regressor(
        joint[:100],
        joint[100:130],
        representation_dim=4,
        hidden_dim=16,
        steps=150,
        batch_size=32,
        learning_rate=5e-3,
        seed=11,
        test_joint_features=joint[130:],
    )

    assert summary.finite
    assert summary.final_train_loss < summary.initial_train_loss
    assert summary.selection_mse < 0.03
    assert summary.test_mse is not None and summary.test_mse < 0.03
    assert len(trace) == 150
