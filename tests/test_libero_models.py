import torch

from ft_ncfm.libero_models import LiberoVLAPolicy, libero_policy_loss


def test_libero_policy_fuses_all_modalities_and_predicts_action() -> None:
    model = LiberoVLAPolicy(
        representation_dim=32,
        visual_dim=16,
        language_dim=8,
        proprio_dim=8,
    )
    batch = {
        "image": torch.rand(2, 6, 128, 128),
        "instruction": torch.tensor([[257, 2, 0], [257, 3, 4]]),
        "proprio": torch.rand(2, 9),
        "action": torch.rand(2, 7),
    }
    representation = model.encode(
        batch["image"], batch["instruction"], batch["proprio"]
    )
    assert representation.shape == (2, 32)
    assert model(batch["image"], batch["instruction"], batch["proprio"]).shape == (2, 7)
    assert torch.isfinite(libero_policy_loss(model, batch))


def test_freeze_encoders_leaves_fusion_and_head_trainable() -> None:
    model = LiberoVLAPolicy(representation_dim=32)
    model.freeze_encoders_for_influence()
    assert not any(parameter.requires_grad for parameter in model.visual_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in model.proprio_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    assert all(parameter.requires_grad for parameter in model.action_head.parameters())
