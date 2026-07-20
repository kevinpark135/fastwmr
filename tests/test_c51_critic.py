"""Tests for online and target FastSAC C51 twin critics."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DistributionalCriticCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinC51Critic,
    TwinC51Critic,
)


def test_c51_twin_shapes_probabilities_and_expected_values() -> None:
    cfg = DistributionalCriticCfg(hidden_dim=16, num_atoms=11, value_min=-5.0, value_max=5.0)
    critic = TwinC51Critic(state_dim=4, action_dim=2, cfg=cfg)
    states = torch.randn(3, 7, 4)
    actions = torch.randn(3, 7, 2)

    logits = critic.stacked_logits(states, actions)
    probabilities = critic.stacked_probabilities(states, actions)
    values = critic.stacked_values(states, actions)

    assert logits.shape == probabilities.shape == (2, 3, 7, 11)
    assert values.shape == (2, 3, 7)
    assert critic.average(states, actions).shape == (3, 7)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3, 7))
    assert torch.all(values >= cfg.value_min)
    assert torch.all(values <= cfg.value_max)
    assert critic.q1.net[0].weight.data_ptr() != critic.q2.net[0].weight.data_ptr()


def test_c51_target_starts_frozen_and_tracks_polyak_updates() -> None:
    cfg = DistributionalCriticCfg(hidden_dim=16, num_atoms=7)
    online = TwinC51Critic(state_dim=4, action_dim=2, cfg=cfg)
    target = TargetTwinC51Critic.from_online(online)
    states = torch.randn(5, 4, requires_grad=True)
    actions = torch.randn(5, 2, requires_grad=True)

    assert torch.equal(target.stacked_logits(states, actions), online.stacked_logits(states, actions).detach())
    assert torch.equal(target.support, online.support)
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())

    target_before = [parameter.clone() for parameter in target.parameters()]
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)
    target.soft_update_from(online, tau=0.25)
    for before, online_parameter, target_parameter in zip(
        target_before, online.parameters(), target.parameters(), strict=True
    ):
        assert torch.allclose(target_parameter, torch.lerp(before, online_parameter, 0.25))


def test_c51_configuration_and_input_validation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DistributionalCriticCfg(num_atoms=1)
    with pytest.raises(ValueError, match="smaller"):
        DistributionalCriticCfg(value_min=1.0, value_max=1.0)

    critic = TwinC51Critic(state_dim=4, action_dim=2, cfg=DistributionalCriticCfg(hidden_dim=16))
    with pytest.raises(ValueError, match="end in dimension 4"):
        critic(torch.randn(3, 5), torch.randn(3, 2))

    target = TargetTwinC51Critic.from_online(critic)
    with pytest.raises(ValueError, match="tau"):
        target.soft_update_from(critic, tau=0.0)
    with pytest.raises(ValueError, match="architectures"):
        target.hard_update_from(
            TwinC51Critic(state_dim=5, action_dim=2, cfg=DistributionalCriticCfg(hidden_dim=16))
        )
