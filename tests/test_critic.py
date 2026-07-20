"""Tests for online and target FastSAC scalar critics."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_CRITIC_CFG,
    DEFAULT_INTERFACE_CFG,
    ScalarCriticCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinScalarCritic,
    TwinScalarCritic,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils.feature_builder import (
    build_control_feature,
)


def test_twin_critic_matches_fastsac_scalar_architecture_and_shapes() -> None:
    critic = TwinScalarCritic(state_dim=6, action_dim=3)
    features = torch.randn(5, 6)
    actions = torch.randn(5, 3)

    q1, q2 = critic(features, actions)
    stacked = critic.stacked(features, actions)

    widths = [layer.out_features for layer in critic.q1.net if isinstance(layer, torch.nn.Linear)]
    assert widths == [768, 384, 192, 1]
    assert q1.shape == q2.shape == (5,)
    assert stacked.shape == (2, 5)
    assert critic.q1.net[0].weight.data_ptr() != critic.q2.net[0].weight.data_ptr()


def test_average_q_is_fastsac_mean_not_clipped_minimum() -> None:
    critic = TwinScalarCritic(state_dim=4, action_dim=2)
    features = torch.randn(7, 4)
    actions = torch.randn(7, 2)

    q1, q2 = critic(features, actions)

    assert torch.allclose(critic.average(features, actions), (q1 + q2) / 2.0)


def test_target_critic_starts_as_exact_frozen_copy() -> None:
    online = TwinScalarCritic(state_dim=4, action_dim=2)
    target = TargetTwinScalarCritic.from_online(online)
    features = torch.randn(3, 4, requires_grad=True)
    actions = torch.randn(3, 2, requires_grad=True)

    online_values = online.stacked(features, actions)
    target_values = target.stacked(features, actions)

    assert torch.equal(target_values, online_values.detach())
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert not target_values.requires_grad


def test_target_soft_and_hard_updates_have_correct_direction() -> None:
    online = TwinScalarCritic(state_dim=4, action_dim=2)
    target = TargetTwinScalarCritic.from_online(online)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(1.0)
        for parameter in target.parameters():
            parameter.zero_()

    target.soft_update_from(online, tau=0.25)
    assert all(torch.allclose(parameter, torch.full_like(parameter, 0.25)) for parameter in target.parameters())
    assert all(torch.all(parameter == 1.0) for parameter in online.parameters())

    target.hard_update_from(online)
    assert all(
        torch.equal(target_parameter, online_parameter)
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True)
    )


def test_fastwmr_critic_uses_actor_feature_and_blocks_estimator_gradient() -> None:
    interface = DEFAULT_INTERFACE_CFG
    critic = TwinScalarCritic(state_dim=interface.critic_state_dim, action_dim=interface.action_dim)
    observations = torch.randn(4, interface.policy_observation_dim, requires_grad=True)
    reconstruction = torch.randn(4, interface.reconstruction_target_dim, requires_grad=True)
    actions = torch.randn(4, interface.action_dim, requires_grad=True)
    features = build_control_feature(observations, reconstruction, cfg=interface)

    critic.average(features, actions).mean().backward()

    assert critic.state_dim == interface.actor_input_dim == interface.critic_state_dim
    assert observations.grad is not None
    assert actions.grad is not None
    assert reconstruction.grad is None


def test_critic_supports_sequence_leading_dimensions() -> None:
    critic = TwinScalarCritic(state_dim=4, action_dim=2)
    features = torch.randn(3, 7, 4)
    actions = torch.randn(3, 7, 2)

    q1, q2 = critic(features, actions)

    assert q1.shape == q2.shape == (3, 7)
    assert critic.stacked(features, actions).shape == (2, 3, 7)


def test_critic_rejects_invalid_configuration_inputs_and_target_update() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        ScalarCriticCfg(hidden_dim=10)

    critic = TwinScalarCritic(state_dim=4, action_dim=2, cfg=DEFAULT_CRITIC_CFG)
    with pytest.raises(ValueError, match="end in dimension 4"):
        critic(torch.randn(3, 5), torch.randn(3, 2))
    with pytest.raises(ValueError, match="leading dimensions"):
        critic(torch.randn(3, 4), torch.randn(2, 2))

    target = TargetTwinScalarCritic.from_online(critic)
    with pytest.raises(ValueError, match="tau"):
        target.soft_update_from(critic, tau=0.0)
    with pytest.raises(ValueError, match="architectures"):
        target.hard_update_from(TwinScalarCritic(state_dim=5, action_dim=2))
