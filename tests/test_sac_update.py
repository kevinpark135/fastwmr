"""Tests for scalar FastSAC losses and optimizer updates."""

import pytest
import torch
from torch import nn

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    EntropyTemperature,
    SACFeatureSource,
    SACTransitionBatch,
    SACUpdater,
    compute_actor_loss,
    compute_critic_loss,
    compute_critic_target,
    compute_temperature_loss,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ScalarCriticCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinScalarCritic,
    TanhGaussianActor,
    TwinScalarCritic,
)


def _batch(batch_size: int = 16, state_dim: int = 4, action_dim: int = 2) -> SACTransitionBatch:
    return SACTransitionBatch(
        states=torch.randn(batch_size, state_dim),
        actions=torch.rand(batch_size, action_dim) * 2.0 - 1.0,
        rewards=torch.randn(batch_size),
        next_states=torch.randn(batch_size, state_dim),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
    )


class _FixedActor(nn.Module):
    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((*states.shape[:-1], 2)), torch.full(states.shape[:-1], -2.0)


class _FixedTarget(nn.Module):
    def average(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.full(states.shape[:-1], 10.0)


class _FixedTwin(nn.Module):
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = states.shape[:-1]
        return torch.full(shape, 1.0), torch.full(shape, 3.0)

    def average(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(states, actions)
        return (q1 + q2) / 2.0


def test_critic_target_bootstraps_truncation_but_not_termination() -> None:
    batch = SACTransitionBatch(
        states=torch.zeros(3, 4),
        actions=torch.zeros(3, 2),
        rewards=torch.ones(3),
        next_states=torch.zeros(3, 4),
        terminated=torch.tensor([True, False, False]),
        truncated=torch.tensor([False, True, False]),
    )

    target = compute_critic_target(batch, _FixedActor(), _FixedTarget(), torch.tensor(0.5), discount=0.9)

    assert torch.allclose(target, torch.tensor([1.0, 10.9, 10.9]))
    assert not target.requires_grad


def test_critic_and_actor_losses_use_both_q_values_by_average() -> None:
    batch = _batch(batch_size=5)
    target = torch.full((5,), 2.0)

    critic_output = compute_critic_loss(batch, _FixedTwin(), target)
    actor_output = compute_actor_loss(batch.states, _FixedActor(), _FixedTwin(), torch.tensor(0.5))

    assert torch.allclose(critic_output.loss, torch.tensor(2.0))
    assert torch.allclose(actor_output.average_q, torch.full((5,), 2.0))
    assert torch.allclose(actor_output.loss, torch.tensor(-3.0))


def test_temperature_loss_matches_fastsac_alpha_objective_and_detaches_policy() -> None:
    log_alpha = torch.tensor(-2.0, requires_grad=True)
    log_probabilities = torch.tensor([-1.0, -3.0], requires_grad=True)

    loss = compute_temperature_loss(log_alpha, log_probabilities, target_entropy=0.0)
    loss.backward()

    expected = 2.0 * torch.exp(torch.tensor(-2.0))
    assert torch.allclose(loss, expected)
    assert log_alpha.grad is not None and log_alpha.grad > 0.0
    assert log_probabilities.grad is None


def test_replay_conversion_selects_baseline_or_fastwmr_features_and_final_state() -> None:
    buffer = TransitionReplayBuffer(
        ReplayBufferSpec(
            capacity=2,
            observation_dim=3,
            action_dim=2,
            privileged_state_dim=2,
            control_feature_dim=5,
            require_temporal_metadata=True,
        )
    )
    buffer.add(
        observations=torch.zeros(1, 3),
        actions=torch.zeros(1, 2),
        rewards=torch.ones(1),
        next_observations=torch.ones(1, 3),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        privileged_states=torch.full((1, 2), 1000.0),
        next_privileged_states=torch.full((1, 2), 2000.0),
        control_features=torch.full((1, 5), 2.0),
        next_control_features=torch.full((1, 5), 3.0),
        estimator_versions=torch.tensor([4]),
        episode_ids=torch.tensor([8]),
        env_ids=torch.tensor([0]),
        timesteps=torch.tensor([7]),
        reset_boundaries=torch.tensor([False]),
        final_observations=torch.full((1, 3), 4.0),
        final_privileged_states=torch.full((1, 2), 3000.0),
        final_control_features=torch.full((1, 5), 5.0),
        final_observation_mask=torch.tensor([True]),
    )
    replay = buffer.chronological()

    baseline = SACTransitionBatch.from_replay(replay, feature_source=SACFeatureSource.POLICY_OBSERVATION)
    fastwmr = SACTransitionBatch.from_replay(replay, feature_source=SACFeatureSource.CONTROL_FEATURE)

    assert baseline.states.shape == (1, 3)
    assert torch.equal(baseline.next_states, torch.full((1, 3), 4.0))
    assert fastwmr.states.shape == (1, 5)
    assert torch.equal(fastwmr.next_states, torch.full((1, 5), 5.0))
    assert not hasattr(fastwmr, "privileged_states")


def test_full_sac_update_changes_online_alpha_and_polyak_target_parameters() -> None:
    torch.manual_seed(3)
    actor = TanhGaussianActor(
        input_dim=4,
        action_dim=2,
        cfg=TanhGaussianActorCfg(hidden_dim=16),
    )
    critic = TwinScalarCritic(state_dim=4, action_dim=2, cfg=ScalarCriticCfg(hidden_dim=16))
    target = TargetTwinScalarCritic.from_online(critic)
    temperature = EntropyTemperature(0.001)
    updater = SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=3e-4),
        discount=0.97,
        target_update_rate=0.25,
        target_entropy=0.0,
    )
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    target_before = [parameter.detach().clone() for parameter in target.parameters()]
    alpha_before = temperature().detach().clone()

    metrics = updater.update(_batch())

    assert any(not torch.equal(before, after) for before, after in zip(actor_before, actor.parameters(), strict=True))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, critic.parameters(), strict=True))
    assert not torch.equal(alpha_before, temperature().detach())
    for before, online, updated_target in zip(target_before, critic.parameters(), target.parameters(), strict=True):
        assert torch.allclose(updated_target, torch.lerp(before, online.detach(), 0.25))
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(torch.isfinite(value) for value in metrics.__dict__.values())


def test_actor_and_temperature_steps_do_not_update_critic_or_each_other() -> None:
    actor = TanhGaussianActor(input_dim=4, action_dim=2, cfg=TanhGaussianActorCfg(hidden_dim=16))
    critic = TwinScalarCritic(state_dim=4, action_dim=2, cfg=ScalarCriticCfg(hidden_dim=16))
    target = TargetTwinScalarCritic.from_online(critic)
    temperature = EntropyTemperature(0.001)
    updater = SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=3e-4),
    )
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    alpha_before = temperature().detach().clone()

    actor_output = updater.update_actor(_batch().states)

    assert all(
        torch.equal(before, after) for before, after in zip(critic_before, critic.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert torch.equal(alpha_before, temperature().detach())

    actor_before_alpha_step = [parameter.detach().clone() for parameter in actor.parameters()]
    updater.update_temperature(actor_output.log_probabilities)
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_before_alpha_step, actor.parameters(), strict=True)
    )


def test_invalid_temperature_and_attached_fastwmr_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        EntropyTemperature(0.0)
    with pytest.raises(ValueError, match="detached"):
        SACTransitionBatch(
            states=torch.randn(2, 4, requires_grad=True),
            actions=torch.randn(2, 2),
            rewards=torch.randn(2),
            next_states=torch.randn(2, 4),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
        )
