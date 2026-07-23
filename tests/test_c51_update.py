"""Tests for categorical projection and C51 FastSAC optimizer updates."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    C51SACUpdater,
    EntropyTemperature,
    SACTransitionBatch,
    compute_c51_critic_target,
    project_categorical_distribution,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DistributionalCriticCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinC51Critic,
    TanhGaussianActor,
    TwinC51Critic,
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


def test_categorical_projection_preserves_exact_interpolated_and_clipped_mass() -> None:
    support = torch.tensor([-1.0, 0.0, 1.0])
    probabilities = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.0, 1.0, 0.0],
            [0.25, 0.5, 0.25],
        ]
    )
    target_atoms = torch.stack((support, torch.full_like(support, 0.5), torch.full_like(support, 5.0)))

    projected = project_categorical_distribution(probabilities, target_atoms, support)

    assert torch.allclose(projected[0], probabilities[0])
    assert torch.allclose(projected[1], torch.tensor([0.0, 0.5, 0.5]))
    assert torch.allclose(projected[2], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(projected.sum(dim=-1), torch.ones(3))


def test_c51_target_uses_each_twin_distribution_and_stops_terminal_bootstrap() -> None:
    cfg = DistributionalCriticCfg(hidden_dim=16, num_atoms=5, value_min=-2.0, value_max=2.0)
    actor = TanhGaussianActor(input_dim=4, action_dim=2, cfg=TanhGaussianActorCfg(hidden_dim=16))
    critic = TwinC51Critic(state_dim=4, action_dim=2, cfg=cfg)
    with torch.no_grad():
        critic.q1.net[-1].bias.copy_(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        critic.q2.net[-1].bias.copy_(torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0]))
        critic.q1.net[-1].weight.zero_()
        critic.q2.net[-1].weight.zero_()
    target = TargetTwinC51Critic.from_online(critic)
    batch = _batch(batch_size=2)
    batch = SACTransitionBatch(
        states=batch.states,
        actions=batch.actions,
        rewards=torch.tensor([0.5, 0.0]),
        next_states=batch.next_states,
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, True]),
    )

    distributions = compute_c51_critic_target(batch, actor, target, torch.tensor(0.01), discount=0.9)

    assert distributions.shape == (2, 2, 5)
    assert torch.allclose(distributions.sum(dim=-1), torch.ones(2, 2))
    assert torch.allclose(distributions[:, 0], torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]]).expand(2, -1))
    assert not torch.allclose(distributions[0, 1], distributions[1, 1])
    assert not distributions.requires_grad


def test_full_c51_sac_update_changes_online_models_and_polyak_target() -> None:
    torch.manual_seed(7)
    actor = TanhGaussianActor(input_dim=4, action_dim=2, cfg=TanhGaussianActorCfg(hidden_dim=16))
    critic = TwinC51Critic(
        state_dim=4,
        action_dim=2,
        cfg=DistributionalCriticCfg(hidden_dim=16, num_atoms=11, value_min=-5.0, value_max=5.0),
    )
    target = TargetTwinC51Critic.from_online(critic)
    temperature = EntropyTemperature(0.001)
    updater = C51SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=3e-4),
        target_update_rate=0.25,
    )
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    target_before = [parameter.detach().clone() for parameter in target.parameters()]

    metrics = updater.update(_batch())

    assert any(not torch.equal(before, after) for before, after in zip(actor_before, actor.parameters(), strict=True))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, critic.parameters(), strict=True))
    for before, online, updated_target in zip(target_before, critic.parameters(), target.parameters(), strict=True):
        assert torch.allclose(updated_target, torch.lerp(before, online.detach(), 0.25))
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(torch.isfinite(value) for value in metrics.__dict__.values())
    assert 0.0 <= metrics.c51_lower_endpoint_mass <= 1.0
    assert 0.0 <= metrics.c51_upper_endpoint_mass <= 1.0
    assert 0.0 <= metrics.c51_target_lower_endpoint_mass <= 1.0
    assert 0.0 <= metrics.c51_target_upper_endpoint_mass <= 1.0
    assert metrics.q_gap_mean >= 0.0
    assert metrics.q_gap_max >= metrics.q_gap_mean
    assert 0.0 <= metrics.policy_action_saturation_fraction <= 1.0
