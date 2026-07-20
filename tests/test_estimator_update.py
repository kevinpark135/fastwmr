"""Tests for standalone FastWMR estimator reconstruction updates."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    EstimatorUpdater,
    WorldStateEstimator,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    EstimatorRolloutBatch,
    EstimatorRolloutCache,
    EstimatorRolloutCacheSpec,
    SequenceReplayBatch,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_INTERFACE_CFG,
    EstimatorLossCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    HistoryEncoder,
    WorldStateDecoder,
)


def _estimator(hidden_dim: int = 8) -> WorldStateEstimator:
    interface = DEFAULT_INTERFACE_CFG
    return WorldStateEstimator(
        HistoryEncoder(
            observation_dim=interface.policy_observation_dim,
            hidden_dim=hidden_dim,
        ),
        WorldStateDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
        ),
    )


def _targets(observations: torch.Tensor) -> torch.Tensor:
    interface = DEFAULT_INTERFACE_CFG
    continuous = torch.tanh(observations[..., : interface.continuous_target_dim])
    contacts = (observations[..., : interface.discrete_target_dim] > 0.0).to(observations.dtype)
    return torch.cat((continuous, contacts), dim=-1)


def _rollout_batch(num_envs: int = 4, steps: int = 5) -> EstimatorRolloutBatch:
    observations = torch.randn(num_envs, steps, DEFAULT_INTERFACE_CFG.policy_observation_dim)
    reset_boundaries = torch.zeros(num_envs, steps, dtype=torch.bool)
    reset_boundaries[:, 0] = True
    return EstimatorRolloutBatch(
        observations=observations,
        privileged_states=_targets(observations),
        reset_boundaries=reset_boundaries,
    )


def _sequence_batch() -> SequenceReplayBatch:
    interface = DEFAULT_INTERFACE_CFG
    batch_size = 2
    burn_in_length = 2
    learning_length = 2
    transition_length = burn_in_length + learning_length
    observations = torch.randn(
        batch_size,
        transition_length + 1,
        interface.policy_observation_dim,
    )
    return SequenceReplayBatch(
        observations=observations,
        privileged_states=_targets(observations),
        stored_control_features=torch.zeros(
            batch_size,
            transition_length + 1,
            interface.control_feature_dim,
        ),
        actions=torch.zeros(batch_size, transition_length, interface.action_dim),
        rewards=torch.zeros(batch_size, transition_length),
        terminated=torch.zeros(batch_size, transition_length, dtype=torch.bool),
        truncated=torch.zeros(batch_size, transition_length, dtype=torch.bool),
        episode_ids=torch.zeros(batch_size, transition_length, dtype=torch.int64),
        env_ids=torch.arange(batch_size, dtype=torch.int64)[:, None].expand(
            batch_size,
            transition_length,
        ),
        timesteps=torch.arange(transition_length, dtype=torch.int64)[None].expand(
            batch_size,
            transition_length,
        ),
        reset_boundaries=torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        ),
        insertion_ids=torch.arange(
            batch_size * transition_length,
            dtype=torch.int64,
        ).reshape(batch_size, transition_length),
        burn_in_length=burn_in_length,
        learning_length=learning_length,
    )


def test_sequence_update_uses_learning_window_and_returns_detached_features() -> None:
    torch.manual_seed(10)
    estimator = _estimator()
    updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=1e-2),
    )

    result = updater.update_sequence(_sequence_batch())

    assert result.reconstructions.shape == (
        2,
        3,
        DEFAULT_INTERFACE_CFG.reconstruction_target_dim,
    )
    assert not result.reconstructions.requires_grad
    assert not result.final_state.hidden.requires_grad
    assert result.metrics.context_exact_fraction == 1.0
    assert result.metrics.estimator_version == 1
    assert result.metrics.gradient_norm > 0.0
    assert set(result.metrics.field_losses) == {
        "base_lin_vel_mse",
        "friction_mse",
        "payload_mass_mse",
        "push_force_torque_mse",
        "foot_contacts_bce",
    }


def test_rollout_reset_boundary_clears_only_selected_context() -> None:
    torch.manual_seed(11)
    estimator = _estimator()
    updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=1e-3),
    )
    observations = torch.randn(2, 3, DEFAULT_INTERFACE_CFG.policy_observation_dim)
    shared_observation = torch.randn(DEFAULT_INTERFACE_CFG.policy_observation_dim)
    observations[0, 2] = shared_observation
    observations[1, 0] = shared_observation
    reset_boundaries = torch.tensor(
        [[True, False, True], [True, False, False]],
        dtype=torch.bool,
    )
    batch = EstimatorRolloutBatch(
        observations=observations,
        privileged_states=_targets(observations),
        reset_boundaries=reset_boundaries,
    )

    result = updater.update_rollout(batch)

    torch.testing.assert_close(result.reconstructions[0, 2], result.reconstructions[1, 0])


def test_repeated_standalone_updates_reduce_reconstruction_loss() -> None:
    torch.manual_seed(12)
    estimator = _estimator(hidden_dim=16)
    updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=2e-2),
        loss_cfg=EstimatorLossCfg(),
    )
    batch = _rollout_batch(num_envs=8, steps=5)

    first_loss = updater.update_rollout(batch).metrics.total_loss
    for _ in range(39):
        final_result = updater.update_rollout(batch)

    assert final_result.metrics.total_loss < first_loss * 0.6
    assert final_result.metrics.estimator_version == 40


def test_cache_update_drains_only_after_successful_optimizer_step() -> None:
    torch.manual_seed(13)
    interface = DEFAULT_INTERFACE_CFG
    cache = EstimatorRolloutCache(
        EstimatorRolloutCacheSpec.fastwmr(capacity_steps=3, num_envs=2)
    )
    for timestep in range(3):
        observations = torch.randn(2, interface.policy_observation_dim)
        cache.add(
            observations,
            _targets(observations),
            torch.full((2,), timestep == 0, dtype=torch.bool),
        )
    estimator = _estimator()
    updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=1e-2),
    )

    result = updater.update_cache(cache)

    assert len(cache) == 0
    assert result.reconstructions.shape == (2, 3, interface.reconstruction_target_dim)
    assert result.metrics.estimator_version == 1
