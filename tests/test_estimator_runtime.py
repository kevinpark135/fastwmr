"""Tests for per-environment FastWMR estimator rollout state."""

import math

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    EstimatorUpdater,
    FastWMREstimatorRuntime,
    WorldStateEstimator,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    EstimatorRolloutCache,
    EstimatorRolloutCacheSpec,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_INTERFACE_CFG,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    HistoryEncoder,
    WorldStateDecoder,
)


def _estimator(hidden_dim: int = 12) -> WorldStateEstimator:
    interface = DEFAULT_INTERFACE_CFG
    return WorldStateEstimator(
        HistoryEncoder(
            interface.policy_observation_dim,
            hidden_dim=hidden_dim,
        ),
        WorldStateDecoder(
            hidden_dim,
            hidden_dim=hidden_dim,
        ),
    )


def _targets(observations: torch.Tensor) -> torch.Tensor:
    interface = DEFAULT_INTERFACE_CFG
    continuous = torch.tanh(observations[..., : interface.continuous_target_dim])
    contacts = (observations[..., : interface.discrete_target_dim] > 0.0).to(observations.dtype)
    return torch.cat((continuous, contacts), dim=-1)


def test_only_done_environment_slices_are_reset() -> None:
    torch.manual_seed(20)
    runtime = FastWMREstimatorRuntime(_estimator(), num_envs=10)
    runtime.step(torch.randn(10, DEFAULT_INTERFACE_CFG.policy_observation_dim))
    before_hidden = runtime.state.hidden.clone()
    before_cell = runtime.state.cell.clone()
    terminated = torch.zeros(10, dtype=torch.bool)
    truncated = torch.zeros(10, dtype=torch.bool)
    terminated[3] = True
    truncated[8] = True

    runtime.reset_done(terminated, truncated)

    assert torch.count_nonzero(runtime.state.hidden[:, 3]) == 0
    assert torch.count_nonzero(runtime.state.cell[:, 3]) == 0
    assert torch.count_nonzero(runtime.state.hidden[:, 8]) == 0
    assert torch.count_nonzero(runtime.state.cell[:, 8]) == 0
    retained = torch.tensor([0, 1, 2, 4, 5, 6, 7, 9])
    torch.testing.assert_close(runtime.state.hidden[:, retained], before_hidden[:, retained])
    torch.testing.assert_close(runtime.state.cell[:, retained], before_cell[:, retained])


def test_long_rollout_does_not_retain_an_autograd_graph() -> None:
    torch.manual_seed(21)
    runtime = FastWMREstimatorRuntime(_estimator(), num_envs=4)

    for _ in range(128):
        result = runtime.step(
            torch.randn(4, DEFAULT_INTERFACE_CFG.policy_observation_dim)
        )
        assert not result.reconstruction.requires_grad
        assert result.reconstruction.grad_fn is None
        assert not runtime.state.hidden.requires_grad
        assert not runtime.state.cell.requires_grad
        assert runtime.state.hidden.grad_fn is None
        assert runtime.state.cell.grad_fn is None

    assert runtime.environment_steps == 128
    assert math.isfinite(runtime.hidden_norm)


def test_reset_boundary_matches_fresh_zero_state_context() -> None:
    torch.manual_seed(22)
    estimator = _estimator()
    runtime = FastWMREstimatorRuntime(estimator, num_envs=2)
    fresh_runtime = FastWMREstimatorRuntime(estimator, num_envs=1)
    runtime.step(torch.randn(2, DEFAULT_INTERFACE_CFG.policy_observation_dim))
    shared_observation = torch.randn(DEFAULT_INTERFACE_CFG.policy_observation_dim)
    observations = torch.stack((shared_observation, torch.randn_like(shared_observation)))

    reset_result = runtime.step(
        observations,
        reset_boundaries=torch.tensor([True, False]),
    )
    fresh_result = fresh_runtime.step(shared_observation.unsqueeze(0))

    torch.testing.assert_close(
        reset_result.reconstruction[0],
        fresh_result.reconstruction[0],
    )
    torch.testing.assert_close(runtime.state.hidden[:, 0], fresh_runtime.state.hidden[:, 0])
    torch.testing.assert_close(runtime.state.cell[:, 0], fresh_runtime.state.cell[:, 0])


def test_estimator_update_rebuilds_runtime_with_current_parameters() -> None:
    torch.manual_seed(23)
    interface = DEFAULT_INTERFACE_CFG
    num_envs = 3
    steps = 5
    estimator = _estimator(hidden_dim=16)
    runtime = FastWMREstimatorRuntime(estimator, num_envs=num_envs)
    cache = EstimatorRolloutCache(
        EstimatorRolloutCacheSpec.fastwmr(steps, num_envs)
    )
    for timestep in range(steps):
        observations = torch.randn(num_envs, interface.policy_observation_dim)
        reset_boundaries = torch.full(
            (num_envs,),
            timestep == 0,
            dtype=torch.bool,
        )
        cache.add(observations, _targets(observations), reset_boundaries)
        runtime.step(observations, reset_boundaries=reset_boundaries)
    cached_batch = cache.chronological()
    old_hidden = runtime.state.hidden.clone()
    updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=2e-2),
    )
    final_reset_mask = torch.tensor([False, True, False])

    synchronized = runtime.update_from_cache(
        updater,
        cache,
        final_reset_mask=final_reset_mask,
    )

    assert len(cache) == 0
    assert runtime.estimator_version == 1
    assert runtime.rebuilds == 1
    assert synchronized.runtime_rebuild.context_exact_fraction == 1.0
    assert not torch.equal(runtime.state.hidden, old_hidden)
    assert torch.count_nonzero(runtime.state.hidden[:, 1]) == 0
    assert torch.count_nonzero(runtime.state.cell[:, 1]) == 0
    reference = FastWMREstimatorRuntime(
        estimator,
        num_envs=num_envs,
        estimator_version=1,
    )
    reference_rebuild = reference.rebuild_from_batch(
        cached_batch,
        estimator_version=1,
        final_reset_mask=final_reset_mask,
    )
    torch.testing.assert_close(
        synchronized.runtime_rebuild.reconstructions,
        reference_rebuild.reconstructions,
    )
    torch.testing.assert_close(runtime.state.hidden, reference.state.hidden)
    torch.testing.assert_close(runtime.state.cell, reference.state.cell)


def test_runtime_version_guard_and_checkpoint_exclusion() -> None:
    runtime = FastWMREstimatorRuntime(_estimator(), num_envs=2, estimator_version=3)

    with pytest.raises(RuntimeError, match="version mismatch"):
        runtime.step(
            torch.randn(2, DEFAULT_INTERFACE_CFG.policy_observation_dim),
            expected_estimator_version=4,
        )

    assert not isinstance(runtime, torch.nn.Module)
    assert not hasattr(runtime, "state_dict")
    runtime.reset_all(estimator_version=4)
    assert runtime.estimator_version == 4
    assert torch.count_nonzero(runtime.state.hidden) == 0
    assert torch.count_nonzero(runtime.state.cell) == 0
