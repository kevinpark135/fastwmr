"""Tests for the short-lived FastWMR estimator rollout cache."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    EstimatorRolloutCache,
    EstimatorRolloutCacheSpec,
)


def _cache(capacity_steps: int = 3) -> EstimatorRolloutCache:
    return EstimatorRolloutCache(
        EstimatorRolloutCacheSpec(
            capacity_steps=capacity_steps,
            num_envs=2,
            observation_dim=3,
            privileged_state_dim=2,
        )
    )


def _add_step(cache: EstimatorRolloutCache, timestep: int) -> None:
    values = torch.tensor([float(timestep), float(timestep + 100)])
    cache.add(
        observations=values[:, None].repeat(1, 3),
        privileged_states=values[:, None].repeat(1, 2),
        reset_boundaries=torch.tensor([timestep == 0, timestep == 2]),
    )


def test_cache_returns_vector_rollout_in_chronological_order() -> None:
    cache = _cache(capacity_steps=3)
    for timestep in range(4):
        _add_step(cache, timestep)

    batch = cache.chronological()

    assert len(cache) == 3
    assert cache.is_full
    assert cache.total_steps == 4
    assert batch.observations.shape == (2, 3, 3)
    assert batch.privileged_states.shape == (2, 3, 2)
    assert batch.reset_boundaries.shape == (2, 3)
    assert torch.equal(batch.observations[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(batch.observations[1, :, 0], torch.tensor([101.0, 102.0, 103.0]))
    assert torch.equal(batch.context_is_exact, torch.tensor([False, False]))


def test_public_cache_reads_remain_isolated_from_zero_copy_runtime_views() -> None:
    cache = _cache(capacity_steps=3)
    for timestep in range(4):
        _add_step(cache, timestep)

    copied = cache.chronological()
    copied.observations.zero_()
    copied.privileged_states.zero_()

    reread = cache.chronological()
    assert torch.equal(reread.observations[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(reread.privileged_states[1, :, 0], torch.tensor([101.0, 102.0, 103.0]))


def test_drain_clears_samples_but_preserves_monotonic_counter() -> None:
    cache = _cache()
    _add_step(cache, 0)
    _add_step(cache, 1)

    batch = cache.drain()

    assert batch.sequence_length == 2
    assert len(cache) == 0
    assert cache.total_steps == 2
    with pytest.raises(RuntimeError, match="empty"):
        cache.chronological()


def test_cache_rejects_malformed_or_nonfinite_steps() -> None:
    cache = _cache()

    with pytest.raises(ValueError, match="observations must have shape"):
        cache.add(
            observations=torch.zeros(2, 4),
            privileged_states=torch.zeros(2, 2),
            reset_boundaries=torch.zeros(2, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="reset_boundaries"):
        cache.add(
            observations=torch.zeros(2, 3),
            privileged_states=torch.zeros(2, 2),
            reset_boundaries=torch.zeros(2),
        )
    observations = torch.zeros(2, 3)
    observations[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        cache.add(
            observations=observations,
            privileged_states=torch.zeros(2, 2),
            reset_boundaries=torch.zeros(2, dtype=torch.bool),
        )
