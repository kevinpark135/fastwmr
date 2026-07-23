"""Tests for the shared FastSAC/FastWMR transition replay contract."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    TransitionReplayBuffer,
)


def _base_transition(start: int, count: int, observation_dim: int = 3, action_dim: int = 2) -> dict:
    values = torch.arange(start, start + count, dtype=torch.float32)
    observations = values[:, None].repeat(1, observation_dim)
    return {
        "observations": observations,
        "actions": values[:, None].repeat(1, action_dim),
        "rewards": values,
        "next_observations": observations + 0.5,
        "terminated": torch.zeros(count, dtype=torch.bool),
        "truncated": torch.zeros(count, dtype=torch.bool),
    }


def test_fastsac_replay_wraps_and_keeps_newest_transitions() -> None:
    buffer = TransitionReplayBuffer(ReplayBufferSpec(capacity=5, observation_dim=3, action_dim=2))
    assert buffer.oldest_insertion_id is None
    assert buffer.newest_insertion_id is None

    buffer.add(**_base_transition(0, 3))
    buffer.add(**_base_transition(3, 4))
    retained = buffer.chronological()

    assert len(buffer) == 5
    assert buffer.is_full
    assert buffer.total_inserted == 7
    assert buffer.oldest_insertion_id == 2
    assert buffer.newest_insertion_id == 6
    assert buffer.oldest_estimator_version is None
    assert buffer.newest_estimator_version is None
    assert torch.equal(retained.rewards, torch.arange(2, 7, dtype=torch.float32))
    assert torch.equal(retained.insertion_ids, torch.arange(2, 7))
    assert retained.privileged_states.shape == (5, 0)
    assert retained.reconstructions.shape == (5, 0)


def test_add_detaches_inputs_and_sampling_has_stable_shapes() -> None:
    buffer = TransitionReplayBuffer(ReplayBufferSpec(capacity=8, observation_dim=3, action_dim=2))
    transition = _base_transition(0, 4)
    transition["observations"].requires_grad_()
    buffer.add(**transition)

    transition["observations"].data.fill_(99.0)
    batch = buffer.sample(3, generator=torch.Generator().manual_seed(7))

    assert batch.observations.shape == (3, 3)
    assert batch.actions.shape == (3, 2)
    assert batch.rewards.shape == (3,)
    assert not batch.observations.requires_grad
    assert torch.all(batch.observations < 99.0)


def test_fastwmr_replay_preserves_every_extended_field() -> None:
    spec = ReplayBufferSpec(
        capacity=4,
        observation_dim=3,
        action_dim=2,
        privileged_state_dim=2,
        reconstruction_dim=5,
        require_temporal_metadata=True,
    )
    buffer = TransitionReplayBuffer(spec)
    transition = _base_transition(0, 2)
    transition["terminated"][0] = True
    transition["truncated"][1] = True
    final_observations = transition["next_observations"] + 10.0
    final_privileged_states = torch.tensor([[11.0, 12.0], [13.0, 14.0]])
    final_reconstructions = torch.arange(20, 30, dtype=torch.float32).reshape(2, 5)

    buffer.add(
        **transition,
        privileged_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        next_privileged_states=torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
        reconstructions=torch.arange(10, dtype=torch.float32).reshape(2, 5),
        next_reconstructions=torch.arange(10, 20, dtype=torch.float32).reshape(2, 5),
        estimator_versions=torch.tensor([7, 8]),
        episode_ids=torch.tensor([11, 12]),
        env_ids=torch.tensor([0, 1]),
        timesteps=torch.tensor([4, 9]),
        reset_boundaries=torch.tensor([False, False]),
        final_observations=final_observations,
        final_privileged_states=final_privileged_states,
        final_reconstructions=final_reconstructions,
        final_observation_mask=torch.tensor([True, True]),
    )
    retained = buffer.chronological()

    assert torch.equal(retained.privileged_states, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(retained.next_privileged_states, torch.tensor([[1.5, 2.5], [3.5, 4.5]]))
    assert retained.reconstructions.shape == (2, 5)
    assert retained.next_reconstructions.shape == (2, 5)
    assert torch.equal(retained.estimator_versions, torch.tensor([7, 8]))
    assert buffer.oldest_estimator_version == 7
    assert buffer.newest_estimator_version == 8
    assert torch.equal(retained.episode_ids, torch.tensor([11, 12]))
    assert torch.equal(retained.env_ids, torch.tensor([0, 1]))
    assert torch.equal(retained.timesteps, torch.tensor([4, 9]))
    assert torch.equal(retained.bootstrap_observations, final_observations)
    assert torch.equal(retained.bootstrap_privileged_states, final_privileged_states)
    assert torch.equal(retained.bootstrap_reconstructions, final_reconstructions)
    assert torch.equal(retained.episode_end, torch.tensor([True, True]))
    assert torch.equal(retained.bootstrap_mask, torch.tensor([0.0, 1.0]))


def test_stored_control_sampling_reserves_fresh_features_without_rejecting_replay() -> None:
    buffer = TransitionReplayBuffer(
        ReplayBufferSpec(
            capacity=8,
            observation_dim=3,
            action_dim=2,
            privileged_state_dim=2,
            reconstruction_dim=5,
            require_temporal_metadata=True,
        )
    )
    transition = _base_transition(0, 4)
    values = torch.arange(4, dtype=torch.float32)
    buffer.add(
        **transition,
        privileged_states=values[:, None].repeat(1, 2),
        next_privileged_states=(values + 1)[:, None].repeat(1, 2),
        reconstructions=values[:, None].repeat(1, 5),
        next_reconstructions=(values + 1)[:, None].repeat(1, 5),
        estimator_versions=torch.arange(4),
        episode_ids=torch.zeros(4, dtype=torch.int64),
        env_ids=torch.arange(4),
        timesteps=torch.zeros(4, dtype=torch.int64),
        reset_boundaries=torch.ones(4, dtype=torch.bool),
    )

    assert buffer.eligible_reconstruction_count(
        current_estimator_version=3,
        max_estimator_feature_age=1,
        recent_transition_horizon=3,
    ) == 2
    assert buffer.available_reconstruction_count(recent_transition_horizon=3) == 3
    batch = buffer.sample_reconstructions(
        2,
        current_estimator_version=3,
        max_estimator_feature_age=1,
        recent_transition_horizon=3,
        minimum_fresh_fraction=1.0,
        generator=torch.Generator().manual_seed(4),
    )

    assert batch.reconstructions.shape == (2, 5)
    assert batch.observations.shape == (2, 3)
    assert batch.bootstrap_observations.shape == (2, 3)
    assert torch.all(batch.estimator_versions >= 2)
    assert torch.all(batch.feature_ages(3) <= 1)
    assert not hasattr(batch, "privileged_states")


def test_fastwmr_spec_requires_extension_fields_and_temporal_metadata() -> None:
    spec = ReplayBufferSpec(
        capacity=4,
        observation_dim=3,
        action_dim=2,
        privileged_state_dim=2,
        reconstruction_dim=5,
        require_temporal_metadata=True,
    )
    buffer = TransitionReplayBuffer(spec)

    with pytest.raises(ValueError, match="privileged_states is required"):
        buffer.add(**_base_transition(0, 1))


def test_invalid_final_observation_mask_is_rejected() -> None:
    buffer = TransitionReplayBuffer(ReplayBufferSpec(capacity=4, observation_dim=3, action_dim=2))
    transition = _base_transition(0, 1)

    with pytest.raises(ValueError, match="terminated or truncated"):
        buffer.add(
            **transition,
            final_observations=torch.ones(1, 3),
            final_observation_mask=torch.tensor([True]),
        )


def test_oversized_insert_retains_only_capacity_newest_values() -> None:
    buffer = TransitionReplayBuffer(ReplayBufferSpec(capacity=3, observation_dim=3, action_dim=2))

    buffer.add(**_base_transition(0, 5))
    retained = buffer.chronological()

    assert torch.equal(retained.rewards, torch.tensor([2.0, 3.0, 4.0]))
    assert buffer.total_inserted == 5
    assert torch.equal(retained.insertion_ids, torch.tensor([2, 3, 4]))

    buffer.reset()

    assert len(buffer) == 0
    assert buffer.total_inserted == 0
    assert buffer.oldest_insertion_id is None
