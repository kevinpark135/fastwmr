"""Tests for FastWMR boundary-safe burn-in sequence sampling."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    TransitionReplayBuffer,
)


def _buffer(capacity: int = 32) -> TransitionReplayBuffer:
    return TransitionReplayBuffer(
        ReplayBufferSpec(
            capacity=capacity,
            observation_dim=3,
            action_dim=2,
            privileged_state_dim=2,
            control_feature_dim=5,
            require_temporal_metadata=True,
        )
    )


def _add_vector_step(
    buffer: TransitionReplayBuffer,
    timestep: int,
    *,
    episode_id: int = 0,
    num_envs: int = 2,
    terminated: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
) -> None:
    env_ids = torch.arange(num_envs, dtype=torch.int64)
    values = env_ids.to(torch.float32) * 100.0 + timestep
    terminated = terminated if terminated is not None else torch.zeros(num_envs, dtype=torch.bool)
    truncated = truncated if truncated is not None else torch.zeros(num_envs, dtype=torch.bool)
    buffer.add(
        observations=values[:, None].repeat(1, 3),
        actions=values[:, None].repeat(1, 2),
        rewards=values,
        next_observations=(values + 1.0)[:, None].repeat(1, 3),
        terminated=terminated,
        truncated=truncated,
        privileged_states=values[:, None].repeat(1, 2),
        next_privileged_states=(values + 1.0)[:, None].repeat(1, 2),
        control_features=values[:, None].repeat(1, 5),
        next_control_features=(values + 1.0)[:, None].repeat(1, 5),
        estimator_versions=torch.zeros(num_envs, dtype=torch.int64),
        episode_ids=torch.full((num_envs,), episode_id, dtype=torch.int64),
        env_ids=env_ids,
        timesteps=torch.full((num_envs,), timestep, dtype=torch.int64),
        reset_boundaries=torch.full((num_envs,), timestep == 0, dtype=torch.bool),
    )


def test_vector_env_interleaving_reconstructs_consecutive_sequences() -> None:
    buffer = _buffer()
    for timestep in range(6):
        _add_vector_step(buffer, timestep)

    sequence = buffer.sample_sequences(
        batch_size=2,
        burn_in_length=2,
        learning_length=2,
        require_episode_start=True,
        generator=torch.Generator().manual_seed(5),
    )

    assert sequence.observations.shape == (2, 5, 3)
    assert sequence.privileged_states.shape == (2, 5, 2)
    assert sequence.stored_control_features.shape == (2, 5, 5)
    assert sequence.actions.shape == (2, 4, 2)
    assert sequence.learning_observations.shape == (2, 3, 3)
    assert sequence.learning_actions.shape == (2, 2, 2)
    assert torch.all(sequence.timesteps[:, 1:] - sequence.timesteps[:, :-1] == 1)
    assert torch.all(sequence.env_ids == sequence.env_ids[:, :1])
    assert torch.all(sequence.episode_ids == sequence.episode_ids[:, :1])
    assert torch.all(sequence.context_is_exact)
    assert torch.equal(sequence.observations[:, -1, 0], sequence.observations[:, -2, 0] + 1.0)


def test_sequences_never_cross_episode_boundaries() -> None:
    buffer = _buffer()
    for timestep in range(3):
        _add_vector_step(
            buffer,
            timestep,
            num_envs=1,
            terminated=torch.tensor([timestep == 2]),
        )
    for timestep in range(3):
        _add_vector_step(buffer, timestep, episode_id=1, num_envs=1)

    assert not buffer.can_sample_sequences(batch_size=1, burn_in_length=2, learning_length=2)
    sequence = buffer.sample_sequences(batch_size=2, burn_in_length=1, learning_length=2)

    assert torch.all(sequence.episode_ids == sequence.episode_ids[:, :1])
    assert torch.all(sequence.timesteps == torch.tensor([[0, 1, 2], [0, 1, 2]]))


def test_terminal_sequence_uses_pre_reset_final_observation() -> None:
    buffer = _buffer()
    _add_vector_step(buffer, 0, num_envs=1)
    value = torch.tensor([1.0])
    buffer.add(
        observations=value[:, None].repeat(1, 3),
        actions=value[:, None].repeat(1, 2),
        rewards=value,
        next_observations=torch.full((1, 3), -1.0),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        privileged_states=value[:, None].repeat(1, 2),
        next_privileged_states=torch.full((1, 2), -1.0),
        control_features=value[:, None].repeat(1, 5),
        next_control_features=torch.full((1, 5), -1.0),
        estimator_versions=torch.tensor([0]),
        episode_ids=torch.tensor([0]),
        env_ids=torch.tensor([0]),
        timesteps=torch.tensor([1]),
        reset_boundaries=torch.tensor([False]),
        final_observations=torch.full((1, 3), 2.0),
        final_privileged_states=torch.full((1, 2), 2.0),
        final_control_features=torch.full((1, 5), 2.0),
    )

    sequence = buffer.sample_sequences(
        batch_size=1,
        burn_in_length=0,
        learning_length=2,
        require_episode_start=True,
    )

    assert torch.equal(sequence.observations[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(sequence.privileged_states[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(sequence.stored_control_features[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(sequence.terminated, torch.tensor([[False, True]]))


def test_sequence_cache_tracks_ring_overwrite_without_stale_starts() -> None:
    buffer = _buffer(capacity=3)
    for timestep in range(3):
        _add_vector_step(buffer, timestep, num_envs=1)
    assert buffer.can_sample_sequences(batch_size=1, burn_in_length=1, learning_length=2)

    _add_vector_step(buffer, 3, num_envs=1)
    sequence = buffer.sample_sequences(batch_size=1, burn_in_length=1, learning_length=2)

    assert torch.equal(sequence.timesteps, torch.tensor([[1, 2, 3]]))
    assert not sequence.context_is_exact.item()
    assert not buffer.can_sample_sequences(
        batch_size=1,
        burn_in_length=1,
        learning_length=2,
        require_episode_start=True,
    )

    buffer.clear()
    assert not buffer.can_sample_sequences(batch_size=1, burn_in_length=1, learning_length=2)


def test_tensor_sequence_index_handles_vector_steps_across_unaligned_wraparound() -> None:
    buffer = _buffer(capacity=5)
    for timestep in range(5):
        _add_vector_step(buffer, timestep, num_envs=2)

    sequence = buffer.sample_sequences(
        batch_size=3,
        burn_in_length=0,
        learning_length=2,
        generator=torch.Generator().manual_seed(12),
    )

    assert torch.all(sequence.timesteps[:, 1] - sequence.timesteps[:, 0] == 1)
    assert torch.all(sequence.env_ids == sequence.env_ids[:, :1])
    assert {
        (int(env_id), int(timestep))
        for env_id, timestep in zip(
            sequence.env_ids[:, 0],
            sequence.timesteps[:, 0],
            strict=True,
        )
    } == {(1, 2), (0, 3), (1, 3)}


def test_sequence_sampling_can_be_limited_to_recent_insertions() -> None:
    buffer = _buffer()
    for timestep in range(8):
        _add_vector_step(buffer, timestep)

    assert buffer.can_sample_sequences(
        batch_size=2,
        burn_in_length=0,
        learning_length=2,
        minimum_insertion_id=12,
    )
    sequence = buffer.sample_sequences(
        batch_size=2,
        burn_in_length=0,
        learning_length=2,
        minimum_insertion_id=12,
        generator=torch.Generator().manual_seed(11),
    )

    assert torch.all(sequence.insertion_ids[:, 0] >= 12)
    assert not buffer.can_sample_sequences(
        batch_size=1,
        burn_in_length=0,
        learning_length=2,
        minimum_insertion_id=15,
    )
