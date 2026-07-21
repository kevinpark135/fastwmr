"""Tests for G1 raw-sequence symmetry augmentation."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    augment_sequence_batch,
    mirror_action,
    mirror_policy_observation,
    mirror_reconstruction_target,
    mirror_sequence_batch,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm \
    import symmetry_utils
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    SequenceReplayBatch,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_INTERFACE_CFG,
)


def _sequence() -> SequenceReplayBatch:
    cfg = DEFAULT_INTERFACE_CFG
    batch_size = 2
    transition_length = 3
    observations = torch.randn(batch_size, transition_length + 1, cfg.policy_observation_dim)
    privileged = torch.randn(
        batch_size,
        transition_length + 1,
        cfg.reconstruction_target_dim,
    )
    privileged[..., -2:] = torch.randint(
        0,
        2,
        privileged[..., -2:].shape,
        dtype=torch.float32,
    )
    stored = torch.cat((observations, privileged), dim=-1)
    return SequenceReplayBatch(
        observations=observations,
        privileged_states=privileged,
        stored_control_features=stored,
        actions=torch.randn(batch_size, transition_length, cfg.action_dim),
        rewards=torch.randn(batch_size, transition_length),
        terminated=torch.zeros(batch_size, transition_length, dtype=torch.bool),
        truncated=torch.zeros(batch_size, transition_length, dtype=torch.bool),
        episode_ids=torch.arange(batch_size, dtype=torch.int64)[:, None].expand(-1, transition_length),
        env_ids=torch.arange(batch_size, dtype=torch.int64)[:, None].expand(-1, transition_length),
        timesteps=torch.arange(transition_length, dtype=torch.int64)[None].expand(batch_size, -1),
        reset_boundaries=torch.tensor([[True, False, False], [True, False, False]]),
        insertion_ids=torch.arange(batch_size * transition_length, dtype=torch.int64).reshape(
            batch_size,
            transition_length,
        ),
        burn_in_length=1,
        learning_length=2,
    )


def test_g1_mirror_transforms_are_involutions() -> None:
    cfg = DEFAULT_INTERFACE_CFG
    action = torch.randn(4, cfg.action_dim)
    observation = torch.randn(4, cfg.policy_observation_dim)
    target = torch.randn(4, cfg.reconstruction_target_dim)

    torch.testing.assert_close(mirror_action(mirror_action(action)), action)
    torch.testing.assert_close(
        mirror_policy_observation(mirror_policy_observation(observation)),
        observation,
    )
    torch.testing.assert_close(
        mirror_reconstruction_target(mirror_reconstruction_target(target)),
        target,
    )


def test_action_mirror_matches_resolved_g1_joint_order() -> None:
    assert symmetry_utils.G1_ACTION_MIRROR_PERMUTATION == (
        1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13,
        16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27,
    )
    assert symmetry_utils.G1_ACTION_MIRROR_SIGNS == (
        1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
        -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
    )


def test_sequence_augmentation_appends_mirrored_raw_recurrent_inputs() -> None:
    sequence = _sequence()
    mirrored = mirror_sequence_batch(sequence)
    augmented = augment_sequence_batch(sequence)

    assert augmented.batch_size == sequence.batch_size * 2
    torch.testing.assert_close(augmented.observations[: sequence.batch_size], sequence.observations)
    torch.testing.assert_close(augmented.observations[sequence.batch_size :], mirrored.observations)
    torch.testing.assert_close(augmented.actions[sequence.batch_size :], mirrored.actions)
    torch.testing.assert_close(
        mirror_sequence_batch(mirrored).observations,
        sequence.observations,
    )
