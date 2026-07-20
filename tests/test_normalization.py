"""Tests for FastSAC running observation normalization."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    FastSACReplayUpdateLoop,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ObservationNormalizationCfg,
    ReplayUpdateCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import TanhGaussianActor
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    RunningObservationNormalizer,
)


def test_parallel_welford_updates_match_population_moments() -> None:
    normalizer = RunningObservationNormalizer(2, ObservationNormalizationCfg(clip=None))
    first = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    second = torch.tensor([[5.0, 12.0], [7.0, 16.0], [9.0, 20.0]])
    all_observations = torch.cat((first, second))

    normalizer.update(first)
    normalizer.update(second)
    normalized = normalizer(all_observations)

    assert normalizer.samples_seen == 5
    assert torch.allclose(normalizer.mean, all_observations.double().mean(dim=0))
    assert torch.allclose(normalizer.variance, all_observations.double().var(dim=0, unbiased=False))
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalizer.denormalize(normalized), all_observations, atol=1e-5)


def test_eval_freezes_statistics_and_state_dict_restores_them() -> None:
    normalizer = RunningObservationNormalizer(3)
    normalizer.update(torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
    expected_mean = normalizer.mean.clone()
    normalizer.eval()

    normalizer.update(torch.full((4, 3), 100.0))
    restored = RunningObservationNormalizer(3)
    restored.load_state_dict(normalizer.state_dict())

    assert normalizer.samples_seen == 2
    assert torch.equal(normalizer.mean, expected_mean)
    assert restored.samples_seen == 2
    assert torch.equal(restored.mean, normalizer.mean)
    assert torch.equal(restored.variance, normalizer.variance)


def test_normalization_clips_and_preserves_input_gradient() -> None:
    normalizer = RunningObservationNormalizer(1, ObservationNormalizationCfg(clip=2.0))
    normalizer.update(torch.tensor([[-1.0], [1.0]]))
    observations = torch.tensor([[0.5], [100.0]], requires_grad=True)

    normalized = normalizer(observations)
    normalized.sum().backward()

    assert normalized[1].item() == 2.0
    assert observations.grad is not None
    assert observations.grad[0].item() > 0.0


class _RecordingUpdater:
    def __init__(self) -> None:
        self.actor = TanhGaussianActor(2, 1, cfg=TanhGaussianActorCfg(hidden_dim=16))
        self.batch = None

    def update(self, batch):
        self.batch = batch
        return object()


def test_replay_stays_raw_while_learner_receives_normalized_states() -> None:
    replay = TransitionReplayBuffer(ReplayBufferSpec(capacity=4, observation_dim=2, action_dim=1))
    replay.add(
        observations=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        actions=torch.zeros(2, 1),
        rewards=torch.zeros(2),
        next_observations=torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
    )
    normalizer = RunningObservationNormalizer(2, ObservationNormalizationCfg(clip=None))
    normalizer.update(torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    updater = _RecordingUpdater()
    loop = FastSACReplayUpdateLoop(
        replay,
        updater,
        ReplayUpdateCfg(random_action_steps=0, minimum_replay_size=2, batch_size=2, num_updates=1),
        learner_device="cpu",
        observation_normalizer=normalizer,
    )

    loop.run_updates(generator=torch.Generator().manual_seed(3))
    retained = replay.chronological()

    assert torch.equal(retained.observations, torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    assert updater.batch is not None
    assert torch.allclose(updater.batch.states.abs(), torch.ones_like(updater.batch.states), atol=1e-5)
