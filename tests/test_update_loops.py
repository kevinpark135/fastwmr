"""Tests for replay warm-up, recurrent burn-in, and update scheduling."""

import torch
from torch import nn

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    EntropyTemperature,
    FastSACReplayUpdateLoop,
    FastWMRSequenceUpdateLoop,
    SACUpdater,
    burn_in_and_unroll,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    SequenceReplayBatch,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ReplayUpdateCfg,
    ScalarCriticCfg,
    SequenceReplayCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinScalarCritic,
    TanhGaussianActor,
    TwinScalarCritic,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    RunningObservationNormalizer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils.temporal_state import (
    RecurrentState,
)


def _updater(state_dim: int = 4, action_dim: int = 2) -> SACUpdater:
    actor = TanhGaussianActor(
        input_dim=state_dim,
        action_dim=action_dim,
        cfg=TanhGaussianActorCfg(hidden_dim=16),
    )
    critic = TwinScalarCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=ScalarCriticCfg(hidden_dim=16),
    )
    target = TargetTwinScalarCritic.from_online(critic)
    temperature = EntropyTemperature(0.001)
    return SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=3e-4),
    )


def _sequence() -> SequenceReplayBatch:
    return SequenceReplayBatch(
        observations=torch.randn(2, 5, 3),
        privileged_states=torch.randn(2, 5, 2),
        stored_reconstructions=torch.randn(2, 5, 4),
        actions=torch.randn(2, 4, 2),
        rewards=torch.randn(2, 4),
        terminated=torch.zeros(2, 4, dtype=torch.bool),
        truncated=torch.zeros(2, 4, dtype=torch.bool),
        episode_ids=torch.zeros(2, 4, dtype=torch.int64),
        env_ids=torch.arange(2, dtype=torch.int64)[:, None].expand(2, 4),
        timesteps=torch.arange(4, dtype=torch.int64)[None].expand(2, 4),
        reset_boundaries=torch.tensor([[True, False, False, False], [True, False, False, False]]),
        insertion_ids=torch.arange(8, dtype=torch.int64).reshape(2, 4),
        burn_in_length=2,
        learning_length=2,
    )


class _ToyEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.Linear(3, 3)
        self.decoder = nn.Linear(3, 2)
        self.grad_modes: list[bool] = []

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentState:
        zeros = torch.zeros(1, batch_size, 3, device=device, dtype=dtype)
        return RecurrentState(hidden=zeros, cell=zeros.clone())

    def forward_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        self.grad_modes.append(torch.is_grad_enabled())
        hidden = state.hidden
        outputs = []
        for timestep in range(observations.shape[1]):
            hidden = hidden + self.recurrent(observations[:, timestep]).unsqueeze(0)
            outputs.append(self.decoder(hidden.squeeze(0)))
        return torch.stack(outputs, dim=1), RecurrentState(hidden=hidden, cell=hidden)


def test_burn_in_is_no_grad_and_learning_unroll_keeps_gradient() -> None:
    estimator = _ToyEstimator()

    output = burn_in_and_unroll(estimator, _sequence())
    output.reconstructions.square().mean().backward()

    assert estimator.grad_modes == [False, True]
    assert output.reconstructions.shape == (2, 3, 2)
    assert output.reconstructions.requires_grad
    assert not output.learning_initial_state.hidden.requires_grad
    assert output.final_state.hidden.requires_grad
    assert torch.all(output.context_is_exact)
    assert estimator.decoder.weight.grad is not None


def test_fastsac_loop_obeys_random_warmup_minimum_replay_and_num_updates() -> None:
    replay = TransitionReplayBuffer(ReplayBufferSpec(capacity=16, observation_dim=4, action_dim=2))
    replay.add(
        observations=torch.randn(8, 4),
        actions=torch.rand(8, 2) * 2.0 - 1.0,
        rewards=torch.randn(8),
        next_observations=torch.randn(8, 4),
        terminated=torch.zeros(8, dtype=torch.bool),
        truncated=torch.zeros(8, dtype=torch.bool),
    )
    loop = FastSACReplayUpdateLoop(
        replay,
        _updater(),
        ReplayUpdateCfg(random_action_steps=2, minimum_replay_size=4, batch_size=4, num_updates=2),
        learner_device="cpu",
    )
    states = torch.randn(3, 4)

    warmup_actions = loop.select_actions(states)
    assert loop.warming_up
    assert loop.run_updates() == []
    assert torch.all(warmup_actions >= loop.updater.actor.action_low)
    assert torch.all(warmup_actions <= loop.updater.actor.action_high)

    loop.advance_environment(2)
    metrics = loop.run_updates(generator=torch.Generator().manual_seed(1))

    assert not loop.warming_up
    assert len(metrics) == 2
    assert loop.gradient_steps == 2


def test_normalizer_freeze_stops_statistics_but_keeps_transform_active() -> None:
    replay = TransitionReplayBuffer(
        ReplayBufferSpec(capacity=8, observation_dim=4, action_dim=2)
    )
    normalizer = RunningObservationNormalizer(4)
    loop = FastSACReplayUpdateLoop(
        replay,
        _updater(),
        ReplayUpdateCfg(
            random_action_steps=0,
            minimum_replay_size=1,
            batch_size=1,
            num_updates=1,
        ),
        learner_device="cpu",
        observation_normalizer=normalizer,
        normalizer_freeze_iteration=2,
    )

    loop.update_observation_statistics(torch.ones(3, 4))
    loop.advance_environment()
    loop.update_observation_statistics(torch.full((2, 4), 3.0))
    mean_at_freeze = normalizer.mean.clone()
    count_at_freeze = normalizer.samples_seen

    loop.advance_environment()
    assert loop.normalization_frozen
    loop.update_observation_statistics(torch.full((5, 4), 100.0))

    assert normalizer.samples_seen == count_at_freeze == 5
    assert torch.equal(normalizer.mean, mean_at_freeze)
    assert torch.isfinite(loop.normalize_observations(torch.ones(1, 4))).all()


def test_fastwmr_loop_samples_sequences_and_updates_only_learning_window() -> None:
    replay = TransitionReplayBuffer(
        ReplayBufferSpec(
            capacity=16,
            observation_dim=3,
            action_dim=2,
            privileged_state_dim=2,
            reconstruction_dim=4,
            require_temporal_metadata=True,
        )
    )
    for timestep in range(6):
        value = torch.tensor([float(timestep)])
        replay.add(
            observations=value[:, None].repeat(1, 3),
            actions=value[:, None].repeat(1, 2),
            rewards=value,
            next_observations=(value + 1.0)[:, None].repeat(1, 3),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([False]),
            privileged_states=value[:, None].repeat(1, 2),
            next_privileged_states=(value + 1.0)[:, None].repeat(1, 2),
            reconstructions=value[:, None].repeat(1, 4),
            next_reconstructions=(value + 1.0)[:, None].repeat(1, 4),
            estimator_versions=torch.tensor([0]),
            episode_ids=torch.tensor([0]),
            env_ids=torch.tensor([0]),
            timesteps=torch.tensor([timestep]),
            reset_boundaries=torch.tensor([timestep == 0]),
        )

    sampled_sequences: list[SequenceReplayBatch] = []

    def process_sequence(sequence: SequenceReplayBatch) -> torch.Tensor:
        sampled_sequences.append(sequence)
        return sequence.learning_stored_reconstructions.detach()

    loop = FastWMRSequenceUpdateLoop(
        replay,
        _updater(),
        ReplayUpdateCfg(random_action_steps=0, minimum_replay_size=4, batch_size=4, num_updates=2),
        SequenceReplayCfg(batch_size=1, burn_in_length=2, learning_length=2),
        process_sequence,
        learner_device="cpu",
    )

    metrics = loop.run_updates(generator=torch.Generator().manual_seed(2))

    assert len(metrics) == 2
    assert loop.gradient_steps == 2
    assert len(sampled_sequences) == 2
    assert all(sequence.burn_in_observations.shape == (1, 2, 3) for sequence in sampled_sequences)
    assert all(sequence.learning_actions.shape == (1, 2, 2) for sequence in sampled_sequences)
