"""Tests for reset-safe IsaacLab adaptation and FastSAC collection."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    EntropyTemperature,
    FastSACReplayUpdateLoop,
    FastSACRolloutCollector,
    SACUpdater,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ReplayUpdateCfg,
    ScalarCriticCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinScalarCritic,
    TanhGaussianActor,
    TwinScalarCritic,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import IsaacLabEnvAdapter


class _ObservationManager:
    def __init__(self, owner: "_RawEnv") -> None:
        self.owner = owner

    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor]:
        del update_history
        return {"policy": self.owner.state.clone()}


class _RawEnv:
    def __init__(self) -> None:
        self.device = "cpu"
        self.num_envs = 2
        self.state = torch.zeros(2, 2)
        self.observation_manager = _ObservationManager(self)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids.long()] = 0.0


class _GymEnv:
    def __init__(self) -> None:
        self.unwrapped = _RawEnv()
        self.steps = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        del seed
        self.unwrapped.state.zero_()
        return self.unwrapped.observation_manager.compute(), {}

    def step(self, actions: torch.Tensor):
        assert actions.shape == (2, 1)
        self.steps += 1
        self.unwrapped.state += 1.0
        terminated = torch.tensor([self.steps == 1, False])
        truncated = torch.zeros(2, dtype=torch.bool)
        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.unwrapped._reset_idx(done_ids)
        observations = self.unwrapped.observation_manager.compute()
        rewards = torch.ones(2)
        return observations, rewards, terminated, truncated, {}

    def close(self) -> None:
        self.closed = True


def _updater() -> SACUpdater:
    actor = TanhGaussianActor(2, 1, cfg=TanhGaussianActorCfg(hidden_dim=16))
    critic = TwinScalarCritic(2, 1, cfg=ScalarCriticCfg(hidden_dim=16))
    target = TargetTwinScalarCritic.from_online(critic)
    temperature = EntropyTemperature()
    return SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=3e-4),
    )


def test_adapter_captures_observation_before_internal_auto_reset() -> None:
    raw_env = _GymEnv()
    env = IsaacLabEnvAdapter(raw_env)
    env.reset(seed=1)

    step = env.step(torch.zeros(2, 1))

    assert torch.equal(step.observations["policy"][0], torch.zeros(2))
    assert torch.equal(step.final_observations["policy"][0], torch.ones(2))
    assert torch.equal(step.final_observation_mask, torch.tensor([True, False]))
    env.close()
    assert raw_env.closed


def test_collector_runs_replay_wraparound_and_gradient_updates() -> None:
    env = IsaacLabEnvAdapter(_GymEnv())
    replay = TransitionReplayBuffer(ReplayBufferSpec(capacity=4, observation_dim=2, action_dim=1))
    loop = FastSACReplayUpdateLoop(
        replay,
        _updater(),
        ReplayUpdateCfg(random_action_steps=0, minimum_replay_size=2, batch_size=2, num_updates=1),
        learner_device="cpu",
    )
    collector = FastSACRolloutCollector(env, replay, loop)
    collector.reset(seed=1)

    results = [collector.collect_step() for _ in range(3)]

    assert replay.total_inserted == 6
    assert len(replay) == 4
    assert replay.is_full
    assert loop.environment_steps == 3
    assert loop.gradient_steps == 3
    assert all(len(result.updates) == 1 for result in results)
    env.close()
