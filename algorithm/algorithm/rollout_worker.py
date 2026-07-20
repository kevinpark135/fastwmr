"""FastSAC vector-environment collection and replay/update coordination."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..buffers import TransitionReplayBuffer
from ..utils.env_wrapper import IsaacLabEnvAdapter
from .fastwmr_agent import FastSACReplayUpdateLoop
from .sac_update import SACUpdateMetrics


@dataclass(frozen=True)
class RolloutStepResult:
    """Diagnostics returned after one vectorized collection step."""

    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    updates: tuple[SACUpdateMetrics, ...]


class FastSACRolloutCollector:
    """Collect baseline policy transitions and immediately run eligible updates."""

    def __init__(
        self,
        env: IsaacLabEnvAdapter,
        replay: TransitionReplayBuffer,
        update_loop: FastSACReplayUpdateLoop,
    ) -> None:
        if replay is not update_loop.replay:
            raise ValueError("Collector and update loop must share the same replay buffer.")
        if replay.spec.privileged_state_dim != 0 or replay.spec.control_feature_dim != 0:
            raise ValueError("FastSACRolloutCollector requires a policy-only replay specification.")
        self.env = env
        self.replay = replay
        self.update_loop = update_loop
        self._observations: dict[str, torch.Tensor] | None = None

    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        self._observations, _ = self.env.reset(seed=seed)
        policy = self.env.policy_observation(self._observations)
        self._validate_policy_shape(policy)
        return policy

    def collect_step(
        self,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> RolloutStepResult:
        if self._observations is None:
            raise RuntimeError("Call reset() before collecting transitions.")
        observations = self.env.policy_observation(self._observations)
        actions = self.update_loop.select_actions(observations, deterministic=deterministic)
        step = self.env.step(actions)
        next_observations = self.env.policy_observation(step.observations)
        final_observations = self.env.policy_observation(step.final_observations)
        self._validate_policy_shape(next_observations)

        self.replay.add(
            observations=observations,
            actions=actions,
            rewards=step.rewards,
            next_observations=next_observations,
            terminated=step.terminated,
            truncated=step.truncated,
            final_observations=final_observations,
            final_observation_mask=step.final_observation_mask,
        )
        self._observations = step.observations
        self.update_loop.advance_environment()
        updates = tuple(self.update_loop.run_updates(generator=generator))
        return RolloutStepResult(
            rewards=step.rewards.detach(),
            terminated=step.terminated.detach(),
            truncated=step.truncated.detach(),
            updates=updates,
        )

    def _validate_policy_shape(self, observations: torch.Tensor) -> None:
        expected = (self.env.num_envs, self.replay.spec.observation_dim)
        if observations.shape != expected:
            raise ValueError(f"Policy observation shape must be {expected}, got {tuple(observations.shape)}.")
