"""IsaacLab environment adapter with reset-safe terminal observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ObservationGroups = dict[str, torch.Tensor]


@dataclass(frozen=True)
class EnvStep:
    """One vector-environment step before replay-specific field selection."""

    observations: ObservationGroups
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    extras: dict[str, Any]
    final_observations: ObservationGroups
    final_observation_mask: torch.Tensor


class IsaacLabEnvAdapter:
    """Expose observation groups and preserve observations immediately before reset.

    ``ManagerBasedRLEnv.step`` resets completed environments before returning
    observations. The adapter temporarily wraps ``_reset_idx`` so Bellman
    successors can use the pre-reset observation instead of the first state of
    the next episode. No IsaacLab source modification is required.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        if not hasattr(self.unwrapped, "observation_manager") or not hasattr(self.unwrapped, "_reset_idx"):
            raise TypeError("IsaacLabEnvAdapter requires a ManagerBasedRLEnv-compatible environment.")

        self._pending_final_observations: ObservationGroups | None = None
        self._pending_final_env_ids: torch.Tensor | None = None
        self._had_instance_reset = "_reset_idx" in vars(self.unwrapped)
        self._instance_reset = vars(self.unwrapped).get("_reset_idx")
        self._original_reset_idx = self.unwrapped._reset_idx

        def capture_then_reset(env_ids: torch.Tensor) -> Any:
            observations = self.unwrapped.observation_manager.compute(update_history=False)
            self._pending_final_observations = self._validate_observations(observations, clone=True)
            self._pending_final_env_ids = env_ids.to(dtype=torch.int64).clone()
            return self._original_reset_idx(env_ids)

        self.unwrapped._reset_idx = capture_then_reset
        self._closed = False

    @property
    def device(self) -> torch.device:
        return torch.device(self.unwrapped.device)

    @property
    def num_envs(self) -> int:
        return int(self.unwrapped.num_envs)

    def reset(self, *, seed: int | None = None) -> tuple[ObservationGroups, dict[str, Any]]:
        self._pending_final_observations = None
        self._pending_final_env_ids = None
        observations, extras = self.env.reset(seed=seed)
        return self._validate_observations(observations), extras

    def step(self, actions: torch.Tensor) -> EnvStep:
        self._pending_final_observations = None
        self._pending_final_env_ids = None
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        observations = self._validate_observations(observations)
        done = terminated | truncated
        final_observations = {name: torch.zeros_like(value) for name, value in observations.items()}

        if torch.any(done):
            if self._pending_final_observations is None or self._pending_final_env_ids is None:
                raise RuntimeError("IsaacLab reset occurred without a captured final observation.")
            expected_ids = done.nonzero(as_tuple=False).squeeze(-1).to(dtype=torch.int64)
            captured_ids = self._pending_final_env_ids.to(device=expected_ids.device)
            if not torch.equal(captured_ids, expected_ids):
                raise RuntimeError("Captured reset indices do not match terminated/truncated environments.")
            for name, destination in final_observations.items():
                if name not in self._pending_final_observations:
                    raise RuntimeError(f"Final observation group {name!r} was not captured.")
                source = self._pending_final_observations[name]
                destination[expected_ids] = source[expected_ids]
        elif self._pending_final_env_ids is not None:
            raise RuntimeError("IsaacLab invoked reset without returning a done flag.")

        return EnvStep(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            extras=extras,
            final_observations=final_observations,
            final_observation_mask=done,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._had_instance_reset:
            self.unwrapped._reset_idx = self._instance_reset
        else:
            delattr(self.unwrapped, "_reset_idx")
        self.env.close()
        self._closed = True

    @staticmethod
    def policy_observation(observations: ObservationGroups) -> torch.Tensor:
        try:
            policy = observations["policy"]
        except KeyError as error:
            raise KeyError("Environment observations must contain a 'policy' group.") from error
        if policy.ndim != 2 or not policy.dtype.is_floating_point:
            raise ValueError("The policy observation must be a floating tensor of shape (num_envs, obs_dim).")
        return policy

    @staticmethod
    def _validate_observations(observations: Any, *, clone: bool = False) -> ObservationGroups:
        if not isinstance(observations, dict):
            raise TypeError("IsaacLab observations must be a dictionary of tensor groups.")
        validated: ObservationGroups = {}
        for name, value in observations.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise TypeError("Observation groups must map string names to tensors.")
            validated[name] = value.detach().clone() if clone else value
        return validated
