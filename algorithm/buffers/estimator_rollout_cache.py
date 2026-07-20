"""Recent ordered rollout cache for online estimator supervision.

This is a short-lived time-axis cache, not an off-policy replay buffer. It keeps
raw policy observations, privileged reconstruction targets, and episode reset
boundaries. Recurrent hidden/cell state is intentionally never stored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import DEFAULT_INTERFACE_CFG, FastWMRInterfaceCfg


@dataclass(frozen=True)
class EstimatorRolloutCacheSpec:
    """Fixed vector-environment and tensor dimensions for a rollout cache."""

    capacity_steps: int
    num_envs: int
    observation_dim: int
    privileged_state_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("capacity_steps", self.capacity_steps),
            ("num_envs", self.num_envs),
            ("observation_dim", self.observation_dim),
            ("privileged_state_dim", self.privileged_state_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

    @classmethod
    def fastwmr(
        cls,
        capacity_steps: int,
        num_envs: int,
        interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    ) -> "EstimatorRolloutCacheSpec":
        """Build the canonical FastWMR rollout-cache contract."""

        return cls(
            capacity_steps=capacity_steps,
            num_envs=num_envs,
            observation_dim=interface.policy_observation_dim,
            privileged_state_dim=interface.reconstruction_target_dim,
        )


@dataclass(frozen=True)
class EstimatorRolloutBatch:
    """Ordered cache contents in batch-first ``(env, time, feature)`` form."""

    observations: torch.Tensor
    privileged_states: torch.Tensor
    reset_boundaries: torch.Tensor

    def __post_init__(self) -> None:
        if self.observations.ndim != 3:
            raise ValueError("observations must have shape (num_envs, steps, observation_dim).")
        if self.privileged_states.ndim != 3:
            raise ValueError(
                "privileged_states must have shape (num_envs, steps, privileged_state_dim)."
            )
        if self.observations.shape[0] <= 0 or self.observations.shape[1] <= 0:
            raise ValueError("Rollout batches must contain at least one environment and timestep.")
        leading_shape = self.observations.shape[:2]
        if self.privileged_states.shape[:2] != leading_shape:
            raise ValueError("Observations and privileged states must share env/time dimensions.")
        if self.reset_boundaries.shape != leading_shape:
            raise ValueError("reset_boundaries must have shape (num_envs, steps).")
        if self.reset_boundaries.dtype is not torch.bool:
            raise TypeError("reset_boundaries must have dtype torch.bool.")
        floating_fields = (self.observations, self.privileged_states)
        if not all(tensor.dtype.is_floating_point for tensor in floating_fields):
            raise TypeError("Rollout observations and privileged states must be floating point.")
        if self.observations.dtype != self.privileged_states.dtype:
            raise ValueError("Rollout observations and privileged states must share a dtype.")
        if not all(torch.isfinite(tensor).all() for tensor in floating_fields):
            raise ValueError("Rollout observations and privileged states must be finite.")
        devices = {tensor.device for tensor in (*floating_fields, self.reset_boundaries)}
        if len(devices) != 1:
            raise ValueError("Every rollout batch tensor must share a device.")

    @property
    def num_envs(self) -> int:
        return self.observations.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.observations.shape[1]

    @property
    def context_is_exact(self) -> torch.Tensor:
        """Whether each cached sequence begins at a real episode reset."""

        return self.reset_boundaries[:, 0]

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> "EstimatorRolloutBatch":
        return EstimatorRolloutBatch(
            observations=self.observations.to(device=device, non_blocking=non_blocking),
            privileged_states=self.privileged_states.to(
                device=device,
                non_blocking=non_blocking,
            ),
            reset_boundaries=self.reset_boundaries.to(
                device=device,
                non_blocking=non_blocking,
            ),
        )


class EstimatorRolloutCache:
    """Preallocated ring of the most recent vectorized rollout timesteps."""

    def __init__(
        self,
        spec: EstimatorRolloutCacheSpec,
        *,
        storage_device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not dtype.is_floating_point:
            raise TypeError(f"Cache dtype must be floating point, got {dtype}.")
        self.spec = spec
        self.storage_device = torch.device(storage_device)
        self.dtype = dtype
        self._position = 0
        self._size = 0
        self._total_steps = 0
        self._observations = torch.empty(
            (spec.capacity_steps, spec.num_envs, spec.observation_dim),
            device=self.storage_device,
            dtype=dtype,
        )
        self._privileged_states = torch.empty(
            (spec.capacity_steps, spec.num_envs, spec.privileged_state_dim),
            device=self.storage_device,
            dtype=dtype,
        )
        self._reset_boundaries = torch.empty(
            (spec.capacity_steps, spec.num_envs),
            device=self.storage_device,
            dtype=torch.bool,
        )

    def __len__(self) -> int:
        """Return the number of cached timesteps, not flattened samples."""

        return self._size

    @property
    def capacity_steps(self) -> int:
        return self.spec.capacity_steps

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity_steps

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def add(
        self,
        observations: torch.Tensor,
        privileged_states: torch.Tensor,
        reset_boundaries: torch.Tensor,
    ) -> None:
        """Append one synchronized vector-environment timestep."""

        self._validate_step(observations, privileged_states, reset_boundaries)
        self._observations[self._position].copy_(
            observations.to(device=self.storage_device, dtype=self.dtype)
        )
        self._privileged_states[self._position].copy_(
            privileged_states.to(device=self.storage_device, dtype=self.dtype)
        )
        self._reset_boundaries[self._position].copy_(
            reset_boundaries.to(device=self.storage_device)
        )
        self._position = (self._position + 1) % self.capacity_steps
        self._size = min(self._size + 1, self.capacity_steps)
        self._total_steps += 1

    def chronological(self) -> EstimatorRolloutBatch:
        """Return a detached copy ordered from oldest to newest timestep."""

        if self._size == 0:
            raise RuntimeError("Cannot read an empty estimator rollout cache.")
        if self._size < self.capacity_steps:
            indices = torch.arange(self._size, device=self.storage_device)
        else:
            indices = torch.cat(
                (
                    torch.arange(self._position, self.capacity_steps, device=self.storage_device),
                    torch.arange(0, self._position, device=self.storage_device),
                )
            )
        return EstimatorRolloutBatch(
            observations=self._observations.index_select(0, indices).transpose(0, 1).clone(),
            privileged_states=self._privileged_states.index_select(0, indices)
            .transpose(0, 1)
            .clone(),
            reset_boundaries=self._reset_boundaries.index_select(0, indices)
            .transpose(0, 1)
            .clone(),
        )

    def drain(self) -> EstimatorRolloutBatch:
        """Return the current ordered chunk and clear it for the next rollout."""

        batch = self.chronological()
        self.clear()
        return batch

    def clear(self) -> None:
        self._position = 0
        self._size = 0

    def _validate_step(
        self,
        observations: torch.Tensor,
        privileged_states: torch.Tensor,
        reset_boundaries: torch.Tensor,
    ) -> None:
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (observations, privileged_states, reset_boundaries)
        ):
            raise TypeError("Every estimator rollout cache field must be a torch.Tensor.")
        expected_observations = (self.spec.num_envs, self.spec.observation_dim)
        expected_privileged = (self.spec.num_envs, self.spec.privileged_state_dim)
        if observations.shape != expected_observations:
            raise ValueError(
                f"observations must have shape {expected_observations}, got {tuple(observations.shape)}."
            )
        if privileged_states.shape != expected_privileged:
            raise ValueError(
                "privileged_states must have shape "
                f"{expected_privileged}, got {tuple(privileged_states.shape)}."
            )
        if reset_boundaries.shape != (self.spec.num_envs,):
            raise ValueError(
                f"reset_boundaries must have shape ({self.spec.num_envs},), "
                f"got {tuple(reset_boundaries.shape)}."
            )
        if not observations.dtype.is_floating_point or not privileged_states.dtype.is_floating_point:
            raise TypeError("observations and privileged_states must be floating point.")
        if reset_boundaries.dtype is not torch.bool:
            raise TypeError("reset_boundaries must have dtype torch.bool.")
        if not torch.isfinite(observations).all() or not torch.isfinite(privileged_states).all():
            raise ValueError("observations and privileged_states must be finite.")
