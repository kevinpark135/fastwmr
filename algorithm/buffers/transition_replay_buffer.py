"""Transition replay shared by the FastSAC baseline and FastWMR.

FastSAC only needs ``(o_t, a_t, r_t, o_{t+1}, terminated, truncated)``.
FastWMR keeps that ordinary off-policy sampling contract, but also records the
raw privileged targets, detached control features, estimator version, and
temporal indexing needed to diagnose stale representations or add sequence
re-inference later. Recurrent hidden/cell state is runtime state and must never
be stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..config import DEFAULT_INTERFACE_CFG, FastWMRInterfaceCfg


@dataclass(frozen=True)
class ReplayBufferSpec:
    """Fixed storage dimensions and validation policy for transition replay."""

    capacity: int
    observation_dim: int
    action_dim: int
    privileged_state_dim: int = 0
    control_feature_dim: int = 0
    require_temporal_metadata: bool = False

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}.")
        if self.observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {self.observation_dim}.")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.privileged_state_dim < 0 or self.control_feature_dim < 0:
            raise ValueError("Optional replay dimensions must be non-negative.")

    @classmethod
    def fastwmr(
        cls,
        capacity: int,
        interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    ) -> "ReplayBufferSpec":
        """Build the full FastWMR replay contract from the shared interface."""

        return cls(
            capacity=capacity,
            observation_dim=interface.policy_observation_dim,
            action_dim=interface.action_dim,
            privileged_state_dim=interface.reconstruction_target_dim,
            control_feature_dim=interface.control_feature_dim,
            require_temporal_metadata=True,
        )

    @property
    def is_fastwmr(self) -> bool:
        return self.privileged_state_dim > 0 or self.control_feature_dim > 0


@dataclass(frozen=True)
class TransitionReplayBatch:
    """A sampled transition batch with base and FastWMR extension fields."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    privileged_states: torch.Tensor
    next_privileged_states: torch.Tensor
    control_features: torch.Tensor
    next_control_features: torch.Tensor
    estimator_versions: torch.Tensor
    episode_ids: torch.Tensor
    env_ids: torch.Tensor
    timesteps: torch.Tensor
    reset_boundaries: torch.Tensor
    final_observations: torch.Tensor
    final_privileged_states: torch.Tensor
    final_control_features: torch.Tensor
    final_observation_mask: torch.Tensor
    insertion_ids: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.observations.shape[0]

    @property
    def episode_end(self) -> torch.Tensor:
        """Mask used to reset recurrent rollout state."""

        return self.terminated | self.truncated

    @property
    def bootstrap_mask(self) -> torch.Tensor:
        """Bellman bootstrap mask; time-limit truncations still bootstrap."""

        return (~self.terminated).to(dtype=self.rewards.dtype)

    @property
    def bootstrap_observations(self) -> torch.Tensor:
        """Use pre-reset final observations when an auto-reset env supplies them."""

        return torch.where(
            self.final_observation_mask.unsqueeze(-1),
            self.final_observations,
            self.next_observations,
        )

    @property
    def bootstrap_privileged_states(self) -> torch.Tensor:
        """Privileged successors aligned with :attr:`bootstrap_observations`."""

        return torch.where(
            self.final_observation_mask.unsqueeze(-1),
            self.final_privileged_states,
            self.next_privileged_states,
        )

    @property
    def bootstrap_control_features(self) -> torch.Tensor:
        """Detached control successors aligned with Bellman bootstrapping."""

        return torch.where(
            self.final_observation_mask.unsqueeze(-1),
            self.final_control_features,
            self.next_control_features,
        )

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "TransitionReplayBatch":
        """Move every field together so metadata cannot drift from transitions."""

        return TransitionReplayBatch(
            **{
                field.name: getattr(self, field.name).to(device=device, non_blocking=non_blocking)
                for field in fields(self)
            }
        )


class TransitionReplayBuffer:
    """Preallocated circular replay with vector-environment batch insertion."""

    def __init__(
        self,
        spec: ReplayBufferSpec,
        *,
        storage_device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not dtype.is_floating_point:
            raise TypeError(f"Replay floating-point dtype must be floating, got {dtype}.")

        self.spec = spec
        self.storage_device = torch.device(storage_device)
        self.dtype = dtype
        self._position = 0
        self._size = 0
        self._total_inserted = 0

        vector = lambda width: torch.empty((spec.capacity, width), dtype=dtype, device=self.storage_device)
        scalar = lambda value_dtype: torch.empty((spec.capacity,), dtype=value_dtype, device=self.storage_device)

        self._observations = vector(spec.observation_dim)
        self._actions = vector(spec.action_dim)
        self._rewards = scalar(dtype)
        self._next_observations = vector(spec.observation_dim)
        self._terminated = scalar(torch.bool)
        self._truncated = scalar(torch.bool)

        # Width-zero tensors keep the sampled batch API uniform for FastSAC.
        self._privileged_states = vector(spec.privileged_state_dim)
        self._next_privileged_states = vector(spec.privileged_state_dim)
        self._control_features = vector(spec.control_feature_dim)
        self._next_control_features = vector(spec.control_feature_dim)

        self._estimator_versions = scalar(torch.int64)
        self._episode_ids = scalar(torch.int64)
        self._env_ids = scalar(torch.int64)
        self._timesteps = scalar(torch.int64)
        self._reset_boundaries = scalar(torch.bool)
        self._final_observations = vector(spec.observation_dim)
        self._final_privileged_states = vector(spec.privileged_state_dim)
        self._final_control_features = vector(spec.control_feature_dim)
        self._final_observation_mask = scalar(torch.bool)
        self._insertion_ids = scalar(torch.int64)

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self.spec.capacity

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def total_inserted(self) -> int:
        """Monotonic transition count, useful for replay-age diagnostics."""

        return self._total_inserted

    def can_sample(self, batch_size: int) -> bool:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        return self._size >= batch_size

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        *,
        privileged_states: torch.Tensor | None = None,
        next_privileged_states: torch.Tensor | None = None,
        control_features: torch.Tensor | None = None,
        next_control_features: torch.Tensor | None = None,
        estimator_versions: torch.Tensor | None = None,
        episode_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        reset_boundaries: torch.Tensor | None = None,
        final_observations: torch.Tensor | None = None,
        final_privileged_states: torch.Tensor | None = None,
        final_control_features: torch.Tensor | None = None,
        final_observation_mask: torch.Tensor | None = None,
    ) -> None:
        """Append one transition or a leading batch of vector-env transitions.

        Inputs are detached and copied onto ``storage_device``. If a single add
        is larger than capacity, only its newest transitions are retained.
        ``final_observations`` refers to the pre-reset state returned separately
        by auto-reset environments; ``next_observations`` remains the raw step
        result so both values are preserved.
        """

        observations = self._as_matrix(observations, self.spec.observation_dim, "observations")
        batch_size = observations.shape[0]
        actions = self._as_matrix(actions, self.spec.action_dim, "actions", batch_size)
        rewards = self._as_vector(rewards, "rewards", batch_size, floating=True)
        next_observations = self._as_matrix(
            next_observations, self.spec.observation_dim, "next_observations", batch_size
        )
        terminated = self._as_vector(terminated, "terminated", batch_size, boolean=True)
        truncated = self._as_vector(truncated, "truncated", batch_size, boolean=True)

        privileged_states = self._optional_matrix(
            privileged_states,
            self.spec.privileged_state_dim,
            "privileged_states",
            batch_size,
            required=self.spec.privileged_state_dim > 0,
        )
        next_privileged_states = self._optional_matrix(
            next_privileged_states,
            self.spec.privileged_state_dim,
            "next_privileged_states",
            batch_size,
            required=self.spec.privileged_state_dim > 0,
        )
        control_features = self._optional_matrix(
            control_features,
            self.spec.control_feature_dim,
            "control_features",
            batch_size,
            required=self.spec.control_feature_dim > 0,
        )
        next_control_features = self._optional_matrix(
            next_control_features,
            self.spec.control_feature_dim,
            "next_control_features",
            batch_size,
            required=self.spec.control_feature_dim > 0,
        )

        metadata_required = self.spec.require_temporal_metadata
        estimator_versions = self._optional_integer_vector(
            estimator_versions, "estimator_versions", batch_size, required=self.spec.control_feature_dim > 0
        )
        episode_ids = self._optional_integer_vector(
            episode_ids, "episode_ids", batch_size, required=metadata_required
        )
        env_ids = self._optional_integer_vector(env_ids, "env_ids", batch_size, required=metadata_required)
        timesteps = self._optional_integer_vector(timesteps, "timesteps", batch_size, required=metadata_required)
        reset_boundaries = self._optional_boolean_vector(
            reset_boundaries, "reset_boundaries", batch_size, required=metadata_required
        )

        if final_observations is None:
            if any(
                value is not None
                for value in (final_privileged_states, final_control_features, final_observation_mask)
            ):
                raise ValueError("Final-state fields and mask require final_observations.")
            final_observations = torch.zeros_like(observations)
            final_privileged_states = torch.zeros_like(privileged_states)
            final_control_features = torch.zeros_like(control_features)
            final_observation_mask = torch.zeros(batch_size, dtype=torch.bool, device=observations.device)
        else:
            final_observations = self._as_matrix(
                final_observations, self.spec.observation_dim, "final_observations", batch_size
            )
            if final_observation_mask is None:
                final_observation_mask = terminated | truncated
            else:
                final_observation_mask = self._as_vector(
                    final_observation_mask, "final_observation_mask", batch_size, boolean=True
                )
            if torch.any(final_observation_mask & ~(terminated | truncated)):
                raise ValueError("final_observation_mask may only mark terminated or truncated transitions.")

            final_privileged_states = self._optional_final_matrix(
                final_privileged_states,
                privileged_states,
                self.spec.privileged_state_dim,
                "final_privileged_states",
                batch_size,
                final_observation_mask,
            )
            final_control_features = self._optional_final_matrix(
                final_control_features,
                control_features,
                self.spec.control_feature_dim,
                "final_control_features",
                batch_size,
                final_observation_mask,
            )

        tensors = (
            observations,
            actions,
            rewards,
            next_observations,
            privileged_states,
            next_privileged_states,
            control_features,
            next_control_features,
            final_observations,
            final_privileged_states,
            final_control_features,
        )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Replay transitions must not contain NaN or infinite floating-point values.")

        original_batch_size = batch_size
        insertion_ids = torch.arange(
            self._total_inserted,
            self._total_inserted + original_batch_size,
            dtype=torch.int64,
            device=observations.device,
        )
        if original_batch_size > self.capacity:
            start = original_batch_size - self.capacity
            batch_size = self.capacity
            observations = observations[start:]
            actions = actions[start:]
            rewards = rewards[start:]
            next_observations = next_observations[start:]
            terminated = terminated[start:]
            truncated = truncated[start:]
            privileged_states = privileged_states[start:]
            next_privileged_states = next_privileged_states[start:]
            control_features = control_features[start:]
            next_control_features = next_control_features[start:]
            estimator_versions = estimator_versions[start:]
            episode_ids = episode_ids[start:]
            env_ids = env_ids[start:]
            timesteps = timesteps[start:]
            reset_boundaries = reset_boundaries[start:]
            final_observations = final_observations[start:]
            final_privileged_states = final_privileged_states[start:]
            final_control_features = final_control_features[start:]
            final_observation_mask = final_observation_mask[start:]
            insertion_ids = insertion_ids[start:]
        indices = (torch.arange(batch_size, device=self.storage_device) + self._position) % self.capacity

        values = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminated": terminated,
            "truncated": truncated,
            "privileged_states": privileged_states,
            "next_privileged_states": next_privileged_states,
            "control_features": control_features,
            "next_control_features": next_control_features,
            "estimator_versions": estimator_versions,
            "episode_ids": episode_ids,
            "env_ids": env_ids,
            "timesteps": timesteps,
            "reset_boundaries": reset_boundaries,
            "final_observations": final_observations,
            "final_privileged_states": final_privileged_states,
            "final_control_features": final_control_features,
            "final_observation_mask": final_observation_mask,
            "insertion_ids": insertion_ids,
        }
        for name, value in values.items():
            storage = getattr(self, f"_{name}")
            storage[indices] = value.detach().to(device=self.storage_device, dtype=storage.dtype)

        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)
        self._total_inserted += original_batch_size

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> TransitionReplayBatch:
        """Sample ordinary independent transitions uniformly with replacement."""

        if not self.can_sample(batch_size):
            raise RuntimeError(f"Cannot sample {batch_size} transitions from replay of size {self._size}.")
        indices = torch.randint(self._size, (batch_size,), generator=generator, device="cpu")
        if self.storage_device.type != "cpu":
            indices = indices.to(self.storage_device)
        batch = self._batch_at(indices)
        return batch if device is None else batch.to(device)

    def chronological(self, *, device: torch.device | str | None = None) -> TransitionReplayBatch:
        """Return retained transitions from oldest to newest for tests/debugging."""

        if self._size == 0:
            indices = torch.empty(0, dtype=torch.int64, device=self.storage_device)
        elif self._size < self.capacity:
            indices = torch.arange(self._size, device=self.storage_device)
        else:
            indices = (torch.arange(self._size, device=self.storage_device) + self._position) % self.capacity
        batch = self._batch_at(indices)
        return batch if device is None else batch.to(device)

    def clear(self) -> None:
        """Drop all retained transitions while preserving monotonic insertion IDs."""

        self._position = 0
        self._size = 0

    def _batch_at(self, indices: torch.Tensor) -> TransitionReplayBatch:
        return TransitionReplayBatch(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_observations[indices],
            terminated=self._terminated[indices],
            truncated=self._truncated[indices],
            privileged_states=self._privileged_states[indices],
            next_privileged_states=self._next_privileged_states[indices],
            control_features=self._control_features[indices],
            next_control_features=self._next_control_features[indices],
            estimator_versions=self._estimator_versions[indices],
            episode_ids=self._episode_ids[indices],
            env_ids=self._env_ids[indices],
            timesteps=self._timesteps[indices],
            reset_boundaries=self._reset_boundaries[indices],
            final_observations=self._final_observations[indices],
            final_privileged_states=self._final_privileged_states[indices],
            final_control_features=self._final_control_features[indices],
            final_observation_mask=self._final_observation_mask[indices],
            insertion_ids=self._insertion_ids[indices],
        )

    @staticmethod
    def _as_matrix(
        value: torch.Tensor,
        width: int,
        name: str,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}.")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        expected = (batch_size, width) if batch_size is not None else (value.shape[0], width)
        if value.ndim != 2 or value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")
        return value

    @staticmethod
    def _as_vector(
        value: torch.Tensor,
        name: str,
        batch_size: int,
        *,
        floating: bool = False,
        boolean: bool = False,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.ndim == 0:
            value = value.unsqueeze(0)
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},) or ({batch_size}, 1), got {tuple(value.shape)}.")
        if floating and not value.dtype.is_floating_point:
            raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}.")
        if boolean and value.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool, got {value.dtype}.")
        return value

    def _optional_matrix(
        self,
        value: torch.Tensor | None,
        width: int,
        name: str,
        batch_size: int,
        *,
        required: bool,
    ) -> torch.Tensor:
        if value is None:
            if required:
                raise ValueError(f"{name} is required by this replay specification.")
            return torch.empty((batch_size, width), dtype=self.dtype, device=self.storage_device)
        return self._as_matrix(value, width, name, batch_size)

    def _optional_final_matrix(
        self,
        value: torch.Tensor | None,
        fallback: torch.Tensor,
        width: int,
        name: str,
        batch_size: int,
        final_observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        if width == 0:
            if value is None:
                return fallback
            return self._as_matrix(value, width, name, batch_size)
        if value is None:
            if torch.any(final_observation_mask):
                raise ValueError(f"{name} is required for marked final FastWMR observations.")
            return torch.zeros_like(fallback)
        return self._as_matrix(value, width, name, batch_size)

    def _optional_integer_vector(
        self,
        value: torch.Tensor | None,
        name: str,
        batch_size: int,
        *,
        required: bool,
    ) -> torch.Tensor:
        if value is None:
            if required:
                raise ValueError(f"{name} is required by this replay specification.")
            return torch.full((batch_size,), -1, dtype=torch.int64, device=self.storage_device)
        value = self._as_vector(value, name, batch_size)
        if value.dtype != torch.int64:
            raise TypeError(f"{name} must have dtype torch.int64, got {value.dtype}.")
        return value

    def _optional_boolean_vector(
        self,
        value: torch.Tensor | None,
        name: str,
        batch_size: int,
        *,
        required: bool,
    ) -> torch.Tensor:
        if value is None:
            if required:
                raise ValueError(f"{name} is required by this replay specification.")
            return torch.zeros(batch_size, dtype=torch.bool, device=self.storage_device)
        return self._as_vector(value, name, batch_size, boolean=True)
