"""Transition replay shared by the FastSAC baseline and FastWMR.

FastSAC only needs ``(o_t, a_t, r_t, o_{t+1}, terminated, truncated)``.
FastWMR extends that contract with raw privileged targets, detached diagnostic
features, estimator versions, and temporal indexing. Its sequence sampler
reconstructs boundary-safe ``B + L`` windows for current-estimator burn-in and
learning. Recurrent hidden/cell state is runtime state and is never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..config import DEFAULT_INTERFACE_CFG, FastWMRInterfaceCfg


TemporalKey = tuple[int, int, int]


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


@dataclass(frozen=True)
class SequenceReplayBatch:
    """Boundary-safe FastWMR sequence with burn-in and learning windows.

    Observation-like tensors contain ``burn_in_length + learning_length + 1``
    values. Transition-like tensors contain one fewer value. Hidden/cell state
    is reconstructed by the current estimator and is never stored here.
    """

    observations: torch.Tensor
    privileged_states: torch.Tensor
    stored_control_features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_ids: torch.Tensor
    env_ids: torch.Tensor
    timesteps: torch.Tensor
    reset_boundaries: torch.Tensor
    insertion_ids: torch.Tensor
    burn_in_length: int
    learning_length: int

    def __post_init__(self) -> None:
        if self.burn_in_length < 0 or self.learning_length <= 0:
            raise ValueError("Sequence burn-in must be non-negative and learning length positive.")
        if self.observations.ndim != 3:
            raise ValueError("observations must have shape (batch, B + L + 1, observation_dim).")
        transition_shape = (self.observations.shape[0], self.transition_length)
        observation_shape = (self.observations.shape[0], self.transition_length + 1)
        if self.observations.shape[:2] != observation_shape:
            raise ValueError("observations must have shape (batch, B + L + 1, observation_dim).")
        if (
            self.privileged_states.ndim != 3
            or self.stored_control_features.ndim != 3
            or self.privileged_states.shape[:2] != observation_shape
            or self.stored_control_features.shape[:2] != observation_shape
        ):
            raise ValueError("Privileged states and stored control features must align with observations.")
        if self.actions.ndim != 3 or self.actions.shape[:2] != transition_shape:
            raise ValueError("actions must have shape (batch, B + L, action_dim).")

        scalar_fields = (
            self.rewards,
            self.terminated,
            self.truncated,
            self.episode_ids,
            self.env_ids,
            self.timesteps,
            self.reset_boundaries,
            self.insertion_ids,
        )
        if any(tensor.shape != transition_shape for tensor in scalar_fields):
            raise ValueError("Sequence transition metadata must have shape (batch, B + L).")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("Sequence termination flags must have dtype torch.bool.")
        if self.reset_boundaries.dtype != torch.bool:
            raise TypeError("Sequence reset_boundaries must have dtype torch.bool.")
        integer_fields = (self.episode_ids, self.env_ids, self.timesteps, self.insertion_ids)
        if any(tensor.dtype != torch.int64 for tensor in integer_fields):
            raise TypeError("Sequence temporal metadata must have dtype torch.int64.")
        if torch.any(self.episode_ids != self.episode_ids[:, :1]) or torch.any(self.env_ids != self.env_ids[:, :1]):
            raise ValueError("Every sampled sequence must stay in one episode and environment.")
        if torch.any(self.timesteps[:, 1:] != self.timesteps[:, :-1] + 1):
            raise ValueError("Sequence timesteps must be consecutive.")
        if torch.any(self.reset_boundaries[:, 1:]):
            raise ValueError("A sampled sequence must not cross a reset boundary.")
        if self.transition_length > 1 and torch.any(
            self.terminated[:, :-1] | self.truncated[:, :-1]
        ):
            raise ValueError("Only the final transition in a sequence may end an episode.")
        floating_fields = (
            self.observations,
            self.privileged_states,
            self.stored_control_features,
            self.actions,
            self.rewards,
        )
        if not all(tensor.dtype.is_floating_point for tensor in floating_fields):
            raise TypeError("Sequence observation, action, and reward fields must be floating point.")
        if not all(torch.isfinite(tensor).all() for tensor in floating_fields):
            raise ValueError("Sequence floating-point fields must be finite.")

    @property
    def batch_size(self) -> int:
        return self.observations.shape[0]

    @property
    def transition_length(self) -> int:
        return self.burn_in_length + self.learning_length

    @property
    def burn_in_observations(self) -> torch.Tensor:
        return self.observations[:, : self.burn_in_length]

    @property
    def learning_observations(self) -> torch.Tensor:
        return self.observations[:, self.burn_in_length :]

    @property
    def learning_privileged_states(self) -> torch.Tensor:
        return self.privileged_states[:, self.burn_in_length :]

    @property
    def learning_stored_control_features(self) -> torch.Tensor:
        return self.stored_control_features[:, self.burn_in_length :]

    @property
    def learning_actions(self) -> torch.Tensor:
        return self.actions[:, self.burn_in_length :]

    @property
    def learning_rewards(self) -> torch.Tensor:
        return self.rewards[:, self.burn_in_length :]

    @property
    def learning_terminated(self) -> torch.Tensor:
        return self.terminated[:, self.burn_in_length :]

    @property
    def learning_truncated(self) -> torch.Tensor:
        return self.truncated[:, self.burn_in_length :]

    @property
    def context_is_exact(self) -> torch.Tensor:
        """True when replay starts at the real episode reset rather than mid-prefix."""

        return self.reset_boundaries[:, 0] & (self.timesteps[:, 0] == 0)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "SequenceReplayBatch":
        tensor_names = (
            "observations",
            "privileged_states",
            "stored_control_features",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "episode_ids",
            "env_ids",
            "timesteps",
            "reset_boundaries",
            "insertion_ids",
        )
        return SequenceReplayBatch(
            **{
                name: getattr(self, name).to(device=device, non_blocking=non_blocking)
                for name in tensor_names
            },
            burn_in_length=self.burn_in_length,
            learning_length=self.learning_length,
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
        self._slot_temporal_keys: list[TemporalKey | None] = [None] * spec.capacity
        self._temporal_index: dict[TemporalKey, int] = {}
        self._valid_sequence_starts: dict[int, set[TemporalKey]] = {}

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

    @property
    def oldest_insertion_id(self) -> int | None:
        """Insertion ID of the oldest retained transition."""

        index = self._oldest_physical_index()
        return None if index is None else int(self._insertion_ids[index].item())

    @property
    def newest_insertion_id(self) -> int | None:
        """Insertion ID of the newest retained transition."""

        index = self._newest_physical_index()
        return None if index is None else int(self._insertion_ids[index].item())

    @property
    def oldest_estimator_version(self) -> int | None:
        """Estimator version attached to the oldest retained transition."""

        if self.spec.control_feature_dim == 0:
            return None
        index = self._oldest_physical_index()
        return None if index is None else int(self._estimator_versions[index].item())

    @property
    def newest_estimator_version(self) -> int | None:
        """Estimator version attached to the newest retained transition."""

        if self.spec.control_feature_dim == 0:
            return None
        index = self._newest_physical_index()
        return None if index is None else int(self._estimator_versions[index].item())

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
        if metadata_required:
            if torch.any(episode_ids < 0) or torch.any(env_ids < 0) or torch.any(timesteps < 0):
                raise ValueError("FastWMR episode_ids, env_ids, and timesteps must be non-negative.")
            if torch.any(reset_boundaries != (timesteps == 0)):
                raise ValueError("reset_boundaries must be true exactly when an episode timestep is zero.")
        if self.spec.control_feature_dim > 0 and torch.any(estimator_versions < 0):
            raise ValueError("FastWMR estimator_versions must be non-negative.")

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
        temporal_keys = self._make_temporal_keys(env_ids, episode_ids, timesteps)
        self._validate_temporal_replacements(indices, temporal_keys)

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
        self._commit_temporal_replacements(indices, temporal_keys)

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

    def can_sample_sequences(
        self,
        batch_size: int,
        burn_in_length: int,
        learning_length: int,
        *,
        require_episode_start: bool = False,
        minimum_insertion_id: int | None = None,
    ) -> bool:
        """Return whether enough distinct boundary-safe windows are retained."""

        self._validate_sequence_request(batch_size, burn_in_length, learning_length)
        starts = self._sequence_starts(
            burn_in_length + learning_length,
            require_episode_start=require_episode_start,
            minimum_insertion_id=minimum_insertion_id,
        )
        return len(starts) >= batch_size

    def sample_sequences(
        self,
        batch_size: int,
        burn_in_length: int,
        learning_length: int,
        *,
        require_episode_start: bool = False,
        minimum_insertion_id: int | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> SequenceReplayBatch:
        """Sample ``B + L`` consecutive transitions without crossing resets."""

        self._validate_sequence_request(batch_size, burn_in_length, learning_length)
        transition_length = burn_in_length + learning_length
        starts = self._sequence_starts(
            transition_length,
            require_episode_start=require_episode_start,
            minimum_insertion_id=minimum_insertion_id,
        )
        if len(starts) < batch_size:
            raise RuntimeError(
                f"Need {batch_size} valid sequences of length {transition_length}, found {len(starts)}."
            )

        choices = torch.randperm(len(starts), generator=generator)[:batch_size].tolist()
        selected_starts = [starts[index] for index in choices]
        physical_indices = [
            [self._temporal_index[(env_id, episode_id, timestep + offset)] for offset in range(transition_length)]
            for env_id, episode_id, timestep in selected_starts
        ]
        indices = torch.tensor(physical_indices, dtype=torch.int64, device=self.storage_device)
        transitions = self._batch_at(indices)
        final_transitions = self._batch_at(indices[:, -1])

        sequence = SequenceReplayBatch(
            observations=torch.cat(
                (transitions.observations, final_transitions.bootstrap_observations.unsqueeze(1)), dim=1
            ),
            privileged_states=torch.cat(
                (
                    transitions.privileged_states,
                    final_transitions.bootstrap_privileged_states.unsqueeze(1),
                ),
                dim=1,
            ),
            stored_control_features=torch.cat(
                (
                    transitions.control_features,
                    final_transitions.bootstrap_control_features.unsqueeze(1),
                ),
                dim=1,
            ),
            actions=transitions.actions,
            rewards=transitions.rewards,
            terminated=transitions.terminated,
            truncated=transitions.truncated,
            episode_ids=transitions.episode_ids,
            env_ids=transitions.env_ids,
            timesteps=transitions.timesteps,
            reset_boundaries=transitions.reset_boundaries,
            insertion_ids=transitions.insertion_ids,
            burn_in_length=burn_in_length,
            learning_length=learning_length,
        )
        return sequence if device is None else sequence.to(device)

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
        self._slot_temporal_keys = [None] * self.capacity
        self._temporal_index.clear()
        self._valid_sequence_starts.clear()

    def reset(self) -> None:
        """Drop replay contents and restart insertion IDs for a fresh run."""

        self.clear()
        self._total_inserted = 0

    def _oldest_physical_index(self) -> int | None:
        if self._size == 0:
            return None
        return self._position if self.is_full else 0

    def _newest_physical_index(self) -> int | None:
        if self._size == 0:
            return None
        return (self._position - 1) % self.capacity

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
    def _make_temporal_keys(
        env_ids: torch.Tensor,
        episode_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> list[TemporalKey | None]:
        values = zip(
            env_ids.detach().cpu().tolist(),
            episode_ids.detach().cpu().tolist(),
            timesteps.detach().cpu().tolist(),
            strict=True,
        )
        return [
            (int(env_id), int(episode_id), int(timestep))
            if env_id >= 0 and episode_id >= 0 and timestep >= 0
            else None
            for env_id, episode_id, timestep in values
        ]

    def _validate_temporal_replacements(
        self,
        indices: torch.Tensor,
        new_keys: list[TemporalKey | None],
    ) -> None:
        slots = indices.detach().cpu().tolist()
        non_null_keys = [key for key in new_keys if key is not None]
        if len(non_null_keys) != len(set(non_null_keys)):
            raise ValueError("A replay add batch contains duplicate temporal keys.")
        replaced_slots = set(slots)
        for key in non_null_keys:
            existing_slot = self._temporal_index.get(key)
            if existing_slot is not None and existing_slot not in replaced_slots:
                raise ValueError(f"Temporal key {key} is already retained in replay.")

    def _commit_temporal_replacements(
        self,
        indices: torch.Tensor,
        new_keys: list[TemporalKey | None],
    ) -> None:
        slots = [int(slot) for slot in indices.detach().cpu().tolist()]
        changed_keys: list[TemporalKey] = []
        for slot in slots:
            old_key = self._slot_temporal_keys[slot]
            if old_key is not None:
                if self._temporal_index.get(old_key) == slot:
                    del self._temporal_index[old_key]
                changed_keys.append(old_key)
        for slot, key in zip(slots, new_keys, strict=True):
            self._slot_temporal_keys[slot] = key
            if key is not None:
                self._temporal_index[key] = slot
                changed_keys.append(key)

        for transition_length, valid_starts in self._valid_sequence_starts.items():
            affected_starts: set[TemporalKey] = set()
            for env_id, episode_id, timestep in changed_keys:
                for offset in range(transition_length):
                    start_timestep = timestep - offset
                    if start_timestep >= 0:
                        affected_starts.add((env_id, episode_id, start_timestep))
            for start in affected_starts:
                if self._is_valid_sequence_start(start, transition_length):
                    valid_starts.add(start)
                else:
                    valid_starts.discard(start)

    def _sequence_starts(
        self,
        transition_length: int,
        *,
        require_episode_start: bool,
        minimum_insertion_id: int | None = None,
    ) -> list[TemporalKey]:
        if transition_length <= 0:
            raise ValueError("Sequence transition length must be positive.")
        if not self.spec.require_temporal_metadata:
            raise RuntimeError("Sequence sampling requires a FastWMR replay specification.")
        if transition_length not in self._valid_sequence_starts:
            self._valid_sequence_starts[transition_length] = {
                key for key in self._temporal_index if self._is_valid_sequence_start(key, transition_length)
            }
        starts = self._valid_sequence_starts[transition_length]
        if require_episode_start:
            starts = {key for key in starts if key[2] == 0}
        if minimum_insertion_id is not None:
            if minimum_insertion_id < 0:
                raise ValueError("minimum_insertion_id must be non-negative.")
            starts = {
                key
                for key in starts
                if int(self._insertion_ids[self._temporal_index[key]].item())
                >= minimum_insertion_id
            }
        return sorted(starts)

    def _is_valid_sequence_start(self, start: TemporalKey, transition_length: int) -> bool:
        env_id, episode_id, start_timestep = start
        for offset in range(transition_length):
            slot = self._temporal_index.get((env_id, episode_id, start_timestep + offset))
            if slot is None:
                return False
            if offset > 0 and bool(self._reset_boundaries[slot].item()):
                return False
            if offset < transition_length - 1 and bool((self._terminated[slot] | self._truncated[slot]).item()):
                return False
        return True

    @staticmethod
    def _validate_sequence_request(batch_size: int, burn_in_length: int, learning_length: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if burn_in_length < 0:
            raise ValueError("burn_in_length must be non-negative.")
        if learning_length <= 0:
            raise ValueError("learning_length must be positive.")

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
