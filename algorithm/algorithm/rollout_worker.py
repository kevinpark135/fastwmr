"""FastSAC collection and FastWMR recurrent rollout runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..buffers import (
    EstimatorRolloutBatch,
    EstimatorRolloutCache,
    TransitionReplayBuffer,
)
from ..config import DEFAULT_INTERFACE_CFG, FastWMRInterfaceCfg
from ..utils.env_wrapper import IsaacLabEnvAdapter
from ..utils.feature_builder import build_control_feature
from ..utils.temporal_state import RecurrentState, RecurrentStateManager
from .estimator_update import EstimatorUpdateResult, EstimatorUpdater, WorldStateEstimator
from .fastwmr_agent import (
    FastSACReplayUpdateLoop,
    FastWMRSequenceFeatureProcessor,
    FastWMRSequenceUpdateLoop,
)
from .sac_update import SACUpdateMetrics


@dataclass(frozen=True)
class RolloutStepResult:
    """Diagnostics returned after one vectorized collection step."""

    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    updates: tuple[SACUpdateMetrics, ...]


@dataclass(frozen=True)
class FastWMRRolloutStepResult:
    """Diagnostics from one integrated estimator and SAC collection step."""

    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    updates: tuple[SACUpdateMetrics, ...]
    estimator_updates: int
    estimator_version: int


@dataclass(frozen=True)
class EstimatorRuntimeStep:
    """One graph-free online reconstruction and runtime-state diagnostic."""

    reconstruction: torch.Tensor
    estimator_version: int
    hidden_norm: float


@dataclass(frozen=True)
class EstimatorRuntimeRebuild:
    """Current-estimator replay of a recent rollout cache."""

    reconstructions: torch.Tensor
    final_state: RecurrentState
    estimator_version: int
    context_exact_fraction: float


@dataclass(frozen=True)
class EstimatorRuntimeUpdate:
    """Estimator optimizer result paired with its rebuilt rollout state."""

    estimator_update: EstimatorUpdateResult
    runtime_rebuild: EstimatorRuntimeRebuild


class FastWMREstimatorRuntime:
    """Manage graph-free per-environment estimator state during collection.

    The estimator parameters are shared across environments, while hidden and
    cell tensors use the vector-environment axis as independent recurrent
    contexts. Parameter updates should happen between collection chunks through
    :meth:`update_from_cache`, which immediately rebuilds runtime state using
    the updated estimator and recent raw observations.
    """

    def __init__(
        self,
        estimator: WorldStateEstimator,
        num_envs: int,
        *,
        observation_transform=None,
        estimator_version: int = 0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if estimator_version < 0:
            raise ValueError("estimator_version must be non-negative.")
        try:
            parameter = next(estimator.parameters())
        except StopIteration as error:
            raise ValueError("The world-state estimator must have trainable parameters.") from error
        if not parameter.dtype.is_floating_point:
            raise TypeError("Estimator parameters must use a floating dtype.")

        self.estimator = estimator
        self.num_envs = num_envs
        self.observation_transform = observation_transform
        self._device = parameter.device
        self._dtype = parameter.dtype
        self._state = RecurrentStateManager(
            estimator.initial_state(
                num_envs,
                device=self._device,
                dtype=self._dtype,
            )
        )
        self._estimator_version = estimator_version
        self._environment_steps = 0
        self._rebuilds = 0
        self._last_reconstruction: torch.Tensor | None = None

    @property
    def state(self) -> RecurrentState:
        return self._state.state

    @property
    def estimator_version(self) -> int:
        return self._estimator_version

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def current_reconstruction(self) -> torch.Tensor:
        """Reconstruction aligned with the runtime's current observation."""

        if self._last_reconstruction is None:
            raise RuntimeError("The estimator runtime has not processed an observation yet.")
        return self._last_reconstruction

    @property
    def environment_steps(self) -> int:
        return self._environment_steps

    @property
    def rebuilds(self) -> int:
        return self._rebuilds

    @property
    def hidden_norm(self) -> float:
        return self._state.hidden_norm

    @torch.no_grad()
    def step(
        self,
        observations: torch.Tensor,
        *,
        reset_boundaries: torch.Tensor | None = None,
        expected_estimator_version: int | None = None,
    ) -> EstimatorRuntimeStep:
        """Advance every environment by one observation without building a graph."""

        if (
            expected_estimator_version is not None
            and expected_estimator_version != self._estimator_version
        ):
            raise RuntimeError(
                "Estimator runtime version mismatch: expected "
                f"{expected_estimator_version}, runtime has {self._estimator_version}."
            )
        self._validate_observations(observations)
        if reset_boundaries is not None:
            self._validate_reset_mask(reset_boundaries)
            self._state.reset(reset_boundaries)

        transformed = self._transform(observations)
        reconstruction, next_state = self.estimator.forward_rollout(
            transformed,
            self._state.state,
        )
        self._state.replace(next_state)
        reconstruction = reconstruction.detach()
        self._validate_reconstruction(reconstruction, expected_time=None)
        self._last_reconstruction = reconstruction
        self._environment_steps += 1
        return EstimatorRuntimeStep(
            reconstruction=reconstruction,
            estimator_version=self._estimator_version,
            hidden_norm=self._state.hidden_norm,
        )

    @torch.no_grad()
    def preview(self, observations: torch.Tensor) -> EstimatorRuntimeStep:
        """Reconstruct successors without mutating recurrent runtime state.

        Auto-reset environments expose terminal observations separately from
        the returned post-reset observations. This preview uses the pre-reset
        recurrent context for those terminal values while leaving the online
        state ready to process the actual next observation.
        """

        self._validate_observations(observations)
        transformed = self._transform(observations)
        reconstruction, _ = self.estimator.forward_rollout(
            transformed,
            self._state.state,
        )
        reconstruction = reconstruction.detach()
        self._validate_reconstruction(reconstruction, expected_time=None)
        return EstimatorRuntimeStep(
            reconstruction=reconstruction,
            estimator_version=self._estimator_version,
            hidden_norm=self._state.hidden_norm,
        )

    def reset_done(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> RecurrentState:
        """Clear only environments whose current episode has ended."""

        state = self._state.reset_done(terminated, truncated)
        if torch.any(terminated | truncated):
            self._last_reconstruction = None
        return state

    def detach_state(self) -> RecurrentState:
        """Explicit collection-chunk graph boundary."""

        return self._state.detach()

    def reset_all(self, *, estimator_version: int | None = None) -> RecurrentState:
        """Fallback synchronization that intentionally discards all context."""

        if estimator_version is not None:
            if estimator_version < self._estimator_version:
                raise ValueError("estimator_version cannot move backwards.")
            self._estimator_version = estimator_version
        self._last_reconstruction = None
        return self._state.clear()

    def restart(self, *, estimator_version: int) -> RecurrentState:
        """Restore a checkpoint version while discarding all ephemeral state."""

        if estimator_version < 0:
            raise ValueError("estimator_version must be non-negative.")
        self._estimator_version = estimator_version
        self._environment_steps = 0
        self._rebuilds = 0
        self._last_reconstruction = None
        return self._state.clear()

    @torch.no_grad()
    def rebuild_from_batch(
        self,
        batch: EstimatorRolloutBatch,
        *,
        estimator_version: int,
        final_reset_mask: torch.Tensor | None = None,
        decode_full_sequence: bool = True,
    ) -> EstimatorRuntimeRebuild:
        """Re-infer runtime context from raw observations with current parameters."""

        if estimator_version < self._estimator_version:
            raise ValueError("estimator_version cannot move backwards during rebuild.")
        batch = self._prepare_batch(batch)
        transformed = self._transform(batch.observations)
        state = self.estimator.initial_state(
            self.num_envs,
            device=self._device,
            dtype=self._dtype,
        )
        encoded_steps: list[torch.Tensor] = []
        final_encoded: torch.Tensor | None = None
        for timestep in range(batch.sequence_length):
            state = state.reset(batch.reset_boundaries[:, timestep])
            encoded, state = self.estimator.encoder.forward_rollout(
                transformed[:, timestep],
                state,
            )
            final_encoded = encoded
            if decode_full_sequence:
                encoded_steps.append(encoded)
        if final_reset_mask is not None:
            self._validate_reset_mask(final_reset_mask)
            state = state.reset(final_reset_mask)
        if final_encoded is None:
            raise RuntimeError("Runtime rebuild requires at least one cached timestep.")
        encoded_history = (
            torch.stack(encoded_steps, dim=1)
            if decode_full_sequence
            else final_encoded.unsqueeze(1)
        )
        reconstructions = self.estimator.decoder(encoded_history).reconstruction.detach()
        self._validate_reconstruction(
            reconstructions,
            expected_time=encoded_history.shape[1],
        )
        final_state = self._state.replace(state)
        self._last_reconstruction = reconstructions[:, -1]
        if final_reset_mask is not None and torch.any(final_reset_mask):
            self._last_reconstruction = None
        self._estimator_version = estimator_version
        self._rebuilds += 1
        return EstimatorRuntimeRebuild(
            reconstructions=reconstructions,
            final_state=final_state,
            estimator_version=estimator_version,
            context_exact_fraction=float(batch.context_is_exact.float().mean()),
        )

    def rebuild_from_cache(
        self,
        cache: EstimatorRolloutCache,
        *,
        estimator_version: int,
        final_reset_mask: torch.Tensor | None = None,
        decode_full_sequence: bool = True,
    ) -> EstimatorRuntimeRebuild:
        return self.rebuild_from_batch(
            cache.chronological(copy=False),
            estimator_version=estimator_version,
            final_reset_mask=final_reset_mask,
            decode_full_sequence=decode_full_sequence,
        )

    def update_from_cache(
        self,
        updater: EstimatorUpdater,
        cache: EstimatorRolloutCache,
        *,
        drain: bool = True,
        final_reset_mask: torch.Tensor | None = None,
    ) -> EstimatorRuntimeUpdate:
        """Update estimator, rebuild current runtime state, then optionally clear cache."""

        if updater.estimator is not self.estimator:
            raise ValueError("Runtime and updater must share the same estimator instance.")
        if updater.observation_transform is not self.observation_transform:
            raise ValueError("Runtime and updater must share the same observation transform.")
        estimator_update = updater.update_cache(cache, drain=False)
        runtime_rebuild = self.rebuild_from_cache(
            cache,
            estimator_version=estimator_update.metrics.estimator_version,
            final_reset_mask=final_reset_mask,
        )
        if drain:
            cache.clear()
        return EstimatorRuntimeUpdate(
            estimator_update=estimator_update,
            runtime_rebuild=runtime_rebuild,
        )

    def _prepare_batch(self, batch: EstimatorRolloutBatch) -> EstimatorRolloutBatch:
        if batch.num_envs != self.num_envs:
            raise ValueError(
                f"Rollout batch has {batch.num_envs} environments, expected {self.num_envs}."
            )
        if batch.observations.shape[-1] != self.estimator.observation_dim:
            raise ValueError("Rollout batch observation width does not match the estimator.")
        if batch.privileged_states.shape[-1] != self.estimator.reconstruction_dim:
            raise ValueError("Rollout batch privileged width does not match the estimator.")
        batch = batch.to(self._device)
        if batch.observations.dtype != self._dtype:
            batch = EstimatorRolloutBatch(
                observations=batch.observations.to(dtype=self._dtype),
                privileged_states=batch.privileged_states.to(dtype=self._dtype),
                reset_boundaries=batch.reset_boundaries,
            )
        return batch

    def _transform(self, observations: torch.Tensor) -> torch.Tensor:
        transformed = (
            self.observation_transform(observations)
            if self.observation_transform is not None
            else observations
        )
        if transformed.shape != observations.shape:
            raise ValueError("The runtime observation transform must preserve tensor shape.")
        if transformed.device != self._device or transformed.dtype != self._dtype:
            raise ValueError("Transformed observations must match estimator device and dtype.")
        if not torch.isfinite(transformed).all():
            raise ValueError("Transformed runtime observations must remain finite.")
        return transformed

    def _validate_observations(self, observations: torch.Tensor) -> None:
        expected_shape = (self.num_envs, self.estimator.observation_dim)
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor.")
        if observations.shape != expected_shape:
            raise ValueError(
                f"observations must have shape {expected_shape}, got {tuple(observations.shape)}."
            )
        if observations.device != self._device or observations.dtype != self._dtype:
            raise ValueError("Runtime observations must match estimator device and dtype.")
        if not torch.isfinite(observations).all():
            raise ValueError("Runtime observations must be finite.")

    def _validate_reset_mask(self, reset_mask: torch.Tensor) -> None:
        if not isinstance(reset_mask, torch.Tensor) or reset_mask.dtype is not torch.bool:
            raise TypeError("reset_boundaries must be a boolean tensor.")
        if reset_mask.shape != (self.num_envs,):
            raise ValueError(
                f"reset_boundaries must have shape ({self.num_envs},), "
                f"got {tuple(reset_mask.shape)}."
            )

    def _validate_reconstruction(
        self,
        reconstruction: torch.Tensor,
        *,
        expected_time: int | None,
    ) -> None:
        leading_shape = (self.num_envs,)
        if expected_time is not None:
            leading_shape = (self.num_envs, expected_time)
        expected_shape = (*leading_shape, self.estimator.reconstruction_dim)
        if reconstruction.shape != expected_shape:
            raise ValueError(
                f"Runtime reconstruction must have shape {expected_shape}, "
                f"got {tuple(reconstruction.shape)}."
            )
        if reconstruction.requires_grad:
            raise RuntimeError("Runtime reconstruction must be detached from autograd.")
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Runtime reconstruction must remain finite.")


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
        self.update_loop.update_observation_statistics(policy)
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
        self.update_loop.update_observation_statistics(next_observations)
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


class FastWMRRolloutCollector:
    """Collect full FastWMR transitions and keep online memory synchronized.

    Raw proprioception and privileged targets are recorded for replay-time
    estimator training. Actor and critics only receive the detached 109D
    control feature. On auto-reset, terminal reconstruction is previewed from
    the old recurrent context before completed environment slices are reset.
    """

    def __init__(
        self,
        env: IsaacLabEnvAdapter,
        replay: TransitionReplayBuffer,
        update_loop: FastWMRSequenceUpdateLoop,
        *,
        interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    ) -> None:
        if replay is not update_loop.replay:
            raise ValueError("Collector and update loop must share the same replay buffer.")
        processor = update_loop.sequence_feature_processor
        if not isinstance(processor, FastWMRSequenceFeatureProcessor):
            raise TypeError(
                "FastWMRRolloutCollector requires FastWMRSequenceFeatureProcessor "
                "to synchronize estimator updates."
            )
        if processor.interface != interface:
            raise ValueError("Collector and sequence processor must share the interface contract.")
        expected_dimensions = (
            interface.policy_observation_dim,
            interface.action_dim,
            interface.reconstruction_target_dim,
            interface.control_feature_dim,
        )
        replay_dimensions = (
            replay.spec.observation_dim,
            replay.spec.action_dim,
            replay.spec.privileged_state_dim,
            replay.spec.control_feature_dim,
        )
        if replay_dimensions != expected_dimensions or not replay.spec.require_temporal_metadata:
            raise ValueError("FastWMR collector requires the complete FastWMR replay contract.")
        if update_loop.updater.actor.input_dim != interface.actor_input_dim:
            raise ValueError("FastWMR actor input width does not match the control feature contract.")
        if update_loop.updater.actor.action_dim != interface.action_dim:
            raise ValueError("FastWMR actor action width does not match the interface contract.")
        if processor.runtime.num_envs != env.num_envs:
            raise ValueError("Environment and estimator runtime environment counts must match.")
        if processor.runtime.device != env.device:
            raise ValueError("Environment and estimator runtime must share a device.")
        actor_device = next(update_loop.updater.actor.parameters()).device
        if actor_device != processor.runtime.device or update_loop.learner_device != actor_device:
            raise ValueError("Actor, learner, and estimator runtime must share a device.")

        self.env = env
        self.replay = replay
        self.update_loop = update_loop
        self.processor = processor
        self.runtime = processor.runtime
        self.rollout_cache = processor.rollout_cache
        self.interface = interface
        self._observations: dict[str, torch.Tensor] | None = None
        self._episode_ids: torch.Tensor | None = None
        self._timesteps: torch.Tensor | None = None
        self._reset_boundaries: torch.Tensor | None = None
        self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        """Reset the environment and initialize every recurrent context."""

        observations, _ = self.env.reset(seed=seed)
        policy, privileged = self._observation_groups(observations)
        if self._episode_ids is None:
            episode_id = 0
        else:
            episode_id = int(self._episode_ids.max().item()) + 1
        self._episode_ids = torch.full_like(self._env_ids, episode_id)
        self._timesteps = torch.zeros_like(self._env_ids)
        self._reset_boundaries = torch.ones(
            self.env.num_envs,
            device=self.env.device,
            dtype=torch.bool,
        )

        self.rollout_cache.clear()
        self.runtime.reset_all(estimator_version=self.processor.estimator_updater.version)
        self._update_observation_statistics(policy)
        self.rollout_cache.add(policy, privileged, self._reset_boundaries)
        runtime_step = self.runtime.step(
            policy,
            reset_boundaries=self._reset_boundaries,
            expected_estimator_version=self.processor.estimator_updater.version,
        )
        self._observations = observations
        return self._build_control_feature(policy, runtime_step.reconstruction)

    def collect_step(
        self,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> FastWMRRolloutStepResult:
        if (
            self._observations is None
            or self._episode_ids is None
            or self._timesteps is None
            or self._reset_boundaries is None
        ):
            raise RuntimeError("Call reset() before collecting FastWMR transitions.")

        observations, privileged = self._observation_groups(self._observations)
        control_features = self._build_control_feature(
            observations,
            self.runtime.current_reconstruction,
        )
        estimator_versions = torch.full_like(
            self._env_ids,
            self.runtime.estimator_version,
        )
        actions = self.update_loop.select_actions(
            control_features,
            deterministic=deterministic,
        )
        step = self.env.step(actions)
        next_observations, next_privileged = self._observation_groups(step.observations)
        final_observations, final_privileged = self._observation_groups(step.final_observations)

        # Preview terminal successors before reset, then advance returned
        # post-reset observations with only completed environment slices zeroed.
        done = step.terminated | step.truncated
        if torch.any(done):
            final_reconstruction = self.runtime.preview(final_observations).reconstruction
            final_control_features = self._build_control_feature(
                final_observations,
                final_reconstruction,
            )
        else:
            final_control_features = torch.zeros_like(control_features)
        self._update_observation_statistics(next_observations)
        self.rollout_cache.add(next_observations, next_privileged, done)
        next_runtime_step = self.runtime.step(
            next_observations,
            reset_boundaries=done,
            expected_estimator_version=self.processor.estimator_updater.version,
        )
        next_control_features = self._build_control_feature(
            next_observations,
            next_runtime_step.reconstruction,
        )

        self.replay.add(
            observations=observations,
            actions=actions,
            rewards=step.rewards,
            next_observations=next_observations,
            terminated=step.terminated,
            truncated=step.truncated,
            privileged_states=privileged,
            next_privileged_states=next_privileged,
            control_features=control_features,
            next_control_features=next_control_features,
            estimator_versions=estimator_versions,
            episode_ids=self._episode_ids,
            env_ids=self._env_ids,
            timesteps=self._timesteps,
            reset_boundaries=self._reset_boundaries,
            final_observations=final_observations,
            final_privileged_states=final_privileged,
            final_control_features=final_control_features,
            final_observation_mask=step.final_observation_mask,
        )

        self._observations = step.observations
        self._episode_ids = self._episode_ids + done.to(dtype=torch.int64)
        self._timesteps = torch.where(done, torch.zeros_like(self._timesteps), self._timesteps + 1)
        self._reset_boundaries = done
        self.update_loop.advance_environment()
        previous_estimator_updates = self.processor.updates
        updates = tuple(self.update_loop.run_updates(generator=generator))
        return FastWMRRolloutStepResult(
            rewards=step.rewards.detach(),
            terminated=step.terminated.detach(),
            truncated=step.truncated.detach(),
            updates=updates,
            estimator_updates=self.processor.updates - previous_estimator_updates,
            estimator_version=self.runtime.estimator_version,
        )

    def _observation_groups(
        self,
        observation_groups: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = self.env.policy_observation(observation_groups)
        privileged = self.env.privileged_observation(observation_groups)
        expected_policy = (self.env.num_envs, self.interface.policy_observation_dim)
        expected_privileged = (self.env.num_envs, self.interface.reconstruction_target_dim)
        if policy.shape != expected_policy:
            raise ValueError(
                f"Policy observation shape must be {expected_policy}, got {tuple(policy.shape)}."
            )
        if privileged.shape != expected_privileged:
            raise ValueError(
                f"Privileged observation shape must be {expected_privileged}, "
                f"got {tuple(privileged.shape)}."
            )
        if policy.device != self.runtime.device or privileged.device != self.runtime.device:
            raise ValueError("FastWMR observation groups must be on the runtime device.")
        if policy.dtype != self.runtime.dtype or privileged.dtype != self.runtime.dtype:
            raise ValueError("FastWMR observation groups must match the estimator dtype.")
        if not torch.isfinite(policy).all() or not torch.isfinite(privileged).all():
            raise ValueError("FastWMR observation groups must remain finite.")
        return policy, privileged

    def _update_observation_statistics(self, observations: torch.Tensor) -> None:
        normalizer = self.processor.observation_normalizer
        if normalizer is not None:
            normalizer.update(observations)

    def _build_control_feature(
        self,
        observations: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        features = build_control_feature(
            observations,
            reconstruction,
            cfg=self.interface,
            normalizer=self.processor.observation_normalizer,
        ).detach()
        if not torch.isfinite(features).all():
            raise FloatingPointError("FastWMR rollout control features must remain finite.")
        return features
