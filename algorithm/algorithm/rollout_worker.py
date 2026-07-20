"""FastSAC collection and FastWMR recurrent rollout runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..buffers import (
    EstimatorRolloutBatch,
    EstimatorRolloutCache,
    TransitionReplayBuffer,
)
from ..utils.env_wrapper import IsaacLabEnvAdapter
from ..utils.temporal_state import RecurrentState, RecurrentStateManager
from .estimator_update import EstimatorUpdateResult, EstimatorUpdater, WorldStateEstimator
from .fastwmr_agent import FastSACReplayUpdateLoop
from .sac_update import SACUpdateMetrics


@dataclass(frozen=True)
class RolloutStepResult:
    """Diagnostics returned after one vectorized collection step."""

    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    updates: tuple[SACUpdateMetrics, ...]


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

    @property
    def state(self) -> RecurrentState:
        return self._state.state

    @property
    def estimator_version(self) -> int:
        return self._estimator_version

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
        self._environment_steps += 1
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

        return self._state.reset_done(terminated, truncated)

    def detach_state(self) -> RecurrentState:
        """Explicit collection-chunk graph boundary."""

        return self._state.detach()

    def reset_all(self, *, estimator_version: int | None = None) -> RecurrentState:
        """Fallback synchronization that intentionally discards all context."""

        if estimator_version is not None:
            if estimator_version < self._estimator_version:
                raise ValueError("estimator_version cannot move backwards.")
            self._estimator_version = estimator_version
        return self._state.clear()

    @torch.no_grad()
    def rebuild_from_batch(
        self,
        batch: EstimatorRolloutBatch,
        *,
        estimator_version: int,
        final_reset_mask: torch.Tensor | None = None,
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
        for timestep in range(batch.sequence_length):
            state = state.reset(batch.reset_boundaries[:, timestep])
            encoded, state = self.estimator.encoder.forward_rollout(
                transformed[:, timestep],
                state,
            )
            encoded_steps.append(encoded)
        if final_reset_mask is not None:
            self._validate_reset_mask(final_reset_mask)
            state = state.reset(final_reset_mask)
        encoded_history = torch.stack(encoded_steps, dim=1)
        reconstructions = self.estimator.decoder(encoded_history).reconstruction.detach()
        self._validate_reconstruction(
            reconstructions,
            expected_time=batch.sequence_length,
        )
        final_state = self._state.replace(state)
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
    ) -> EstimatorRuntimeRebuild:
        return self.rebuild_from_batch(
            cache.chronological(),
            estimator_version=estimator_version,
            final_reset_mask=final_reset_mask,
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
