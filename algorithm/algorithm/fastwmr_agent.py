"""Replay warm-up and update loops for FastSAC and sequence-based FastWMR."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..buffers import EstimatorRolloutCache, SequenceReplayBatch, TransitionReplayBuffer
from ..config import (
    DEFAULT_INTERFACE_CFG,
    FastWMRInterfaceCfg,
    ReplayUpdateCfg,
    SequenceReplayCfg,
)
from ..utils.feature_builder import build_control_feature
from ..utils.normalization import RunningObservationNormalizer
from .estimator_update import EstimatorUpdateResult, EstimatorUpdater
from .sac_update import SACFeatureSource, SACTransitionBatch, SACUpdateMetrics, SACUpdater

if TYPE_CHECKING:
    from .rollout_worker import EstimatorRuntimeRebuild, FastWMREstimatorRuntime


SequenceFeatureProcessor = Callable[[SequenceReplayBatch], torch.Tensor]
SequenceAugmentation = Callable[[SequenceReplayBatch], SequenceReplayBatch]


class GradientBoundaryError(RuntimeError):
    """Raised when an optimizer phase writes gradients outside its ownership."""


@dataclass(frozen=True)
class GradientBoundaryReport:
    """Diagnostics from one complete set of FastWMR gradient checks."""

    enabled: bool
    checks: int
    estimator_gradient_norm: float | torch.Tensor | None
    cutoff_enabled: bool = True
    policy_estimator_gradient_norm: float | torch.Tensor | None = None


@dataclass(frozen=True)
class FastWMRAgentUpdateResult:
    """Estimator and SAC diagnostics produced by one ordered agent update."""

    estimator_update: EstimatorUpdateResult
    sac_update: SACUpdateMetrics
    update_order: tuple[str, ...]
    gradient_boundary: GradientBoundaryReport


class FastWMRGradientGuard:
    """Enforce optimizer ownership and estimator-to-SAC gradient cutoff."""

    def __init__(
        self,
        estimator_updater: EstimatorUpdater,
        sac_updater: SACUpdater,
        *,
        enabled: bool = True,
    ) -> None:
        self.estimator_updater = estimator_updater
        self.sac_updater = sac_updater
        self.enabled = enabled
        self._checks = 0
        self._validate_optimizer_ownership()

    def begin_update(self) -> None:
        """Clear stale gradients before ownership checks begin."""

        self._checks = 0
        self.clear_all()
        self._assert_all_clean("update start")

    def after_estimator(self, *, require_gradient: bool = True) -> float | None:
        gradient_norm = None
        if self.enabled:
            if require_gradient:
                gradient_norm = self._require_finite_gradient(
                    self.estimator_updater.estimator,
                    "estimator",
                    "estimator update",
                )
            else:
                self._assert_no_gradient(
                    self.estimator_updater.estimator,
                    "estimator",
                    "frozen estimator evaluation",
                )
            self._assert_no_gradient(self.sac_updater.actor, "actor", "estimator update")
            self._assert_no_gradient(self.sac_updater.critic, "critic", "estimator update")
            self._assert_no_gradient(
                self.sac_updater.temperature,
                "temperature",
                "estimator update",
            )
            self._assert_no_gradient(
                self.sac_updater.target_critic,
                "target critic",
                "estimator update",
            )
            self._checks += 1

        # The estimator optimizer leaves gradients available for diagnostics.
        # Remove them before critic or actor backward can run.
        self.estimator_updater.optimizer.zero_grad(set_to_none=True)
        self._assert_no_gradient(self.estimator_updater.estimator, "estimator", "SAC boundary")
        return gradient_norm

    def before_sac(self, batch: SACTransitionBatch) -> None:
        if batch.states.requires_grad or batch.next_states.requires_grad:
            raise GradientBoundaryError("SAC features must be detached from the estimator graph.")
        self._assert_no_gradient(self.estimator_updater.estimator, "estimator", "SAC start")
        if self.enabled:
            self._checks += 1

    def after_critic(self) -> None:
        if not self.enabled:
            return
        self._assert_no_gradient(self.estimator_updater.estimator, "estimator", "critic update")
        self._assert_no_gradient(self.sac_updater.actor, "actor", "critic update")
        self._assert_no_gradient(self.sac_updater.temperature, "temperature", "critic update")
        self._assert_no_gradient(self.sac_updater.target_critic, "target critic", "critic update")
        self._checks += 1

    def after_actor(self) -> None:
        if not self.enabled:
            return
        self._assert_no_gradient(self.estimator_updater.estimator, "estimator", "actor update")
        self._assert_no_gradient(self.sac_updater.critic, "critic", "actor update")
        self._assert_no_gradient(self.sac_updater.temperature, "temperature", "actor update")
        self._assert_no_gradient(self.sac_updater.target_critic, "target critic", "actor update")
        self._require_finite_gradient(self.sac_updater.actor, "actor", "actor update")
        self._checks += 1

    def after_temperature(self) -> None:
        if not self.enabled:
            return
        self._assert_no_gradient(self.estimator_updater.estimator, "estimator", "temperature update")
        self._assert_no_gradient(self.sac_updater.critic, "critic", "temperature update")
        self._assert_no_gradient(
            self.sac_updater.target_critic,
            "target critic",
            "temperature update",
        )
        self._require_finite_gradient(
            self.sac_updater.temperature,
            "temperature",
            "temperature update",
        )
        self._checks += 1

    def after_target(self) -> None:
        if not self.enabled:
            return
        self._assert_no_gradient(self.estimator_updater.estimator, "estimator", "target update")
        self._assert_no_gradient(self.sac_updater.target_critic, "target critic", "target update")
        self._checks += 1

    def report(
        self,
        estimator_gradient_norm: float | None,
        *,
        cutoff_enabled: bool = True,
        policy_estimator_gradient_norm: float | None = None,
    ) -> GradientBoundaryReport:
        return GradientBoundaryReport(
            enabled=self.enabled,
            checks=self._checks,
            estimator_gradient_norm=estimator_gradient_norm,
            cutoff_enabled=cutoff_enabled,
            policy_estimator_gradient_norm=policy_estimator_gradient_norm,
        )

    def clear_all(self) -> None:
        """Leave no parameter gradients live between integrated updates."""

        self.estimator_updater.optimizer.zero_grad(set_to_none=True)
        self.sac_updater.critic_optimizer.zero_grad(set_to_none=True)
        self.sac_updater.actor_optimizer.zero_grad(set_to_none=True)
        self.sac_updater.temperature_optimizer.zero_grad(set_to_none=True)
        for parameter in self.sac_updater.target_critic.parameters():
            parameter.grad = None

    def _assert_all_clean(self, phase: str) -> None:
        if not self.enabled:
            return
        for name, module in self._modules().items():
            self._assert_no_gradient(module, name, phase)
        self._checks += 1

    def _validate_optimizer_ownership(self) -> None:
        modules = {
            "estimator": self.estimator_updater.estimator,
            "critic": self.sac_updater.critic,
            "actor": self.sac_updater.actor,
            "temperature": self.sac_updater.temperature,
        }
        optimizers = {
            "estimator": self.estimator_updater.optimizer,
            "critic": self.sac_updater.critic_optimizer,
            "actor": self.sac_updater.actor_optimizer,
            "temperature": self.sac_updater.temperature_optimizer,
        }
        if len({id(optimizer) for optimizer in optimizers.values()}) != len(optimizers):
            raise ValueError("FastWMR requires four distinct optimizer instances.")

        owned_parameter_ids: dict[str, set[int]] = {}
        for name, module in modules.items():
            module_ids = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
            optimizer_ids = {
                id(parameter)
                for group in optimizers[name].param_groups
                for parameter in group["params"]
            }
            if not module_ids or optimizer_ids != module_ids:
                raise ValueError(f"The {name} optimizer must own exactly the trainable {name} parameters.")
            owned_parameter_ids[name] = module_ids

        names = tuple(owned_parameter_ids)
        for index, name in enumerate(names):
            for other_name in names[index + 1 :]:
                if owned_parameter_ids[name] & owned_parameter_ids[other_name]:
                    raise ValueError(f"{name} and {other_name} optimizer parameters must be disjoint.")
        if any(parameter.requires_grad for parameter in self.sac_updater.target_critic.parameters()):
            raise ValueError("Target critic parameters must remain frozen.")

    def _modules(self) -> dict[str, nn.Module]:
        return {
            "estimator": self.estimator_updater.estimator,
            "critic": self.sac_updater.critic,
            "actor": self.sac_updater.actor,
            "temperature": self.sac_updater.temperature,
            "target critic": self.sac_updater.target_critic,
        }

    @staticmethod
    def _assert_no_gradient(module: nn.Module, name: str, phase: str) -> None:
        for parameter in module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise GradientBoundaryError(f"{phase} left a non-finite gradient on {name}.")
            if torch.count_nonzero(gradient).item() != 0:
                raise GradientBoundaryError(f"{phase} leaked a gradient into {name}.")

    @staticmethod
    def _require_finite_gradient(module: nn.Module, name: str, phase: str) -> float:
        squared_norm: torch.Tensor | None = None
        for parameter in module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise GradientBoundaryError(f"{phase} produced a non-finite {name} gradient.")
            contribution = gradient.detach().square().sum()
            squared_norm = contribution if squared_norm is None else squared_norm + contribution
        if squared_norm is None:
            raise GradientBoundaryError(f"{phase} did not produce a {name} gradient.")
        return float(torch.sqrt(squared_norm))


class FastWMRSequenceFeatureProcessor:
    """Update the estimator and rebuild detached SAC features from raw replay.

    Every call performs three synchronized operations: optimize the estimator
    on a boundary-safe replay sequence, re-infer that sequence with the new
    parameters, and rebuild online recurrent memory from the recent rollout
    cache. The policy observation can be normalized for SAC without changing
    the raw observation history seen by the recurrent estimator.
    """

    def __init__(
        self,
        estimator_updater: EstimatorUpdater,
        runtime: "FastWMREstimatorRuntime",
        rollout_cache: EstimatorRolloutCache,
        *,
        interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
        observation_normalizer: RunningObservationNormalizer | None = None,
        gradient_cutoff: bool = True,
        estimator_frozen: bool = False,
    ) -> None:
        if estimator_updater.estimator is not runtime.estimator:
            raise ValueError("Estimator updater and runtime must share the estimator instance.")
        if estimator_updater.observation_transform is not runtime.observation_transform:
            raise ValueError("Estimator updater and runtime must share the observation transform.")
        if estimator_updater.interface != interface:
            raise ValueError("Estimator updater and sequence processor must share the interface contract.")
        if estimator_updater.version != runtime.estimator_version:
            raise ValueError("Estimator updater and runtime versions must match before integration.")
        if rollout_cache.spec.num_envs != runtime.num_envs:
            raise ValueError("Estimator rollout cache and runtime environment counts must match.")
        if rollout_cache.spec.observation_dim != interface.policy_observation_dim:
            raise ValueError("Estimator rollout cache observation width does not match the interface.")
        if rollout_cache.spec.privileged_state_dim != interface.reconstruction_target_dim:
            raise ValueError("Estimator rollout cache target width does not match the interface.")
        if observation_normalizer is not None:
            if observation_normalizer.observation_dim != interface.policy_observation_dim:
                raise ValueError("Observation normalizer width does not match the policy observation.")
            parameter = next(estimator_updater.estimator.parameters())
            if observation_normalizer.mean.device != parameter.device:
                raise ValueError("Observation normalizer and estimator must share a device.")

        self.estimator_updater = estimator_updater
        self.runtime = runtime
        self.rollout_cache = rollout_cache
        self.interface = interface
        self.observation_normalizer = observation_normalizer
        self.gradient_cutoff = gradient_cutoff
        self.estimator_frozen = estimator_frozen
        if estimator_frozen and not gradient_cutoff:
            raise ValueError("A frozen estimator cannot be combined with disabled gradient cutoff.")
        self.updates = 0
        self.last_estimator_update: EstimatorUpdateResult | None = None
        self.last_runtime_rebuild: EstimatorRuntimeRebuild | None = None

    def __call__(
        self,
        sequence: SequenceReplayBatch,
        *,
        synchronize_runtime: bool = True,
        validate_values: bool = True,
    ) -> torch.Tensor:
        parameter = next(self.estimator_updater.estimator.parameters())
        if sequence.observations.device != parameter.device:
            raise ValueError("Replay sequence and estimator must share a device.")
        if sequence.observations.dtype != parameter.dtype:
            raise ValueError("Replay sequence and estimator must share a floating dtype.")

        estimator_update = (
            self.estimator_updater.evaluate_sequence(
                sequence,
                validate_values=validate_values,
            )
            if self.estimator_frozen
            else self.estimator_updater.update_sequence(
                sequence,
                validate_values=validate_values,
            )
        )
        reconstructions = self.estimator_updater.reconstruct_sequence(
            sequence,
            detach=self.gradient_cutoff,
            validate_values=validate_values,
        )
        runtime_rebuild = None
        if not self.estimator_frozen and self.gradient_cutoff and synchronize_runtime:
            runtime_rebuild = self.synchronize_runtime()

        features = build_control_feature(
            sequence.learning_observations,
            reconstructions,
            cfg=self.interface,
            normalizer=self.observation_normalizer,
            detach_reconstruction=self.gradient_cutoff,
        )
        if self.gradient_cutoff:
            features = features.detach()
        if validate_values and not torch.isfinite(features).all():
            raise FloatingPointError("Current-estimator control features must remain finite.")

        self.last_estimator_update = estimator_update
        self.last_runtime_rebuild = runtime_rebuild
        self.updates += 1
        return features

    def finalize_policy_estimator_step(
        self,
        *,
        synchronize_runtime: bool = True,
        validate_values: bool = True,
    ) -> torch.Tensor:
        """Step no-cutoff SAC gradients and rebuild runtime with new weights."""

        if self.gradient_cutoff or self.estimator_frozen:
            raise RuntimeError("Policy estimator gradients exist only in no-cutoff training.")
        gradient_norm = self.estimator_updater.step_external_gradients(
            validate_values=validate_values,
        )
        version = self.estimator_updater.version
        if synchronize_runtime:
            self.last_runtime_rebuild = self.synchronize_runtime()
        if self.last_estimator_update is not None:
            self.last_estimator_update = replace(
                self.last_estimator_update,
                metrics=replace(self.last_estimator_update.metrics, estimator_version=version),
            )
        return gradient_norm

    def synchronize_runtime(self) -> "EstimatorRuntimeRebuild | None":
        """Rebuild rollout state once after a bundle of estimator updates."""

        version = self.estimator_updater.version
        if self.estimator_frozen or self.runtime.estimator_version == version:
            return None
        rebuild = self._rebuild_runtime(version)
        self.last_runtime_rebuild = rebuild
        return rebuild

    def _rebuild_runtime(self, estimator_version: int) -> "EstimatorRuntimeRebuild | None":
        if len(self.rollout_cache) > 0:
            return self.runtime.rebuild_from_cache(
                self.rollout_cache,
                estimator_version=estimator_version,
                decode_full_sequence=False,
            )
        self.runtime.reset_all(estimator_version=estimator_version)
        return None


class FastWMRAgent:
    """Own the ordered estimator and FastSAC optimizer lifecycle."""

    UPDATE_ORDER = ("estimator", "critic", "actor", "temperature", "target")

    def __init__(
        self,
        sac_updater: SACUpdater,
        feature_processor: FastWMRSequenceFeatureProcessor,
        *,
        verify_gradient_boundaries: bool = True,
        validation_interval: int = 1,
        initial_validation_updates: int = 0,
    ) -> None:
        if feature_processor.interface.actor_input_dim != sac_updater.actor.input_dim:
            raise ValueError("FastWMR agent feature and actor input dimensions must match.")
        if not feature_processor.gradient_cutoff and verify_gradient_boundaries:
            raise ValueError("Gradient-boundary checks require gradient cutoff to be enabled.")
        if validation_interval <= 0:
            raise ValueError("validation_interval must be positive.")
        if initial_validation_updates < 0:
            raise ValueError("initial_validation_updates must be non-negative.")
        self.sac_updater = sac_updater
        self.feature_processor = feature_processor
        self.verify_gradient_boundaries = verify_gradient_boundaries
        self.validation_interval = validation_interval
        self.initial_validation_updates = initial_validation_updates
        self.gradient_guard = FastWMRGradientGuard(
            feature_processor.estimator_updater,
            sac_updater,
            enabled=verify_gradient_boundaries,
        )
        self.update_steps = 0
        self.last_update: FastWMRAgentUpdateResult | None = None

    def update(
        self,
        sequence: SequenceReplayBatch,
        *,
        synchronize_runtime: bool = True,
    ) -> FastWMRAgentUpdateResult:
        """Run estimator, critic, actor, alpha, and target phases in order."""

        completed_phases: list[str] = []
        validate_values = self._should_validate_update()
        self.gradient_guard.enabled = self.verify_gradient_boundaries and validate_values
        self.gradient_guard.begin_update()
        try:
            learning_features = self.feature_processor(
                sequence,
                synchronize_runtime=synchronize_runtime,
                validate_values=validate_values,
            )
            completed_phases.append("estimator")
            estimator_update = self.feature_processor.last_estimator_update
            if estimator_update is None:
                raise RuntimeError("Feature processing did not produce an estimator update result.")
            estimator_gradient_norm = self.gradient_guard.after_estimator(
                require_gradient=not self.feature_processor.estimator_frozen,
            )

            batch = _build_sequence_learning_batch(
                sequence,
                learning_features,
                actor_input_dim=self.sac_updater.actor.input_dim,
                require_detached=self.feature_processor.gradient_cutoff,
                validate_values=validate_values,
            )
            if self.feature_processor.gradient_cutoff:
                self.gradient_guard.before_sac(batch)

            if self.feature_processor.gradient_cutoff:
                critic_output = self.sac_updater.update_critic(batch)
            else:
                critic_output = self.sac_updater.update_critic(batch, retain_graph=True)
            completed_phases.append("critic")
            if self.feature_processor.gradient_cutoff:
                self.gradient_guard.after_critic()

            if self.feature_processor.gradient_cutoff:
                actor_output = self.sac_updater.update_actor(batch.states)
            else:
                actor_output = self.sac_updater.update_actor(
                    batch.states,
                    allow_state_gradients=True,
                )
            completed_phases.append("actor")
            policy_estimator_gradient_norm = None
            if self.feature_processor.gradient_cutoff:
                self.gradient_guard.after_actor()
            else:
                policy_estimator_gradient_norm = (
                    self.feature_processor.finalize_policy_estimator_step(
                        synchronize_runtime=synchronize_runtime,
                        validate_values=validate_values,
                    )
                )
                estimator_update = self.feature_processor.last_estimator_update
                if estimator_update is None:
                    raise RuntimeError("No-cutoff estimator finalization lost its update result.")

            temperature_loss = self.sac_updater.update_temperature(
                actor_output.log_probabilities
            )
            completed_phases.append("temperature")
            self.gradient_guard.after_temperature()

            self.sac_updater.update_target()
            completed_phases.append("target")
            self.gradient_guard.after_target()

            if tuple(completed_phases) != self.UPDATE_ORDER:
                raise RuntimeError(f"FastWMR update order drifted to {tuple(completed_phases)}.")
            sac_metrics = SACUpdateMetrics(
                critic_loss=critic_output.loss.detach(),
                actor_loss=actor_output.loss.detach(),
                temperature_loss=temperature_loss.detach(),
                temperature=self.sac_updater.temperature().detach(),
                target_q_mean=critic_output.target.mean().detach(),
                target_q_std=critic_output.target.std(unbiased=False).detach(),
                q1_mean=critic_output.q1.mean().detach(),
                q1_std=critic_output.q1.std(unbiased=False).detach(),
                q2_mean=critic_output.q2.mean().detach(),
                q2_std=critic_output.q2.std(unbiased=False).detach(),
                policy_entropy=(-actor_output.log_probabilities.mean()).detach(),
            )
            result = FastWMRAgentUpdateResult(
                estimator_update=estimator_update,
                sac_update=sac_metrics,
                update_order=tuple(completed_phases),
                gradient_boundary=self.gradient_guard.report(
                    estimator_gradient_norm,
                    cutoff_enabled=self.feature_processor.gradient_cutoff,
                    policy_estimator_gradient_norm=policy_estimator_gradient_norm,
                ),
            )
            self.update_steps += 1
            self.last_update = result
            return result
        finally:
            self.gradient_guard.clear_all()

    def _should_validate_update(self) -> bool:
        return (
            self.update_steps < self.initial_validation_updates
            or self.update_steps % self.validation_interval == 0
        )


def _build_sequence_learning_batch(
    sequence: SequenceReplayBatch,
    learning_features: torch.Tensor,
    *,
    actor_input_dim: int,
    require_detached: bool = True,
    validate_values: bool = True,
) -> SACTransitionBatch:
    expected_shape = (
        sequence.batch_size,
        sequence.learning_length + 1,
        actor_input_dim,
    )
    if learning_features.shape != expected_shape:
        raise ValueError(
            f"Sequence feature processor must return shape {expected_shape}, "
            f"got {tuple(learning_features.shape)}."
        )
    if require_detached and learning_features.requires_grad:
        raise ValueError("SAC learning features must be detached after estimator update.")
    if validate_values and not torch.isfinite(learning_features).all():
        raise ValueError("SAC learning features must be finite.")
    return SACTransitionBatch(
        states=learning_features[:, :-1],
        actions=sequence.learning_actions,
        rewards=sequence.learning_rewards,
        next_states=learning_features[:, 1:],
        terminated=sequence.learning_terminated,
        truncated=sequence.learning_truncated,
        allow_state_gradients=not require_detached,
    )


class FastSACReplayUpdateLoop:
    """Ordinary transition sampling after random-action replay warm-up."""

    def __init__(
        self,
        replay: TransitionReplayBuffer,
        updater: SACUpdater,
        cfg: ReplayUpdateCfg,
        *,
        learner_device: torch.device | str,
        observation_normalizer: RunningObservationNormalizer | None = None,
    ) -> None:
        self.replay = replay
        self.updater = updater
        self.cfg = cfg
        self.learner_device = torch.device(learner_device)
        self.observation_normalizer = observation_normalizer
        if observation_normalizer is not None:
            if observation_normalizer.observation_dim != updater.actor.input_dim:
                raise ValueError("Observation normalizer and actor input dimensions must match.")
            if observation_normalizer.mean.device != self.learner_device:
                raise ValueError("Observation normalizer must be on learner_device.")
        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def warming_up(self) -> bool:
        return self.environment_steps < self.cfg.random_action_steps

    @property
    def ready(self) -> bool:
        return (
            not self.warming_up
            and len(self.replay) >= self.cfg.minimum_replay_size
            and self.replay.can_sample(self.cfg.batch_size)
        )

    def advance_environment(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("Environment step increment must be positive.")
        self.environment_steps += steps

    def update_observation_statistics(self, observations: torch.Tensor) -> None:
        """Record raw rollout observations exactly once at collection time."""

        if self.observation_normalizer is not None:
            self.observation_normalizer.update(observations)

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Apply current statistics without mutating them."""

        if self.observation_normalizer is None:
            return observations
        return self.observation_normalizer(observations)

    @torch.no_grad()
    def select_actions(self, states: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """Use uniform bounded actions during warm-up, then the learned actor."""

        actor = self.updater.actor
        if states.shape[-1] != actor.input_dim:
            raise ValueError("Action-selection state dimension does not match the actor.")
        if self.warming_up:
            random_values = torch.rand((*states.shape[:-1], actor.action_dim), device=states.device)
            return actor.action_low + random_values * (actor.action_high - actor.action_low)
        return actor.act(self.normalize_observations(states), deterministic=deterministic)

    def run_updates(self, *, generator: torch.Generator | None = None) -> list[SACUpdateMetrics]:
        if not self.ready:
            return []
        metrics: list[SACUpdateMetrics] = []
        for _ in range(self.cfg.num_updates):
            replay_batch = self.replay.sample(self.cfg.batch_size, generator=generator)
            batch = SACTransitionBatch.from_replay(
                replay_batch,
                feature_source=SACFeatureSource.POLICY_OBSERVATION,
            ).to(self.learner_device)
            batch = self._normalize_transition_batch(batch)
            metrics.append(self.updater.update(batch))
            self.gradient_steps += 1
        return metrics

    def _normalize_transition_batch(self, batch: SACTransitionBatch) -> SACTransitionBatch:
        if self.observation_normalizer is None:
            return batch
        return SACTransitionBatch(
            states=self.normalize_observations(batch.states),
            actions=batch.actions,
            rewards=batch.rewards,
            next_states=self.normalize_observations(batch.next_states),
            terminated=batch.terminated,
            truncated=batch.truncated,
        )


class FastWMRSequenceUpdateLoop(FastSACReplayUpdateLoop):
    """Sequence replay that delegates current-estimator reconstruction.

    ``sequence_feature_processor`` must run burn-in/current-estimator inference,
    finish any estimator update for the full sequence, and return detached
    control features for the ``L + 1`` learning observations.
    """

    def __init__(
        self,
        replay: TransitionReplayBuffer,
        updater: SACUpdater,
        cfg: ReplayUpdateCfg,
        sequence_cfg: SequenceReplayCfg,
        sequence_feature_processor: SequenceFeatureProcessor,
        *,
        learner_device: torch.device | str,
        verify_gradient_boundaries: bool = True,
        validation_interval: int = 1,
        initial_validation_updates: int = 0,
        sequence_augmentation: SequenceAugmentation | None = None,
    ) -> None:
        super().__init__(replay, updater, cfg, learner_device=learner_device)
        self.sequence_cfg = sequence_cfg
        self.sequence_feature_processor = sequence_feature_processor
        self.sequence_augmentation = sequence_augmentation
        self.agent = (
            FastWMRAgent(
                updater,
                sequence_feature_processor,
                verify_gradient_boundaries=verify_gradient_boundaries,
                validation_interval=validation_interval,
                initial_validation_updates=initial_validation_updates,
            )
            if isinstance(sequence_feature_processor, FastWMRSequenceFeatureProcessor)
            else None
        )
        self.last_agent_updates: tuple[FastWMRAgentUpdateResult, ...] = ()

    @property
    def ready(self) -> bool:
        minimum_insertion_id = self._minimum_sequence_insertion_id()
        return (
            not self.warming_up
            and len(self.replay) >= self.cfg.minimum_replay_size
            and self.replay.can_sample_sequences(
                self.sequence_cfg.batch_size,
                self.sequence_cfg.burn_in_length,
                self.sequence_cfg.learning_length,
                require_episode_start=self.sequence_cfg.require_episode_start,
                minimum_insertion_id=minimum_insertion_id,
            )
        )

    def run_updates(self, *, generator: torch.Generator | None = None) -> list[SACUpdateMetrics]:
        self.last_agent_updates = ()
        if not self.ready:
            return []
        metrics: list[SACUpdateMetrics] = []
        agent_updates: list[FastWMRAgentUpdateResult] = []
        minimum_insertion_id = self._minimum_sequence_insertion_id()
        for _ in range(self.cfg.num_updates):
            sequence = self.replay.sample_sequences(
                self.sequence_cfg.batch_size,
                self.sequence_cfg.burn_in_length,
                self.sequence_cfg.learning_length,
                require_episode_start=self.sequence_cfg.require_episode_start,
                minimum_insertion_id=minimum_insertion_id,
                device=self.learner_device,
                generator=generator,
            )
            if self.sequence_augmentation is not None:
                sequence = self.sequence_augmentation(sequence)
            if self.agent is not None:
                agent_update = self.agent.update(
                    sequence,
                    synchronize_runtime=False,
                )
                agent_updates.append(agent_update)
                metrics.append(agent_update.sac_update)
            else:
                learning_features = self.sequence_feature_processor(sequence)
                batch = self._build_learning_batch(sequence, learning_features)
                metrics.append(self.updater.update(batch))
            self.gradient_steps += 1
        if agent_updates:
            self.sequence_feature_processor.synchronize_runtime()
        self.last_agent_updates = tuple(agent_updates)
        return metrics

    def _minimum_sequence_insertion_id(self) -> int | None:
        horizon = self.sequence_cfg.recent_transition_horizon
        if horizon is None:
            return None
        return max(0, self.replay.total_inserted - horizon)

    def _build_learning_batch(
        self,
        sequence: SequenceReplayBatch,
        learning_features: torch.Tensor,
    ) -> SACTransitionBatch:
        return _build_sequence_learning_batch(
            sequence,
            learning_features,
            actor_input_dim=self.updater.actor.input_dim,
        )
