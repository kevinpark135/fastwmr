"""Replay warm-up and update loops for FastSAC and sequence-based FastWMR."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..buffers import EstimatorRolloutCache, SequenceReplayBatch, TransitionReplayBuffer
from ..config import (
    DEFAULT_FASTWMR_V2_CFG,
    DEFAULT_INTERFACE_CFG,
    FastWMRInterfaceCfg,
    FastWMRV2Cfg,
    ReplayUpdateCfg,
    SequenceReplayCfg,
)
from ..utils.feature_builder import build_control_feature
from ..utils.normalization import RunningObservationNormalizer
from ..utils.profiling import StageProfiler
from .estimator_update import EMAControlEstimator, EstimatorUpdateResult, EstimatorUpdater
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


class ReconstructionGateState(str, Enum):
    """One-way routing state for a qualified stationary reconstruction snapshot."""

    CLOSED = "closed"
    RAMPING = "ramping"
    OPEN = "open"


class FastWMRV2EstimatorController:
    """Qualify one online estimator and freeze its snapshot for control."""

    def __init__(
        self,
        estimator_updater: EstimatorUpdater,
        ema_estimator: EMAControlEstimator,
        runtime: "FastWMREstimatorRuntime",
        rollout_cache: EstimatorRolloutCache,
        *,
        cfg: FastWMRV2Cfg = DEFAULT_FASTWMR_V2_CFG,
        interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
        observation_normalizer: RunningObservationNormalizer | None = None,
        estimator_frozen: bool = False,
        validation_interval: int = 1,
        initial_validation_updates: int = 0,
    ) -> None:
        if estimator_updater.estimator is not ema_estimator.online_estimator:
            raise ValueError("Estimator updater must own the EMA online estimator.")
        if runtime.estimator is not ema_estimator.control_estimator:
            raise ValueError("Runtime must use the EMA control estimator.")
        if runtime.estimator_version != ema_estimator.version:
            raise ValueError("Runtime and EMA control-estimator versions must match.")
        if rollout_cache.spec.num_envs != runtime.num_envs:
            raise ValueError("Rollout cache and runtime environment counts must match.")
        if validation_interval <= 0 or initial_validation_updates < 0:
            raise ValueError("Estimator validation schedule is invalid.")
        self.estimator_updater = estimator_updater
        self.ema_estimator = ema_estimator
        self.runtime = runtime
        self.rollout_cache = rollout_cache
        self.cfg = cfg
        self.interface = interface
        unknown_control_fields = set(cfg.control_reconstruction_fields).difference(
            interface.reconstruction_layout.names
        )
        if unknown_control_fields:
            raise ValueError(
                "Control reconstruction fields are absent from the interface: "
                + ", ".join(sorted(unknown_control_fields))
            )
        self.control_reconstruction_fields = cfg.control_reconstruction_fields
        self.observation_normalizer = observation_normalizer
        self.estimator_frozen = estimator_frozen
        self.validation_interval = validation_interval
        self.initial_validation_updates = initial_validation_updates
        self.estimator_updates = 0
        self.estimator_attempts = 0
        self.estimator_triggers = 0
        self.last_estimator_update: EstimatorUpdateResult | None = None
        self.last_gate_validation: EstimatorUpdateResult | None = None
        self.last_runtime_rebuild: EstimatorRuntimeRebuild | None = None
        self.gate_state = ReconstructionGateState.CLOSED
        self.gate_quality_ema: float | None = None
        self.gate_base_velocity_rmse_ema: float | None = None
        self.gate_contact_bce_ema: float | None = None
        self.gate_quality_passes = 0
        self.gate_quality_failures = 0
        self.gate_validation_checks = 0
        self.snapshot_active = False
        self.snapshot_estimator_version: int | None = None
        self.snapshot_replay_resets = 0
        self.snapshot_gate_updates = 0
        self._reconstruction_gate = 0.0
        self._gate_target = 0.0
        self._gate_hard_sync_pending = False
        self._snapshot_activation_pending = False

    @property
    def updates(self) -> int:
        return self.estimator_updates

    @property
    def control_estimator_version(self) -> int:
        return self.ema_estimator.version

    @property
    def reconstruction_gate(self) -> float:
        return self._reconstruction_gate

    @property
    def online_estimator_frozen(self) -> bool:
        return self.estimator_frozen or (
            self.snapshot_active
            and self.cfg.freeze_online_estimator_after_snapshot
        )

    @property
    def gate_validation_due(self) -> bool:
        return (
            not self.snapshot_active
            and self.estimator_attempts
            >= (
                self.gate_validation_checks + 1
            )
            * self.cfg.reconstruction_gate_validation_interval
        )

    def update_sequence(self, sequence: SequenceReplayBatch) -> EstimatorUpdateResult:
        """Run one slow estimator step without re-inferring SAC features."""

        validate_values = self._should_validate_attempt()
        result = (
            self.estimator_updater.evaluate_sequence(
                sequence,
                validate_values=validate_values,
            )
            if self.online_estimator_frozen
            else self.estimator_updater.update_sequence(
                sequence,
                validate_values=validate_values,
            )
        )
        self.estimator_attempts += 1
        if not self.online_estimator_frozen:
            self.estimator_updates += 1
            self._advance_gate_state()
        self.last_estimator_update = result
        return result

    def validate_reconstruction_gate(
        self,
        sequence: SequenceReplayBatch,
    ) -> EstimatorUpdateResult:
        """Update the gate quality EMA from a separately sampled replay sequence."""

        result = self.estimator_updater.evaluate_sequence(sequence)
        quality = float(result.metrics.total_loss)
        base_velocity_rmse = math.sqrt(
            max(
                0.0,
                float(
                    result.metrics.physical_field_losses[
                        "base_lin_vel_mse"
                    ]
                ),
            )
        )
        contact_bce = float(
            result.metrics.physical_field_losses["foot_contacts_bce"]
        )
        decay = self.cfg.reconstruction_gate_quality_ema_decay
        self.gate_quality_ema = (
            quality
            if self.gate_quality_ema is None
            else decay * self.gate_quality_ema + (1.0 - decay) * quality
        )
        self.gate_base_velocity_rmse_ema = (
            base_velocity_rmse
            if self.gate_base_velocity_rmse_ema is None
            else decay * self.gate_base_velocity_rmse_ema
            + (1.0 - decay) * base_velocity_rmse
        )
        self.gate_contact_bce_ema = (
            contact_bce
            if self.gate_contact_bce_ema is None
            else decay * self.gate_contact_bce_ema
            + (1.0 - decay) * contact_bce
        )
        self.gate_validation_checks += 1
        quality_passed = (
            self.gate_quality_ema
            <= self.cfg.reconstruction_gate_quality_threshold
            and self.gate_base_velocity_rmse_ema
            <= self.cfg.reconstruction_gate_base_velocity_rmse_threshold
            and self.gate_contact_bce_ema
            <= self.cfg.reconstruction_gate_contact_bce_threshold
        )
        if quality_passed:
            self.gate_quality_passes += 1
            self.gate_quality_failures = 0
        else:
            self.gate_quality_passes = 0
            self.gate_quality_failures += 1

        ready = (
            not self.snapshot_active
            and self.estimator_updates
            >= self.cfg.reconstruction_gate_start_updates
            and self.gate_quality_passes
            >= self.cfg.reconstruction_gate_quality_patience
        )
        if self._gate_target == 0.0 and ready:
            self._gate_hard_sync_pending = True
            self._set_gate_target(1.0)
        self.last_gate_validation = result
        return result

    def synchronize_control_estimator(self) -> "EstimatorRuntimeRebuild | None":
        """EMA-sync before qualification, then freeze one hard control snapshot."""

        self.estimator_triggers += 1
        if self._gate_hard_sync_pending:
            version = self.ema_estimator.hard_sync(advance_version=True)
            self._gate_hard_sync_pending = False
            self.snapshot_active = True
            self.snapshot_estimator_version = version
            self._snapshot_activation_pending = True
        elif self.snapshot_active or self.estimator_frozen:
            self.last_runtime_rebuild = None
            return None
        else:
            version = self.ema_estimator.update()
        if len(self.rollout_cache) > 0:
            rebuild = self.runtime.rebuild_from_cache(
                self.rollout_cache,
                estimator_version=version,
                decode_full_sequence=False,
            )
        else:
            self.runtime.reset_all(estimator_version=version)
            rebuild = None
        self.last_runtime_rebuild = rebuild
        return rebuild

    def advance_frozen_snapshot(self) -> None:
        """Advance a qualified one-way gate without changing estimator weights."""

        if not self.snapshot_active:
            raise RuntimeError("Cannot advance the snapshot gate before qualification.")
        self.estimator_triggers += 1
        self.snapshot_gate_updates += 1
        self._advance_gate_state()

    def consume_snapshot_activation(self) -> bool:
        """Return a one-shot signal used by the learner to clear pre-gate replay."""

        activated = self._snapshot_activation_pending
        self._snapshot_activation_pending = False
        return activated

    def record_snapshot_replay_reset(self) -> None:
        if not self.snapshot_active:
            raise RuntimeError("Replay cannot reset before a control snapshot is active.")
        if self.snapshot_replay_resets != 0:
            raise RuntimeError("Snapshot replay may reset only once.")
        self.snapshot_replay_resets += 1

    def clear_transient_state(self) -> None:
        self.last_estimator_update = None
        self.last_gate_validation = None
        self.last_runtime_rebuild = None

    def _advance_gate_state(self) -> None:
        warmup = self.cfg.reconstruction_gate_warmup_updates
        if warmup == 0:
            self._reconstruction_gate = self._gate_target
        else:
            step = 1.0 / warmup
            if self._reconstruction_gate < self._gate_target:
                self._reconstruction_gate = min(
                    self._gate_target,
                    self._reconstruction_gate + step,
                )
            elif self._reconstruction_gate > self._gate_target:
                self._reconstruction_gate = max(
                    self._gate_target,
                    self._reconstruction_gate - step,
                )
        self._refresh_gate_state()

    def _set_gate_target(self, target: float) -> None:
        if target not in (0.0, 1.0):
            raise ValueError("Gate target must be zero or one.")
        self._gate_target = target
        if self.cfg.reconstruction_gate_warmup_updates == 0:
            self._reconstruction_gate = target
        self._refresh_gate_state()

    def _refresh_gate_state(self) -> None:
        if self._reconstruction_gate <= 0.0 and self._gate_target == 0.0:
            self._reconstruction_gate = 0.0
            self.gate_state = ReconstructionGateState.CLOSED
        elif self._reconstruction_gate >= 1.0 and self._gate_target == 1.0:
            self._reconstruction_gate = 1.0
            self.gate_state = ReconstructionGateState.OPEN
        elif self._gate_target == 1.0:
            self.gate_state = ReconstructionGateState.RAMPING
        else:
            raise RuntimeError("A one-way reconstruction gate cannot close.")

    def _should_validate_attempt(self) -> bool:
        return (
            self.estimator_attempts < self.initial_validation_updates
            or self.estimator_attempts % self.validation_interval == 0
        )


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
            sac_metrics = self.sac_updater.metrics_from_outputs(
                critic_output,
                actor_output,
                temperature_loss,
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
        normalizer_freeze_iteration: int | None = None,
    ) -> None:
        if normalizer_freeze_iteration is not None and normalizer_freeze_iteration < 0:
            raise ValueError("normalizer_freeze_iteration must be non-negative.")
        self.replay = replay
        self.updater = updater
        self.cfg = cfg
        self.learner_device = torch.device(learner_device)
        self.observation_normalizer = observation_normalizer
        self.normalizer_freeze_iteration = normalizer_freeze_iteration
        if observation_normalizer is not None:
            if observation_normalizer.observation_dim != updater.actor.input_dim:
                raise ValueError("Observation normalizer and actor input dimensions must match.")
            if observation_normalizer.mean.device != self.learner_device:
                raise ValueError("Observation normalizer must be on learner_device.")
        self.environment_steps = 0
        self.gradient_steps = 0
        self.profiler = StageProfiler(self.learner_device)

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

    @property
    def normalization_frozen(self) -> bool:
        """Whether rollout statistics have reached their configured freeze point."""

        return (
            self.normalizer_freeze_iteration is not None
            and self.environment_steps >= self.normalizer_freeze_iteration
        )

    def advance_environment(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("Environment step increment must be positive.")
        self.environment_steps += steps

    def update_observation_statistics(self, observations: torch.Tensor) -> None:
        """Record raw rollout observations exactly once at collection time."""

        if self.observation_normalizer is not None and not self.normalization_frozen:
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
            with self.profiler.measure("replay_sample"):
                replay_batch = self.replay.sample(self.cfg.batch_size, generator=generator)
            with self.profiler.measure("transfer"):
                batch = SACTransitionBatch.from_replay(
                    replay_batch,
                    feature_source=SACFeatureSource.POLICY_OBSERVATION,
                ).to(self.learner_device)
            batch = self._normalize_transition_batch(batch)
            with self.profiler.measure("sac_update"):
                metrics.append(self.updater.update(batch))
            self.gradient_steps += 1
        return metrics

    def drain_profile_metrics(self) -> dict[str, int | float]:
        return self.profiler.drain_metrics()

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
        normalizer_freeze_iteration: int | None = None,
    ) -> None:
        super().__init__(
            replay,
            updater,
            cfg,
            learner_device=learner_device,
            normalizer_freeze_iteration=normalizer_freeze_iteration,
        )
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
                episode_start_fraction=self.sequence_cfg.episode_start_fraction,
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
            with self.profiler.measure("sequence_sample_transfer"):
                sequence = self.replay.sample_sequences(
                    self.sequence_cfg.batch_size,
                    self.sequence_cfg.burn_in_length,
                    self.sequence_cfg.learning_length,
                    require_episode_start=self.sequence_cfg.require_episode_start,
                    episode_start_fraction=self.sequence_cfg.episode_start_fraction,
                    minimum_insertion_id=minimum_insertion_id,
                    device=self.learner_device,
                    generator=generator,
                )
            if self.sequence_augmentation is not None:
                sequence = self.sequence_augmentation(sequence)
            if self.agent is not None:
                with self.profiler.measure("integrated_estimator_sac_update"):
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
            with self.profiler.measure("runtime_rebuild"):
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


class FastWMRV2UpdateLoop(FastSACReplayUpdateLoop):
    """FastSAC transition updates plus a low-frequency sequence estimator."""

    def __init__(
        self,
        replay: TransitionReplayBuffer,
        updater: SACUpdater,
        cfg: ReplayUpdateCfg,
        sequence_cfg: SequenceReplayCfg,
        estimator_controller: FastWMRV2EstimatorController,
        *,
        learner_device: torch.device | str,
        v2_cfg: FastWMRV2Cfg = DEFAULT_FASTWMR_V2_CFG,
        sequence_augmentation: SequenceAugmentation | None = None,
        normalizer_freeze_iteration: int | None = None,
    ) -> None:
        super().__init__(
            replay,
            updater,
            cfg,
            learner_device=learner_device,
            normalizer_freeze_iteration=normalizer_freeze_iteration,
        )
        if estimator_controller.cfg != v2_cfg:
            raise ValueError("V2 update loop and estimator controller configs must match.")
        if replay.spec.reconstruction_dim != estimator_controller.interface.reconstruction_target_dim:
            raise ValueError("Stored reconstruction width must match the estimator contract.")
        self.sequence_cfg = sequence_cfg
        self.estimator_controller = estimator_controller
        self.sequence_feature_processor = estimator_controller
        self.v2_cfg = v2_cfg
        self.sequence_augmentation = sequence_augmentation
        self.sac_updates_since_estimator = 0
        self.last_estimator_updates: tuple[EstimatorUpdateResult, ...] = ()
        self.last_feature_age_mean: torch.Tensor | None = None
        self.last_feature_age_max: torch.Tensor | None = None
        self.last_eligible_features = 0
        self.last_rejected_features = 0
        self.last_full_transition_count = 0
        self.last_fresh_features = 0
        self.last_stale_features = 0
        self.last_sampled_fresh_fraction: torch.Tensor | None = None
        self.last_reconstruction_masked_fraction: torch.Tensor | None = None
        self.last_reconstruction_confidence_mean: torch.Tensor | None = None
        self.last_reconstruction_confidence_min: torch.Tensor | None = None
        self.last_reconstruction_confidence_max: torch.Tensor | None = None
        self.last_snapshot_replay_reset = False

    @property
    def ready(self) -> bool:
        if self.warming_up or len(self.replay) < self.cfg.minimum_replay_size:
            return False
        return self.replay.can_sample_reconstructions(
            self.cfg.batch_size,
            current_estimator_version=self.estimator_controller.control_estimator_version,
            max_estimator_feature_age=self.v2_cfg.max_estimator_feature_age,
            recent_transition_horizon=self.v2_cfg.stored_feature_replay_horizon,
        )

    @property
    def estimator_ready(self) -> bool:
        return self.replay.can_sample_sequences(
            self.sequence_cfg.batch_size,
            self.sequence_cfg.burn_in_length,
            self.sequence_cfg.learning_length,
            require_episode_start=self.sequence_cfg.require_episode_start,
            episode_start_fraction=self.sequence_cfg.episode_start_fraction,
            minimum_insertion_id=self._minimum_sequence_insertion_id(),
        )

    def run_updates(self, *, generator: torch.Generator | None = None) -> list[SACUpdateMetrics]:
        self.last_estimator_updates = ()
        self.last_snapshot_replay_reset = False
        if not self.ready:
            return []
        metrics: list[SACUpdateMetrics] = []
        estimator_updates: list[EstimatorUpdateResult] = []
        for _ in range(self.cfg.num_updates):
            current_version = self.estimator_controller.control_estimator_version
            reconstruction_gate = self.estimator_controller.reconstruction_gate
            minimum_fresh_fraction = (
                reconstruction_gate * self.v2_cfg.fresh_reconstruction_fraction
            )
            with self.profiler.measure("stored_feature_sample"):
                replay_batch = self.replay.sample_reconstructions(
                    self.cfg.batch_size,
                    current_estimator_version=current_version,
                    max_estimator_feature_age=self.v2_cfg.max_estimator_feature_age,
                    recent_transition_horizon=self.v2_cfg.stored_feature_replay_horizon,
                    minimum_fresh_fraction=minimum_fresh_fraction,
                    generator=generator,
                )
                feature_ages = replay_batch.feature_ages(current_version)
                freshness = (
                    torch.ones_like(feature_ages, dtype=torch.bool)
                    if self.v2_cfg.max_estimator_feature_age is None
                    else feature_ages <= self.v2_cfg.max_estimator_feature_age
                )
            with self.profiler.measure("transfer"):
                replay_batch = replay_batch.to(self.learner_device)
                batch = SACTransitionBatch.from_stored_reconstruction_replay(
                    replay_batch,
                    interface=self.estimator_controller.interface,
                    normalizer=self.estimator_controller.observation_normalizer,
                    reconstruction_gate=reconstruction_gate,
                    reconstruction_freshness=freshness.to(self.learner_device),
                    reconstruction_fields=(
                        self.estimator_controller.control_reconstruction_fields
                    ),
                )
            with self.profiler.measure("sac_update"):
                metrics.append(self.updater.update(batch))
            self.gradient_steps += 1
            self.sac_updates_since_estimator += 1
            self.last_feature_age_mean = feature_ages.float().mean()
            self.last_feature_age_max = feature_ages.max()
            self.last_sampled_fresh_fraction = freshness.float().mean()
            confidence = reconstruction_gate * freshness.float()
            self.last_reconstruction_masked_fraction = (confidence == 0.0).float().mean()
            self.last_reconstruction_confidence_mean = confidence.mean()
            self.last_reconstruction_confidence_min = confidence.min()
            self.last_reconstruction_confidence_max = confidence.max()

            estimator_trigger_ready = (
                self.estimator_controller.snapshot_active
                and self.estimator_controller.cfg.freeze_online_estimator_after_snapshot
            ) or self.estimator_ready
            if (
                self.sac_updates_since_estimator >= self.v2_cfg.estimator_update_interval
                and estimator_trigger_ready
            ):
                trigger_updates, replay_reset = self._run_estimator_trigger(
                    generator=generator
                )
                estimator_updates.extend(trigger_updates)
                self.sac_updates_since_estimator = 0
                if replay_reset:
                    self.last_snapshot_replay_reset = True
                    break

        self.last_fresh_features = self.replay.eligible_reconstruction_count(
            current_estimator_version=self.estimator_controller.control_estimator_version,
            max_estimator_feature_age=self.v2_cfg.max_estimator_feature_age,
            recent_transition_horizon=self.v2_cfg.stored_feature_replay_horizon,
        )
        self.last_eligible_features = self.replay.available_reconstruction_count(
            recent_transition_horizon=self.v2_cfg.stored_feature_replay_horizon,
        )
        self.last_full_transition_count = len(self.replay)
        self.last_stale_features = self.last_eligible_features - self.last_fresh_features
        self.last_rejected_features = len(self.replay) - self.last_eligible_features
        self.last_estimator_updates = tuple(estimator_updates)
        return metrics

    def state_dict(self) -> dict[str, bool | int | float | str | None]:
        return {
            "sac_updates_since_estimator": self.sac_updates_since_estimator,
            "estimator_updates": self.estimator_controller.estimator_updates,
            "estimator_attempts": self.estimator_controller.estimator_attempts,
            "estimator_triggers": self.estimator_controller.estimator_triggers,
            "control_estimator_version": (
                self.estimator_controller.control_estimator_version
            ),
            "gate_state": self.estimator_controller.gate_state.value,
            "reconstruction_gate": self.estimator_controller.reconstruction_gate,
            "gate_target": self.estimator_controller._gate_target,
            "gate_quality_ema": self.estimator_controller.gate_quality_ema,
            "gate_base_velocity_rmse_ema": (
                self.estimator_controller.gate_base_velocity_rmse_ema
            ),
            "gate_contact_bce_ema": self.estimator_controller.gate_contact_bce_ema,
            "gate_quality_passes": self.estimator_controller.gate_quality_passes,
            "gate_quality_failures": self.estimator_controller.gate_quality_failures,
            "gate_validation_checks": self.estimator_controller.gate_validation_checks,
            "snapshot_active": self.estimator_controller.snapshot_active,
            "snapshot_estimator_version": (
                self.estimator_controller.snapshot_estimator_version
            ),
            "snapshot_replay_resets": (
                self.estimator_controller.snapshot_replay_resets
            ),
            "snapshot_gate_updates": self.estimator_controller.snapshot_gate_updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        required = {
            "sac_updates_since_estimator",
            "estimator_updates",
            "estimator_attempts",
            "estimator_triggers",
            "control_estimator_version",
            "gate_state",
            "reconstruction_gate",
            "gate_target",
            "gate_quality_ema",
            "gate_base_velocity_rmse_ema",
            "gate_contact_bce_ema",
            "gate_quality_passes",
            "gate_quality_failures",
            "gate_validation_checks",
            "snapshot_active",
            "snapshot_estimator_version",
            "snapshot_replay_resets",
            "snapshot_gate_updates",
        }
        if state.keys() != required:
            raise ValueError("FastWMR v2 scheduler state is incomplete or invalid.")
        integer_names = {
            "sac_updates_since_estimator",
            "estimator_updates",
            "estimator_attempts",
            "estimator_triggers",
            "control_estimator_version",
            "gate_quality_passes",
            "gate_quality_failures",
            "gate_validation_checks",
            "snapshot_replay_resets",
            "snapshot_gate_updates",
        }
        values = {name: int(state[name]) for name in integer_names}
        if any(value < 0 for value in values.values()):
            raise ValueError("FastWMR v2 scheduler counters must be non-negative.")
        gate_state = ReconstructionGateState(str(state["gate_state"]))
        quality_value = state["gate_quality_ema"]
        gate_quality_ema = None if quality_value is None else float(quality_value)
        if gate_quality_ema is not None and gate_quality_ema < 0.0:
            raise ValueError("Gate quality EMA must be non-negative.")
        base_quality_value = state["gate_base_velocity_rmse_ema"]
        gate_base_velocity_rmse_ema = (
            None if base_quality_value is None else float(base_quality_value)
        )
        contact_quality_value = state["gate_contact_bce_ema"]
        gate_contact_bce_ema = (
            None if contact_quality_value is None else float(contact_quality_value)
        )
        if (
            gate_base_velocity_rmse_ema is not None
            and gate_base_velocity_rmse_ema < 0.0
        ):
            raise ValueError("Gate base-velocity RMSE EMA must be non-negative.")
        if gate_contact_bce_ema is not None and gate_contact_bce_ema < 0.0:
            raise ValueError("Gate contact BCE EMA must be non-negative.")
        reconstruction_gate = float(state["reconstruction_gate"])
        gate_target = float(state["gate_target"])
        if not 0.0 <= reconstruction_gate <= 1.0:
            raise ValueError("Reconstruction gate must lie in [0, 1].")
        if gate_target not in (0.0, 1.0):
            raise ValueError("Gate target must be zero or one.")
        if gate_target == 0.0 and reconstruction_gate != 0.0:
            raise ValueError("A one-way gate cannot retain a closing value.")
        expected_state = (
            ReconstructionGateState.CLOSED
            if reconstruction_gate == 0.0 and gate_target == 0.0
            else ReconstructionGateState.OPEN
            if reconstruction_gate == 1.0 and gate_target == 1.0
            else ReconstructionGateState.RAMPING
        )
        if gate_state is not expected_state:
            raise ValueError("Gate state does not match its value and target.")
        snapshot_active_value = state["snapshot_active"]
        if not isinstance(snapshot_active_value, bool):
            raise ValueError("Snapshot activity must be a boolean.")
        snapshot_active = snapshot_active_value
        snapshot_version_value = state["snapshot_estimator_version"]
        snapshot_estimator_version = (
            None
            if snapshot_version_value is None
            else int(snapshot_version_value)
        )
        if snapshot_estimator_version is not None and snapshot_estimator_version < 0:
            raise ValueError("Snapshot estimator version must be non-negative.")
        if snapshot_active:
            if gate_target != 1.0:
                raise ValueError("An active snapshot requires a one-way open gate target.")
            if snapshot_estimator_version != values["control_estimator_version"]:
                raise ValueError(
                    "Snapshot and control-estimator versions must match."
                )
            if values["snapshot_replay_resets"] > 1:
                raise ValueError("Snapshot replay may reset at most once.")
        elif (
            snapshot_estimator_version is not None
            or gate_target != 0.0
            or values["snapshot_replay_resets"] != 0
            or values["snapshot_gate_updates"] != 0
        ):
            raise ValueError("Inactive snapshots cannot retain snapshot state.")

        self.sac_updates_since_estimator = values["sac_updates_since_estimator"]
        self.estimator_controller.estimator_updates = values["estimator_updates"]
        self.estimator_controller.estimator_attempts = values["estimator_attempts"]
        self.estimator_controller.estimator_triggers = values["estimator_triggers"]
        self.estimator_controller.gate_state = gate_state
        self.estimator_controller._reconstruction_gate = reconstruction_gate
        self.estimator_controller._gate_target = gate_target
        self.estimator_controller.gate_quality_ema = gate_quality_ema
        self.estimator_controller.gate_base_velocity_rmse_ema = (
            gate_base_velocity_rmse_ema
        )
        self.estimator_controller.gate_contact_bce_ema = gate_contact_bce_ema
        self.estimator_controller.gate_quality_passes = values["gate_quality_passes"]
        self.estimator_controller.gate_quality_failures = values[
            "gate_quality_failures"
        ]
        self.estimator_controller.gate_validation_checks = values[
            "gate_validation_checks"
        ]
        self.estimator_controller.snapshot_active = snapshot_active
        self.estimator_controller.snapshot_estimator_version = (
            snapshot_estimator_version
        )
        self.estimator_controller.snapshot_replay_resets = values[
            "snapshot_replay_resets"
        ]
        self.estimator_controller.snapshot_gate_updates = values[
            "snapshot_gate_updates"
        ]
        self.estimator_controller._gate_hard_sync_pending = False
        self.estimator_controller._snapshot_activation_pending = False
        self.estimator_controller.ema_estimator.restart(
            version=values["control_estimator_version"]
        )
        self.estimator_controller.clear_transient_state()
        self.last_estimator_updates = ()
        self.last_snapshot_replay_reset = False

    def _run_estimator_trigger(
        self,
        *,
        generator: torch.Generator | None,
    ) -> tuple[list[EstimatorUpdateResult], bool]:
        if (
            self.estimator_controller.snapshot_active
            and self.v2_cfg.freeze_online_estimator_after_snapshot
        ):
            self.estimator_controller.advance_frozen_snapshot()
            return [], False

        updates: list[EstimatorUpdateResult] = []
        minimum_insertion_id = self._minimum_sequence_insertion_id()
        for _ in range(self.v2_cfg.estimator_updates_per_trigger):
            with self.profiler.measure("estimator_sequence_sample_transfer"):
                sequence = self.replay.sample_sequences(
                    self.sequence_cfg.batch_size,
                    self.sequence_cfg.burn_in_length,
                    self.sequence_cfg.learning_length,
                    require_episode_start=self.sequence_cfg.require_episode_start,
                    episode_start_fraction=self.sequence_cfg.episode_start_fraction,
                    minimum_insertion_id=minimum_insertion_id,
                    device=self.learner_device,
                    generator=generator,
                )
                if self.sequence_augmentation is not None:
                    sequence = self.sequence_augmentation(sequence)
            with self.profiler.measure("estimator_update"):
                updates.append(self.estimator_controller.update_sequence(sequence))
        if self.estimator_controller.gate_validation_due:
            with self.profiler.measure("estimator_gate_validation"):
                validation_sequence = self.replay.sample_sequences(
                    self.sequence_cfg.batch_size,
                    self.sequence_cfg.burn_in_length,
                    self.sequence_cfg.learning_length,
                    require_episode_start=self.sequence_cfg.require_episode_start,
                    episode_start_fraction=self.sequence_cfg.episode_start_fraction,
                    minimum_insertion_id=minimum_insertion_id,
                    device=self.learner_device,
                    generator=generator,
                )
                self.estimator_controller.validate_reconstruction_gate(
                    validation_sequence
                )
        with self.profiler.measure("ema_sync_runtime_rebuild"):
            self.estimator_controller.synchronize_control_estimator()
        snapshot_activated = self.estimator_controller.consume_snapshot_activation()
        replay_reset = snapshot_activated and self.v2_cfg.reset_replay_on_snapshot
        if replay_reset:
            self.replay.clear()
            self.estimator_controller.record_snapshot_replay_reset()
        return updates, replay_reset

    def _minimum_sequence_insertion_id(self) -> int | None:
        horizon = self.sequence_cfg.recent_transition_horizon
        if horizon is None:
            return None
        return max(0, self.replay.total_inserted - horizon)
