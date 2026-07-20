"""FastWMR world-state estimator inference and supervised updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from ..buffers import EstimatorRolloutBatch, EstimatorRolloutCache, SequenceReplayBatch
from ..config import (
    DEFAULT_ESTIMATOR_LOSS_CFG,
    DEFAULT_INTERFACE_CFG,
    EstimatorLossCfg,
    FastWMRInterfaceCfg,
    TargetKind,
)
from ..networks import DecoderOutput, HistoryEncoder, WorldStateDecoder
from ..utils.temporal_state import RecurrentState


ObservationTransform = Callable[[torch.Tensor], torch.Tensor]


class RecurrentSequenceEstimator(Protocol):
    """Minimal interface needed for replay-time recurrent re-inference."""

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentState: ...

    def forward_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]: ...


@dataclass(frozen=True)
class EstimatorPrediction:
    """Latent sequence, structured decoder output, and final recurrent state."""

    encoded_history: torch.Tensor
    decoded_state: DecoderOutput
    final_state: RecurrentState

    def __post_init__(self) -> None:
        if self.encoded_history.shape[:-1] != self.decoded_state.continuous.shape[:-1]:
            raise ValueError("Encoded history and decoded state must share leading dimensions.")

    @property
    def reconstruction(self) -> torch.Tensor:
        return self.decoded_state.reconstruction


class WorldStateEstimator(nn.Module):
    """Compose the recurrent history encoder and multi-head decoder."""

    def __init__(self, encoder: HistoryEncoder, decoder: WorldStateDecoder) -> None:
        super().__init__()
        if encoder.output_dim != decoder.input_dim:
            raise ValueError(
                "Encoder output and decoder input dimensions must match, got "
                f"{encoder.output_dim} and {decoder.input_dim}."
            )
        self.encoder = encoder
        self.decoder = decoder

    @property
    def observation_dim(self) -> int:
        return self.encoder.observation_dim

    @property
    def reconstruction_dim(self) -> int:
        return self.decoder.output_dim

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RecurrentState:
        return self.encoder.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        return self.forward_sequence(observations, state)

    def predict_rollout(
        self,
        observation: torch.Tensor,
        state: RecurrentState,
    ) -> EstimatorPrediction:
        encoded_history, final_state = self.encoder.forward_rollout(observation, state)
        return EstimatorPrediction(
            encoded_history=encoded_history,
            decoded_state=self.decoder(encoded_history),
            final_state=final_state,
        )

    def forward_rollout(
        self,
        observation: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        prediction = self.predict_rollout(observation, state)
        return prediction.reconstruction, prediction.final_state

    def predict_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> EstimatorPrediction:
        encoded_history, final_state = self.encoder.forward_sequence(observations, state)
        return EstimatorPrediction(
            encoded_history=encoded_history,
            decoded_state=self.decoder(encoded_history),
            final_state=final_state,
        )

    def forward_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        prediction = self.predict_sequence(observations, state)
        return prediction.reconstruction, prediction.final_state


@dataclass(frozen=True)
class BurnInUnrollOutput:
    """Learning-window reconstructions and recurrent context diagnostics."""

    reconstructions: torch.Tensor
    learning_initial_state: RecurrentState
    final_state: RecurrentState
    context_is_exact: torch.Tensor


def burn_in_and_unroll(
    estimator: RecurrentSequenceEstimator,
    sequence: SequenceReplayBatch,
) -> BurnInUnrollOutput:
    """Rebuild context with no-grad burn-in, then unroll ``L + 1`` with grad."""

    observations = sequence.observations
    state = estimator.initial_state(
        sequence.batch_size,
        device=observations.device,
        dtype=observations.dtype,
    )
    if sequence.burn_in_length > 0:
        with torch.no_grad():
            _, state = estimator.forward_sequence(sequence.burn_in_observations, state)
    learning_initial_state = state.detach()
    reconstructions, final_state = estimator.forward_sequence(
        sequence.learning_observations,
        learning_initial_state,
    )

    expected_leading_shape = sequence.learning_observations.shape[:-1]
    if reconstructions.ndim < 3 or reconstructions.shape[:-1] != expected_leading_shape:
        raise ValueError(
            "Estimator reconstructions must preserve batch/time dimensions, got "
            f"{tuple(reconstructions.shape)} for observations "
            f"{tuple(sequence.learning_observations.shape)}."
        )
    return BurnInUnrollOutput(
        reconstructions=reconstructions,
        learning_initial_state=learning_initial_state,
        final_state=final_state,
        context_is_exact=sequence.context_is_exact,
    )


@dataclass(frozen=True)
class EstimatorLossOutput:
    """Differentiable reconstruction-loss components."""

    total_loss: torch.Tensor
    continuous_mse: torch.Tensor
    discrete_bce: torch.Tensor
    latent_l1: torch.Tensor
    field_losses: Mapping[str, torch.Tensor]


def compute_estimator_loss(
    prediction: EstimatorPrediction,
    privileged_targets: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    cfg: EstimatorLossCfg = DEFAULT_ESTIMATOR_LOSS_CFG,
) -> EstimatorLossOutput:
    """Compute WMR's weighted MSE, BCE-with-logits, and latent L1 loss."""

    if not isinstance(privileged_targets, torch.Tensor):
        raise TypeError("privileged_targets must be a torch.Tensor.")
    if not privileged_targets.dtype.is_floating_point:
        raise TypeError("privileged_targets must have a floating dtype.")
    expected_shape = (*prediction.encoded_history.shape[:-1], interface.reconstruction_target_dim)
    if privileged_targets.shape != expected_shape:
        raise ValueError(
            f"privileged_targets must have shape {expected_shape}, "
            f"got {tuple(privileged_targets.shape)}."
        )
    if not torch.isfinite(privileged_targets).all():
        raise ValueError("privileged_targets must be finite.")
    if prediction.decoded_state.continuous.shape[-1] != interface.continuous_target_dim:
        raise ValueError("Continuous decoder width does not match the reconstruction contract.")
    if prediction.decoded_state.discrete_logits.shape[-1] != interface.discrete_target_dim:
        raise ValueError("Discrete decoder width does not match the reconstruction contract.")

    continuous_targets = _select_target_kind(privileged_targets, interface, TargetKind.CONTINUOUS)
    discrete_targets = _select_target_kind(privileged_targets, interface, TargetKind.DISCRETE)
    if torch.any((discrete_targets < 0.0) | (discrete_targets > 1.0)):
        raise ValueError("Discrete reconstruction targets must lie in [0, 1].")

    continuous_mse = F.mse_loss(prediction.decoded_state.continuous, continuous_targets)
    discrete_bce = F.binary_cross_entropy_with_logits(
        prediction.decoded_state.discrete_logits,
        discrete_targets,
    )
    latent_l1 = prediction.encoded_history.abs().mean()
    total_loss = (
        cfg.continuous_weight * continuous_mse
        + cfg.discrete_weight * discrete_bce
        + cfg.latent_l1_weight * latent_l1
    )
    field_losses = _compute_field_losses(
        prediction.decoded_state,
        privileged_targets,
        interface,
    )
    return EstimatorLossOutput(
        total_loss=total_loss,
        continuous_mse=continuous_mse,
        discrete_bce=discrete_bce,
        latent_l1=latent_l1,
        field_losses=field_losses,
    )


@dataclass(frozen=True)
class EstimatorUpdateMetrics:
    """Detached scalar diagnostics from one estimator optimizer step."""

    total_loss: float
    continuous_mse: float
    discrete_bce: float
    latent_l1: float
    gradient_norm: float
    context_exact_fraction: float
    field_losses: Mapping[str, float]
    estimator_version: int


@dataclass(frozen=True)
class EstimatorUpdateResult:
    """Detached reconstruction sequence and metrics produced by one update."""

    reconstructions: torch.Tensor
    final_state: RecurrentState
    metrics: EstimatorUpdateMetrics


class EstimatorUpdater:
    """Train a world-state estimator from rollout chunks or replay sequences."""

    def __init__(
        self,
        estimator: WorldStateEstimator,
        optimizer: torch.optim.Optimizer,
        *,
        interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
        loss_cfg: EstimatorLossCfg = DEFAULT_ESTIMATOR_LOSS_CFG,
        observation_transform: ObservationTransform | None = None,
    ) -> None:
        if estimator.observation_dim != interface.policy_observation_dim:
            raise ValueError("Estimator observation width does not match the policy contract.")
        if estimator.decoder.continuous_dim != interface.continuous_target_dim:
            raise ValueError("Estimator continuous output width does not match the target contract.")
        if estimator.decoder.discrete_dim != interface.discrete_target_dim:
            raise ValueError("Estimator discrete output width does not match the target contract.")
        self.estimator = estimator
        self.optimizer = optimizer
        self.interface = interface
        self.loss_cfg = loss_cfg
        self.observation_transform = observation_transform
        self.version = 0

    def update_sequence(self, sequence: SequenceReplayBatch) -> EstimatorUpdateResult:
        """Run no-grad burn-in and update on the complete ``L + 1`` window."""

        observations = sequence.observations
        state = self.estimator.initial_state(
            sequence.batch_size,
            device=observations.device,
            dtype=observations.dtype,
        )
        if sequence.burn_in_length > 0:
            burn_in_observations = self._transform(sequence.burn_in_observations)
            with torch.no_grad():
                _, state = self.estimator.encoder.forward_sequence(burn_in_observations, state)
        state = state.detach()
        learning_observations = self._transform(sequence.learning_observations)
        prediction = self.estimator.predict_sequence(learning_observations, state)
        return self._apply_update(
            prediction,
            sequence.learning_privileged_states,
            sequence.context_is_exact,
        )

    def update_rollout(self, batch: EstimatorRolloutBatch) -> EstimatorUpdateResult:
        """Update from an ordered rollout while resetting asynchronous episodes."""

        observations = self._transform(batch.observations)
        state = self.estimator.initial_state(
            batch.num_envs,
            device=observations.device,
            dtype=observations.dtype,
        )
        encoded_steps: list[torch.Tensor] = []
        for timestep in range(batch.sequence_length):
            state = state.reset(batch.reset_boundaries[:, timestep])
            encoded, state = self.estimator.encoder.forward_rollout(
                observations[:, timestep],
                state,
            )
            encoded_steps.append(encoded)
        encoded_history = torch.stack(encoded_steps, dim=1)
        prediction = EstimatorPrediction(
            encoded_history=encoded_history,
            decoded_state=self.estimator.decoder(encoded_history),
            final_state=state,
        )
        return self._apply_update(
            prediction,
            batch.privileged_states,
            batch.context_is_exact,
        )

    def update_cache(
        self,
        cache: EstimatorRolloutCache,
        *,
        drain: bool = True,
    ) -> EstimatorUpdateResult:
        """Move a cached chunk to the estimator device and update it once."""

        batch = cache.chronological()
        parameter = next(self.estimator.parameters())
        batch = batch.to(parameter.device)
        if batch.observations.dtype != parameter.dtype:
            batch = EstimatorRolloutBatch(
                observations=batch.observations.to(dtype=parameter.dtype),
                privileged_states=batch.privileged_states.to(dtype=parameter.dtype),
                reset_boundaries=batch.reset_boundaries,
            )
        result = self.update_rollout(batch)
        if drain:
            cache.clear()
        return result

    def _transform(self, observations: torch.Tensor) -> torch.Tensor:
        transformed = (
            self.observation_transform(observations)
            if self.observation_transform is not None
            else observations
        )
        if transformed.shape != observations.shape:
            raise ValueError("The estimator observation transform must preserve tensor shape.")
        if not transformed.dtype.is_floating_point or not torch.isfinite(transformed).all():
            raise ValueError("Transformed estimator observations must be finite and floating point.")
        return transformed

    def _apply_update(
        self,
        prediction: EstimatorPrediction,
        privileged_targets: torch.Tensor,
        context_is_exact: torch.Tensor,
    ) -> EstimatorUpdateResult:
        losses = compute_estimator_loss(
            prediction,
            privileged_targets,
            interface=self.interface,
            cfg=self.loss_cfg,
        )
        if not torch.isfinite(losses.total_loss):
            raise FloatingPointError("Estimator loss is not finite.")

        reconstructions = prediction.reconstruction.detach()
        final_state = prediction.final_state.detach()
        self.optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        gradient_norm = self._gradient_norm()
        self.optimizer.step()
        self.version += 1

        metrics = EstimatorUpdateMetrics(
            total_loss=float(losses.total_loss.detach()),
            continuous_mse=float(losses.continuous_mse.detach()),
            discrete_bce=float(losses.discrete_bce.detach()),
            latent_l1=float(losses.latent_l1.detach()),
            gradient_norm=gradient_norm,
            context_exact_fraction=float(context_is_exact.float().mean()),
            field_losses={
                name: float(value.detach()) for name, value in losses.field_losses.items()
            },
            estimator_version=self.version,
        )
        return EstimatorUpdateResult(
            reconstructions=reconstructions,
            final_state=final_state,
            metrics=metrics,
        )

    def _gradient_norm(self) -> float:
        squared_norm = torch.zeros((), device=next(self.estimator.parameters()).device)
        for parameter in self.estimator.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Estimator gradient is not finite.")
            squared_norm = squared_norm + parameter.grad.detach().square().sum()
        return float(torch.sqrt(squared_norm))


def _select_target_kind(
    targets: torch.Tensor,
    interface: FastWMRInterfaceCfg,
    kind: TargetKind,
) -> torch.Tensor:
    fields = [
        targets[..., interface.reconstruction_layout.field_slice(field.name)]
        for field in interface.reconstruction_layout.fields
        if field.kind is kind
    ]
    if not fields:
        raise ValueError(f"Reconstruction layout has no {kind.value} targets.")
    return torch.cat(fields, dim=-1)


def _compute_field_losses(
    decoded_state: DecoderOutput,
    targets: torch.Tensor,
    interface: FastWMRInterfaceCfg,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    continuous_offset = 0
    discrete_offset = 0
    for field in interface.reconstruction_layout.fields:
        target = targets[..., interface.reconstruction_layout.field_slice(field.name)]
        if field.kind is TargetKind.CONTINUOUS:
            prediction = decoded_state.continuous[
                ..., continuous_offset : continuous_offset + field.width
            ]
            losses[f"{field.name}_mse"] = F.mse_loss(prediction, target)
            continuous_offset += field.width
        else:
            logits = decoded_state.discrete_logits[
                ..., discrete_offset : discrete_offset + field.width
            ]
            losses[f"{field.name}_bce"] = F.binary_cross_entropy_with_logits(logits, target)
            discrete_offset += field.width
    return losses
