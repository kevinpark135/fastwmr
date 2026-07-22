"""FastSAC actor, critic, temperature, and target-network updates.

The learner sees only a prepared SAC feature tensor. In baseline mode this is
the policy observation; in FastWMR mode it is the stored detached control
feature ``x_t``. Privileged reconstruction targets and recurrent hidden state
are intentionally absent from :class:`SACTransitionBatch`.

The categorical target construction follows Holosoma's FastSAC C51 update:
https://github.com/amazon-far/holosoma
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from ..buffers import StoredControlReplayBatch, TransitionReplayBatch
from ..networks import (
    TargetTwinC51Critic,
    TargetTwinScalarCritic,
    TanhGaussianActor,
    TwinC51Critic,
    TwinScalarCritic,
)


class SACFeatureSource(str, Enum):
    """Replay field exposed as the state input to actor and critics."""

    POLICY_OBSERVATION = "policy_observation"
    CONTROL_FEATURE = "control_feature"


@dataclass(frozen=True)
class SACTransitionBatch:
    """Minimal detached transition contract consumed by SAC updates."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    allow_state_gradients: bool = False

    def __post_init__(self) -> None:
        tensors = (self.states, self.actions, self.rewards, self.next_states, self.terminated, self.truncated)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("Every SAC transition field must be a torch.Tensor.")
        if self.states.ndim < 2 or self.next_states.shape != self.states.shape:
            raise ValueError("states and next_states must have the same shape (..., state_dim).")
        if self.actions.ndim != self.states.ndim or self.actions.shape[:-1] != self.states.shape[:-1]:
            raise ValueError("actions must share the states leading dimensions.")
        if self.states.shape[-1] <= 0 or self.actions.shape[-1] <= 0:
            raise ValueError("SAC state and action dimensions must be positive.")

        leading_shape = self.states.shape[:-1]
        if self.rewards.shape != leading_shape:
            raise ValueError(f"rewards must have shape {leading_shape}, got {tuple(self.rewards.shape)}.")
        if self.terminated.shape != leading_shape or self.truncated.shape != leading_shape:
            raise ValueError("terminated and truncated must share the states leading dimensions.")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have dtype torch.bool.")
        if not self.states.dtype.is_floating_point or not self.actions.dtype.is_floating_point:
            raise TypeError("SAC states and actions must have floating-point dtypes.")
        if not self.rewards.dtype.is_floating_point:
            raise TypeError("SAC rewards must have a floating-point dtype.")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("Every SAC transition field must be on the same device.")
        if not self.allow_state_gradients and (
            self.states.requires_grad or self.next_states.requires_grad
        ):
            raise ValueError("SAC replay features must be detached before constructing a batch.")
        finite_tensors = (self.states, self.actions, self.rewards, self.next_states)
        if not all(torch.isfinite(tensor).all() for tensor in finite_tensors):
            raise ValueError("SAC transition floating-point fields must be finite.")

    @classmethod
    def from_replay(
        cls,
        replay: TransitionReplayBatch,
        *,
        feature_source: SACFeatureSource,
    ) -> "SACTransitionBatch":
        """Remove non-SAC fields and select the correct auto-reset successor."""

        if feature_source is SACFeatureSource.POLICY_OBSERVATION:
            states = replay.observations
            next_states = replay.bootstrap_observations
        elif feature_source is SACFeatureSource.CONTROL_FEATURE:
            if replay.control_features.shape[-1] == 0:
                raise ValueError("Replay batch does not contain FastWMR control features.")
            states = replay.control_features
            next_states = replay.bootstrap_control_features
        else:
            raise ValueError(f"Unsupported SAC feature source {feature_source!r}.")

        return cls(
            states=states.detach(),
            actions=replay.actions.detach(),
            rewards=replay.rewards.detach(),
            next_states=next_states.detach(),
            terminated=replay.terminated.detach(),
            truncated=replay.truncated.detach(),
        )

    @classmethod
    def from_stored_control_replay(
        cls,
        replay: StoredControlReplayBatch,
    ) -> "SACTransitionBatch":
        """Build the v2 SAC batch without transferring estimator-only fields."""

        return cls(
            states=replay.control_features.detach(),
            actions=replay.actions.detach(),
            rewards=replay.rewards.detach(),
            next_states=replay.bootstrap_control_features.detach(),
            terminated=replay.terminated.detach(),
            truncated=replay.truncated.detach(),
        )

    @property
    def bootstrap_mask(self) -> torch.Tensor:
        """Time-limit truncations bootstrap; true terminations do not."""

        return (~self.terminated).to(dtype=self.rewards.dtype)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "SACTransitionBatch":
        return SACTransitionBatch(
            states=self.states.to(device, non_blocking=non_blocking),
            actions=self.actions.to(device, non_blocking=non_blocking),
            rewards=self.rewards.to(device, non_blocking=non_blocking),
            next_states=self.next_states.to(device, non_blocking=non_blocking),
            terminated=self.terminated.to(device, non_blocking=non_blocking),
            truncated=self.truncated.to(device, non_blocking=non_blocking),
        )


class EntropyTemperature(nn.Module):
    """Positive SAC entropy temperature represented by a trainable log alpha."""

    def __init__(self, initial_temperature: float = 0.001) -> None:
        super().__init__()
        if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be finite and positive.")
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_temperature), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.log_alpha.exp()


@dataclass(frozen=True)
class CriticLossOutput:
    loss: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class C51CriticLossOutput:
    """Categorical critic loss plus scalar expectations used for metrics."""

    loss: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    target: torch.Tensor
    target_distributions: torch.Tensor


@dataclass(frozen=True)
class ActorLossOutput:
    loss: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    average_q: torch.Tensor


@dataclass(frozen=True)
class SACUpdateMetrics:
    critic_loss: torch.Tensor
    actor_loss: torch.Tensor
    temperature_loss: torch.Tensor
    temperature: torch.Tensor
    target_q_mean: torch.Tensor
    target_q_std: torch.Tensor
    q1_mean: torch.Tensor
    q1_std: torch.Tensor
    q2_mean: torch.Tensor
    q2_std: torch.Tensor
    policy_entropy: torch.Tensor


@torch.no_grad()
def compute_critic_target(
    batch: SACTransitionBatch,
    actor: TanhGaussianActor,
    target_critic: TargetTwinScalarCritic,
    temperature: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Compute one-step FastSAC targets using average target Q1/Q2."""

    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be in [0, 1], got {discount}.")
    _validate_temperature(temperature)
    next_actions, next_log_probabilities = actor.sample(batch.next_states)
    next_q = target_critic.average(batch.next_states, next_actions)
    soft_next_value = next_q - temperature * next_log_probabilities
    target = batch.rewards + discount * batch.bootstrap_mask * soft_next_value
    if target.shape != batch.rewards.shape or not torch.isfinite(target).all():
        raise RuntimeError("Critic target has an invalid shape or non-finite value.")
    return target


def compute_critic_loss(
    batch: SACTransitionBatch,
    critic: TwinScalarCritic,
    target: torch.Tensor,
) -> CriticLossOutput:
    """Return the sum of mean Q1 and Q2 squared Bellman errors."""

    if target.shape != batch.rewards.shape:
        raise ValueError(f"target must have shape {tuple(batch.rewards.shape)}, got {tuple(target.shape)}.")
    target = target.detach()
    q1, q2 = critic(batch.states, batch.actions)
    loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    _require_finite_scalar(loss, "critic loss")
    return CriticLossOutput(loss=loss, q1=q1, q2=q2, target=target)


def project_categorical_distribution(
    probabilities: torch.Tensor,
    target_atoms: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project shifted categorical mass back onto an evenly spaced support."""

    if probabilities.shape != target_atoms.shape:
        raise ValueError("probabilities and target_atoms must have the same shape.")
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("support must be one-dimensional with at least two atoms.")
    if probabilities.shape[-1] != support.numel():
        raise ValueError("The distribution atom dimension must match support.")
    if probabilities.device != target_atoms.device or probabilities.device != support.device:
        raise ValueError("probabilities, target_atoms, and support must share a device.")
    if not all(tensor.dtype.is_floating_point for tensor in (probabilities, target_atoms, support)):
        raise TypeError("C51 projection inputs must have floating-point dtypes.")
    if not all(torch.isfinite(tensor).all() for tensor in (probabilities, target_atoms, support)):
        raise ValueError("C51 projection inputs must be finite.")

    spacing = support[1] - support[0]
    if spacing <= 0.0 or not torch.allclose(
        support[1:] - support[:-1],
        spacing.expand_as(support[1:]),
    ):
        raise ValueError("support must be strictly increasing and evenly spaced.")

    positions = (target_atoms.clamp(support[0], support[-1]) - support[0]) / spacing
    lower = positions.floor().to(torch.long)
    upper = positions.ceil().to(torch.long)
    lower_weight = upper.to(positions.dtype) - positions
    upper_weight = positions - lower.to(positions.dtype)
    exact_atom = lower == upper
    lower_weight = torch.where(exact_atom, torch.ones_like(lower_weight), lower_weight)
    upper_weight = torch.where(exact_atom, torch.zeros_like(upper_weight), upper_weight)

    projected = torch.zeros_like(probabilities)
    projected.scatter_add_(-1, lower, probabilities * lower_weight)
    projected.scatter_add_(-1, upper, probabilities * upper_weight)
    return projected


@torch.no_grad()
def compute_c51_critic_target(
    batch: SACTransitionBatch,
    actor: TanhGaussianActor,
    target_critic: TargetTwinC51Critic,
    temperature: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Build one projected target distribution for each target Q head."""

    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be in [0, 1], got {discount}.")
    _validate_temperature(temperature)
    next_actions, next_log_probabilities = actor.sample(batch.next_states)
    next_probabilities = target_critic.stacked_probabilities(batch.next_states, next_actions)
    continuation = discount * batch.bootstrap_mask
    shifted_atoms = batch.rewards.unsqueeze(-1) + continuation.unsqueeze(-1) * (
        target_critic.support - temperature * next_log_probabilities.unsqueeze(-1)
    )
    shifted_atoms = shifted_atoms.unsqueeze(0).expand_as(next_probabilities)
    projected = project_categorical_distribution(next_probabilities, shifted_atoms, target_critic.support)
    if not torch.allclose(
        projected.sum(dim=-1),
        torch.ones_like(projected[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Projected C51 target distributions do not sum to one.")
    return projected


def compute_c51_critic_loss(
    batch: SACTransitionBatch,
    critic: TwinC51Critic,
    target_distributions: torch.Tensor,
) -> C51CriticLossOutput:
    """Return the summed twin cross-entropy against projected C51 targets."""

    expected_shape = (2, *batch.rewards.shape, critic.cfg.num_atoms)
    if target_distributions.shape != expected_shape:
        raise ValueError(
            f"target_distributions must have shape {expected_shape}, got {tuple(target_distributions.shape)}."
        )
    target_distributions = target_distributions.detach()
    logits = critic.stacked_logits(batch.states, batch.actions)
    cross_entropy = -(target_distributions * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loss = cross_entropy.reshape(2, -1).mean(dim=-1).sum()
    _require_finite_scalar(loss, "C51 critic loss")

    values = critic.values_from_probabilities(logits.softmax(dim=-1))
    target_values = critic.values_from_probabilities(target_distributions)
    return C51CriticLossOutput(
        loss=loss,
        q1=values[0],
        q2=values[1],
        target=target_values,
        target_distributions=target_distributions,
    )


def compute_actor_loss(
    states: torch.Tensor,
    actor: TanhGaussianActor,
    critic: TwinScalarCritic | TwinC51Critic,
    temperature: torch.Tensor,
    *,
    allow_state_gradients: bool = False,
) -> ActorLossOutput:
    """Compute ``mean(alpha * log_pi - average(Q1, Q2))``."""

    if states.requires_grad and not allow_state_gradients:
        raise ValueError("Actor-update states must be detached from the estimator.")
    _validate_temperature(temperature)
    actions, log_probabilities = actor.sample(states)
    average_q = critic.average(states, actions)
    loss = (temperature.detach() * log_probabilities - average_q).mean()
    _require_finite_scalar(loss, "actor loss")
    return ActorLossOutput(
        loss=loss,
        actions=actions,
        log_probabilities=log_probabilities,
        average_q=average_q,
    )


def compute_temperature_loss(
    log_alpha: torch.Tensor,
    log_probabilities: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """FastSAC alpha objective with policy log probabilities detached."""

    if log_alpha.numel() != 1 or not log_alpha.dtype.is_floating_point:
        raise ValueError("log_alpha must be one floating-point scalar.")
    if not math.isfinite(target_entropy):
        raise ValueError("target_entropy must be finite.")
    if not log_probabilities.dtype.is_floating_point or not torch.isfinite(log_probabilities).all():
        raise ValueError("log_probabilities must be finite floating-point values.")
    alpha = log_alpha.exp()
    loss = (-alpha * (log_probabilities.detach() + target_entropy)).mean()
    _require_finite_scalar(loss, "temperature loss")
    return loss


class SACUpdater:
    """Optimizer coordinator for one scalar-critic FastSAC gradient update."""

    def __init__(
        self,
        *,
        actor: TanhGaussianActor,
        critic: TwinScalarCritic,
        target_critic: TargetTwinScalarCritic,
        temperature: EntropyTemperature,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temperature_optimizer: torch.optim.Optimizer,
        discount: float = 0.97,
        target_update_rate: float = 0.005,
        target_entropy: float = 0.0,
    ) -> None:
        if actor.input_dim != critic.state_dim or critic.state_dim != target_critic.state_dim:
            raise ValueError("Actor, online critic, and target critic state dimensions must match.")
        if actor.action_dim != critic.action_dim or critic.action_dim != target_critic.action_dim:
            raise ValueError("Actor, online critic, and target critic action dimensions must match.")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if not 0.0 < target_update_rate <= 1.0:
            raise ValueError("target_update_rate must be in (0, 1].")
        if not math.isfinite(target_entropy):
            raise ValueError("target_entropy must be finite.")

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temperature = temperature
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.temperature_optimizer = temperature_optimizer
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.target_entropy = target_entropy

    def update(self, batch: SACTransitionBatch) -> SACUpdateMetrics:
        """Perform critic, actor, alpha, and target updates in FastSAC order."""

        if batch.states.shape[-1] != self.actor.input_dim or batch.actions.shape[-1] != self.actor.action_dim:
            raise ValueError("SAC batch dimensions do not match the configured actor and critics.")

        critic_output = self.update_critic(batch)
        actor_output = self.update_actor(batch.states)
        alpha_loss = self.update_temperature(actor_output.log_probabilities)
        self.update_target()
        return SACUpdateMetrics(
            critic_loss=critic_output.loss.detach(),
            actor_loss=actor_output.loss.detach(),
            temperature_loss=alpha_loss.detach(),
            temperature=self.temperature().detach(),
            target_q_mean=critic_output.target.mean().detach(),
            target_q_std=critic_output.target.std(unbiased=False).detach(),
            q1_mean=critic_output.q1.mean().detach(),
            q1_std=critic_output.q1.std(unbiased=False).detach(),
            q2_mean=critic_output.q2.mean().detach(),
            q2_std=critic_output.q2.std(unbiased=False).detach(),
            policy_entropy=(-actor_output.log_probabilities.mean()).detach(),
        )

    def update_critic(
        self,
        batch: SACTransitionBatch,
        *,
        retain_graph: bool = False,
    ) -> CriticLossOutput:
        """Step Q1/Q2 while keeping the actor and target critic untouched."""

        target = compute_critic_target(
            batch,
            self.actor,
            self.target_critic,
            self.temperature().detach(),
            self.discount,
        )
        critic_output = compute_critic_loss(batch, self.critic, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_output.loss.backward(retain_graph=retain_graph)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        return critic_output

    def update_actor(
        self,
        states: torch.Tensor,
        *,
        allow_state_gradients: bool = False,
    ) -> ActorLossOutput:
        """Step the actor through action gradients while freezing Q parameters."""

        self.actor_optimizer.zero_grad(set_to_none=True)
        with _freeze_parameters(self.critic):
            actor_output = compute_actor_loss(
                states,
                self.actor,
                self.critic,
                self.temperature(),
                allow_state_gradients=allow_state_gradients,
            )
            actor_output.loss.backward()
        self.actor_optimizer.step()
        return actor_output

    def update_temperature(self, log_probabilities: torch.Tensor) -> torch.Tensor:
        """Step log alpha without sending gradients back into the policy."""

        self.temperature_optimizer.zero_grad(set_to_none=True)
        alpha_loss = compute_temperature_loss(
            self.temperature.log_alpha,
            log_probabilities,
            self.target_entropy,
        )
        alpha_loss.backward()
        self.temperature_optimizer.step()
        return alpha_loss

    def update_target(self) -> None:
        """Apply one Polyak update to both frozen target Q-networks."""

        self.target_critic.soft_update_from(self.critic, self.target_update_rate)


class C51SACUpdater(SACUpdater):
    """FastSAC optimizer coordinator using twin categorical critics."""

    critic: TwinC51Critic
    target_critic: TargetTwinC51Critic

    def __init__(
        self,
        *,
        actor: TanhGaussianActor,
        critic: TwinC51Critic,
        target_critic: TargetTwinC51Critic,
        temperature: EntropyTemperature,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temperature_optimizer: torch.optim.Optimizer,
        discount: float = 0.97,
        target_update_rate: float = 0.005,
        target_entropy: float = 0.0,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,  # type: ignore[arg-type]
            target_critic=target_critic,  # type: ignore[arg-type]
            temperature=temperature,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
            discount=discount,
            target_update_rate=target_update_rate,
            target_entropy=target_entropy,
        )

    def update_critic(
        self,
        batch: SACTransitionBatch,
        *,
        retain_graph: bool = False,
    ) -> C51CriticLossOutput:
        target_distributions = compute_c51_critic_target(
            batch,
            self.actor,
            self.target_critic,
            self.temperature().detach(),
            self.discount,
        )
        critic_output = compute_c51_critic_loss(batch, self.critic, target_distributions)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_output.loss.backward(retain_graph=retain_graph)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        return critic_output


def _validate_temperature(temperature: torch.Tensor) -> None:
    if not isinstance(temperature, torch.Tensor) or temperature.numel() != 1:
        raise ValueError("temperature must be a scalar tensor.")
    if not temperature.dtype.is_floating_point or not torch.isfinite(temperature).all() or temperature.item() <= 0.0:
        raise ValueError("temperature must be finite and positive.")


def _require_finite_scalar(value: torch.Tensor, name: str) -> None:
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must be one finite scalar.")


@contextmanager
def _freeze_parameters(module: nn.Module) -> Iterator[None]:
    original_flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original_flags, strict=True):
            parameter.requires_grad_(requires_grad)
