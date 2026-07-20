"""Numerically stable running normalization for raw policy observations."""

from __future__ import annotations

import torch
from torch import nn

from ..config import DEFAULT_OBSERVATION_NORMALIZATION_CFG, ObservationNormalizationCfg


class RunningObservationNormalizer(nn.Module):
    """Track population moments with parallel Welford updates.

    Statistics use float64 even when observations and networks use float32.
    Calling :meth:`eval` freezes statistics while normalization remains active.
    The moments are registered buffers, so regular ``state_dict`` checkpointing
    preserves train/evaluation behavior.
    """

    def __init__(
        self,
        observation_dim: int,
        cfg: ObservationNormalizationCfg = DEFAULT_OBSERVATION_NORMALIZATION_CFG,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if not cfg.enabled:
            raise ValueError("Do not construct a normalizer when normalization is disabled.")
        self.observation_dim = observation_dim
        self.cfg = cfg
        self.register_buffer("mean", torch.zeros(observation_dim, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(observation_dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))

    @property
    def samples_seen(self) -> int:
        return int(self.count.item())

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        """Merge one arbitrary-leading-dimension observation batch."""

        self._validate(observations, require_finite=True)
        if not self.training:
            return
        samples = observations.detach().reshape(-1, self.observation_dim).to(
            device=self.mean.device,
            dtype=torch.float64,
        )
        if samples.shape[0] == 0:
            raise ValueError("Cannot update observation statistics from an empty batch.")
        batch_mean = samples.mean(dim=0)
        batch_variance = samples.var(dim=0, unbiased=False)
        batch_count = torch.as_tensor(samples.shape[0], device=self.mean.device, dtype=torch.float64)
        previous_count = self.count.to(dtype=torch.float64)
        total_count = previous_count + batch_count
        delta = batch_mean - self.mean
        merged_mean = self.mean + delta * (batch_count / total_count)
        previous_m2 = self.variance * previous_count
        batch_m2 = batch_variance * batch_count
        correction = delta.square() * (previous_count * batch_count / total_count)
        self.mean.copy_(merged_mean)
        self.variance.copy_(((previous_m2 + batch_m2 + correction) / total_count).clamp_min_(0.0))
        self.count.add_(samples.shape[0])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalize without changing statistics."""

        self._validate(observations)
        mean = self.mean.to(device=observations.device, dtype=observations.dtype)
        variance = self.variance.to(device=observations.device, dtype=observations.dtype).clamp_min(0.0)
        normalized = (observations - mean) * torch.rsqrt(variance + self.cfg.epsilon)
        if self.cfg.clip is not None:
            normalized = normalized.clamp(-self.cfg.clip, self.cfg.clip)
        return normalized

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Invert normalization for unclipped values and diagnostics."""

        self._validate(normalized, field_name="normalized")
        mean = self.mean.to(device=normalized.device, dtype=normalized.dtype)
        variance = self.variance.to(device=normalized.device, dtype=normalized.dtype).clamp_min(0.0)
        return normalized * torch.sqrt(variance + self.cfg.epsilon) + mean

    @torch.no_grad()
    def reset_statistics(self) -> None:
        self.mean.zero_()
        self.variance.fill_(1.0)
        self.count.zero_()

    def _validate(
        self,
        observations: torch.Tensor,
        *,
        field_name: str = "observations",
        require_finite: bool = False,
    ) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError(f"{field_name} must be a torch.Tensor.")
        if not observations.dtype.is_floating_point:
            raise TypeError(f"{field_name} must have a floating-point dtype.")
        if observations.ndim < 1 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f"{field_name} must end in dimension {self.observation_dim}.")
        if require_finite and not torch.isfinite(observations).all():
            raise ValueError(f"{field_name} must contain only finite values.")
