"""FastSAC tanh-Gaussian actor shared by the baseline and FastWMR.

The baseline passes normalized policy observations directly. FastWMR passes the
shared control feature ``x_t`` built by :mod:`algorithm.utils.feature_builder`;
that builder, rather than this network, owns the estimator gradient cutoff.
Ground-truth privileged state is therefore never part of this API.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DEFAULT_ACTOR_CFG, TanhGaussianActorCfg


class TanhGaussianActor(nn.Module):
    """Reparameterized diagonal Gaussian policy with bounded actions.

    The policy samples ``u ~ Normal(mean, std)`` and returns
    ``a = tanh(u) * action_scale + action_bias``. Sample log-probabilities
    include both the tanh and affine-scale change-of-variables corrections.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        cfg: TanhGaussianActorCfg = DEFAULT_ACTOR_CFG,
        action_low: torch.Tensor | float = -1.0,
        action_high: torch.Tensor | float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.cfg = cfg

        first_width = cfg.hidden_dim
        second_width = cfg.hidden_dim // 2
        third_width = cfg.hidden_dim // 4
        widths = (input_dim, first_width, second_width, third_width)
        layers: list[nn.Module] = []
        for source_width, target_width in zip(widths[:-1], widths[1:], strict=True):
            layers.append(nn.Linear(source_width, target_width))
            layers.append(nn.LayerNorm(target_width) if cfg.use_layer_norm else nn.Identity())
            layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(third_width, action_dim)
        self.log_std_head = nn.Linear(third_width, action_dim)

        # The reference implementation starts with a zero-mean policy and a
        # state-independent midpoint log std, while the trunk learns features.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

        low = self._action_bound(action_low, action_dim, "action_low")
        high = self._action_bound(action_high, action_dim, "action_high")
        if torch.any(low >= high):
            raise ValueError("Every action_low value must be smaller than action_high.")
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    @property
    def action_low(self) -> torch.Tensor:
        return self.action_bias - self.action_scale

    @property
    def action_high(self) -> torch.Tensor:
        return self.action_bias + self.action_scale

    def forward(self, control_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return deterministic action, pre-tanh mean, and bounded log std."""

        self._validate_input(control_feature)
        latent = self.trunk(control_feature)
        mean = self.mean_head(latent)
        raw_log_std = torch.tanh(self.log_std_head(latent))
        log_std = self.cfg.log_std_min + 0.5 * (self.cfg.log_std_max - self.cfg.log_std_min) * (
            raw_log_std + 1.0
        )
        deterministic_action = self._squash_and_scale(mean)
        return deterministic_action, mean, log_std

    def sample(self, control_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a differentiable action and return its corrected log probability.

        ``log_probability`` has shape ``control_feature.shape[:-1]`` because the
        independent action dimensions are summed. ``Normal.rsample`` preserves
        the reparameterization path needed by the SAC actor objective.
        """

        _, mean, log_std = self(control_feature)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = distribution.rsample()
        action = self._squash_and_scale(pre_tanh_action)

        # log(1 - tanh(u)^2) in a stable form. The affine term is also part of
        # the transformed density when joint-specific action bounds are used.
        log_tanh_jacobian = 2.0 * (math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action))
        log_scale_jacobian = torch.log(self.action_scale)
        log_probability = (
            distribution.log_prob(pre_tanh_action) - log_tanh_jacobian - log_scale_jacobian
        ).sum(dim=-1)
        return action, log_probability

    @torch.no_grad()
    def act(self, control_feature: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """Select rollout actions without retaining an autograd graph."""

        if deterministic:
            action, _, _ = self(control_feature)
            return action
        action, _ = self.sample(control_feature)
        return action

    def _squash_and_scale(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh_action) * self.action_scale + self.action_bias

    def _validate_input(self, control_feature: torch.Tensor) -> None:
        if not isinstance(control_feature, torch.Tensor):
            raise TypeError("control_feature must be a torch.Tensor.")
        if not control_feature.dtype.is_floating_point:
            raise TypeError(f"control_feature must have a floating dtype, got {control_feature.dtype}.")
        if control_feature.ndim < 1 or control_feature.shape[-1] != self.input_dim:
            raise ValueError(
                f"control_feature must end in dimension {self.input_dim}, got {tuple(control_feature.shape)}."
            )

    @staticmethod
    def _action_bound(value: torch.Tensor | float, action_dim: int, name: str) -> torch.Tensor:
        bound = torch.as_tensor(value, dtype=torch.float32)
        if bound.ndim == 0:
            bound = bound.expand(action_dim).clone()
        if bound.shape != (action_dim,):
            raise ValueError(f"{name} must be scalar or have shape ({action_dim},), got {tuple(bound.shape)}.")
        if not torch.isfinite(bound).all():
            raise ValueError(f"{name} must contain only finite values.")
        return bound.detach().clone()
