"""Twin scalar critics and frozen target critics for FastSAC/FastWMR.

Both online Q-functions consume ``(x_t, a_t)`` where ``x_t`` is exactly the
same control feature consumed by the actor. FastWMR ground-truth privileged
state is deliberately absent from every public method in this module.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import DEFAULT_CRITIC_CFG, ScalarCriticCfg


class ScalarQNetwork(nn.Module):
    """One scalar state-action value network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        cfg: ScalarCriticCfg = DEFAULT_CRITIC_CFG,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg

        first_width = cfg.hidden_dim
        second_width = cfg.hidden_dim // 2
        third_width = cfg.hidden_dim // 4
        widths = (state_dim + action_dim, first_width, second_width, third_width)
        layers: list[nn.Module] = []
        for source_width, target_width in zip(widths[:-1], widths[1:], strict=True):
            layers.append(nn.Linear(source_width, target_width))
            layers.append(nn.LayerNorm(target_width) if cfg.use_layer_norm else nn.Identity())
            layers.append(nn.SiLU())
        layers.append(nn.Linear(third_width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, control_feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape ``control_feature.shape[:-1]``."""

        self._validate_inputs(control_feature, action)
        critic_input = torch.cat((control_feature, action), dim=-1)
        return self.net(critic_input).squeeze(-1)

    def _validate_inputs(self, control_feature: torch.Tensor, action: torch.Tensor) -> None:
        if not isinstance(control_feature, torch.Tensor) or not isinstance(action, torch.Tensor):
            raise TypeError("control_feature and action must both be torch.Tensor instances.")
        if not control_feature.dtype.is_floating_point or not action.dtype.is_floating_point:
            raise TypeError("control_feature and action must both have floating-point dtypes.")
        if control_feature.ndim < 1 or control_feature.shape[-1] != self.state_dim:
            raise ValueError(
                f"control_feature must end in dimension {self.state_dim}, got {tuple(control_feature.shape)}."
            )
        if action.ndim < 1 or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must end in dimension {self.action_dim}, got {tuple(action.shape)}.")
        if control_feature.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "control_feature and action leading dimensions must match, got "
                f"{control_feature.shape[:-1]} and {action.shape[:-1]}."
            )


class TwinScalarCritic(nn.Module):
    """Two independently initialized scalar Q-functions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        cfg: ScalarCriticCfg = DEFAULT_CRITIC_CFG,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.q1 = ScalarQNetwork(state_dim, action_dim, cfg=cfg)
        self.q2 = ScalarQNetwork(state_dim, action_dim, cfg=cfg)

    def forward(
        self,
        control_feature: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(control_feature, action), self.q2(control_feature, action)

    def stacked(self, control_feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-values with the twin axis first: ``(2, ...)``."""

        return torch.stack(self(control_feature, action), dim=0)

    def average(self, control_feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """FastSAC aggregation, which intentionally uses mean Q instead of min Q."""

        q1, q2 = self(control_feature, action)
        return 0.5 * (q1 + q2)


class TargetTwinScalarCritic(TwinScalarCritic):
    """Non-trainable twin critic updated only from an online critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        cfg: ScalarCriticCfg = DEFAULT_CRITIC_CFG,
    ) -> None:
        super().__init__(state_dim, action_dim, cfg=cfg)
        self.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_online(cls, online: TwinScalarCritic) -> "TargetTwinScalarCritic":
        """Create an exact frozen copy on the online critic's device and dtype."""

        target = cls(online.state_dim, online.action_dim, cfg=online.cfg)
        reference_parameter = next(online.parameters())
        target.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        target.hard_update_from(online)
        return target

    def train(self, mode: bool = True) -> "TargetTwinScalarCritic":
        """Keep target LayerNorm modules in evaluation mode."""

        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self,
        control_feature: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(control_feature, action)

    @torch.no_grad()
    def hard_update_from(self, online: TwinScalarCritic) -> None:
        """Replace all target parameters and buffers with online values."""

        self._validate_source(online)
        self.load_state_dict(online.state_dict())
        self.requires_grad_(False)

    @torch.no_grad()
    def soft_update_from(self, online: TwinScalarCritic, tau: float) -> None:
        """Polyak update ``target <- (1 - tau) * target + tau * online``."""

        self._validate_source(online)
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        for target_parameter, online_parameter in zip(self.parameters(), online.parameters(), strict=True):
            target_parameter.lerp_(online_parameter, tau)

        # LayerNorm has no running statistics today, but copying buffers keeps
        # this update correct if the architecture later gains stateful buffers.
        for target_buffer, online_buffer in zip(self.buffers(), online.buffers(), strict=True):
            target_buffer.copy_(online_buffer)

    def _validate_source(self, online: TwinScalarCritic) -> None:
        if not isinstance(online, TwinScalarCritic) or isinstance(online, TargetTwinScalarCritic):
            raise TypeError("online must be a trainable TwinScalarCritic.")
        if self.state_dim != online.state_dim or self.action_dim != online.action_dim or self.cfg != online.cfg:
            raise ValueError("Online and target critic architectures must match exactly.")
