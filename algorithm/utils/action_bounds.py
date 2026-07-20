"""Joint-limit-aware FastSAC action bounds in raw policy coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionBounds:
    """One shared low/high action vector used by all parallel environments."""

    low: torch.Tensor
    high: torch.Tensor

    def __post_init__(self) -> None:
        if self.low.ndim != 1 or self.high.shape != self.low.shape:
            raise ValueError("Action bounds must be equal one-dimensional tensors.")
        if self.low.device != self.high.device or self.low.dtype != self.high.dtype:
            raise ValueError("Action bounds must share device and dtype.")
        if not self.low.dtype.is_floating_point:
            raise TypeError("Action bounds must have a floating-point dtype.")
        if not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all():
            raise ValueError("Action bounds must be finite.")
        if torch.any(self.low >= self.high):
            raise ValueError("Every lower action bound must be smaller than its upper bound.")

    @property
    def action_dim(self) -> int:
        return self.low.shape[0]

    @property
    def scale(self) -> torch.Tensor:
        return 0.5 * (self.high - self.low)

    @property
    def bias(self) -> torch.Tensor:
        return 0.5 * (self.high + self.low)

    def to(self, device: torch.device | str) -> "ActionBounds":
        return ActionBounds(self.low.to(device), self.high.to(device))


def symmetric_joint_limit_action_bounds(
    joint_position_limits: torch.Tensor,
    default_joint_positions: torch.Tensor,
    environment_action_scale: torch.Tensor | float,
) -> ActionBounds:
    """Reproduce the FastSAC reference's per-joint symmetric tanh scale.

    IsaacLab applies ``target = default + environment_action_scale * raw``.
    The actor therefore uses
    ``max(abs(lower-default), abs(upper-default)) / environment_action_scale``
    as a symmetric raw-action magnitude. Leading dimensions represent parallel
    environments and are reduced to one shared maximum-safe reference vector.
    """

    if not isinstance(joint_position_limits, torch.Tensor) or not isinstance(
        default_joint_positions, torch.Tensor
    ):
        raise TypeError("Joint limits and default positions must be tensors.")
    if not joint_position_limits.dtype.is_floating_point or not default_joint_positions.dtype.is_floating_point:
        raise TypeError("Joint limits and default positions must be floating point.")
    if joint_position_limits.shape[:-1] != default_joint_positions.shape or joint_position_limits.shape[-1] != 2:
        raise ValueError("Joint limits must have shape (..., action_dim, 2) aligned with default positions.")
    if default_joint_positions.ndim < 1 or default_joint_positions.shape[-1] <= 0:
        raise ValueError("Default joint positions must end in a non-empty action dimension.")
    if (
        joint_position_limits.device != default_joint_positions.device
        or joint_position_limits.dtype != default_joint_positions.dtype
    ):
        raise ValueError("Joint limits and default positions must share a device and dtype.")
    if not torch.isfinite(joint_position_limits).all() or not torch.isfinite(default_joint_positions).all():
        raise ValueError("Joint limits and default positions must be finite.")

    lower, upper = joint_position_limits.unbind(dim=-1)
    if torch.any(lower >= upper):
        raise ValueError("Every lower joint limit must be smaller than its upper limit.")
    if torch.any(default_joint_positions < lower) or torch.any(default_joint_positions > upper):
        raise ValueError("Default joint positions must lie inside their joint limits.")

    scale = torch.as_tensor(
        environment_action_scale,
        device=default_joint_positions.device,
        dtype=default_joint_positions.dtype,
    )
    try:
        scale = torch.broadcast_to(scale, default_joint_positions.shape)
    except RuntimeError as error:
        raise ValueError("Environment action scale is not broadcastable to default joint positions.") from error
    if not torch.isfinite(scale).all() or torch.any(scale <= 0.0):
        raise ValueError("Environment action scale must be finite and strictly positive.")

    magnitude = torch.maximum((lower - default_joint_positions).abs(), (upper - default_joint_positions).abs())
    raw_magnitude = magnitude / scale
    if raw_magnitude.ndim > 1:
        raw_magnitude = raw_magnitude.flatten(0, -2).amax(dim=0)
    if torch.any(raw_magnitude <= 0.0):
        raise ValueError("Every joint must have a positive action range around its default position.")
    return ActionBounds(low=-raw_magnitude, high=raw_magnitude)
