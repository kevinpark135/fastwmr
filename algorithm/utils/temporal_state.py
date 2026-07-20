"""Termination masks and rollout-only recurrent state for FastWMR."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_done_flags(terminated: torch.Tensor, truncated: torch.Tensor) -> None:
    if terminated.dtype is not torch.bool or truncated.dtype is not torch.bool:
        raise TypeError("terminated and truncated must both be boolean tensors.")
    if terminated.shape != truncated.shape:
        raise ValueError(
            f"terminated and truncated must have equal shapes, got {terminated.shape} and {truncated.shape}."
        )
    if terminated.device != truncated.device:
        raise ValueError("terminated and truncated must be on the same device.")
    if terminated.ndim != 1:
        raise ValueError(f"Done flags must have shape (num_envs,), got {terminated.shape}.")


def episode_end_mask(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """Return environments whose recurrent state must be reset."""

    _validate_done_flags(terminated, truncated)
    return terminated | truncated


def bellman_bootstrap_mask(
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``1`` where a Bellman target may bootstrap.

    Time-limit truncation ends the recurrent episode but does not represent an
    MDP terminal state, so it intentionally remains bootstrap-eligible.
    """

    _validate_done_flags(terminated, truncated)
    return (~terminated).to(dtype=dtype)


@dataclass(frozen=True)
class RecurrentState:
    """LSTM hidden/cell tensors with shape ``(layers, envs, hidden_dim)``."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def __post_init__(self) -> None:
        if self.hidden.shape != self.cell.shape:
            raise ValueError(f"Hidden and cell shapes differ: {self.hidden.shape} vs {self.cell.shape}.")
        if self.hidden.ndim != 3:
            raise ValueError(
                "Recurrent tensors must have shape (num_layers, num_envs, hidden_dim), "
                f"got {self.hidden.shape}."
            )
        if self.hidden.device != self.cell.device or self.hidden.dtype != self.cell.dtype:
            raise ValueError("Hidden and cell tensors must share a device and dtype.")

    @classmethod
    def zeros(
        cls,
        *,
        num_layers: int,
        num_envs: int,
        hidden_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "RecurrentState":
        if min(num_layers, num_envs, hidden_dim) <= 0:
            raise ValueError("num_layers, num_envs, and hidden_dim must all be positive.")
        shape = (num_layers, num_envs, hidden_dim)
        return cls(
            hidden=torch.zeros(shape, device=device, dtype=dtype),
            cell=torch.zeros(shape, device=device, dtype=dtype),
        )

    @property
    def num_envs(self) -> int:
        return self.hidden.shape[1]

    def detach(self) -> "RecurrentState":
        """Cut the graph between rollout chunks without changing values."""

        return RecurrentState(self.hidden.detach(), self.cell.detach())

    def reset(self, reset_mask: torch.Tensor) -> "RecurrentState":
        """Return state with only selected environments zeroed."""

        if reset_mask.dtype is not torch.bool:
            raise TypeError("reset_mask must be a boolean tensor.")
        if reset_mask.shape != (self.num_envs,):
            raise ValueError(f"reset_mask must have shape ({self.num_envs},), got {reset_mask.shape}.")
        keep = (~reset_mask).to(device=self.hidden.device, dtype=self.hidden.dtype).view(1, self.num_envs, 1)
        return RecurrentState(self.hidden * keep, self.cell * keep)

    def reset_done(self, terminated: torch.Tensor, truncated: torch.Tensor) -> "RecurrentState":
        """Reset terminated and time-limit-truncated environments."""

        return self.reset(episode_end_mask(terminated, truncated))
