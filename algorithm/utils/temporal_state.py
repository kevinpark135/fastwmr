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


class RecurrentStateManager:
    """Own detached per-environment recurrent state during data collection.

    This is deliberately a plain Python object rather than an ``nn.Module``.
    Runtime hidden/cell values therefore stay outside model ``state_dict`` and
    checkpoints. Every replacement is detached to prevent collection graphs
    from growing across environment steps.
    """

    def __init__(self, initial_state: RecurrentState) -> None:
        if not isinstance(initial_state, RecurrentState):
            raise TypeError("initial_state must be a RecurrentState.")
        self._shape = initial_state.hidden.shape
        self._device = initial_state.hidden.device
        self._dtype = initial_state.hidden.dtype
        self._state = self._validated_detached(initial_state)

    @classmethod
    def zeros(
        cls,
        *,
        num_layers: int,
        num_envs: int,
        hidden_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "RecurrentStateManager":
        return cls(
            RecurrentState.zeros(
                num_layers=num_layers,
                num_envs=num_envs,
                hidden_dim=hidden_dim,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def state(self) -> RecurrentState:
        return self._state

    @property
    def num_envs(self) -> int:
        return self._shape[1]

    @property
    def hidden_norm(self) -> float:
        """Return the combined hidden/cell RMS value for runtime diagnostics."""

        squared_mean = 0.5 * (
            self._state.hidden.square().mean() + self._state.cell.square().mean()
        )
        return float(torch.sqrt(squared_mean))

    def replace(self, state: RecurrentState) -> RecurrentState:
        """Install a finite state after detaching it from any autograd graph."""

        self._state = self._validated_detached(state)
        return self._state

    def reset(self, reset_mask: torch.Tensor) -> RecurrentState:
        self._state = self._validated_detached(self._state.reset(reset_mask))
        return self._state

    def reset_done(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> RecurrentState:
        self._state = self._validated_detached(
            self._state.reset_done(terminated, truncated)
        )
        return self._state

    def detach(self) -> RecurrentState:
        self._state = self._state.detach()
        return self._state

    def clear(self) -> RecurrentState:
        """Reset every environment without changing the fixed state contract."""

        self._state = RecurrentState(
            hidden=torch.zeros_like(self._state.hidden),
            cell=torch.zeros_like(self._state.cell),
        )
        return self._state

    def _validated_detached(self, state: RecurrentState) -> RecurrentState:
        if not isinstance(state, RecurrentState):
            raise TypeError("state must be a RecurrentState.")
        if state.hidden.shape != self._shape:
            raise ValueError(
                f"Runtime state shape is fixed at {self._shape}, got {state.hidden.shape}."
            )
        if state.hidden.device != self._device or state.hidden.dtype != self._dtype:
            raise ValueError("Runtime state device and dtype cannot change after initialization.")
        if not torch.isfinite(state.hidden).all() or not torch.isfinite(state.cell).all():
            raise FloatingPointError("Runtime recurrent state must remain finite.")
        return state.detach()
