"""Recurrent history encoder :math:`E_psi` for FastWMR.

The encoder owns network parameters but deliberately does not own mutable
hidden state. Rollout workers keep one :class:`RecurrentState` per vectorized
environment, while replay learning reconstructs temporary state from ordered
raw-observation sequences using the current encoder parameters.
"""

from __future__ import annotations

import torch
from torch import nn

from ..utils.temporal_state import RecurrentState


class HistoryEncoder(nn.Module):
    """LSTM encoder that compresses proprioceptive history into latent features.

    Args:
        observation_dim: Width of one policy observation.
        hidden_dim: Width of the recurrent feature. WMR uses 256 by default.
        num_layers: Number of stacked LSTM layers.

    Hidden and cell tensors always follow PyTorch's
    ``(num_layers, batch_or_envs, hidden_dim)`` layout. Keeping them explicit
    prevents state from leaking between independent rollout or replay calls.
    """

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.recurrent = nn.LSTM(
            input_size=observation_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    @property
    def output_dim(self) -> int:
        """Width of each encoded history feature."""

        return self.hidden_dim

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RecurrentState:
        """Create a zero recurrent state for rollout or replay inference."""

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"dtype must be a floating torch dtype, got {dtype}.")
        return RecurrentState.zeros(
            num_layers=self.num_layers,
            num_envs=batch_size,
            hidden_dim=self.hidden_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Encode a ``(batch, time, observation_dim)`` sequence."""

        return self.forward_sequence(observations, state)

    def forward_rollout(
        self,
        observation: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Advance one timestep for every environment in a rollout batch.

        Returns a feature tensor with shape ``(num_envs, hidden_dim)`` and the
        corresponding next recurrent state. Episode-end slices are reset by the
        rollout worker through :meth:`RecurrentState.reset_done`.
        """

        self._validate_observations(observation, expected_ndim=2, name="observation")
        self._validate_state(
            state,
            batch_size=observation.shape[0],
            device=observation.device,
        )
        features, next_state = self._run_lstm(observation.unsqueeze(1), state)
        return features[:, 0], next_state

    def forward_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Encode an ordered replay sequence with explicit initial context.

        ``observations`` has shape ``(batch, time, observation_dim)``. This API
        is used for both no-grad burn-in and gradient-tracked learning unrolls;
        the caller owns that gradient boundary.
        """

        self._validate_observations(observations, expected_ndim=3, name="observations")
        if observations.shape[1] <= 0:
            raise ValueError("observations must contain at least one timestep.")
        self._validate_state(
            state,
            batch_size=observations.shape[0],
            device=observations.device,
        )
        return self._run_lstm(observations, state)

    def _run_lstm(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        features, (hidden, cell) = self.recurrent(observations, (state.hidden, state.cell))
        return features, RecurrentState(hidden=hidden, cell=cell)

    def _validate_observations(
        self,
        observations: torch.Tensor,
        *,
        expected_ndim: int,
        name: str,
    ) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not observations.dtype.is_floating_point:
            raise TypeError(f"{name} must have a floating dtype, got {observations.dtype}.")
        if observations.ndim != expected_ndim or observations.shape[-1] != self.observation_dim:
            leading_dims = "batch, " if expected_ndim == 2 else "batch, time, "
            raise ValueError(
                f"{name} must have shape ({leading_dims}{self.observation_dim}), "
                f"got {tuple(observations.shape)}."
            )

    def _validate_state(
        self,
        state: RecurrentState,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if not isinstance(state, RecurrentState):
            raise TypeError("state must be a RecurrentState.")
        expected_shape = (self.num_layers, batch_size, self.hidden_dim)
        if state.hidden.shape != expected_shape:
            raise ValueError(
                f"state must have shape {expected_shape}, got {tuple(state.hidden.shape)}."
            )
        if state.hidden.device != device:
            raise ValueError(
                f"state and observations must share a device, got {state.hidden.device} and {device}."
            )
