"""Multi-head world-state decoder :math:`D_psi` for FastWMR.

Continuous world-state fields and discrete contacts use independent MLP heads.
The discrete head exposes logits for numerically stable supervised learning and
probabilities for the reconstructed control feature used at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import DEFAULT_INTERFACE_CFG


@dataclass(frozen=True)
class DecoderOutput:
    """Structured continuous predictions and discrete contact logits."""

    continuous: torch.Tensor
    discrete_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.continuous.ndim < 1 or self.discrete_logits.ndim < 1:
            raise ValueError("Decoder outputs must have at least one dimension.")
        if self.continuous.shape[:-1] != self.discrete_logits.shape[:-1]:
            raise ValueError(
                "Continuous and discrete decoder batch shapes must match, got "
                f"{self.continuous.shape[:-1]} and {self.discrete_logits.shape[:-1]}."
            )
        if self.continuous.device != self.discrete_logits.device:
            raise ValueError("Continuous predictions and discrete logits must share a device.")
        if self.continuous.dtype != self.discrete_logits.dtype:
            raise ValueError("Continuous predictions and discrete logits must share a dtype.")

    @property
    def discrete_probabilities(self) -> torch.Tensor:
        """Return contact probabilities while preserving estimator gradients."""

        return torch.sigmoid(self.discrete_logits)

    @property
    def reconstruction(self) -> torch.Tensor:
        """Return the deployable reconstructed state ``[continuous, probabilities]``.

        The ordering matches the current FastWMR target contract: all continuous
        fields precede the discrete foot-contact fields. Actor/critic gradient
        cutoff is applied later by ``build_control_feature``.
        """

        return torch.cat((self.continuous, self.discrete_probabilities), dim=-1)


class WorldStateDecoder(nn.Module):
    """Decode recurrent features into continuous state and contact predictions.

    Both heads consist of two fully connected layers with an ELU activation,
    following WMR's multi-head decoder design. They do not share parameters, so
    regression and contact-classification losses meet only in the encoder.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        continuous_dim: int = DEFAULT_INTERFACE_CFG.continuous_target_dim,
        discrete_dim: int = DEFAULT_INTERFACE_CFG.discrete_target_dim,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("continuous_dim", continuous_dim),
            ("discrete_dim", discrete_dim),
            ("hidden_dim", hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        self.input_dim = input_dim
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        self.hidden_dim = hidden_dim
        self.continuous_head = self._make_head(continuous_dim)
        self.discrete_head = self._make_head(discrete_dim)

    @property
    def output_dim(self) -> int:
        """Total reconstructed-state width exposed to the control feature."""

        return self.continuous_dim + self.discrete_dim

    def forward(self, encoded_history: torch.Tensor) -> DecoderOutput:
        """Decode rollout or sequence features while preserving leading dims."""

        self._validate_input(encoded_history)
        return DecoderOutput(
            continuous=self.continuous_head(encoded_history),
            discrete_logits=self.discrete_head(encoded_history),
        )

    def _make_head(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(alpha=1.0),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def _validate_input(self, encoded_history: torch.Tensor) -> None:
        if not isinstance(encoded_history, torch.Tensor):
            raise TypeError("encoded_history must be a torch.Tensor.")
        if not encoded_history.dtype.is_floating_point:
            raise TypeError(
                f"encoded_history must have a floating dtype, got {encoded_history.dtype}."
            )
        if encoded_history.ndim < 2 or encoded_history.shape[-1] != self.input_dim:
            raise ValueError(
                f"encoded_history must have shape (..., {self.input_dim}) with at least "
                f"one batch dimension, got {tuple(encoded_history.shape)}."
            )
