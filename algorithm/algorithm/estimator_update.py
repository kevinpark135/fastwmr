"""Current-estimator replay inference with burn-in and truncated BPTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..buffers import SequenceReplayBatch
from ..utils.temporal_state import RecurrentState


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
    """Rebuild context with no-grad burn-in, then unroll ``L + 1`` with grad.

    Mid-episode sequence starts use a zero-state burn-in approximation. When
    ``sequence.context_is_exact`` is true, the sampled prefix starts at the real
    episode reset and the reconstructed context is exact for that prefix.
    """

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
            f"{tuple(reconstructions.shape)} for observations {tuple(sequence.learning_observations.shape)}."
        )
    return BurnInUnrollOutput(
        reconstructions=reconstructions,
        learning_initial_state=learning_initial_state,
        final_state=final_state,
        context_is_exact=sequence.context_is_exact,
    )
