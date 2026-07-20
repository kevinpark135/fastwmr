"""Tests for the FastWMR recurrent history encoder."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    HistoryEncoder,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    RecurrentState,
)


def test_initial_state_matches_encoder_contract() -> None:
    encoder = HistoryEncoder(observation_dim=6, hidden_dim=8, num_layers=2)

    state = encoder.initial_state(batch_size=4, device="cpu", dtype=torch.float32)

    assert state.hidden.shape == (2, 4, 8)
    assert state.cell.shape == (2, 4, 8)
    assert state.hidden.dtype is torch.float32
    assert torch.count_nonzero(state.hidden) == 0
    assert torch.count_nonzero(state.cell) == 0
    assert encoder.output_dim == 8


def test_rollout_and_sequence_paths_are_equivalent() -> None:
    torch.manual_seed(4)
    encoder = HistoryEncoder(observation_dim=3, hidden_dim=5, num_layers=2)
    observations = torch.randn(4, 6, 3)
    initial_state = encoder.initial_state(
        batch_size=4,
        device=observations.device,
        dtype=observations.dtype,
    )

    sequence_features, sequence_state = encoder.forward_sequence(observations, initial_state)
    rollout_state = initial_state
    rollout_features = []
    for timestep in range(observations.shape[1]):
        features, rollout_state = encoder.forward_rollout(observations[:, timestep], rollout_state)
        rollout_features.append(features)

    assert sequence_features.shape == (4, 6, 5)
    torch.testing.assert_close(torch.stack(rollout_features, dim=1), sequence_features)
    torch.testing.assert_close(rollout_state.hidden, sequence_state.hidden)
    torch.testing.assert_close(rollout_state.cell, sequence_state.cell)


def test_sequence_path_propagates_gradients_through_time() -> None:
    encoder = HistoryEncoder(observation_dim=3, hidden_dim=5)
    observations = torch.randn(2, 4, 3, requires_grad=True)
    state = encoder.initial_state(
        batch_size=2,
        device=observations.device,
        dtype=observations.dtype,
    )

    features, final_state = encoder(observations, state)
    (features.square().mean() + final_state.hidden.square().mean()).backward()

    assert observations.grad is not None
    assert torch.isfinite(observations.grad).all()
    assert encoder.recurrent.weight_ih_l0.grad is not None
    assert encoder.recurrent.weight_hh_l0.grad is not None


def test_rollout_state_can_reset_only_finished_environments() -> None:
    encoder = HistoryEncoder(observation_dim=3, hidden_dim=4)
    observations = torch.randn(3, 3)
    state = encoder.initial_state(
        batch_size=3,
        device=observations.device,
        dtype=observations.dtype,
    )

    _, next_state = encoder.forward_rollout(observations, state)
    reset_state = next_state.reset_done(
        terminated=torch.tensor([False, True, False]),
        truncated=torch.tensor([False, False, False]),
    )

    assert torch.count_nonzero(reset_state.hidden[:, 1]) == 0
    assert torch.count_nonzero(reset_state.cell[:, 1]) == 0
    torch.testing.assert_close(reset_state.hidden[:, 0], next_state.hidden[:, 0])
    torch.testing.assert_close(reset_state.hidden[:, 2], next_state.hidden[:, 2])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"observation_dim": 0}, "observation_dim"),
        ({"observation_dim": 3, "hidden_dim": 0}, "hidden_dim"),
        ({"observation_dim": 3, "num_layers": 0}, "num_layers"),
    ],
)
def test_invalid_dimensions_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HistoryEncoder(**kwargs)


def test_invalid_observation_or_state_shape_is_rejected() -> None:
    encoder = HistoryEncoder(observation_dim=3, hidden_dim=4)
    state = encoder.initial_state(batch_size=2, device="cpu", dtype=torch.float32)

    with pytest.raises(ValueError, match="observation must have shape"):
        encoder.forward_rollout(torch.randn(2, 1, 3), state)
    with pytest.raises(ValueError, match="observations must contain at least one timestep"):
        encoder.forward_sequence(torch.empty(2, 0, 3), state)

    wrong_state = RecurrentState.zeros(
        num_layers=1,
        num_envs=3,
        hidden_dim=4,
        device="cpu",
    )
    with pytest.raises(ValueError, match="state must have shape"):
        encoder.forward_rollout(torch.randn(2, 3), wrong_state)
