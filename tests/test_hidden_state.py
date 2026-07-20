"""Tests for per-environment recurrent state and done semantics."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils.temporal_state import (
    RecurrentState,
    bellman_bootstrap_mask,
    episode_end_mask,
)


def test_terminated_and_truncated_have_different_bootstrap_semantics() -> None:
    terminated = torch.tensor([False, True, False, False])
    truncated = torch.tensor([False, False, True, False])

    assert torch.equal(episode_end_mask(terminated, truncated), torch.tensor([False, True, True, False]))
    assert torch.equal(bellman_bootstrap_mask(terminated, truncated), torch.tensor([1.0, 0.0, 1.0, 1.0]))


def test_only_done_environments_are_reset() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1.0
    state = RecurrentState(hidden=hidden, cell=-hidden)

    reset_state = state.reset(torch.tensor([False, True, False]))

    assert torch.equal(reset_state.hidden[:, 0], state.hidden[:, 0])
    assert torch.count_nonzero(reset_state.hidden[:, 1]) == 0
    assert torch.equal(reset_state.hidden[:, 2], state.hidden[:, 2])
    assert torch.count_nonzero(reset_state.cell[:, 1]) == 0


def test_detach_cuts_rollout_graph() -> None:
    hidden = torch.ones((1, 2, 3), requires_grad=True)
    state = RecurrentState(hidden=hidden * 2.0, cell=hidden * 3.0)

    detached = state.detach()

    assert not detached.hidden.requires_grad
    assert not detached.cell.requires_grad
    assert torch.equal(detached.hidden, state.hidden)
