"""Tests for FastSAC joint-limit-aware action scaling."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import TanhGaussianActor
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    symmetric_joint_limit_action_bounds,
)


def test_reference_bounds_use_furthest_limit_and_environment_scale() -> None:
    limits = torch.tensor([[-1.0, 2.0], [-4.0, 3.0]])
    defaults = torch.tensor([0.5, -1.0])

    bounds = symmetric_joint_limit_action_bounds(limits, defaults, torch.tensor([0.5, 2.0]))

    assert torch.equal(bounds.low, torch.tensor([-3.0, -2.0]))
    assert torch.equal(bounds.high, torch.tensor([3.0, 2.0]))
    assert torch.equal(bounds.bias, torch.zeros(2))


def test_zero_centered_actor_maps_to_default_joint_pose() -> None:
    limits = torch.tensor([[-1.0, 2.0], [-4.0, 3.0]])
    defaults = torch.tensor([0.5, -1.0])
    environment_scale = torch.tensor([0.5, 2.0])
    bounds = symmetric_joint_limit_action_bounds(limits, defaults, environment_scale)
    actor = TanhGaussianActor(3, 2, action_low=bounds.low, action_high=bounds.high)

    raw_actions = actor.act(torch.randn(4, 3), deterministic=True)
    joint_targets = defaults + environment_scale * raw_actions

    assert torch.equal(raw_actions, torch.zeros_like(raw_actions))
    assert torch.equal(joint_targets, defaults.expand_as(joint_targets))


def test_parallel_environment_ranges_reduce_to_one_shared_bound() -> None:
    limits = torch.tensor(
        [
            [[-1.0, 1.0], [-2.0, 2.0]],
            [[-1.5, 1.5], [-3.0, 3.0]],
        ]
    )
    defaults = torch.zeros(2, 2)

    bounds = symmetric_joint_limit_action_bounds(limits, defaults, 0.5)

    assert torch.equal(bounds.scale, torch.tensor([3.0, 6.0]))


def test_joint_limit_bounds_reject_invalid_contracts() -> None:
    limits = torch.tensor([[-1.0, 1.0]])
    defaults = torch.tensor([0.0])

    with pytest.raises(ValueError, match="strictly positive"):
        symmetric_joint_limit_action_bounds(limits, defaults, 0.0)
    with pytest.raises(ValueError, match="inside"):
        symmetric_joint_limit_action_bounds(limits, torch.tensor([2.0]), 1.0)
    with pytest.raises(ValueError, match="aligned"):
        symmetric_joint_limit_action_bounds(torch.zeros(1, 3), defaults, 1.0)
