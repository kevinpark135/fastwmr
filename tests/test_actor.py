"""Tests for the FastSAC tanh-Gaussian actor and FastWMR routing contract."""

import math

import pytest
import torch
import torch.nn.functional as F

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_ACTOR_CFG,
    DEFAULT_INTERFACE_CFG,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TanhGaussianActor,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils.feature_builder import (
    build_control_feature,
)


def test_default_actor_matches_fastsac_architecture_and_std_cap() -> None:
    actor = TanhGaussianActor(input_dim=6, action_dim=3)
    features = torch.randn(5, 6)

    deterministic_action, mean, log_std = actor(features)

    assert [layer.out_features for layer in actor.trunk if isinstance(layer, torch.nn.Linear)] == [512, 256, 128]
    assert deterministic_action.shape == mean.shape == log_std.shape == (5, 3)
    assert torch.all(log_std >= DEFAULT_ACTOR_CFG.log_std_min)
    assert torch.all(log_std <= DEFAULT_ACTOR_CFG.log_std_max)
    assert torch.all(log_std.exp() <= 1.0)
    assert torch.equal(deterministic_action, torch.zeros_like(deterministic_action))


def test_reparameterized_sample_is_bounded_and_has_finite_log_probability() -> None:
    actor = TanhGaussianActor(input_dim=4, action_dim=2)
    features = torch.randn(32, 4, requires_grad=True)

    actions, log_probability = actor.sample(features)
    (actions.square().mean() + log_probability.mean()).backward()

    assert actions.shape == (32, 2)
    assert log_probability.shape == (32,)
    assert torch.all(actions > actor.action_low)
    assert torch.all(actions < actor.action_high)
    assert torch.isfinite(log_probability).all()
    assert features.grad is not None
    assert actor.mean_head.weight.grad is not None
    assert actor.log_std_head.weight.grad is not None


def test_log_probability_includes_stable_tanh_and_scale_corrections() -> None:
    low = torch.tensor([-2.0, 1.0])
    high = torch.tensor([2.0, 5.0])
    actor = TanhGaussianActor(input_dim=3, action_dim=2, action_low=low, action_high=high)
    features = torch.randn(4, 3)
    _, mean, log_std = actor(features)

    torch.manual_seed(17)
    action, actual_log_probability = actor.sample(features)
    torch.manual_seed(17)
    distribution = torch.distributions.Normal(mean, log_std.exp())
    pre_tanh_action = distribution.rsample()
    log_tanh_jacobian = 2.0 * (math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action))
    expected_log_probability = (
        distribution.log_prob(pre_tanh_action) - log_tanh_jacobian - torch.log(actor.action_scale)
    ).sum(dim=-1)

    assert torch.allclose(actual_log_probability, expected_log_probability)
    assert torch.all(action > low)
    assert torch.all(action < high)


def test_fastwmr_feature_builder_blocks_actor_gradient_to_estimator() -> None:
    interface = DEFAULT_INTERFACE_CFG
    actor = TanhGaussianActor(input_dim=interface.actor_input_dim, action_dim=interface.action_dim)
    observations = torch.randn(3, interface.policy_observation_dim, requires_grad=True)
    reconstruction = torch.randn(3, interface.reconstruction_target_dim, requires_grad=True)
    features = build_control_feature(observations, reconstruction, cfg=interface)

    actions, log_probability = actor.sample(features)
    (actions.mean() + log_probability.mean()).backward()

    assert observations.grad is not None
    assert reconstruction.grad is None


def test_actor_supports_sequence_leading_dimensions_and_deterministic_act() -> None:
    actor = TanhGaussianActor(input_dim=4, action_dim=2)
    features = torch.randn(3, 7, 4)

    expected, _, _ = actor(features)
    actual = actor.act(features, deterministic=True)
    sampled, log_probability = actor.sample(features)

    assert torch.equal(actual, expected)
    assert not actual.requires_grad
    assert sampled.shape == (3, 7, 2)
    assert log_probability.shape == (3, 7)


def test_actor_rejects_invalid_configuration_and_bounds() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        TanhGaussianActorCfg(hidden_dim=10)
    with pytest.raises(ValueError, match="smaller"):
        TanhGaussianActorCfg(log_std_min=0.0, log_std_max=0.0)
    with pytest.raises(ValueError, match="smaller than action_high"):
        TanhGaussianActor(input_dim=3, action_dim=2, action_low=1.0, action_high=1.0)

    actor = TanhGaussianActor(input_dim=3, action_dim=2)
    with pytest.raises(ValueError, match="end in dimension 3"):
        actor(torch.randn(4, 2))
