"""Tests for actor/critic feature routing and estimator gradient cutoff."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_algorithm.config import (
    DEFAULT_INTERFACE_CFG,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_algorithm.utils.feature_builder import (
    build_control_feature,
    build_critic_input,
)


def test_reconstruction_is_detached_from_control_losses() -> None:
    cfg = DEFAULT_INTERFACE_CFG
    observation = torch.randn(4, cfg.policy_observation_dim, requires_grad=True)
    reconstruction = torch.randn(4, cfg.reconstruction_target_dim, requires_grad=True)

    control_feature = build_control_feature(observation, reconstruction, cfg=cfg)
    control_feature.square().mean().backward()

    assert control_feature.shape == (4, cfg.control_feature_dim)
    assert observation.grad is not None
    assert reconstruction.grad is None


def test_actor_and_critic_share_the_same_control_feature() -> None:
    cfg = DEFAULT_INTERFACE_CFG
    observation = torch.zeros(8, cfg.policy_observation_dim)
    reconstruction = torch.zeros(8, cfg.reconstruction_target_dim)
    action = torch.zeros(8, cfg.action_dim)

    actor_input = build_control_feature(observation, reconstruction, cfg=cfg)
    critic_input = build_critic_input(actor_input, action, cfg=cfg)

    assert actor_input.shape[-1] == cfg.control_feature_dim
    assert critic_input.shape[-1] == cfg.critic_input_dim
    assert torch.equal(critic_input[:, : cfg.control_feature_dim], actor_input)
