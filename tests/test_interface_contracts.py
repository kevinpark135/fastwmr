"""Pure tests for dimensions shared by the estimator, actor, and critics."""

import pytest

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ControlFeatureMode,
    DEFAULT_INTERFACE_CFG,
    FastWMRInterfaceCfg,
    TensorFieldSpec,
    TensorLayoutSpec,
)


def test_default_interface_dimensions() -> None:
    cfg = DEFAULT_INTERFACE_CFG

    assert cfg.action_dim == 29
    assert cfg.policy_observation_dim == 96
    assert cfg.reconstruction_target_dim == 13
    assert cfg.control_feature_dim == 109
    assert cfg.actor_input_dim == cfg.critic_state_dim == cfg.control_feature_dim
    assert cfg.critic_input_dim == 138


def test_reconstruction_only_ablation_changes_both_control_inputs() -> None:
    cfg = FastWMRInterfaceCfg(control_feature_mode=ControlFeatureMode.RECONSTRUCTION_ONLY)

    assert cfg.control_feature_dim == cfg.reconstruction_target_dim == 13
    assert cfg.critic_input_dim == cfg.control_feature_dim + cfg.action_dim == 42


def test_tensor_layout_rejects_ambiguous_fields() -> None:
    with pytest.raises(ValueError, match="unique"):
        TensorLayoutSpec((TensorFieldSpec("duplicate", 1), TensorFieldSpec("duplicate", 2)))

    with pytest.raises(ValueError, match="positive width"):
        TensorFieldSpec("invalid", 0)
