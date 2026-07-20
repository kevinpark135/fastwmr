"""Tests for the ordered FastWMR observation contract."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_INTERFACE_CFG,
    TargetKind,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_env_cfg import (
    G1FastWMREnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.observations import (
    FastWMRObservationsCfg,
    G1_29DOF_JOINT_PATTERNS,
    POLICY_TERM_NAMES,
    PRIVILEGED_TERM_NAMES,
    assemble_privileged_reconstruction_target,
    privileged_friction,
)


_GROUP_METADATA_FIELDS = {
    "enable_corruption",
    "concatenate_terms",
    "concatenate_dim",
    "history_length",
    "flatten_history_dim",
}


def _term_names(group: object) -> tuple[str, ...]:
    return tuple(
        name
        for name in vars(group)
        if not name.startswith("_") and name not in _GROUP_METADATA_FIELDS
    )


def test_policy_observation_order_and_dimension() -> None:
    layout = DEFAULT_INTERFACE_CFG.policy_observation_layout
    observations = FastWMRObservationsCfg()

    assert POLICY_TERM_NAMES == (
        "base_ang_vel",
        "projected_gravity",
        "velocity_command",
        "joint_pos",
        "joint_vel",
        "previous_action",
    )
    assert layout.names == POLICY_TERM_NAMES
    assert _term_names(observations.policy) == POLICY_TERM_NAMES
    assert layout.dim == 96


def test_privileged_target_order_and_loss_partition() -> None:
    layout = DEFAULT_INTERFACE_CFG.reconstruction_layout
    observations = FastWMRObservationsCfg()

    assert PRIVILEGED_TERM_NAMES == (
        "base_lin_vel",
        "friction",
        "payload_mass",
        "push_force_torque",
        "foot_contacts",
    )
    assert layout.names == PRIVILEGED_TERM_NAMES
    assert _term_names(observations.privileged) == PRIVILEGED_TERM_NAMES
    assert layout.dim == 13
    assert layout.select_kind(TargetKind.CONTINUOUS).dim == 11
    assert layout.select_kind(TargetKind.DISCRETE).dim == 2
    assert layout.field_slice("foot_contacts") == slice(11, 13)


def test_policy_contract_excludes_privileged_signals() -> None:
    privileged_only = {"base_lin_vel", "friction", "payload_mass", "push_force_torque", "foot_contacts"}

    assert privileged_only.isdisjoint(POLICY_TERM_NAMES)
    assert "height_scan" not in POLICY_TERM_NAMES


def test_environment_uses_the_same_29dof_contract() -> None:
    env_cfg = G1FastWMREnvCfg()

    assert env_cfg.scene.robot.spawn.usd_path.endswith("/Unitree/G1/g1.usd")
    assert tuple(env_cfg.actions.joint_pos.joint_names) == G1_29DOF_JOINT_PATTERNS
    assert _term_names(env_cfg.observations.policy) == POLICY_TERM_NAMES
    assert _term_names(env_cfg.observations.privileged) == PRIVILEGED_TERM_NAMES


def test_privileged_target_assembly_uses_the_canonical_order() -> None:
    batch_size = 2
    target = assemble_privileged_reconstruction_target(
        base_lin_vel=torch.full((batch_size, 3), 1.0),
        friction=torch.full((batch_size, 1), 2.0),
        payload_mass=torch.full((batch_size, 1), 3.0),
        push_force_torque=torch.full((batch_size, 6), 4.0),
        foot_contacts=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )

    assert target.shape == (batch_size, 13)
    assert torch.equal(target[:, 0:3], torch.full((batch_size, 3), 1.0))
    assert torch.equal(target[:, 3:4], torch.full((batch_size, 1), 2.0))
    assert torch.equal(target[:, 4:5], torch.full((batch_size, 1), 3.0))
    assert torch.equal(target[:, 5:11], torch.full((batch_size, 6), 4.0))
    assert torch.equal(target[:, 11:13], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_privileged_target_rejects_non_binary_contacts() -> None:
    with pytest.raises(ValueError, match="binary"):
        assemble_privileged_reconstruction_target(
            base_lin_vel=torch.zeros(1, 3),
            friction=torch.zeros(1, 1),
            payload_mass=torch.zeros(1, 1),
            push_force_torque=torch.zeros(1, 6),
            foot_contacts=torch.tensor([[0.25, 1.0]]),
        )


def test_privileged_dr_reader_requires_initialized_exact_buffer() -> None:
    env = SimpleNamespace(num_envs=2, device="cpu", observation_manager=object())
    with pytest.raises(RuntimeError, match="startup event"):
        privileged_friction(env)

    env.fastwmr_friction = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="must have shape"):
        privileged_friction(env)

    env.fastwmr_friction = torch.zeros(2, 1, dtype=torch.float64)
    with pytest.raises(TypeError, match="torch.float32"):
        privileged_friction(env)
