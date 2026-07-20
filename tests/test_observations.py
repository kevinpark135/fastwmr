"""Tests for the ordered FastWMR observation contract."""

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_algorithm.config import (
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
