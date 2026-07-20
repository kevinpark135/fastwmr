"""Task registry and config-level checks that do not launch Isaac Sim."""

import gymnasium as gym

import isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.baseline_env_cfg import (
    G1FastSACBaselineEnvCfg,
    G1FastSACBaselineEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_env_cfg import (
    G1FastWMREnvCfg,
    G1FastWMREnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.observations import (
    G1_29DOF_JOINT_PATTERNS,
)


TASK_ENTRY_POINTS = {
    "Isaac-Velocity-G1-FastWMR-v0": "fastwmr_env_cfg:G1FastWMREnvCfg",
    "Isaac-Velocity-G1-FastWMR-Play-v0": "fastwmr_env_cfg:G1FastWMREnvCfg_PLAY",
    "Isaac-Velocity-G1-FastSAC-Baseline-v0": "baseline_env_cfg:G1FastSACBaselineEnvCfg",
    "Isaac-Velocity-G1-FastSAC-Baseline-Play-v0": "baseline_env_cfg:G1FastSACBaselineEnvCfg_PLAY",
}


def test_all_fastwmr_tasks_are_registered() -> None:
    for task_id, expected_suffix in TASK_ENTRY_POINTS.items():
        spec = gym.spec(task_id)

        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"].endswith(expected_suffix)


def test_training_configs_share_the_physical_task() -> None:
    fastwmr_cfg = G1FastWMREnvCfg()
    baseline_cfg = G1FastSACBaselineEnvCfg()

    assert tuple(fastwmr_cfg.actions.joint_pos.joint_names) == G1_29DOF_JOINT_PATTERNS
    assert fastwmr_cfg.scene.robot.spawn.activate_contact_sensors
    assert baseline_cfg.actions.joint_pos == fastwmr_cfg.actions.joint_pos
    assert baseline_cfg.scene.robot == fastwmr_cfg.scene.robot
    assert baseline_cfg.scene.terrain == fastwmr_cfg.scene.terrain
    assert baseline_cfg.rewards == fastwmr_cfg.rewards
    assert baseline_cfg.terminations == fastwmr_cfg.terminations


def test_baseline_exposes_no_privileged_group() -> None:
    fastwmr_cfg = G1FastWMREnvCfg()
    baseline_cfg = G1FastSACBaselineEnvCfg()

    assert hasattr(fastwmr_cfg.observations, "policy")
    assert hasattr(fastwmr_cfg.observations, "privileged")
    assert hasattr(baseline_cfg.observations, "policy")
    assert not hasattr(baseline_cfg.observations, "privileged")


def test_play_configs_disable_policy_noise_and_pushes() -> None:
    for cfg in (G1FastWMREnvCfg_PLAY(), G1FastSACBaselineEnvCfg_PLAY()):
        assert not cfg.observations.policy.enable_corruption
        assert cfg.events.base_external_force_torque is None
        assert cfg.events.push_robot is None
        assert cfg.scene.terrain.terrain_generator is not None
        assert not cfg.scene.terrain.terrain_generator.curriculum
