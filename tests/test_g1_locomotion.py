"""Contracts for the Holosoma-aligned G1 locomotion preset."""

import pytest

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_env_cfg import (
    G1FastWMREnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.g1_locomotion import (
    G1_LOCOMOTION_ACTION_SCALE,
    G1_LOCOMOTION_CFG,
    G1_LOCOMOTION_TERMINATION_BODY_NAMES,
    G1_LOCOMOTION_TERRAINS_CFG,
)


def test_locomotion_initial_pose_matches_holosoma_g1() -> None:
    init = G1_LOCOMOTION_CFG.init_state

    assert init.pos == (0.0, 0.0, 0.8)
    assert init.joint_pos[".*_hip_pitch_joint"] == pytest.approx(-0.312)
    assert init.joint_pos[".*_knee_joint"] == pytest.approx(0.669)
    assert init.joint_pos[".*_ankle_pitch_joint"] == pytest.approx(-0.363)
    assert init.joint_pos["left_shoulder_roll_joint"] == pytest.approx(0.2)
    assert init.joint_pos["right_shoulder_roll_joint"] == pytest.approx(-0.2)
    assert init.joint_pos["left_elbow_joint"] == pytest.approx(0.6)
    assert init.joint_pos["right_elbow_joint"] == pytest.approx(0.6)


def test_locomotion_pd_gains_and_effort_limits_match_holosoma_g1() -> None:
    actuator = G1_LOCOMOTION_CFG.actuators["locomotion_pd"]

    assert set(G1_LOCOMOTION_CFG.actuators) == {"locomotion_pd", "hands"}
    assert actuator.stiffness[".*_hip_(pitch|yaw)_joint"] == pytest.approx(40.179238471)
    assert actuator.stiffness[".*_hip_roll_joint"] == pytest.approx(99.098427777)
    assert actuator.stiffness[".*_knee_joint"] == pytest.approx(99.098427777)
    assert actuator.stiffness[".*_ankle_.*_joint"] == pytest.approx(28.501246196)
    assert actuator.stiffness[".*_(shoulder|elbow).*_joint"] == pytest.approx(14.250623098)
    assert actuator.damping[".*_knee_joint"] == pytest.approx(6.308801854)
    assert actuator.damping[".*_ankle_.*_joint"] == pytest.approx(1.814445687)
    assert actuator.effort_limit[".*_knee_joint"] == pytest.approx(139.0)
    assert actuator.effort_limit[".*_wrist_(pitch|yaw)_joint"] == pytest.approx(5.0)


def test_environment_applies_locomotion_action_and_contact_contracts() -> None:
    cfg = G1FastWMREnvCfg()
    termination_sensor = cfg.terminations.base_contact.params["sensor_cfg"]

    assert cfg.actions.joint_pos.scale == pytest.approx(G1_LOCOMOTION_ACTION_SCALE)
    assert tuple(termination_sensor.body_names) == G1_LOCOMOTION_TERMINATION_BODY_NAMES
    assert cfg.terminations.base_contact.params["threshold"] == pytest.approx(1.0)


def test_environment_starts_with_holosoma_style_easy_mix() -> None:
    cfg = G1FastWMREnvCfg()
    terrains = cfg.scene.terrain.terrain_generator.sub_terrains

    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert cfg.commands.base_velocity.rel_standing_envs == pytest.approx(0.2)
    assert set(terrains) == {"flat", "rough", "low_obstacles"}
    assert terrains["flat"].proportion == pytest.approx(0.2)
    assert terrains["rough"].proportion == pytest.approx(0.6)
    assert terrains["low_obstacles"].proportion == pytest.approx(0.2)
    assert G1_LOCOMOTION_TERRAINS_CFG.sub_terrains["rough"].noise_range == (-0.05, 0.0)
