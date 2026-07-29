"""Tests for the minimal FastSAC/FastWMR reward contract."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_env_cfg import (
    G1FastWMREnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.rewards import (
    FASTSAC_REWARD_TERM_NAMES,
    G1_29DOF_POSE_WEIGHTS,
    FastSACMinimalRewardsCfg,
    base_stability_l2,
    close_feet_xy,
    expected_foot_height,
    feet_phase_exp,
    feet_orientation_error,
    terrain_relative_foot_heights,
    weighted_joint_pose_l2,
)


def _term_names(group: object) -> tuple[str, ...]:
    return tuple(name for name, value in vars(group).items() if not name.startswith("_") and value is not None)


def test_minimal_reward_contract_has_nine_terms() -> None:
    rewards = FastSACMinimalRewardsCfg()

    assert _term_names(rewards) == FASTSAC_REWARD_TERM_NAMES
    assert len(FASTSAC_REWARD_TERM_NAMES) == 9
    assert rewards.track_lin_vel.weight == 2.0
    assert rewards.track_ang_vel.weight == 1.5
    assert rewards.base_stability.weight == -1.0
    assert rewards.base_stability.params["orientation_scale"] == 10.0
    assert rewards.action_rate.weight == -2.0
    assert rewards.feet_phase.weight == 5.0
    assert rewards.joint_pose.weight == -0.5
    assert rewards.close_feet.weight == -10.0
    assert rewards.feet_orientation.weight == -5.0
    assert rewards.alive.weight == 10.0


def test_reward_functions_and_holosoma_parameters_are_wired() -> None:
    rewards = G1FastWMREnvCfg().rewards

    assert rewards.base_stability.func is base_stability_l2
    assert rewards.feet_phase.func is feet_phase_exp
    assert rewards.feet_phase.params["gait_command_name"] == "gait_phase"
    assert rewards.feet_phase.params["swing_height"] == 0.09
    assert rewards.feet_phase.params["tracking_sigma"] == 0.008
    assert rewards.feet_phase.params["height_sensor_cfg"].name == "height_scanner"
    assert rewards.joint_pose.func is weighted_joint_pose_l2
    assert rewards.joint_pose.params["pose_weights"] == G1_29DOF_POSE_WEIGHTS
    assert rewards.close_feet.func is close_feet_xy
    assert rewards.close_feet.params["threshold"] == 0.15
    assert rewards.feet_orientation.func is feet_orientation_error


def test_pose_weights_match_the_fixed_29dof_action_order() -> None:
    assert len(G1_29DOF_POSE_WEIGHTS) == 29
    assert G1_29DOF_POSE_WEIGHTS[0:2] == (0.01, 0.01)
    assert G1_29DOF_POSE_WEIGHTS[2] == 50.0
    assert G1_29DOF_POSE_WEIGHTS[13:15] == (5.0, 5.0)
    assert G1_29DOF_POSE_WEIGHTS[17:19] == (5.0, 5.0)


def test_holosoma_bezier_foot_profile_rises_and_falls() -> None:
    phase = torch.tensor([-torch.pi, 0.0, torch.pi])

    height = expected_foot_height(phase, 0.09)

    torch.testing.assert_close(height, torch.tensor([0.0, 0.09, 0.0]))


def test_terrain_relative_foot_height_uses_nearest_finite_ray() -> None:
    feet = torch.tensor([[[0.02, 0.0, 0.10], [0.98, 0.0, 0.20]]])
    ray_hits = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [float("inf")] * 3]]
    )

    heights = terrain_relative_foot_heights(feet, ray_hits, torch.tensor([0.0]))

    torch.testing.assert_close(heights, torch.tensor([[0.10, 0.10]]))
