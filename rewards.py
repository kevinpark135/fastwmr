# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal FastSAC locomotion rewards for the G1 FastWMR task.

The term set follows the FastSAC paper and Holosoma's G1-29DoF FastSAC
configuration. Holosoma's separate base angular-velocity and orientation
penalties are combined into one base-stability term so the task stays below ten
active rewards. Reward weights are expressed per second; IsaacLab multiplies
each term by the environment step duration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .observations import G1_29DOF_JOINT_PATTERNS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


G1_29DOF_POSE_WEIGHTS = (
    0.01,
    0.01,
    50.0,
    1.0,
    1.0,
    50.0,
    5.0,
    5.0,
    50.0,
    0.01,
    0.01,
    50.0,
    50.0,
    5.0,
    5.0,
    50.0,
    50.0,
    5.0,
    5.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
)
"""Holosoma pose weights reordered into IsaacLab's resolved 29-joint order."""

DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
DEFAULT_CONTROLLED_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(G1_29DOF_JOINT_PATTERNS))
DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
DEFAULT_CONTACT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])


def base_stability_l2(
    env: "ManagerBasedRLEnv",
    orientation_scale: float,
    asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity and tilt in one minimal term."""

    asset = env.scene[asset_cfg.name]
    angular_error = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=-1)
    orientation_error = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=-1)
    return angular_error + orientation_scale * orientation_error


def weighted_joint_pose_l2(
    env: "ManagerBasedRLEnv",
    pose_weights: tuple[float, ...],
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
) -> torch.Tensor:
    """Penalize weighted deviation from the configured default G1 pose."""

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != len(pose_weights):
        raise ValueError(
            f"pose_weights has {len(pose_weights)} values, but asset_cfg resolved "
            f"{joint_pos.shape[-1]} joints."
        )
    weights = joint_pos.new_tensor(pose_weights)
    return torch.sum(torch.square(joint_pos - default_joint_pos) * weights, dim=-1)


def swing_foot_height_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = DEFAULT_FEET_CFG,
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
) -> torch.Tensor:
    """Reward swing feet for clearing the support foot by the target height.

    Holosoma uses a phase-conditioned foot-height trajectory. FastWMR's fixed
    96D policy observation intentionally has no gait phase, so this equivalent
    uses foot contact and relative kinematics without an exteroceptive terrain
    sensor or a hidden clock.
    """

    if target_height <= 0.0 or std <= 0.0:
        raise ValueError("target_height and std must be positive.")

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]

    force_history = contact_sensor.data.net_forces_w_history
    if force_history is None:
        raise RuntimeError("Foot-height reward requires contact-force history.")
    contacts = force_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1) > 1.0
    stance = contacts.to(dtype=foot_z.dtype)
    stance_count = stance.sum(dim=-1, keepdim=True)
    support_z = (foot_z * stance).sum(dim=-1, keepdim=True) / stance_count.clamp_min(1.0)
    support_z = torch.where(stance_count > 0, support_z, foot_z.amin(dim=-1, keepdim=True))
    height_error = torch.square(foot_z - support_z - target_height)

    swing = ~contacts
    per_foot_reward = torch.exp(-height_error / std) * swing
    swing_count = swing.sum(dim=-1)
    tracked_swing = per_foot_reward.sum(dim=-1) / swing_count.clamp_min(1)

    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :3], dim=-1) > 0.1
    return torch.where(moving & (swing_count > 0), tracked_swing, torch.ones_like(tracked_swing))


def close_feet_xy(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Penalize lateral foot separation below the crossing-safe threshold."""

    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    if feet_pos_w.shape[1] != 2:
        raise ValueError(f"Expected two feet, got {feet_pos_w.shape[1]} bodies.")
    relative_w = feet_pos_w[:, 0] - feet_pos_w[:, 1]
    relative_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), relative_w)
    return (relative_yaw[:, 1].abs() < threshold).to(dtype=torch.float32)


def feet_orientation_error(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Penalize both feet for tilting away from the gravity-aligned plane."""

    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    if foot_quat_w.shape[1] != 2:
        raise ValueError(f"Expected two feet, got {foot_quat_w.shape[1]} bodies.")
    gravity_w = foot_quat_w.new_zeros((*foot_quat_w.shape[:-1], 3))
    gravity_w[..., 2] = -1.0
    gravity_b = quat_apply_inverse(foot_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3))
    tilt = torch.linalg.vector_norm(gravity_b[:, :2], dim=-1).reshape(foot_quat_w.shape[:2])
    return tilt.sum(dim=-1)


@configclass
class FastSACMinimalRewardsCfg:
    """Nine-term G1 reward shared by FastSAC and FastWMR training."""

    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    base_stability = RewTerm(
        func=base_stability_l2,
        weight=-1.0,
        params={"orientation_scale": 10.0, "asset_cfg": DEFAULT_ROBOT_CFG},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2.0)
    swing_foot_height = RewTerm(
        func=swing_foot_height_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "target_height": 0.09,
            "std": 0.008,
            "asset_cfg": DEFAULT_FEET_CFG,
            "sensor_cfg": DEFAULT_CONTACT_SENSOR_CFG,
        },
    )
    joint_pose = RewTerm(
        func=weighted_joint_pose_l2,
        weight=-0.5,
        params={"pose_weights": G1_29DOF_POSE_WEIGHTS, "asset_cfg": DEFAULT_CONTROLLED_JOINT_CFG},
    )
    close_feet = RewTerm(
        func=close_feet_xy,
        weight=-10.0,
        params={"threshold": 0.15, "asset_cfg": DEFAULT_FEET_CFG},
    )
    feet_orientation = RewTerm(
        func=feet_orientation_error,
        weight=-5.0,
        params={"asset_cfg": DEFAULT_FEET_CFG},
    )
    alive = RewTerm(func=mdp.is_alive, weight=10.0)


FASTSAC_REWARD_TERM_NAMES = (
    "track_lin_vel",
    "track_ang_vel",
    "base_stability",
    "action_rate",
    "swing_foot_height",
    "joint_pose",
    "close_feet",
    "feet_orientation",
    "alive",
)
"""Canonical reward order used by config tests and reward logging."""
