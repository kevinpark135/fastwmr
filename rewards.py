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
DEFAULT_HEIGHT_SENSOR_CFG = SceneEntityCfg("height_scanner")


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


def expected_foot_height(
    phase: torch.Tensor,
    swing_height: float,
) -> torch.Tensor:
    """Map gait phase to Holosoma's cubic Bézier foot-height profile."""

    if swing_height <= 0.0:
        raise ValueError("swing_height must be positive.")
    normalized_phase = (phase + torch.pi) / (2.0 * torch.pi)

    def cubic_bezier(start: torch.Tensor, end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        blend = x**3 + 3.0 * x**2 * (1.0 - x)
        return start + (end - start) * blend

    zeros = torch.zeros_like(normalized_phase)
    peaks = torch.full_like(normalized_phase, swing_height)
    rising = cubic_bezier(zeros, peaks, 2.0 * normalized_phase)
    falling = cubic_bezier(peaks, zeros, 2.0 * normalized_phase - 1.0)
    return torch.where(normalized_phase <= 0.5, rising, falling)


def terrain_relative_foot_heights(
    foot_positions_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    fallback_terrain_height: torch.Tensor,
) -> torch.Tensor:
    """Estimate each foot's terrain clearance from its nearest scanner ray."""

    if foot_positions_w.ndim != 3 or foot_positions_w.shape[1:] != (2, 3):
        raise ValueError("foot_positions_w must have shape (N, 2, 3).")
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[0] != foot_positions_w.shape[0] or ray_hits_w.shape[-1] != 3:
        raise ValueError("ray_hits_w must have shape (N, R, 3).")
    if fallback_terrain_height.shape != (foot_positions_w.shape[0],):
        raise ValueError("fallback_terrain_height must have shape (N,).")

    finite_hits = torch.isfinite(ray_hits_w).all(dim=-1)
    safe_hit_xy = torch.where(
        finite_hits.unsqueeze(-1),
        ray_hits_w[..., :2],
        torch.zeros_like(ray_hits_w[..., :2]),
    )
    distance_squared = torch.sum(
        torch.square(
            foot_positions_w[:, :, None, :2] - safe_hit_xy[:, None, :, :]
        ),
        dim=-1,
    )
    distance_squared.masked_fill_(~finite_hits[:, None, :], torch.inf)
    nearest_ray = distance_squared.argmin(dim=-1)
    terrain_height = torch.gather(ray_hits_w[..., 2], 1, nearest_ray)
    terrain_height = torch.where(
        finite_hits.any(dim=-1, keepdim=True),
        terrain_height,
        fallback_terrain_height.unsqueeze(1),
    )
    return foot_positions_w[..., 2] - terrain_height


def feet_phase_exp(
    env: "ManagerBasedRLEnv",
    gait_command_name: str,
    swing_height: float,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = DEFAULT_FEET_CFG,
    height_sensor_cfg: SceneEntityCfg = DEFAULT_HEIGHT_SENSOR_CFG,
) -> torch.Tensor:
    """Reward terrain-relative foot heights that follow the shared gait phase."""

    if swing_height <= 0.0 or tracking_sigma <= 0.0:
        raise ValueError("swing_height and tracking_sigma must be positive.")
    asset = env.scene[asset_cfg.name]
    foot_positions_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    if foot_positions_w.shape[1] != 2:
        raise ValueError(f"Expected two feet, got {foot_positions_w.shape[1]} bodies.")
    height_sensor = env.scene[height_sensor_cfg.name]
    foot_heights = terrain_relative_foot_heights(
        foot_positions_w,
        height_sensor.data.ray_hits_w.torch,
        env.scene.env_origins[:, 2],
    )

    phase = env.command_manager.get_command(gait_command_name)
    if phase.shape != foot_heights.shape:
        raise ValueError(
            f"Gait phase must have shape {tuple(foot_heights.shape)}, got {tuple(phase.shape)}."
        )
    target_heights = expected_foot_height(phase, swing_height)
    total_error = torch.square(foot_heights - target_heights).sum(dim=-1)
    return torch.exp(-total_error / tracking_sigma)


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
    feet_phase = RewTerm(
        func=feet_phase_exp,
        weight=5.0,
        params={
            "gait_command_name": "gait_phase",
            "swing_height": 0.09,
            "tracking_sigma": 0.008,
            "asset_cfg": DEFAULT_FEET_CFG,
            "height_sensor_cfg": DEFAULT_HEIGHT_SENSOR_CFG,
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
    "feet_phase",
    "joint_pose",
    "close_feet",
    "feet_orientation",
    "alive",
)
"""Canonical reward order used by config tests and reward logging."""
