# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""FastWMR policy observations and privileged reconstruction targets.

Term declaration order is part of the algorithm interface. Keep it synchronized
with :mod:`algorithm.config`; changing an item here must also change the
interface-contract tests and the decoder output layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .algorithm.config import DEFAULT_INTERFACE_CFG
from .randomization import (
    FASTWMR_FRICTION_ATTR,
    FASTWMR_PAYLOAD_MASS_ATTR,
    FASTWMR_PUSH_FORCE_TORQUES_ATTR,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


G1_29DOF_JOINT_PATTERNS = (
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
)
"""Regexes resolving to the 29 body joints in the G1 29-DoF asset."""

FOOT_BODY_PATTERNS = (".*_ankle_roll_link",)

DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
DEFAULT_CONTROLLED_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(G1_29DOF_JOINT_PATTERNS))
DEFAULT_CONTACT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_PATTERNS))


def _num_envs(env: "ManagerBasedEnv") -> int:
    return int(env.num_envs if hasattr(env, "num_envs") else env.episode_length_buf.shape[0])


def _device(env: "ManagerBasedEnv") -> torch.device:
    return torch.device(env.device) if hasattr(env, "device") else env.episode_length_buf.device


def _privileged_buffer(env: "ManagerBasedEnv", name: str, width: int) -> torch.Tensor:
    """Read a per-environment DR record, returning zeros in isolated tests.

    The environment startup event creates these buffers before observations are
    evaluated. The fallback keeps standalone observation-function tests usable
    without constructing an IsaacLab event manager.
    """

    value = getattr(env, name, None)
    if value is None:
        return torch.zeros((_num_envs(env), width), device=_device(env), dtype=torch.float32)

    tensor = torch.as_tensor(value, device=_device(env), dtype=torch.float32)
    if tensor.numel() != _num_envs(env) * width:
        raise ValueError(
            f"{name} must contain {_num_envs(env) * width} values for shape "
            f"({_num_envs(env)}, {width}), got shape {tuple(tensor.shape)}."
        )
    return tensor.reshape(_num_envs(env), width)


def privileged_friction(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return the scalar friction coefficient applied to each environment."""

    return _privileged_buffer(env, FASTWMR_FRICTION_ATTR, 1)


def privileged_payload_mass(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return the randomized additional payload mass in kilograms."""

    return _privileged_buffer(env, FASTWMR_PAYLOAD_MASS_ATTR, 1)


def privileged_push_force_torque(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return applied body-frame force xyz followed by torque xyz."""

    return _privileged_buffer(env, FASTWMR_PUSH_FORCE_TORQUES_ATTR, 6)


def privileged_foot_contacts(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return a two-bit left/right foot contact target as float32."""

    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    if forces is None:
        raise RuntimeError("The contact sensor must retain force history for FastWMR contact targets.")
    forces_torch = forces.torch[:, :, sensor_cfg.body_ids, :]
    contacts = forces_torch.norm(dim=-1).amax(dim=1) > threshold
    if contacts.shape[-1] != 2:
        raise ValueError(f"Expected exactly two foot contact sensors, got shape {tuple(contacts.shape)}.")
    return contacts.to(dtype=torch.float32)


@configclass
class FastWMRObservationsCfg:
    """Observation groups exposed by the FastWMR environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable proprioception ``o_t`` in the canonical 96D order."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": DEFAULT_CONTROLLED_JOINT_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": DEFAULT_CONTROLLED_JOINT_CFG},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Simulator-only reconstruction target ``s_t`` in canonical 13D order."""

        # Continuous fields must precede discrete fields for decoder loss slicing.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        friction = ObsTerm(func=privileged_friction)
        payload_mass = ObsTerm(func=privileged_payload_mass)
        push_force_torque = ObsTerm(func=privileged_push_force_torque)
        foot_contacts = ObsTerm(
            func=privileged_foot_contacts,
            params={"sensor_cfg": DEFAULT_CONTACT_SENSOR_CFG, "threshold": 1.0},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class FastSACObservationsCfg:
    """Policy-only observation schema for the FastSAC control baseline."""

    policy: FastWMRObservationsCfg.PolicyCfg = FastWMRObservationsCfg.PolicyCfg()


POLICY_TERM_NAMES = DEFAULT_INTERFACE_CFG.policy_observation_layout.names
PRIVILEGED_TERM_NAMES = DEFAULT_INTERFACE_CFG.reconstruction_layout.names
"""Public order constants used by config checks and observation tests."""
