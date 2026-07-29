# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomotion-specific robot and terrain presets.

The stock Isaac Lab 29-DoF G1 preset is tuned for locomanipulation.  FastWMR
instead follows Holosoma's G1 locomotion pose, actuator gains, limits, and
terrain mix while retaining the orientation required by Isaac Lab's USD asset.
"""

from copy import deepcopy

import isaaclab.terrains as terrain_gen
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab_assets import G1_29DOF_CFG


G1_LOCOMOTION_ACTION_SCALE = 0.25
"""Joint-position residual scale used by Holosoma's G1 locomotion policy."""

G1_LOCOMOTION_TERMINATION_BODY_NAMES = (
    "pelvis",
    ".*_hip_.*_link",
    ".*_shoulder_.*_link",
)
"""Bodies whose non-foot ground contact terminates a locomotion episode."""

G1_LOCOMOTION_TERMINATION_BODY_GROUPS = {
    "pelvis_contact": ("pelvis",),
    "hip_contact": (".*_hip_.*_link",),
    "shoulder_contact": (".*_shoulder_.*_link",),
}
"""Named contact groups used for both termination and fall diagnostics."""


G1_LOCOMOTION_CFG = G1_29DOF_CFG.copy()
G1_LOCOMOTION_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8),
    # The Isaac Lab USD is rotated relative to Holosoma's URDF.
    rot=G1_29DOF_CFG.init_state.rot,
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_ankle_roll_joint": 0.0,
        "waist_.*_joint": 0.0,
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.6,
        "left_wrist_.*_joint": 0.0,
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.6,
        "right_wrist_.*_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)
G1_LOCOMOTION_CFG.actuators = {
    "locomotion_pd": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
            "waist_.*_joint",
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
        ],
        effort_limit={
            ".*_hip_pitch_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_yaw_joint": 88.0,
            ".*_knee_joint": 139.0,
            ".*_ankle_.*_joint": 50.0,
            "waist_yaw_joint": 88.0,
            "waist_(roll|pitch)_joint": 50.0,
            ".*_(shoulder|elbow).*_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_(pitch|yaw)_joint": 5.0,
        },
        velocity_limit={
            ".*_hip_(pitch|yaw)_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_knee_joint": 20.0,
            ".*_ankle_.*_joint": 37.0,
            "waist_yaw_joint": 32.0,
            "waist_(roll|pitch)_joint": 37.0,
            ".*_(shoulder|elbow).*_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_(pitch|yaw)_joint": 22.0,
        },
        stiffness={
            ".*_hip_(pitch|yaw)_joint": 40.179238471,
            ".*_hip_roll_joint": 99.098427777,
            ".*_knee_joint": 99.098427777,
            ".*_ankle_.*_joint": 28.501246196,
            "waist_yaw_joint": 40.179238471,
            "waist_(roll|pitch)_joint": 28.501246196,
            ".*_(shoulder|elbow).*_joint": 14.250623098,
            ".*_wrist_roll_joint": 14.250623098,
            ".*_wrist_(pitch|yaw)_joint": 16.778327481,
        },
        damping={
            ".*_hip_(pitch|yaw)_joint": 2.557889765,
            ".*_hip_roll_joint": 6.308801854,
            ".*_knee_joint": 6.308801854,
            ".*_ankle_.*_joint": 1.814445687,
            "waist_yaw_joint": 2.557889765,
            "waist_(roll|pitch)_joint": 1.814445687,
            ".*_(shoulder|elbow).*_joint": 0.907222843,
            ".*_wrist_roll_joint": 0.907222843,
            ".*_wrist_(pitch|yaw)_joint": 1.068141502,
        },
        armature={
            ".*_hip_(pitch|yaw)_joint": 0.010177520,
            ".*_hip_roll_joint": 0.025101925,
            ".*_knee_joint": 0.025101925,
            ".*_ankle_.*_joint": 0.007219450,
            "waist_yaw_joint": 0.010177520,
            "waist_(roll|pitch)_joint": 0.007219450,
            ".*_(shoulder|elbow).*_joint": 0.003609725,
            ".*_wrist_roll_joint": 0.003609725,
            ".*_wrist_(pitch|yaw)_joint": 0.00425,
        },
        friction=0.0,
    ),
    # The Isaac Lab USD includes 14 coupled hand joints outside the 29D policy
    # action. Keep their stock actuator so no articulation joint is left free.
    "hands": deepcopy(G1_29DOF_CFG.actuators["hands"]),
}


G1_LOCOMOTION_TERRAINS_CFG = deepcopy(ROUGH_TERRAINS_CFG)
G1_LOCOMOTION_TERRAINS_CFG.sub_terrains = {
    # Holosoma's G1 mix: flat 20%, rough 60%, low obstacles 20%.
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
    "rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.6,
        noise_range=(-0.05, 0.0),
        noise_step=0.005,
        border_width=0.25,
    ),
    "low_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.2,
        grid_width=0.45,
        grid_height_range=(0.0, 0.05),
        platform_width=2.0,
    ),
}


__all__ = [
    "G1_LOCOMOTION_ACTION_SCALE",
    "G1_LOCOMOTION_CFG",
    "G1_LOCOMOTION_TERMINATION_BODY_NAMES",
    "G1_LOCOMOTION_TERMINATION_BODY_GROUPS",
    "G1_LOCOMOTION_TERRAINS_CFG",
]
