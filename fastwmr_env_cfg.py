# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_CFG  # isort: skip

from .curriculum import FastWMRCurriculumCfg
from .observations import FastWMRObservationsCfg, G1_29DOF_JOINT_PATTERNS
from .randomization import (
    initialize_fastwmr_dr_buffers,
    randomize_and_record_friction,
    randomize_and_record_payload_mass,
    sample_apply_record_external_wrench,
)
from .rewards import FastSACMinimalRewardsCfg


@configclass
class G1FastWMREnvCfg(LocomotionVelocityRoughEnvCfg):
    """FastWMR G1 velocity task.

    This is the single environment config kept for FastWMR. It inherits
    IsaacLab's rough-terrain velocity base config because FastWMR is intended to
    train under terrain, friction, push, and payload variation, but the public
    task name is FastWMR.
    """

    rewards: FastSACMinimalRewardsCfg = FastSACMinimalRewardsCfg()
    observations: FastWMRObservationsCfg = FastWMRObservationsCfg()
    curriculum: FastWMRCurriculumCfg = FastWMRCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # biped yaw control is harder than quadruped — relax the per-episode-mean yaw
        # threshold to 0.8 rad/s (defaults work for quadrupeds).
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        # Scene
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Contact rewards and the privileged foot-contact target both require
        # PhysX contact reporters on the robot bodies.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"

        # Action, joint-position observation, and joint-velocity observation
        # must resolve against the same 29 body joints in articulation order.
        self.actions.joint_pos.joint_names = list(G1_29DOF_JOINT_PATTERNS)
        self.actions.joint_pos.preserve_order = False

        # Each FastWMR DR event owns sample -> physics application -> recording.
        # Disable the inherited terms because their internal samples are not
        # available to the privileged reconstruction target.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.initialize_fastwmr_dr_buffers = EventTerm(
            func=initialize_fastwmr_dr_buffers,
            mode="startup",
            params={"nominal_friction": 0.8},
        )
        self.events.randomize_fastwmr_friction = EventTerm(
            func=randomize_and_record_friction,
            mode="startup",
            params={
                "friction_range": (0.2, 1.5),
                "restitution": 0.0,
                "num_buckets": 64,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            },
        )
        self.events.randomize_fastwmr_payload = EventTerm(
            func=randomize_and_record_payload_mass,
            mode="startup",
            params={
                "payload_mass_range": (-5.0, 5.0),
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "min_mass": 1.0,
            },
        )
        self.events.base_external_force_torque = EventTerm(
            func=sample_apply_record_external_wrench,
            mode="reset",
            params={
                "force_range": (-50.0, 50.0),
                "torque_range": (-10.0, 10.0),
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            },
        )
        self.events.push_robot = None
        # G1 has precise initial pose — don't scale joint defaults randomly on reset
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"


@configclass
class G1FastWMREnvCfg_PLAY(G1FastWMREnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.randomize_fastwmr_friction.params["friction_range"] = (0.8, 0.8)
        self.events.randomize_fastwmr_payload.params["payload_mass_range"] = (0.0, 0.0)
        # Remove external disturbances while retaining nominal DR records.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.penalty_weights = None
