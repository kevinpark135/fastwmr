# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,
    LocomotionVelocityRoughEnvCfg,
    TerminationsCfg,
)

from .curriculum import FastWMRCurriculumCfg
from .gait import GaitPhaseCommandCfg
from .g1_locomotion import (
    G1_LOCOMOTION_ACTION_SCALE,
    G1_LOCOMOTION_CFG,
    G1_LOCOMOTION_TERMINATION_BODY_GROUPS,
    G1_LOCOMOTION_TERRAINS_CFG,
)
from .observations import FastWMRObservationsCfg, G1_29DOF_JOINT_PATTERNS
from .randomization import (
    initialize_fastwmr_dr_buffers,
    randomize_and_record_friction,
    randomize_and_record_payload_mass,
    sample_apply_record_external_wrench,
)
from .rewards import FastSACMinimalRewardsCfg


@configclass
class FastWMRCommandsCfg(CommandsCfg):
    """Velocity command plus the gait phase shared by policy and reward."""

    gait_phase = GaitPhaseCommandCfg()


@configclass
class FastWMRTerminationsCfg(TerminationsCfg):
    """Split illegal contacts so each G1 fall source is observable."""

    base_contact = None
    pelvis_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(G1_LOCOMOTION_TERMINATION_BODY_GROUPS["pelvis_contact"]),
            ),
            "threshold": 1.0,
        },
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(G1_LOCOMOTION_TERMINATION_BODY_GROUPS["hip_contact"]),
            ),
            "threshold": 1.0,
        },
    )
    shoulder_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(G1_LOCOMOTION_TERMINATION_BODY_GROUPS["shoulder_contact"]),
            ),
            "threshold": 1.0,
        },
    )


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
    commands: FastWMRCommandsCfg = FastWMRCommandsCfg()
    terminations: FastWMRTerminationsCfg = FastWMRTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # biped yaw control is harder than quadruped — relax the per-episode-mean yaw
        # threshold to 0.8 rad/s (defaults work for quadrupeds).
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        # Scene
        self.scene.robot = G1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Contact rewards and the privileged foot-contact target both require
        # PhysX contact reporters on the robot bodies.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.scene.terrain.terrain_generator = deepcopy(G1_LOCOMOTION_TERRAINS_CFG)
        # Begin at the easiest generated row. Isaac Lab's terrain curriculum
        # promotes successful environments instead of spawning G1 at level 0-5.
        self.scene.terrain.max_init_terrain_level = 0

        # Action, joint-position observation, and joint-velocity observation
        # must resolve against the same 29 body joints in articulation order.
        self.actions.joint_pos.joint_names = list(G1_29DOF_JOINT_PATTERNS)
        self.actions.joint_pos.preserve_order = False
        self.actions.joint_pos.scale = G1_LOCOMOTION_ACTION_SCALE

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
                "friction_range": (0.5, 1.25),
                "restitution": 0.0,
                "num_buckets": 64,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            },
        )
        self.events.randomize_fastwmr_payload = EventTerm(
            func=randomize_and_record_payload_mass,
            mode="startup",
            params={
                "payload_mass_range": (-1.0, 3.0),
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "min_mass": 1.0,
            },
        )
        self.events.base_external_force_torque = EventTerm(
            func=sample_apply_record_external_wrench,
            mode="reset",
            params={
                "force_range": (-20.0, 20.0),
                "torque_range": (-5.0, 5.0),
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "warmup_steps": 500,
                "ramp_steps": 1000,
            },
        )
        self.events.push_robot = None
        # G1 has precise initial pose — don't scale joint defaults randomly on reset
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.2

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
        self.commands.gait_phase.randomize_phase = False
        self.commands.gait_phase.frequency_randomization_width = 0.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.randomize_fastwmr_friction.params["friction_range"] = (0.8, 0.8)
        self.events.randomize_fastwmr_payload.params["payload_mass_range"] = (0.0, 0.0)
        # Remove external disturbances while retaining nominal DR records.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.penalty_weights = None
