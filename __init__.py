# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-G1-FastWMR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fastwmr_env_cfg:G1FastWMREnvCfg",
    },
)


gym.register(
    id="Isaac-Velocity-G1-FastSAC-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1FastSACBaselineEnvCfg",
    },
)


gym.register(
    id="Isaac-Velocity-G1-FastSAC-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1FastSACBaselineEnvCfg_PLAY",
    },
)


gym.register(
    id="Isaac-Velocity-G1-FastWMR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fastwmr_env_cfg:G1FastWMREnvCfg_PLAY",
    },
)
