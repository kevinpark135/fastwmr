# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""FastSAC baseline task sharing the FastWMR G1 environment.

The baseline changes only the observation schema: it receives deployable 96D
proprioception without the privileged reconstruction target. Terrain, robot,
actions, rewards, events, commands, and termination rules are inherited from
the corresponding FastWMR training/play configs.
"""

from isaaclab.utils.configclass import configclass

from .fastwmr_env_cfg import G1FastWMREnvCfg, G1FastWMREnvCfg_PLAY
from .observations import FastSACObservationsCfg


@configclass
class G1FastSACBaselineEnvCfg(G1FastWMREnvCfg):
    """Training config for the policy-only FastSAC baseline."""

    observations: FastSACObservationsCfg = FastSACObservationsCfg()


@configclass
class G1FastSACBaselineEnvCfg_PLAY(G1FastWMREnvCfg_PLAY):
    """Evaluation config for the policy-only FastSAC baseline."""

    observations: FastSACObservationsCfg = FastSACObservationsCfg()
