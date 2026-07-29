"""Tests for the shared Holosoma-style gait phase contract."""

import math

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_env_cfg import (
    G1FastWMREnvCfg,
    G1FastWMREnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.gait import (
    GaitPhaseCommand,
    opposite_phase,
    wrap_phase,
)


def test_gait_command_matches_holosoma_g1_defaults() -> None:
    train_cfg = G1FastWMREnvCfg().commands.gait_phase
    play_cfg = G1FastWMREnvCfg_PLAY().commands.gait_phase

    assert train_cfg.class_type is GaitPhaseCommand
    assert train_cfg.gait_period == 1.0
    assert train_cfg.frequency_randomization_width == 0.2
    assert train_cfg.randomize_phase
    assert train_cfg.resampling_time_range == (10.0, 10.0)
    assert not play_cfg.randomize_phase
    assert play_cfg.frequency_randomization_width == 0.0


def test_opposite_phase_stays_half_a_cycle_apart() -> None:
    left = torch.tensor([-math.pi, -1.0, 0.0, 1.0, math.pi])
    right = opposite_phase(left)

    difference = wrap_phase(right - left)
    torch.testing.assert_close(difference.abs(), torch.full_like(left, math.pi))
    assert torch.all(right >= -math.pi)
    assert torch.all(right < math.pi)
