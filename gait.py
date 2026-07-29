# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Holosoma-style gait phase state for locomotion observations and rewards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def wrap_phase(angle: torch.Tensor) -> torch.Tensor:
    """Wrap phase angles to the half-open interval ``[-pi, pi)``."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def opposite_phase(left_phase: torch.Tensor) -> torch.Tensor:
    """Return the right-foot phase half a gait cycle from the left foot."""

    return wrap_phase(left_phase + math.pi)


class GaitPhaseCommand(CommandTerm):
    """Maintain synchronized left/right gait phases for every environment."""

    cfg: GaitPhaseCommandCfg

    def __init__(self, cfg: "GaitPhaseCommandCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        if cfg.gait_period <= 0.0:
            raise ValueError("gait_period must be positive.")
        mean_frequency = 1.0 / cfg.gait_period
        if cfg.frequency_randomization_width < 0.0:
            raise ValueError("frequency_randomization_width must be non-negative.")
        if cfg.frequency_randomization_width >= mean_frequency:
            raise ValueError(
                "frequency_randomization_width must be smaller than the mean gait frequency."
            )

        self.phase = torch.zeros((self.num_envs, 2), device=self.device)
        self.phase_offset = torch.zeros_like(self.phase)
        self.gait_frequency = torch.full(
            (self.num_envs, 1),
            mean_frequency,
            device=self.device,
        )
        self.phase_dt = torch.full(
            (self.num_envs, 1),
            2.0 * math.pi * env.step_dt * mean_frequency,
            device=self.device,
        )
        self._mean_frequency = mean_frequency

    @property
    def command(self) -> torch.Tensor:
        """Return left/right phases with shape ``(num_envs, 2)``."""

        return self.phase

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self._as_index_tensor(env_ids)
        if ids.numel() == 0:
            return

        # Phase offsets are episode state. Periodic command resampling changes
        # frequency only, matching Holosoma's locomotion command/gait coupling.
        reset_ids = ids[self.command_counter[ids] == 0]
        if reset_ids.numel() > 0:
            if self.cfg.randomize_phase:
                left = torch.empty(reset_ids.numel(), device=self.device).uniform_(
                    -math.pi,
                    math.pi,
                )
            else:
                left = torch.zeros(reset_ids.numel(), device=self.device)
            self.phase_offset[reset_ids, 0] = left
            self.phase_offset[reset_ids, 1] = opposite_phase(left)
            self.phase[reset_ids] = self.phase_offset[reset_ids]

        width = self.cfg.frequency_randomization_width
        if width > 0.0:
            frequency = torch.empty((ids.numel(), 1), device=self.device).uniform_(
                self._mean_frequency - width,
                self._mean_frequency + width,
            )
        else:
            frequency = torch.full(
                (ids.numel(), 1),
                self._mean_frequency,
                device=self.device,
            )
        self.gait_frequency[ids] = frequency
        self.phase_dt[ids] = 2.0 * math.pi * self._env.step_dt * frequency

    def _update_command(self) -> None:
        phase = (
            self._env.episode_length_buf.unsqueeze(1) * self.phase_dt
            + self.phase_offset
        )
        self.phase.copy_(wrap_phase(phase))

        velocity_command = self._env.command_manager.get_command(
            self.cfg.velocity_command_name
        )
        standing = torch.logical_and(
            torch.linalg.vector_norm(velocity_command[:, :2], dim=1) < 0.01,
            velocity_command[:, 2].abs() < 0.01,
        )
        if torch.any(standing):
            self.phase[standing] = self.cfg.stand_phase_value

    def _as_index_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[
                env_ids
            ]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class GaitPhaseCommandCfg(CommandTermCfg):
    """Configuration for the shared gait phase clock."""

    class_type: type[GaitPhaseCommand] = GaitPhaseCommand
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    gait_period: float = 1.0
    frequency_randomization_width: float = 0.2
    randomize_phase: bool = True
    stand_phase_value: float = math.pi
    velocity_command_name: str = "base_velocity"
    debug_vis: bool = False
    cmd_kind: str | None = "command/gait_phase"
    element_names: list[str] | None = ["left_phase", "right_phase"]
