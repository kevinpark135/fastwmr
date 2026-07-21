# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-stability curriculum for the FastSAC minimal penalties.

Physical domain randomization remains at full configured strength. Only the
selected negative reward weights are scaled, so the curriculum cannot silently
change the estimator's privileged reconstruction targets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PENALTY_TARGET_WEIGHTS = {
    "base_stability": -1.0,
    "action_rate": -2.0,
    "joint_pose": -0.5,
    "close_feet": -10.0,
    "feet_orientation": -5.0,
}
"""Final penalty weights shared by FastSAC and FastWMR."""

DEFAULT_PENALTY_SCALES = (0.1, 0.3, 0.6, 1.0)
DEFAULT_EPISODE_LENGTH_THRESHOLDS = (0.25, 0.5, 0.75)
PENALTY_CURRICULUM_STATE_ATTR = "fastwmr_penalty_curriculum_state"


def _validate_schedule(
    scales: Sequence[float],
    episode_length_thresholds: Sequence[float],
    ema_decay: float,
    min_completed_episodes: int,
) -> None:
    if len(scales) < 1 or len(episode_length_thresholds) != len(scales) - 1:
        raise ValueError("Penalty curriculum requires one fewer threshold than scales.")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in scales):
        raise ValueError("Penalty scales must be finite values in (0, 1].")
    if any(left >= right for left, right in zip(scales, scales[1:])):
        raise ValueError("Penalty scales must be strictly increasing.")
    if scales[-1] != 1.0:
        raise ValueError("The final penalty scale must equal one.")
    if any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in episode_length_thresholds
    ):
        raise ValueError("Episode-length thresholds must be finite values in (0, 1].")
    if any(
        left >= right
        for left, right in zip(episode_length_thresholds, episode_length_thresholds[1:])
    ):
        raise ValueError("Episode-length thresholds must be strictly increasing.")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1).")
    if min_completed_episodes <= 0:
        raise ValueError("min_completed_episodes must be positive.")


def _resolve_env_ids(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | slice,
) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _apply_penalty_scale(
    env: "ManagerBasedRLEnv",
    target_weights: Mapping[str, float],
    scale: float,
) -> None:
    for term_name, target_weight in target_weights.items():
        if not math.isfinite(target_weight) or target_weight >= 0.0:
            raise ValueError(f"Penalty target {term_name!r} must have a finite negative weight.")
        term_cfg = copy(env.reward_manager.get_term_cfg(term_name))
        term_cfg.weight = target_weight * scale
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def penalty_weight_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | slice,
    *,
    target_weights: Mapping[str, float] = PENALTY_TARGET_WEIGHTS,
    scales: Sequence[float] = DEFAULT_PENALTY_SCALES,
    episode_length_thresholds: Sequence[float] = DEFAULT_EPISODE_LENGTH_THRESHOLDS,
    ema_decay: float = 0.9,
    min_completed_episodes: int = 64,
) -> dict[str, float | int]:
    """Increase penalty weights as completed episode lengths become stable.

    IsaacLab calls curriculum terms immediately before resetting completed
    environments, while their previous episode lengths are still available.
    The scale never decreases, avoiding reward non-stationarity after a harder
    level has been reached.
    """

    _validate_schedule(scales, episode_length_thresholds, ema_decay, min_completed_episodes)
    ids = _resolve_env_ids(env, env_ids)
    state = getattr(env, PENALTY_CURRICULUM_STATE_ATTR, None)
    if state is None:
        state = {
            "level": 0,
            "scale": float(scales[0]),
            "episode_length_ema": 0.0,
            "completed_episodes": 0,
        }
        setattr(env, PENALTY_CURRICULUM_STATE_ATTR, state)
        _apply_penalty_scale(env, target_weights, float(scales[0]))

    if ids.numel() > 0:
        lengths = env.episode_length_buf[ids].to(dtype=torch.float32)
        completed = lengths > 0
        if torch.any(completed):
            fractions = lengths[completed] / float(env.max_episode_length)
            batch_mean = float(fractions.clamp(0.0, 1.0).mean().item())
            previous_count = int(state["completed_episodes"])
            previous_ema = float(state["episode_length_ema"])
            state["episode_length_ema"] = (
                batch_mean
                if previous_count == 0
                else ema_decay * previous_ema + (1.0 - ema_decay) * batch_mean
            )
            state["completed_episodes"] = previous_count + int(completed.sum().item())

    level = int(state["level"])
    if int(state["completed_episodes"]) >= min_completed_episodes:
        while (
            level < len(episode_length_thresholds)
            and float(state["episode_length_ema"]) >= episode_length_thresholds[level]
        ):
            level += 1
    if level != int(state["level"]):
        state["level"] = level
        state["scale"] = float(scales[level])
        _apply_penalty_scale(env, target_weights, float(scales[level]))

    return dict(state)


def penalty_curriculum_state(env: object) -> dict[str, float | int] | None:
    """Return a copy of the current curriculum diagnostics when initialized."""

    state = getattr(env, PENALTY_CURRICULUM_STATE_ATTR, None)
    return None if state is None else dict(state)


@configclass
class FastWMRCurriculumCfg:
    """Terrain progression plus the shared FastSAC penalty curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    penalty_weights = CurrTerm(
        func=penalty_weight_curriculum,
        params={
            "target_weights": PENALTY_TARGET_WEIGHTS,
            "scales": DEFAULT_PENALTY_SCALES,
            "episode_length_thresholds": DEFAULT_EPISODE_LENGTH_THRESHOLDS,
            "ema_decay": 0.9,
            "min_completed_episodes": 64,
        },
    )
