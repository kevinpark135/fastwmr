"""Tests for episode-length-driven FastSAC penalty scaling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.curriculum import (
    PENALTY_TARGET_WEIGHTS,
    penalty_curriculum_state,
    penalty_weight_curriculum,
    terrain_curriculum_state,
)


class _RewardManager:
    def __init__(self) -> None:
        self.terms = {
            name: SimpleNamespace(weight=target)
            for name, target in PENALTY_TARGET_WEIGHTS.items()
        }

    def get_term_cfg(self, name: str):
        return self.terms[name]

    def set_term_cfg(self, name: str, cfg) -> None:
        self.terms[name] = cfg


def _environment() -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=4,
        device="cpu",
        max_episode_length=100,
        episode_length_buf=torch.zeros(4, dtype=torch.int64),
        reward_manager=_RewardManager(),
    )


def test_penalty_curriculum_starts_weak_and_only_moves_up() -> None:
    env = _environment()
    kwargs = {
        "scales": (0.1, 0.5, 1.0),
        "episode_length_thresholds": (0.4, 0.8),
        "ema_decay": 0.0,
        "min_completed_episodes": 2,
    }

    initial = penalty_weight_curriculum(env, slice(None), **kwargs)
    assert initial["scale"] == pytest.approx(0.1)
    assert env.reward_manager.terms["action_rate"].weight == pytest.approx(-0.2)

    env.episode_length_buf[:] = torch.tensor([50, 60, 0, 0])
    middle = penalty_weight_curriculum(env, torch.tensor([0, 1]), **kwargs)
    assert middle["level"] == 1
    assert middle["scale"] == pytest.approx(0.5)

    env.episode_length_buf[:] = torch.tensor([90, 95, 0, 0])
    final = penalty_weight_curriculum(env, torch.tensor([0, 1]), **kwargs)
    assert final["level"] == 2
    assert final["scale"] == pytest.approx(1.0)
    assert env.reward_manager.terms["close_feet"].weight == pytest.approx(-10.0)

    env.episode_length_buf[:] = torch.tensor([10, 10, 0, 0])
    unchanged = penalty_weight_curriculum(env, torch.tensor([0, 1]), **kwargs)
    assert unchanged["level"] == 2
    assert penalty_curriculum_state(env) == unchanged


def test_terrain_curriculum_state_summarizes_vector_levels() -> None:
    env = SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                terrain_levels=torch.tensor([0, 1, 2, 5]),
                max_terrain_level=10,
            )
        )
    )

    state = terrain_curriculum_state(env)

    assert state == {
        "level_mean": pytest.approx(2.0),
        "level_min": 0,
        "level_max": 5,
        "level_cap": 9,
    }
    assert terrain_curriculum_state(SimpleNamespace()) is None


@pytest.mark.parametrize(
    ("scales", "thresholds", "message"),
    (
        ((0.2, 0.1, 1.0), (0.4, 0.8), "strictly increasing"),
        ((0.1, 0.5), (0.3, 0.6), "one fewer threshold"),
        ((0.1, 0.9), (0.4,), "final penalty scale"),
    ),
)
def test_penalty_curriculum_rejects_invalid_schedules(scales, thresholds, message) -> None:
    with pytest.raises(ValueError, match=message):
        penalty_weight_curriculum(
            _environment(),
            slice(None),
            scales=scales,
            episode_length_thresholds=thresholds,
        )
