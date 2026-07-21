"""Tests for structured training metrics and asynchronous episode statistics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EpisodeStatisticsTracker,
    TrainingMetricsLogger,
    fastwmr_agent_metrics_dict,
    format_console_metrics,
)


def test_episode_statistics_track_asynchronous_vector_resets() -> None:
    tracker = EpisodeStatisticsTracker(2, device="cpu")

    first = tracker.update(
        torch.tensor([1.0, 2.0]),
        torch.tensor([False, False]),
        torch.tensor([False, False]),
    )
    second = tracker.update(
        torch.tensor([3.0, 4.0]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    )
    third = tracker.update(
        torch.tensor([5.0, 6.0]),
        torch.tensor([False, False]),
        torch.tensor([False, True]),
    )

    assert first.count == 0
    assert second.count == 1
    assert second.mean_return == pytest.approx(4.0)
    assert second.mean_length == pytest.approx(2.0)
    assert third.count == 1
    assert third.mean_return == pytest.approx(12.0)
    assert third.mean_length == pytest.approx(3.0)


def test_training_logger_writes_flat_finite_jsonl_and_appends(tmp_path) -> None:
    with TrainingMetricsLogger(tmp_path, mode="fastwmr") as logger:
        record = logger.log(7, {"replay/size": 32, "rollout/reward_mean": 1.25})
        with pytest.raises(ValueError, match="finite"):
            logger.log(8, {"sac/critic_loss": float("nan")})

    with TrainingMetricsLogger(tmp_path, mode="fastwmr", append=True) as logger:
        logger.log(9, {"replay/size": 40})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [item["step"] for item in records] == [7, 9]
    assert record["mode"] == "fastwmr"
    assert record["replay/size"] == 32
    assert "reward=1.2500" in format_console_metrics(record)


def test_fastwmr_metrics_include_estimator_and_gradient_boundary_fields() -> None:
    sac_update = SimpleNamespace(
        critic_loss=torch.tensor(1.0),
        actor_loss=torch.tensor(2.0),
        temperature_loss=torch.tensor(3.0),
        temperature=torch.tensor(0.01),
        target_q_mean=torch.tensor(4.0),
        target_q_std=torch.tensor(0.4),
        q1_mean=torch.tensor(5.0),
        q1_std=torch.tensor(0.5),
        q2_mean=torch.tensor(6.0),
        q2_std=torch.tensor(0.6),
        policy_entropy=torch.tensor(7.0),
    )
    estimator_metrics = SimpleNamespace(
        total_loss=8.0,
        continuous_mse=9.0,
        discrete_bce=10.0,
        latent_l1=11.0,
        gradient_norm=12.0,
        context_exact_fraction=1.0,
        estimator_version=13,
        field_losses={"base_linear_velocity": 0.25},
    )
    update = SimpleNamespace(
        sac_update=sac_update,
        estimator_update=SimpleNamespace(metrics=estimator_metrics),
        gradient_boundary=SimpleNamespace(
            checks=4,
            enabled=True,
            estimator_gradient_norm=0.0,
        ),
    )

    metrics = fastwmr_agent_metrics_dict(update)

    assert metrics["sac/critic_loss"] == pytest.approx(1.0)
    assert metrics["sac/q1_std"] == pytest.approx(0.5)
    assert metrics["estimator/version"] == 13
    assert metrics["estimator/field/base_linear_velocity"] == pytest.approx(0.25)
    assert metrics["gradient_boundary/enabled"] == 1
    assert metrics["gradient_boundary/estimator_gradient_norm"] == pytest.approx(0.0)
