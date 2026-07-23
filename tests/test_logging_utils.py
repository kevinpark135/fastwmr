"""Tests for structured training metrics and asynchronous episode statistics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EpisodeStatisticsTracker,
    TrainingMetricsLogger,
    fastwmr_agent_metrics_dict,
    fastwmr_v2_metrics_dict,
    format_console_metrics,
    format_console_metrics_header,
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
        record = logger.log(
            7,
            {
                "replay/size": 32,
                "rollout/reward_mean": 1.25,
                "curriculum/terrain_level_mean": 0.5,
                "curriculum/terrain_level_max": 2,
                "curriculum/penalty_level": 1,
                "curriculum/penalty_scale": 0.3,
                "checkpoint/saved": 1,
            },
        )
        with pytest.raises(ValueError, match="finite"):
            logger.log(8, {"sac/critic_loss": float("nan")})

    with TrainingMetricsLogger(tmp_path, mode="fastwmr", append=True) as logger:
        logger.log(9, {"replay/size": 40})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [item["step"] for item in records] == [7, 9]
    assert record["mode"] == "fastwmr"
    assert record["replay/size"] == 32
    tensorboard = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    replay_events = tensorboard.Scalars("replay/size")
    assert [event.step for event in replay_events] == [7, 9]
    assert [event.value for event in replay_events] == pytest.approx([32.0, 40.0])
    assert "elapsed_seconds" in tensorboard.Tags()["scalars"]
    header = format_console_metrics_header("fastwmr")
    row = format_console_metrics(record)
    assert "Terr avg/max" in header
    assert "Pen lvl/x" not in header
    assert "1.2500" in row
    assert "0.50/2" in row
    assert "yes" in row
    assert len(header.splitlines()[-1]) == len(row.splitlines()[0])


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
        q_gap_mean=torch.tensor(0.2),
        q_gap_max=torch.tensor(0.8),
        policy_entropy=torch.tensor(7.0),
        policy_action_saturation_fraction=torch.tensor(0.1),
        c51_lower_endpoint_mass=torch.tensor(0.02),
        c51_upper_endpoint_mass=torch.tensor(0.03),
        c51_target_lower_endpoint_mass=torch.tensor(0.04),
        c51_target_upper_endpoint_mass=torch.tensor(0.05),
        c51_distribution_entropy=torch.tensor(2.0),
    )
    estimator_metrics = SimpleNamespace(
        total_loss=8.0,
        continuous_mse=9.0,
        discrete_bce=10.0,
        latent_l1=11.0,
        gradient_norm=12.0,
        context_exact_fraction=1.0,
        estimator_version=13,
        field_losses={"base_linear_velocity_mse": 0.25},
        physical_field_losses={"base_linear_velocity_mse": 4.0},
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
    assert metrics["estimator/field_normalized/base_linear_velocity_mse"] == pytest.approx(
        0.25
    )
    assert metrics["estimator/field_normalized/base_linear_velocity_rmse"] == 0.5
    assert metrics["estimator/field_physical/base_linear_velocity_rmse"] == 2.0
    assert metrics["sac/q_gap_mean"] == pytest.approx(0.2)
    assert metrics["sac/c51_lower_endpoint_mass"] == pytest.approx(0.02)
    assert metrics["gradient_boundary/enabled"] == 1
    assert metrics["gradient_boundary/estimator_gradient_norm"] == pytest.approx(0.0)


def test_v2_metrics_report_full_replay_freshness_and_confidence() -> None:
    controller = SimpleNamespace(
        estimator_updates=3,
        estimator_attempts=4,
        estimator_triggers=2,
        control_estimator_version=3,
        reconstruction_gate=0.5,
        gate_state=SimpleNamespace(value="ramping"),
        gate_quality_passes=2,
        gate_quality_failures=0,
        gate_validation_checks=4,
        gate_quality_ema=0.4,
        last_gate_validation=None,
    )
    update_loop = SimpleNamespace(
        estimator_controller=controller,
        sac_updates_since_estimator=0,
        last_eligible_features=64_000,
        last_rejected_features=0,
        last_full_transition_count=64_000,
        last_fresh_features=16_384,
        last_stale_features=47_616,
        last_feature_age_mean=torch.tensor(500.0),
        last_feature_age_max=torch.tensor(900),
        last_sampled_fresh_fraction=torch.tensor(0.5),
        last_reconstruction_masked_fraction=torch.tensor(0.5),
        last_reconstruction_confidence_mean=torch.tensor(0.25),
        last_reconstruction_confidence_min=torch.tensor(0.0),
        last_reconstruction_confidence_max=torch.tensor(0.5),
    )

    metrics = fastwmr_v2_metrics_dict(update_loop)

    assert metrics["replay/full_transition_count"] == 64_000
    assert metrics["replay/fresh_reconstruction_count"] == 16_384
    assert metrics["replay/stale_reconstruction_count"] == 47_616
    assert metrics["replay/sampled_fresh_fraction"] == pytest.approx(0.5)
    assert metrics["replay/sampled_stale_fraction"] == pytest.approx(0.5)
    assert metrics["representation/confidence_mean"] == pytest.approx(0.25)
