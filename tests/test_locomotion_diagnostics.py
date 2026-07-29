"""Tests for interval locomotion diagnostics and TensorBoard grouping."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    LOCOMOTION_PIN_METRICS,
    LOCOMOTION_TENSORBOARD_LAYOUT,
    LocomotionDiagnosticStep,
    LocomotionDiagnosticsTracker,
)


def _step() -> LocomotionDiagnosticStep:
    return LocomotionDiagnosticStep(
        velocity_command=torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
        linear_velocity_yaw=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        yaw_rate=torch.tensor([0.0, 0.0]),
        base_height=torch.tensor([0.8, 0.7]),
        base_tilt=torch.tensor([0.1, 0.2]),
        joint_position_error=torch.tensor([[1.0, 0.0], [3.0, 4.0]]),
        action_saturated=torch.tensor([[True, False], [True, True]]),
        torque_clipped=torch.tensor([[False, True], [True, True]]),
        foot_contacts=torch.tensor([[True, False], [True, True]]),
        foot_xy_speed=torch.tensor([[0.2, 0.4], [0.0, 0.0]]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        pelvis_contact=torch.tensor([False, True]),
        hip_contact=torch.tensor([False, False]),
        shoulder_contact=torch.tensor([False, False]),
    )


def test_locomotion_tracker_aggregates_valid_state_and_all_termination_rows() -> None:
    tracker = LocomotionDiagnosticsTracker(2, device="cpu")

    tracker.update(_step())
    metrics = tracker.drain()

    assert metrics["locomotion/linear_velocity_rmse"] == pytest.approx(
        1.0 / torch.sqrt(torch.tensor(2.0)).item()
    )
    assert metrics["locomotion/yaw_rate_rmse"] == pytest.approx(0.5)
    assert metrics["locomotion/joint_tracking_rmse"] == pytest.approx(
        1.0 / torch.sqrt(torch.tensor(2.0)).item()
    )
    assert metrics["locomotion/base_height_mean"] == pytest.approx(0.8)
    assert metrics["locomotion/action_saturation_fraction"] == pytest.approx(0.5)
    assert metrics["locomotion/torque_clipping_fraction"] == pytest.approx(0.5)
    assert metrics["locomotion/single_support_fraction"] == pytest.approx(1.0)
    assert metrics["locomotion/contacting_foot_slip_speed_mean"] == pytest.approx(0.2)
    assert metrics["locomotion/non_timeout_termination_rate"] == pytest.approx(0.5)
    assert metrics["locomotion/pelvis_contact_rate"] == pytest.approx(0.5)
    assert metrics["locomotion/hip_contact_rate"] == pytest.approx(0.0)
    assert tracker.drain() == {}


def test_locomotion_tracker_skips_support_metrics_without_moving_samples() -> None:
    tracker = LocomotionDiagnosticsTracker(2, device="cpu")
    step = _step()
    standing_step = LocomotionDiagnosticStep(
        **{
            **vars(step),
            "velocity_command": torch.zeros_like(step.velocity_command),
        }
    )

    tracker.update(standing_step)
    metrics = tracker.drain()

    assert "locomotion/single_support_fraction" not in metrics
    assert metrics["locomotion/left_foot_contact_fraction"] == pytest.approx(1.0)


def test_locomotion_metrics_define_one_category_and_eight_pin_cards() -> None:
    assert len(LOCOMOTION_PIN_METRICS) == 8
    assert LOCOMOTION_PIN_METRICS[0] == "episode/return_mean"
    assert all(
        name.startswith("locomotion/")
        for name in LOCOMOTION_PIN_METRICS[1:]
    )
    assert set(LOCOMOTION_TENSORBOARD_LAYOUT) == {"Locomotion Pins"}
