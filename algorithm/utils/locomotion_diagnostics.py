"""Interval diagnostics for G1 locomotion training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat


LOCOMOTION_PIN_METRICS = (
    "episode/return_mean",
    "locomotion/linear_velocity_rmse",
    "locomotion/yaw_rate_rmse",
    "locomotion/non_timeout_termination_rate",
    "locomotion/base_height_mean",
    "locomotion/single_support_fraction",
    "locomotion/joint_tracking_rmse",
    "locomotion/torque_clipping_fraction",
)
"""Eight high-signal cards recommended for TensorBoard pinning."""

LOCOMOTION_TENSORBOARD_LAYOUT = {
    "Locomotion Pins": {
        "Performance": [
            "Multiline",
            [
                "episode/return_mean",
                "locomotion/linear_velocity_rmse",
                "locomotion/yaw_rate_rmse",
            ],
        ],
        "Stability": [
            "Multiline",
            [
                "locomotion/base_height_mean",
                "locomotion/base_tilt_mean",
                "locomotion/non_timeout_termination_rate",
            ],
        ],
        "Gait": [
            "Multiline",
            [
                "locomotion/single_support_fraction",
                "locomotion/double_support_fraction",
                "locomotion/airborne_fraction",
                "locomotion/contacting_foot_slip_speed_mean",
            ],
        ],
        "Control": [
            "Multiline",
            [
                "locomotion/joint_tracking_rmse",
                "locomotion/action_saturation_fraction",
                "locomotion/torque_clipping_fraction",
            ],
        ],
    }
}
"""TensorBoard Custom Scalars layout grouping the main locomotion signals."""


@dataclass(frozen=True)
class LocomotionDiagnosticStep:
    """One vector-environment snapshot consumed by the interval tracker."""

    velocity_command: torch.Tensor
    linear_velocity_yaw: torch.Tensor
    yaw_rate: torch.Tensor
    base_height: torch.Tensor
    base_tilt: torch.Tensor
    joint_position_error: torch.Tensor
    action_saturated: torch.Tensor
    torque_clipped: torch.Tensor
    foot_contacts: torch.Tensor
    foot_xy_speed: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    pelvis_contact: torch.Tensor
    hip_contact: torch.Tensor
    shoulder_contact: torch.Tensor


class IsaacLabLocomotionDiagnosticSource:
    """Read graph-free locomotion state from a configured Isaac Lab environment."""

    _FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
    _CONTACT_TERMS = ("pelvis_contact", "hip_contact", "shoulder_contact")

    def __init__(
        self,
        env: object,
        *,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        action_term_name: str = "joint_pos",
        command_name: str = "base_velocity",
        contact_sensor_name: str = "contact_forces",
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.action_term = env.action_manager.get_term(action_term_name)
        self.command_name = command_name
        self.contact_sensor = env.scene[contact_sensor_name]

        action_dim = int(self.action_term.action_dim)
        expected_bounds_shape = (action_dim,)
        if action_low.shape != expected_bounds_shape or action_high.shape != expected_bounds_shape:
            raise ValueError(f"Action bounds must both have shape {expected_bounds_shape}.")
        if action_low.device != action_high.device or action_low.device != torch.device(env.device):
            raise ValueError("Action bounds must share the environment device.")
        if not torch.all(action_low < action_high):
            raise ValueError("Every locomotion action lower bound must be below its upper bound.")
        self._action_center = ((action_low + action_high) * 0.5).detach()
        self._action_half_range = ((action_high - action_low) * 0.5).detach()

        joint_ids = self.action_term._joint_ids
        if isinstance(joint_ids, slice):
            joint_ids = list(range(self.robot.num_joints))[joint_ids]
        self._joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
        if self._joint_ids.numel() != action_dim:
            raise ValueError("The locomotion action term must resolve one action per controlled joint.")

        foot_body_ids, foot_body_names = self.robot.find_bodies(
            list(self._FOOT_NAMES),
            preserve_order=True,
        )
        foot_sensor_ids, foot_sensor_names = self.contact_sensor.find_sensors(
            list(self._FOOT_NAMES),
            preserve_order=True,
        )
        if tuple(foot_body_names) != self._FOOT_NAMES or tuple(foot_sensor_names) != self._FOOT_NAMES:
            raise ValueError("Locomotion diagnostics require exact left/right ankle-roll links.")
        self._foot_body_ids = foot_body_ids
        self._foot_sensor_ids = foot_sensor_ids

        active_terminations = set(env.termination_manager.active_terms)
        missing_terminations = set(self._CONTACT_TERMS) - active_terminations
        if missing_terminations:
            raise ValueError(
                "Locomotion diagnostics require split contact termination terms; "
                f"missing {sorted(missing_terminations)}."
            )

    @torch.no_grad()
    def sample(
        self,
        *,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        contact_threshold: float = 1.0,
        action_saturation_threshold: float = 0.95,
    ) -> LocomotionDiagnosticStep:
        """Capture one post-step snapshot, masking reset rows in the tracker."""

        if contact_threshold <= 0.0:
            raise ValueError("contact_threshold must be positive.")
        if not 0.0 < action_saturation_threshold <= 1.0:
            raise ValueError("action_saturation_threshold must lie in (0, 1].")

        robot_data = self.robot.data
        command = self.env.command_manager.get_command(self.command_name)
        linear_velocity_yaw = quat_apply_inverse(
            yaw_quat(robot_data.root_quat_w.torch),
            robot_data.root_lin_vel_w.torch[:, :3],
        )[:, :2]
        base_height = robot_data.root_pos_w.torch[:, 2] - self.env.scene.env_origins[:, 2]
        base_tilt = torch.linalg.vector_norm(
            robot_data.projected_gravity_b.torch[:, :2],
            dim=-1,
        )

        joint_positions = robot_data.joint_pos.torch[:, self._joint_ids]
        joint_targets = self.action_term.processed_actions
        normalized_action = (
            self.env.action_manager.action - self._action_center
        ) / self._action_half_range
        action_saturated = normalized_action.abs() >= action_saturation_threshold

        computed_torque = robot_data.computed_torque.torch[:, self._joint_ids]
        applied_torque = robot_data.applied_torque.torch[:, self._joint_ids]
        clipping_tolerance = 1.0e-5 + 1.0e-4 * computed_torque.abs()
        torque_clipped = (computed_torque - applied_torque).abs() > clipping_tolerance

        contact_history = self.contact_sensor.data.net_forces_w_history
        if contact_history is None:
            raise RuntimeError("Locomotion diagnostics require contact-force history.")
        contact_forces = contact_history.torch[:, :, self._foot_sensor_ids, :]
        foot_contacts = contact_forces.norm(dim=-1).amax(dim=1) > contact_threshold
        foot_xy_speed = torch.linalg.vector_norm(
            robot_data.body_lin_vel_w.torch[:, self._foot_body_ids, :2],
            dim=-1,
        )

        return LocomotionDiagnosticStep(
            velocity_command=command.detach(),
            linear_velocity_yaw=linear_velocity_yaw.detach(),
            yaw_rate=robot_data.root_ang_vel_w.torch[:, 2].detach(),
            base_height=base_height.detach(),
            base_tilt=base_tilt.detach(),
            joint_position_error=(joint_targets - joint_positions).detach(),
            action_saturated=action_saturated.detach(),
            torque_clipped=torque_clipped.detach(),
            foot_contacts=foot_contacts.detach(),
            foot_xy_speed=foot_xy_speed.detach(),
            terminated=terminated.detach(),
            truncated=truncated.detach(),
            pelvis_contact=self.env.termination_manager.get_term("pelvis_contact").detach(),
            hip_contact=self.env.termination_manager.get_term("hip_contact").detach(),
            shoulder_contact=self.env.termination_manager.get_term("shoulder_contact").detach(),
        )


class LocomotionDiagnosticsTracker:
    """Aggregate locomotion diagnostics over one training logging interval."""

    _RMSE_METRICS = (
        "linear_velocity_rmse",
        "yaw_rate_rmse",
        "joint_tracking_rmse",
    )
    _MEAN_METRICS = (
        "base_height_mean",
        "base_tilt_mean",
        "action_saturation_fraction",
        "torque_clipping_fraction",
        "left_foot_contact_fraction",
        "right_foot_contact_fraction",
        "single_support_fraction",
        "double_support_fraction",
        "airborne_fraction",
        "contacting_foot_slip_speed_mean",
        "non_timeout_termination_rate",
        "timeout_rate",
        "pelvis_contact_rate",
        "hip_contact_rate",
        "shoulder_contact_rate",
    )

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        moving_command_threshold: float = 0.05,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if not math.isfinite(moving_command_threshold) or moving_command_threshold < 0.0:
            raise ValueError("moving_command_threshold must be finite and non-negative.")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.moving_command_threshold = moving_command_threshold
        names = (*self._RMSE_METRICS, *self._MEAN_METRICS)
        self._sums = {
            name: torch.zeros((), device=self.device, dtype=torch.float64)
            for name in names
        }
        self._counts = {name: 0 for name in names}

    @torch.no_grad()
    def update(self, step: LocomotionDiagnosticStep) -> None:
        self._validate_step(step)
        valid = ~(step.terminated | step.truncated)

        self._accumulate_squared(
            "linear_velocity_rmse",
            step.linear_velocity_yaw[valid] - step.velocity_command[valid, :2],
        )
        self._accumulate_squared(
            "yaw_rate_rmse",
            step.yaw_rate[valid] - step.velocity_command[valid, 2],
        )
        self._accumulate_squared("joint_tracking_rmse", step.joint_position_error[valid])
        self._accumulate_mean("base_height_mean", step.base_height[valid])
        self._accumulate_mean("base_tilt_mean", step.base_tilt[valid])
        self._accumulate_mean(
            "action_saturation_fraction",
            step.action_saturated[valid],
        )
        self._accumulate_mean("torque_clipping_fraction", step.torque_clipped[valid])
        self._accumulate_mean(
            "left_foot_contact_fraction",
            step.foot_contacts[valid, 0],
        )
        self._accumulate_mean(
            "right_foot_contact_fraction",
            step.foot_contacts[valid, 1],
        )

        moving = (
            torch.linalg.vector_norm(step.velocity_command[:, :2], dim=-1)
            > self.moving_command_threshold
        ) | (step.velocity_command[:, 2].abs() > self.moving_command_threshold)
        moving_valid = moving & valid
        support_count = step.foot_contacts[moving_valid].sum(dim=-1)
        self._accumulate_mean("single_support_fraction", support_count == 1)
        self._accumulate_mean("double_support_fraction", support_count == 2)
        self._accumulate_mean("airborne_fraction", support_count == 0)

        contacting_valid = step.foot_contacts & valid.unsqueeze(-1)
        self._accumulate_mean(
            "contacting_foot_slip_speed_mean",
            step.foot_xy_speed[contacting_valid],
        )

        self._accumulate_mean("non_timeout_termination_rate", step.terminated)
        self._accumulate_mean("timeout_rate", step.truncated)
        self._accumulate_mean("pelvis_contact_rate", step.pelvis_contact)
        self._accumulate_mean("hip_contact_rate", step.hip_contact)
        self._accumulate_mean("shoulder_contact_rate", step.shoulder_contact)

    def drain(self, *, prefix: str = "locomotion/") -> dict[str, float]:
        """Return interval metrics and clear all accumulated samples."""

        if not prefix:
            raise ValueError("Locomotion metric prefix must not be empty.")
        metrics: dict[str, float] = {}
        for name in self._RMSE_METRICS:
            if self._counts[name]:
                mean_square = float((self._sums[name] / self._counts[name]).item())
                metrics[f"{prefix}{name}"] = math.sqrt(max(0.0, mean_square))
        for name in self._MEAN_METRICS:
            if self._counts[name]:
                metrics[f"{prefix}{name}"] = float(
                    (self._sums[name] / self._counts[name]).item()
                )
        self.reset()
        return metrics

    def reset(self) -> None:
        for value in self._sums.values():
            value.zero_()
        for name in self._counts:
            self._counts[name] = 0

    def _validate_step(self, step: LocomotionDiagnosticStep) -> None:
        expected_vector = (self.num_envs,)
        expected_matrix_prefix = (self.num_envs,)
        boolean_names = (
            "action_saturated",
            "torque_clipped",
            "foot_contacts",
            "terminated",
            "truncated",
            "pelvis_contact",
            "hip_contact",
            "shoulder_contact",
        )
        for name, value in vars(step).items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.device != self.device:
                raise ValueError(f"{name} must be on {self.device}.")
            if value.shape[:1] != expected_matrix_prefix:
                raise ValueError(f"{name} must start with shape {expected_matrix_prefix}.")
            if name in boolean_names:
                if value.dtype is not torch.bool:
                    raise TypeError(f"{name} must be boolean.")
            elif not value.dtype.is_floating_point or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values.")

        if step.velocity_command.shape != (self.num_envs, 3):
            raise ValueError("velocity_command must have shape (num_envs, 3).")
        if step.linear_velocity_yaw.shape != (self.num_envs, 2):
            raise ValueError("linear_velocity_yaw must have shape (num_envs, 2).")
        if step.foot_contacts.shape != (self.num_envs, 2):
            raise ValueError("foot_contacts must have shape (num_envs, 2).")
        if step.foot_xy_speed.shape != (self.num_envs, 2):
            raise ValueError("foot_xy_speed must have shape (num_envs, 2).")
        for name in (
            "yaw_rate",
            "base_height",
            "base_tilt",
            "terminated",
            "truncated",
            "pelvis_contact",
            "hip_contact",
            "shoulder_contact",
        ):
            if getattr(step, name).shape != expected_vector:
                raise ValueError(f"{name} must have shape {expected_vector}.")
        if (
            step.joint_position_error.ndim != 2
            or step.action_saturated.shape != step.joint_position_error.shape
            or step.torque_clipped.shape != step.joint_position_error.shape
        ):
            raise ValueError(
                "Joint error, action saturation, and torque clipping tensors must "
                "share shape (num_envs, action_dim)."
            )

    def _accumulate_squared(self, name: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        self._sums[name].add_(values.double().square().sum())
        self._counts[name] += values.numel()

    def _accumulate_mean(self, name: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        self._sums[name].add_(values.double().sum())
        self._counts[name] += values.numel()


__all__ = [
    "IsaacLabLocomotionDiagnosticSource",
    "LOCOMOTION_PIN_METRICS",
    "LOCOMOTION_TENSORBOARD_LAYOUT",
    "LocomotionDiagnosticStep",
    "LocomotionDiagnosticsTracker",
]
