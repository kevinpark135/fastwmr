"""Structured metrics and episode statistics for FastWMR experiments."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


_CONSOLE_HEADERS = (
    "Step",
    "Reward",
    "Replay",
    "Updates",
    "Critic",
    "Actor",
    "Alpha",
    "Estimator",
    "Terr avg/max",
    "Elapsed",
    "Ckpt",
)
_CONSOLE_WIDTHS = (8, 9, 9, 9, 8, 8, 9, 9, 12, 8, 4)


@dataclass(frozen=True)
class CompletedEpisodeStatistics:
    """Episode aggregates completed during one vector-environment step."""

    count: int
    return_sum: float
    length_sum: int

    @property
    def mean_return(self) -> float | None:
        return self.return_sum / self.count if self.count else None

    @property
    def mean_length(self) -> float | None:
        return self.length_sum / self.count if self.count else None


class EpisodeStatisticsTracker:
    """Track per-environment returns and lengths across asynchronous resets."""

    def __init__(self, num_envs: int, *, device: torch.device | str) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._returns = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        self._lengths = torch.zeros(num_envs, device=self.device, dtype=torch.int64)

    @torch.no_grad()
    def update(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> CompletedEpisodeStatistics:
        expected_shape = (self.num_envs,)
        if rewards.shape != expected_shape or terminated.shape != expected_shape or truncated.shape != expected_shape:
            raise ValueError(f"Episode tracker inputs must all have shape {expected_shape}.")
        if not rewards.dtype.is_floating_point or not torch.isfinite(rewards).all():
            raise ValueError("Episode rewards must be finite floating-point values.")
        if terminated.dtype is not torch.bool or truncated.dtype is not torch.bool:
            raise TypeError("Episode termination flags must be boolean tensors.")
        if any(tensor.device != self.device for tensor in (rewards, terminated, truncated)):
            raise ValueError("Episode tracker inputs must share its device.")

        self._returns.add_(rewards.to(dtype=torch.float64))
        self._lengths.add_(1)
        done = terminated | truncated
        completed_returns = self._returns[done]
        completed_lengths = self._lengths[done]
        statistics = CompletedEpisodeStatistics(
            count=int(done.sum().item()),
            return_sum=float(completed_returns.sum().item()),
            length_sum=int(completed_lengths.sum().item()),
        )
        self._returns[done] = 0.0
        self._lengths[done] = 0
        return statistics

    def reset(self) -> None:
        self._returns.zero_()
        self._lengths.zero_()


class TrainingMetricsLogger:
    """Write finite scalar records to JSONL and TensorBoard together."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        mode: str,
        append: bool = False,
        tensorboard_purge_step: int | None = None,
    ) -> None:
        if not mode:
            raise ValueError("Logging mode must not be empty.")
        if tensorboard_purge_step is not None and tensorboard_purge_step < 0:
            raise ValueError("TensorBoard purge step must be non-negative.")
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.path = self.run_directory / "metrics.jsonl"
        self.tensorboard_directory = self.run_directory / "tensorboard"
        self.mode = mode
        self._start_time = time.monotonic()
        self._file = self.path.open("a" if append else "w", encoding="utf-8")
        self._tensorboard = SummaryWriter(
            log_dir=str(self.tensorboard_directory),
            purge_step=tensorboard_purge_step,
            max_queue=100,
            flush_secs=10,
        )
        self.records_written = 0

    def log(self, step: int, metrics: Mapping[str, int | float]) -> Mapping[str, int | float | str]:
        if step < 0:
            raise ValueError("Logging step must be non-negative.")
        clean_metrics = _finite_scalars(metrics)
        record: dict[str, int | float | str] = {
            "step": step,
            "mode": self.mode,
            "elapsed_seconds": time.monotonic() - self._start_time,
            **clean_metrics,
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()
        self._tensorboard.add_scalar("elapsed_seconds", record["elapsed_seconds"], step)
        for name, value in clean_metrics.items():
            self._tensorboard.add_scalar(name, value, step)
        self.records_written += 1
        return record

    def close(self) -> None:
        self._tensorboard.close()
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TrainingMetricsLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def sac_metrics_dict(metrics: object, *, prefix: str = "sac/") -> dict[str, float]:
    """Convert one SAC metrics object into JSON-safe scalar fields."""

    names = (
        "critic_loss",
        "actor_loss",
        "temperature_loss",
        "temperature",
        "target_q_mean",
        "target_q_std",
        "q1_mean",
        "q1_std",
        "q2_mean",
        "q2_std",
        "policy_entropy",
    )
    output = {
        f"{prefix}{name}": _scalar(getattr(metrics, name), name)
        for name in names
    }
    optional_names = (
        "q_gap_mean",
        "q_gap_max",
        "policy_action_saturation_fraction",
        "c51_lower_endpoint_mass",
        "c51_upper_endpoint_mass",
        "c51_target_lower_endpoint_mass",
        "c51_target_upper_endpoint_mass",
        "c51_distribution_entropy",
    )
    for name in optional_names:
        value = getattr(metrics, name, None)
        if value is not None:
            output[f"{prefix}{name}"] = _scalar(value, name)
    return output


def fastwmr_agent_metrics_dict(update: object) -> dict[str, float | int]:
    """Combine SAC, estimator, and gradient-boundary diagnostics."""

    output: dict[str, float | int] = sac_metrics_dict(update.sac_update)
    output.update(estimator_metrics_dict(update.estimator_update))
    output.update(
        {
            "gradient_boundary/checks": int(update.gradient_boundary.checks),
            "gradient_boundary/enabled": int(update.gradient_boundary.enabled),
            "gradient_boundary/cutoff_enabled": int(
                getattr(update.gradient_boundary, "cutoff_enabled", True)
            ),
        }
    )
    if update.gradient_boundary.estimator_gradient_norm is not None:
        output["gradient_boundary/estimator_gradient_norm"] = float(
            update.gradient_boundary.estimator_gradient_norm
        )
    policy_gradient_norm = getattr(
        update.gradient_boundary,
        "policy_estimator_gradient_norm",
        None,
    )
    if policy_gradient_norm is not None:
        output["gradient_boundary/policy_estimator_gradient_norm"] = float(
            policy_gradient_norm
        )
    return output


def estimator_metrics_dict(update: object) -> dict[str, float | int]:
    """Convert one standalone estimator result into logging scalars."""

    estimator = update.metrics
    output: dict[str, float | int] = {
        "estimator/total_loss": float(estimator.total_loss),
        "estimator/continuous_mse": float(estimator.continuous_mse),
        "estimator/discrete_bce": float(estimator.discrete_bce),
        "estimator/latent_l1": float(estimator.latent_l1),
        "estimator/gradient_norm": float(estimator.gradient_norm),
        "estimator/context_exact_fraction": float(estimator.context_exact_fraction),
        "estimator/version": int(estimator.estimator_version),
    }
    for name, value in estimator.field_losses.items():
        output[f"estimator/field_normalized/{name}"] = float(value)
        if name.endswith("_mse"):
            output[f"estimator/field_normalized/{name[:-4]}_rmse"] = math.sqrt(
                max(0.0, float(value))
            )
    for name, value in getattr(estimator, "physical_field_losses", {}).items():
        output[f"estimator/field_physical/{name}"] = float(value)
        if name.endswith("_mse"):
            output[f"estimator/field_physical/{name[:-4]}_rmse"] = math.sqrt(
                max(0.0, float(value))
            )
    return output


def fastwmr_v2_metrics_dict(update_loop: object) -> dict[str, float | int]:
    """Expose two-timescale scheduling, gate, and feature-age diagnostics."""

    controller = update_loop.estimator_controller
    metrics: dict[str, float | int] = {
        "v2/estimator_updates": int(controller.estimator_updates),
        "v2/estimator_attempts": int(controller.estimator_attempts),
        "v2/estimator_triggers": int(controller.estimator_triggers),
        "v2/control_estimator_version": int(controller.control_estimator_version),
        "v2/sac_updates_since_estimator": int(update_loop.sac_updates_since_estimator),
        "v2/reconstruction_gate": float(controller.reconstruction_gate),
        "v2/gate_state": {
            "closed": 0,
            "ramping": 1,
            "open": 2,
        }[controller.gate_state.value],
        "v2/gate_quality_passes": int(controller.gate_quality_passes),
        "v2/gate_validation_checks": int(controller.gate_validation_checks),
        "v2/eligible_features": int(update_loop.last_eligible_features),
        "v2/rejected_features": int(update_loop.last_rejected_features),
    }
    if controller.gate_quality_ema is not None:
        metrics["v2/gate_quality_ema"] = float(controller.gate_quality_ema)
    if controller.last_gate_validation is not None:
        metrics["v2/gate_validation_loss"] = float(
            controller.last_gate_validation.metrics.total_loss
        )
    if update_loop.last_feature_age_mean is not None:
        metrics["v2/feature_age_mean"] = float(update_loop.last_feature_age_mean)
    if update_loop.last_feature_age_max is not None:
        metrics["v2/feature_age_max"] = int(update_loop.last_feature_age_max)
    return metrics


def format_console_metrics_header(mode: str) -> str:
    """Return the title and fixed-width header for the training progress table."""

    if not mode:
        raise ValueError("Console metrics mode must not be empty.")
    separator = _console_separator()
    return "\n".join(
        (
            f"[{mode}] Training progress",
            separator,
            _console_row(_CONSOLE_HEADERS),
            separator,
        )
    )


def format_console_metrics(record: Mapping[str, int | float | str]) -> str:
    """Return one fixed-width progress row followed by a visual separator."""

    values = (
        _format_count(record.get("step")),
        _format_float(record.get("rollout/reward_mean"), ".4f"),
        _format_count(record.get("replay/size")),
        _format_count(record.get("learner/gradient_steps")),
        _format_float(record.get("sac/critic_loss"), ".4f"),
        _format_float(record.get("sac/actor_loss"), ".4f"),
        _format_float(record.get("sac/temperature"), ".6f"),
        _format_float(record.get("estimator/total_loss"), ".4f"),
        _format_terrain_curriculum(record),
        _format_duration(record.get("learner/wallclock_seconds")),
        "yes" if int(record.get("checkpoint/saved", 0)) else "",
    )
    return f"{_console_row(values)}\n{_console_separator()}"


def _console_separator() -> str:
    return "+" + "+".join("-" * (width + 2) for width in _CONSOLE_WIDTHS) + "+"


def _console_row(values: tuple[str, ...]) -> str:
    if len(values) != len(_CONSOLE_WIDTHS):
        raise ValueError("Console table row does not match its configured columns.")
    cells = (
        f" {value:>{width}} "
        for value, width in zip(values, _CONSOLE_WIDTHS, strict=True)
    )
    return "|" + "|".join(cells) + "|"


def _format_count(value: object) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_float(value: object, format_spec: str) -> str:
    if value is None:
        return "-"
    return format(float(value), format_spec)


def _format_terrain_curriculum(record: Mapping[str, int | float | str]) -> str:
    mean = record.get("curriculum/terrain_level_mean")
    maximum = record.get("curriculum/terrain_level_max")
    if mean is None or maximum is None:
        return "-"
    return f"{float(mean):.2f}/{int(maximum)}"


def _format_duration(value: object) -> str:
    if value is None:
        return "-"
    total_seconds = max(0, int(float(value)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _finite_scalars(metrics: Mapping[str, int | float]) -> dict[str, int | float]:
    clean: dict[str, int | float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric names must be non-empty strings.")
        if isinstance(value, bool):
            clean[name] = int(value)
        elif isinstance(value, int):
            clean[name] = value
        elif isinstance(value, float) and math.isfinite(value):
            clean[name] = value
        else:
            raise ValueError(f"Metric {name!r} must be one finite int or float, got {value!r}.")
    return clean


def _scalar(value: object, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"Metric {name!r} must be one finite tensor scalar.")
        return float(value.item())
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"Metric {name!r} must be finite.")
    return scalar
