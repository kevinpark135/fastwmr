"""Structured metrics and episode statistics for FastWMR experiments."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch


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
    """Append finite scalar records to a machine-readable JSONL run log."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        mode: str,
        append: bool = False,
    ) -> None:
        if not mode:
            raise ValueError("Logging mode must not be empty.")
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.path = self.run_directory / "metrics.jsonl"
        self.mode = mode
        self._start_time = time.monotonic()
        self._file = self.path.open("a" if append else "w", encoding="utf-8")
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
        self.records_written += 1
        return record

    def close(self) -> None:
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
    return {
        f"{prefix}{name}": _scalar(getattr(metrics, name), name)
        for name in names
    }


def fastwmr_agent_metrics_dict(update: object) -> dict[str, float | int]:
    """Combine SAC, estimator, and gradient-boundary diagnostics."""

    output: dict[str, float | int] = sac_metrics_dict(update.sac_update)
    estimator = update.estimator_update.metrics
    output.update(
        {
            "estimator/total_loss": float(estimator.total_loss),
            "estimator/continuous_mse": float(estimator.continuous_mse),
            "estimator/discrete_bce": float(estimator.discrete_bce),
            "estimator/latent_l1": float(estimator.latent_l1),
            "estimator/gradient_norm": float(estimator.gradient_norm),
            "estimator/context_exact_fraction": float(estimator.context_exact_fraction),
            "estimator/version": int(estimator.estimator_version),
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
    for name, value in estimator.field_losses.items():
        output[f"estimator/field/{name}"] = float(value)
    return output


def format_console_metrics(record: Mapping[str, int | float | str]) -> str:
    """Return one compact progress line from a structured log record."""

    parts = [f"[{record['mode']}]", f"step={record['step']}"]
    preferred = (
        ("rollout/reward_mean", "reward", ".4f"),
        ("replay/size", "replay", ".0f"),
        ("learner/gradient_steps", "updates", ".0f"),
        ("sac/critic_loss", "critic", ".4f"),
        ("sac/actor_loss", "actor", ".4f"),
        ("sac/temperature", "alpha", ".6f"),
        ("estimator/total_loss", "estimator", ".4f"),
    )
    for key, label, format_spec in preferred:
        if key in record:
            parts.append(f"{label}={format(float(record[key]), format_spec)}")
    return " ".join(parts)


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
