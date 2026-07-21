"""Serializable robustness-evaluation records and multi-seed summaries."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


EVALUATION_FORMAT_VERSION = 1


class EvaluationCondition(str, Enum):
    """Nominal and out-of-distribution rollout conditions from stage 9."""

    NOMINAL_ROUGH = "nominal_rough"
    FRICTION_LOW = "friction_low"
    FRICTION_HIGH = "friction_high"
    PAYLOAD_HEAVY = "payload_heavy"
    STRONG_PUSH = "strong_push"
    OBSERVATION_NOISE = "observation_noise"
    OBSERVATION_MASKING = "observation_masking"


@dataclass(frozen=True)
class EvaluationRecord:
    """One fixed-budget rollout result for one checkpoint and random seed."""

    mode: str
    variant: str
    condition: str
    seed: int
    checkpoint: str
    checkpoint_environment_steps: int
    evaluation_steps: int
    num_envs: int
    wallclock_seconds: float
    metrics: dict[str, float]
    reconstruction_correlations: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mode or not self.variant or not self.checkpoint:
            raise ValueError("Evaluation mode, variant, and checkpoint must not be empty.")
        EvaluationCondition(self.condition)
        if self.seed < 0 or self.checkpoint_environment_steps < 0:
            raise ValueError("Evaluation seed and checkpoint steps must be non-negative.")
        if self.evaluation_steps <= 0 or self.num_envs <= 0 or self.wallclock_seconds <= 0.0:
            raise ValueError("Evaluation budget and wall-clock duration must be positive.")
        for name, value in {**self.metrics, **self.reconstruction_correlations}.items():
            if not name or not math.isfinite(float(value)):
                raise ValueError(f"Evaluation metric {name!r} must be finite.")


def write_evaluation_record(path: str | Path, record: EvaluationRecord) -> Path:
    """Atomically write one evaluation record as JSON."""

    payload = {"format_version": EVALUATION_FORMAT_VERSION, **asdict(record)}
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_evaluation_record(path: str | Path) -> EvaluationRecord:
    """Load and validate one stage-9 JSON result."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.pop("format_version", None) != EVALUATION_FORMAT_VERSION:
        raise ValueError(f"Unsupported evaluation record format: {resolved}")
    return EvaluationRecord(**payload)


def aggregate_evaluation_records(
    records: list[EvaluationRecord],
    *,
    minimum_seeds: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate each mode/condition group into sample mean and standard deviation."""

    if minimum_seeds <= 0:
        raise ValueError("minimum_seeds must be positive.")
    groups: dict[tuple[str, str, str], list[EvaluationRecord]] = {}
    for record in records:
        groups.setdefault((record.mode, record.variant, record.condition), []).append(record)
    rows: list[dict[str, Any]] = []
    for (mode, variant, condition), group in sorted(groups.items()):
        seeds = {record.seed for record in group}
        if len(seeds) < minimum_seeds:
            raise ValueError(
                f"{mode}/{variant}/{condition} has {len(seeds)} unique seeds; "
                f"at least {minimum_seeds} are required."
            )
        metric_names = set(group[0].metrics)
        correlation_names = set(group[0].reconstruction_correlations)
        if any(set(record.metrics) != metric_names for record in group):
            raise ValueError(f"{mode}/{variant}/{condition} metric schemas do not match.")
        if any(set(record.reconstruction_correlations) != correlation_names for record in group):
            raise ValueError(f"{mode}/{variant}/{condition} reconstruction schemas do not match.")
        flattened = {
            **{
                f"metric/{name}": [record.metrics[name] for record in group]
                for name in metric_names
            },
            **{
                f"reconstruction_correlation/{name}": [
                    record.reconstruction_correlations[name] for record in group
                ]
                for name in correlation_names
            },
            "budget/wallclock_seconds": [record.wallclock_seconds for record in group],
            "budget/environment_steps": [
                float(record.evaluation_steps * record.num_envs) for record in group
            ],
        }
        row: dict[str, Any] = {
            "mode": mode,
            "variant": variant,
            "condition": condition,
            "seeds": len(seeds),
        }
        for name, values in sorted(flattened.items()):
            row[f"{name}/mean"] = statistics.fmean(values)
            row[f"{name}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_evaluation_summary(
    output_directory: str | Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write machine-readable JSON/CSV plus a compact Markdown table."""

    if not rows:
        raise ValueError("At least one aggregate row is required.")
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = _write_text(
        directory / "summary.json",
        json.dumps({"format_version": EVALUATION_FORMAT_VERSION, "rows": rows}, indent=2) + "\n",
    )
    fieldnames = sorted({name for row in rows for name in row})
    csv_path = directory / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    display_metrics = (
        "metric/return_mean",
        "metric/fall_rate",
        "metric/linear_tracking_error",
        "metric/yaw_tracking_error",
        "reconstruction_correlation/overall",
        "budget/wallclock_seconds",
    )
    header = ["mode", "variant", "condition", "seeds", *display_metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        values = [
            str(row["mode"]),
            str(row["variant"]),
            str(row["condition"]),
            str(row["seeds"]),
        ]
        for name in display_metrics:
            mean = row.get(f"{name}/mean")
            std = row.get(f"{name}/std")
            values.append("n/a" if mean is None else f"{mean:.5g} +/- {std:.3g}")
        lines.append("| " + " | ".join(values) + " |")
    markdown_path = _write_text(directory / "summary.md", "\n".join(lines) + "\n")
    return json_path, csv_path, markdown_path


def _write_text(path: str | Path, text: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, resolved)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return resolved
