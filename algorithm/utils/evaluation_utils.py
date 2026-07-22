"""Serializable robustness-evaluation records and multi-seed summaries."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


EVALUATION_FORMAT_VERSION = 2


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
    """One fixed-budget rollout for one training and evaluation seed pair."""

    mode: str
    variant: str
    condition: str
    training_seed: int
    evaluation_seed: int
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
        if self.training_seed < 0 or self.evaluation_seed < 0:
            raise ValueError("Training and evaluation seeds must be non-negative.")
        if self.checkpoint_environment_steps < 0:
            raise ValueError("Checkpoint steps must be non-negative.")
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


def training_seed_from_config(config: object) -> int:
    """Read the training seed recorded in a checkpoint configuration."""

    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint config must be a mapping with recorded arguments.")
    arguments = config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Checkpoint config is missing its recorded training arguments.")
    seed = arguments.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Checkpoint config must contain a non-negative integer training seed.")
    return seed


def aggregate_evaluation_records(
    records: list[EvaluationRecord],
    *,
    minimum_training_seeds: int = 3,
    minimum_evaluation_seeds: int = 1,
) -> list[dict[str, Any]]:
    """Aggregate evaluation runs hierarchically across independent training seeds."""

    if minimum_training_seeds <= 0 or minimum_evaluation_seeds <= 0:
        raise ValueError("Minimum training and evaluation seed counts must be positive.")
    groups: dict[tuple[str, str, str], list[EvaluationRecord]] = {}
    for record in records:
        groups.setdefault((record.mode, record.variant, record.condition), []).append(record)
    rows: list[dict[str, Any]] = []
    for (mode, variant, condition), group in sorted(groups.items()):
        records_by_training_seed: dict[int, list[EvaluationRecord]] = {}
        seen_seed_pairs: set[tuple[int, int]] = set()
        for record in group:
            seed_pair = (record.training_seed, record.evaluation_seed)
            if seed_pair in seen_seed_pairs:
                raise ValueError(
                    f"{mode}/{variant}/{condition} contains duplicate seed pair {seed_pair}."
                )
            seen_seed_pairs.add(seed_pair)
            records_by_training_seed.setdefault(record.training_seed, []).append(record)

        if len(records_by_training_seed) < minimum_training_seeds:
            raise ValueError(
                f"{mode}/{variant}/{condition} has {len(records_by_training_seed)} "
                f"training seeds; at least {minimum_training_seeds} are required."
            )
        evaluation_seed_sets = {
            training_seed: {record.evaluation_seed for record in training_records}
            for training_seed, training_records in records_by_training_seed.items()
        }
        for training_seed, evaluation_seeds in evaluation_seed_sets.items():
            if len(evaluation_seeds) < minimum_evaluation_seeds:
                raise ValueError(
                    f"{mode}/{variant}/{condition} training seed {training_seed} has "
                    f"{len(evaluation_seeds)} evaluation seeds; at least "
                    f"{minimum_evaluation_seeds} are required."
                )
        reference_evaluation_seeds = next(iter(evaluation_seed_sets.values()))
        if any(
            evaluation_seeds != reference_evaluation_seeds
            for evaluation_seeds in evaluation_seed_sets.values()
        ):
            raise ValueError(
                f"{mode}/{variant}/{condition} must use the same evaluation seeds "
                "for every training seed."
            )

        metric_names = set(group[0].metrics)
        correlation_names = set(group[0].reconstruction_correlations)
        if any(set(record.metrics) != metric_names for record in group):
            raise ValueError(f"{mode}/{variant}/{condition} metric schemas do not match.")
        if any(set(record.reconstruction_correlations) != correlation_names for record in group):
            raise ValueError(f"{mode}/{variant}/{condition} reconstruction schemas do not match.")
        ordered_training_groups = [
            records_by_training_seed[seed] for seed in sorted(records_by_training_seed)
        ]
        flattened = {
            **{
                f"metric/{name}": [
                    statistics.fmean(record.metrics[name] for record in training_group)
                    for training_group in ordered_training_groups
                ]
                for name in metric_names
            },
            **{
                f"reconstruction_correlation/{name}": [
                    statistics.fmean(
                        record.reconstruction_correlations[name]
                        for record in training_group
                    )
                    for training_group in ordered_training_groups
                ]
                for name in correlation_names
            },
            "budget/wallclock_seconds": [
                statistics.fmean(record.wallclock_seconds for record in training_group)
                for training_group in ordered_training_groups
            ],
            "budget/environment_steps": [
                statistics.fmean(
                    float(record.evaluation_steps * record.num_envs)
                    for record in training_group
                )
                for training_group in ordered_training_groups
            ],
        }
        row: dict[str, Any] = {
            "mode": mode,
            "variant": variant,
            "condition": condition,
            "training_seeds": len(records_by_training_seed),
            "evaluation_seeds_per_training_seed": len(reference_evaluation_seeds),
            "evaluation_runs": len(group),
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
    header = [
        "mode",
        "variant",
        "condition",
        "training_seeds",
        "evaluation_seeds_per_training_seed",
        "evaluation_runs",
        *display_metrics,
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        values = [
            str(row["mode"]),
            str(row["variant"]),
            str(row["condition"]),
            str(row["training_seeds"]),
            str(row["evaluation_seeds_per_training_seed"]),
            str(row["evaluation_runs"]),
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
