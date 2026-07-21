"""Tests for stage-9 evaluation records and multi-seed tables."""

import json

import pytest

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EvaluationRecord,
    aggregate_evaluation_records,
    load_evaluation_record,
    write_evaluation_record,
    write_evaluation_summary,
)


def _record(seed: int, value: float = 1.0) -> EvaluationRecord:
    return EvaluationRecord(
        mode="fastwmr",
        variant="default",
        condition="nominal_rough",
        seed=seed,
        checkpoint="checkpoint.pt",
        checkpoint_environment_steps=100,
        evaluation_steps=20,
        num_envs=4,
        wallclock_seconds=value + 1.0,
        metrics={
            "return_mean": value,
            "fall_rate": 0.1 * value,
            "linear_tracking_error": 0.2,
            "yaw_tracking_error": 0.3,
        },
        reconstruction_correlations={"overall": 0.5},
    )


def test_evaluation_record_round_trip_and_summary(tmp_path) -> None:
    paths = [
        write_evaluation_record(tmp_path / f"seed_{seed}.json", _record(seed, float(seed)))
        for seed in (1, 2, 3)
    ]
    records = [load_evaluation_record(path) for path in paths]
    rows = aggregate_evaluation_records(records)
    output_paths = write_evaluation_summary(tmp_path / "summary", rows)

    assert len(rows) == 1
    assert rows[0]["metric/return_mean/mean"] == pytest.approx(2.0)
    assert rows[0]["metric/return_mean/std"] == pytest.approx(1.0)
    assert all(path.is_file() for path in output_paths)
    assert json.loads(output_paths[0].read_text())["format_version"] == 1
    assert "+/-" in output_paths[2].read_text()


def test_evaluation_aggregate_requires_three_unique_seeds() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        aggregate_evaluation_records([_record(1), _record(2)])
