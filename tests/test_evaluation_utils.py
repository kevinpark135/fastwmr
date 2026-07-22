"""Tests for stage-9 evaluation records and multi-seed tables."""

import json

import pytest

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EvaluationRecord,
    aggregate_evaluation_records,
    load_evaluation_record,
    training_seed_from_config,
    write_evaluation_record,
    write_evaluation_summary,
)


def _record(
    training_seed: int,
    evaluation_seed: int,
    value: float = 1.0,
) -> EvaluationRecord:
    return EvaluationRecord(
        mode="fastwmr",
        variant="default",
        condition="nominal_rough",
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
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
    paths = []
    for training_seed in (1, 2, 3):
        for evaluation_seed, offset in ((10, -0.5), (11, 0.5)):
            paths.append(
                write_evaluation_record(
                    tmp_path / f"train_{training_seed}_eval_{evaluation_seed}.json",
                    _record(training_seed, evaluation_seed, training_seed + offset),
                )
            )
    records = [load_evaluation_record(path) for path in paths]
    rows = aggregate_evaluation_records(records, minimum_evaluation_seeds=2)
    output_paths = write_evaluation_summary(tmp_path / "summary", rows)

    assert len(rows) == 1
    assert rows[0]["metric/return_mean/mean"] == pytest.approx(2.0)
    assert rows[0]["metric/return_mean/std"] == pytest.approx(1.0)
    assert rows[0]["training_seeds"] == 3
    assert rows[0]["evaluation_seeds_per_training_seed"] == 2
    assert rows[0]["evaluation_runs"] == 6
    assert all(path.is_file() for path in output_paths)
    assert json.loads(output_paths[0].read_text())["format_version"] == 2
    assert "+/-" in output_paths[2].read_text()


def test_evaluation_aggregate_requires_three_training_seeds() -> None:
    records = [
        _record(training_seed, evaluation_seed)
        for training_seed in (1, 2)
        for evaluation_seed in (10, 11, 12)
    ]
    with pytest.raises(ValueError, match="training seeds; at least 3"):
        aggregate_evaluation_records(records)


def test_evaluation_aggregate_requires_matching_evaluation_seeds() -> None:
    records = [
        _record(training_seed, evaluation_seed)
        for training_seed, evaluation_seeds in ((1, (10, 11)), (2, (10, 11)), (3, (10,)))
        for evaluation_seed in evaluation_seeds
    ]
    with pytest.raises(ValueError, match="same evaluation seeds"):
        aggregate_evaluation_records(records)


def test_training_seed_is_read_from_checkpoint_config() -> None:
    assert training_seed_from_config({"arguments": {"seed": 42}}) == 42
    with pytest.raises(ValueError, match="training seed"):
        training_seed_from_config({"arguments": {"seed": True}})
