"""Tests for the lightweight training CLI before IsaacLab launcher setup."""

from __future__ import annotations

import pytest

from script.cli_args import (
    FASTWMR_TASK,
    build_play_parser,
    build_train_parser,
    validate_play_args,
    validate_train_args,
)


def test_train_cli_defaults_to_fastwmr_and_validates() -> None:
    args = build_train_parser().parse_args([])

    validate_train_args(args)

    assert args.task == FASTWMR_TASK
    assert args.resume is None
    assert args.checkpoint_interval > 0
    assert args.burn_in_length > 0
    assert args.learning_length > 0


def test_train_cli_accepts_resume_and_sequence_overrides(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    args = build_train_parser().parse_args(
        [
            "--resume",
            str(checkpoint),
            "--run-name",
            "continued",
            "--checkpoint-interval",
            "0",
            "--burn-in-length",
            "0",
            "--learning-length",
            "4",
            "--sequence-batch-size",
            "8",
        ]
    )

    validate_train_args(args)

    assert args.resume == checkpoint
    assert args.checkpoint_interval == 0
    assert args.burn_in_length == 0


def test_train_cli_accepts_fastwmr_ablation_matrix() -> None:
    args = build_train_parser().parse_args(
        [
            "--control-feature-mode",
            "reconstruction_only",
            "--recent-replay-horizon",
            "4096",
            "--use-symmetry",
            "--freeze-estimator",
        ]
    )

    validate_train_args(args)

    assert args.control_feature_mode == "reconstruction_only"
    assert args.recent_replay_horizon == 4096
    assert args.use_symmetry
    assert args.freeze_estimator


def test_train_cli_rejects_frozen_no_cutoff_combination() -> None:
    args = build_train_parser().parse_args(
        ["--freeze-estimator", "--disable-gradient-cutoff"]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_train_args(args)


def test_train_cli_accepts_positive_wallclock_budget() -> None:
    args = build_train_parser().parse_args(["--wallclock-limit-s", "3600"])

    validate_train_args(args)

    assert args.wallclock_limit_s == 3600.0


def test_play_cli_validates_fixed_budget(tmp_path) -> None:
    args = build_play_parser().parse_args(
        ["--checkpoint", str(tmp_path / "checkpoint.pt"), "--variant", "no_cutoff"]
    )

    validate_play_args(args)

    assert args.steps == 1000
    assert args.variant == "no_cutoff"


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("--checkpoint-interval", "-1", "--checkpoint-interval"),
        ("--burn-in-length", "-1", "--burn-in-length"),
        ("--estimator-cache-steps", "0", "--estimator-cache-steps"),
        ("--run-name", "nested/run", "--run-name"),
    ),
)
def test_train_cli_rejects_invalid_resume_and_sequence_values(argument, value, message) -> None:
    args = build_train_parser().parse_args([argument, value])

    with pytest.raises(ValueError, match=message):
        validate_train_args(args)
