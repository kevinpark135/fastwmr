"""Tests for the lightweight training CLI before IsaacLab launcher setup."""

from __future__ import annotations

import pytest

from script.cli_args import FASTWMR_TASK, build_train_parser, validate_train_args


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
