"""Tests for the lightweight training CLI before IsaacLab launcher setup."""

from __future__ import annotations

import pytest

from script.cli_args import (
    FASTSAC_BASELINE_TASK,
    FASTWMR_TASK,
    build_play_parser,
    build_train_parser,
    resolve_max_estimator_feature_age,
    resolve_network_hidden_dims,
    validate_play_args,
    validate_train_args,
)


def test_train_cli_defaults_to_fastwmr_and_validates() -> None:
    args = build_train_parser().parse_args([])

    validate_train_args(args)

    assert args.task == FASTWMR_TASK
    assert args.resume is None
    assert args.log_interval == 5
    assert args.checkpoint_interval == 50
    assert args.log_dir is None
    assert args.fastwmr_version == "v2"
    assert args.estimator_update_interval == 8
    assert args.max_estimator_feature_age is None
    assert resolve_max_estimator_feature_age(args) == 256
    assert args.fresh_reconstruction_fraction == pytest.approx(0.5)
    assert args.stored_feature_replay_horizon is None
    assert args.reconstruction_gate_quality_threshold == pytest.approx(0.45)
    assert args.reconstruction_gate_base_velocity_rmse_threshold == pytest.approx(
        0.65
    )
    assert args.reconstruction_gate_contact_bce_threshold == pytest.approx(0.55)
    assert tuple(args.control_reconstruction_fields) == (
        "base_lin_vel",
        "foot_contacts",
    )
    assert not args.continue_online_estimator_after_snapshot
    assert not args.keep_pre_snapshot_replay
    assert resolve_network_hidden_dims(args) == (512, 768)
    assert args.normalizer_freeze_iteration is None
    assert args.burn_in_length == 32
    assert args.learning_length > 0
    assert args.episode_start_fraction == pytest.approx(0.25)
    assert args.recent_replay_horizon == 200_000


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


def test_train_cli_accepts_narrow_feature_freshness_horizon() -> None:
    args = build_train_parser().parse_args(
        [
            "--num-envs",
            "64",
            "--batch-size",
            "8192",
            "--num-updates",
            "8",
            "--estimator-update-interval",
            "8",
            "--max-estimator-feature-age",
            "127",
        ]
    )

    validate_train_args(args)

    assert resolve_max_estimator_feature_age(args) == 127


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


def test_train_cli_accepts_fastsac_with_shared_penalty_override() -> None:
    args = build_train_parser().parse_args(
        [
            "--task",
            FASTSAC_BASELINE_TASK,
            "--penalty-min-completed-episodes",
            "1000000",
        ]
    )

    validate_train_args(args)

    assert args.penalty_min_completed_episodes == 1_000_000


def test_train_cli_rejects_fastsac_recent_replay_ablation() -> None:
    args = build_train_parser().parse_args(
        [
            "--task",
            FASTSAC_BASELINE_TASK,
            "--recent-replay-horizon",
            "4096",
        ]
    )

    with pytest.raises(ValueError, match="require the FastWMR task"):
        validate_train_args(args)


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


def test_train_cli_accepts_normalizer_freeze_from_start() -> None:
    args = build_train_parser().parse_args(
        ["--normalizer-freeze-iteration", "0"]
    )

    validate_train_args(args)

    assert args.normalizer_freeze_iteration == 0


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
        ("--validation-interval", "0", "--validation-interval"),
        ("--initial-validation-updates", "-1", "--initial-validation-updates"),
        (
            "--normalizer-freeze-iteration",
            "-1",
            "--normalizer-freeze-iteration",
        ),
        ("--estimator-update-interval", "0", "--estimator-update-interval"),
        ("--max-estimator-feature-age", "-1", "--max-estimator-feature-age"),
        ("--control-estimator-tau", "0", "--control-estimator-tau"),
        (
            "--reconstruction-gate-base-velocity-rmse-threshold",
            "0",
            "--reconstruction-gate-base-velocity-rmse-threshold",
        ),
        (
            "--reconstruction-gate-contact-bce-threshold",
            "0",
            "--reconstruction-gate-contact-bce-threshold",
        ),
        ("--estimator-cache-steps", "0", "--estimator-cache-steps"),
        ("--episode-start-fraction", "1.1", "--episode-start-fraction"),
        ("--run-name", "nested/run", "--run-name"),
    ),
)
def test_train_cli_rejects_invalid_resume_and_sequence_values(argument, value, message) -> None:
    args = build_train_parser().parse_args([argument, value])

    with pytest.raises(ValueError, match=message):
        validate_train_args(args)
