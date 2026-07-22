"""Command-line arguments shared by FastWMR training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


FASTSAC_BASELINE_TASK = "Isaac-Velocity-G1-FastSAC-Baseline-v0"
FASTWMR_TASK = "Isaac-Velocity-G1-FastWMR-v0"
FASTSAC_BASELINE_PLAY_TASK = "Isaac-Velocity-G1-FastSAC-Baseline-Play-v0"
FASTWMR_PLAY_TASK = "Isaac-Velocity-G1-FastWMR-Play-v0"
TRAIN_TASKS = (FASTSAC_BASELINE_TASK, FASTWMR_TASK)
EVALUATION_CONDITIONS = (
    "nominal_rough",
    "friction_low",
    "friction_high",
    "payload_heavy",
    "strong_push",
    "observation_noise",
    "observation_masking",
)


def build_train_parser() -> argparse.ArgumentParser:
    """Create the algorithm parser before IsaacLab adds launcher arguments."""

    parser = argparse.ArgumentParser(description="Train FastSAC or FastWMR on the Rough G1 task.")
    parser.add_argument("--task", choices=TRAIN_TASKS, default=FASTWMR_TASK)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--wallclock-limit-s",
        type=float,
        default=None,
        help="Optional learner wall-clock budget shared by FastSAC/FastWMR comparisons.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed stored in checkpoints and evaluation metadata.",
    )
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--replay-storage-device", default="cpu")
    parser.add_argument("--random-action-steps", type=int, default=10)
    parser.add_argument("--minimum-replay-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-updates", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--critic-type", choices=("c51", "scalar"), default="c51")
    parser.add_argument("--num-atoms", type=int, default=101)
    parser.add_argument("--value-min", type=float, default=-20.0)
    parser.add_argument("--value-max", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--target-update-rate", type=float, default=0.005)
    parser.add_argument("--initial-temperature", type=float, default=0.001)
    parser.add_argument("--target-entropy", type=float, default=0.0)
    parser.add_argument("--normalization-epsilon", type=float, default=1e-5)
    parser.add_argument("--normalization-clip", type=float, default=10.0)
    parser.add_argument("--disable-observation-normalization", action="store_true")
    parser.add_argument("--disable-joint-limit-action-bounds", action="store_true")
    parser.add_argument(
        "--use-soft-joint-limits",
        action="store_true",
        help="Derive actor scaling from IsaacLab soft limits instead of physical joint limits.",
    )
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--log-dir", default="logs/fastwmr")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--disable-final-checkpoint", action="store_true")
    parser.add_argument("--estimator-hidden-dim", type=int, default=256)
    parser.add_argument("--estimator-num-layers", type=int, default=1)
    parser.add_argument("--estimator-learning-rate", type=float, default=3e-4)
    parser.add_argument("--estimator-weight-decay", type=float, default=1e-3)
    parser.add_argument("--estimator-cache-steps", type=int, default=64)
    parser.add_argument("--fastwmr-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--estimator-update-interval", type=int, default=8)
    parser.add_argument("--estimator-updates-per-trigger", type=int, default=1)
    parser.add_argument("--max-estimator-feature-age", type=int, default=100)
    parser.add_argument("--disable-feature-age-filter", action="store_true")
    parser.add_argument("--stored-feature-replay-horizon", type=int, default=200_000)
    parser.add_argument("--control-estimator-tau", type=float, default=0.005)
    parser.add_argument("--reconstruction-gate-start-updates", type=int, default=0)
    parser.add_argument("--reconstruction-gate-warmup-updates", type=int, default=1_000)
    parser.add_argument("--sequence-batch-size", type=int, default=256)
    parser.add_argument("--burn-in-length", type=int, default=16)
    parser.add_argument("--learning-length", type=int, default=8)
    parser.add_argument("--require-episode-start", action="store_true")
    parser.add_argument("--disable-gradient-boundary-checks", action="store_true")
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=500,
        help="Run synchronized value and gradient checks every N learner updates.",
    )
    parser.add_argument(
        "--initial-validation-updates",
        type=int,
        default=16,
        help="Validate every learner update during this initial window.",
    )
    parser.add_argument(
        "--control-feature-mode",
        choices=("obs_and_reconstruction", "reconstruction_only"),
        default="obs_and_reconstruction",
        help="FastWMR actor/critic input used by the shat-only ablation.",
    )
    parser.add_argument("--freeze-estimator", action="store_true")
    parser.add_argument("--disable-gradient-cutoff", action="store_true")
    parser.add_argument("--recent-replay-horizon", type=int, default=None)
    parser.add_argument("--use-symmetry", action="store_true")
    parser.add_argument("--disable-penalty-curriculum", action="store_true")
    parser.add_argument(
        "--penalty-scales",
        type=float,
        nargs=4,
        default=(0.1, 0.3, 0.6, 1.0),
        metavar=("S0", "S1", "S2", "S3"),
    )
    parser.add_argument(
        "--penalty-length-thresholds",
        type=float,
        nargs=3,
        default=(0.25, 0.5, 0.75),
        metavar=("T0", "T1", "T2"),
    )
    parser.add_argument("--penalty-ema-decay", type=float, default=0.9)
    parser.add_argument("--penalty-min-completed-episodes", type=int, default=64)
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Optional episode duration override, useful for reset-path smoke tests.",
    )
    parser.add_argument(
        "--rough-debug",
        action="store_true",
        help="Use terrain level zero and disable corruption, pushes, and terrain curriculum.",
    )
    return parser


def validate_train_args(args: argparse.Namespace) -> None:
    """Reject invalid schedules before expensive simulator construction."""

    positive_fields = (
        "num_envs",
        "steps",
        "replay_capacity",
        "minimum_replay_size",
        "batch_size",
        "num_updates",
        "hidden_dim",
        "log_interval",
        "estimator_hidden_dim",
        "estimator_num_layers",
        "estimator_cache_steps",
        "estimator_update_interval",
        "estimator_updates_per_trigger",
        "stored_feature_replay_horizon",
        "sequence_batch_size",
        "learning_length",
        "validation_interval",
        "penalty_min_completed_episodes",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.random_action_steps < 0:
        raise ValueError("--random-action-steps must be non-negative.")
    if args.wallclock_limit_s is not None and args.wallclock_limit_s <= 0.0:
        raise ValueError("--wallclock-limit-s must be positive when provided.")
    if args.burn_in_length < 0:
        raise ValueError("--burn-in-length must be non-negative.")
    if args.initial_validation_updates < 0:
        raise ValueError("--initial-validation-updates must be non-negative.")
    if args.max_estimator_feature_age < 0:
        raise ValueError("--max-estimator-feature-age must be non-negative.")
    if not 0.0 < args.control_estimator_tau <= 1.0:
        raise ValueError("--control-estimator-tau must be in (0, 1].")
    if args.reconstruction_gate_start_updates < 0:
        raise ValueError("--reconstruction-gate-start-updates must be non-negative.")
    if args.reconstruction_gate_warmup_updates < 0:
        raise ValueError("--reconstruction-gate-warmup-updates must be non-negative.")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative.")
    if args.recent_replay_horizon is not None and args.recent_replay_horizon <= 0:
        raise ValueError("--recent-replay-horizon must be positive when provided.")
    if args.freeze_estimator and args.disable_gradient_cutoff:
        raise ValueError("--freeze-estimator cannot be combined with --disable-gradient-cutoff.")
    if args.fastwmr_version == "v2" and args.disable_gradient_cutoff:
        raise ValueError("--disable-gradient-cutoff is only available in FastWMR v1.")
    fastwmr_ablation_requested = (
        args.control_feature_mode != "obs_and_reconstruction"
        or args.freeze_estimator
        or args.disable_gradient_cutoff
        or args.recent_replay_horizon is not None
        or args.use_symmetry
    )
    if args.task == FASTSAC_BASELINE_TASK and fastwmr_ablation_requested:
        raise ValueError("FastWMR estimator and sequence ablations require the FastWMR task.")
    if len(args.penalty_scales) != len(args.penalty_length_thresholds) + 1:
        raise ValueError("--penalty-scales must contain one more value than thresholds.")
    if any(value <= 0.0 or value > 1.0 for value in args.penalty_scales):
        raise ValueError("--penalty-scales values must be in (0, 1].")
    if any(left >= right for left, right in zip(args.penalty_scales, args.penalty_scales[1:])):
        raise ValueError("--penalty-scales must be strictly increasing.")
    if args.penalty_scales[-1] != 1.0:
        raise ValueError("The final --penalty-scales value must be 1.0.")
    if any(value <= 0.0 or value > 1.0 for value in args.penalty_length_thresholds):
        raise ValueError("--penalty-length-thresholds values must be in (0, 1].")
    if any(
        left >= right
        for left, right in zip(
            args.penalty_length_thresholds,
            args.penalty_length_thresholds[1:],
        )
    ):
        raise ValueError("--penalty-length-thresholds must be strictly increasing.")
    if not 0.0 <= args.penalty_ema_decay < 1.0:
        raise ValueError("--penalty-ema-decay must be in [0, 1).")
    if args.minimum_replay_size < args.batch_size:
        raise ValueError("--minimum-replay-size must be at least --batch-size.")
    if args.replay_capacity < args.minimum_replay_size:
        raise ValueError("--replay-capacity must be at least --minimum-replay-size.")
    if args.hidden_dim % 4 != 0:
        raise ValueError("--hidden-dim must be divisible by four.")
    if args.num_atoms < 2:
        raise ValueError("--num-atoms must be at least two.")
    if args.value_min >= args.value_max:
        raise ValueError("--value-min must be smaller than --value-max.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Optimizer learning rate must be positive and weight decay non-negative.")
    if args.estimator_learning_rate <= 0.0 or args.estimator_weight_decay < 0.0:
        raise ValueError(
            "Estimator learning rate must be positive and weight decay non-negative."
        )
    if args.normalization_epsilon <= 0.0 or args.normalization_clip <= 0.0:
        raise ValueError("Observation normalization epsilon and clip must be positive.")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        raise ValueError("--episode-length-s must be positive when provided.")
    if args.run_name is not None:
        if not args.run_name or Path(args.run_name).name != args.run_name:
            raise ValueError("--run-name must be one non-empty path component.")
    if args.resume is not None and args.task not in TRAIN_TASKS:
        raise ValueError("--resume requires a supported training task.")


def build_play_parser() -> argparse.ArgumentParser:
    """Create the checkpoint robustness-evaluation parser."""

    parser = argparse.ArgumentParser(description="Evaluate a FastSAC or FastWMR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=(FASTSAC_BASELINE_PLAY_TASK, FASTWMR_PLAY_TASK),
        default=None,
    )
    parser.add_argument("--condition", choices=EVALUATION_CONDITIONS, default="nominal_rough")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation rollout seed; the training seed is read from the checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--observation-noise-std", type=float, default=0.2)
    parser.add_argument("--observation-mask-probability", type=float, default=0.2)
    parser.add_argument("--stochastic", action="store_true")
    return parser


def validate_play_args(args: argparse.Namespace) -> None:
    """Validate rollout budgets and policy-observation perturbations."""

    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not args.variant or Path(args.variant).name != args.variant:
        raise ValueError("--variant must be one non-empty path component.")
    if args.observation_noise_std < 0.0:
        raise ValueError("--observation-noise-std must be non-negative.")
    if not 0.0 <= args.observation_mask_probability <= 1.0:
        raise ValueError("--observation-mask-probability must be in [0, 1].")
