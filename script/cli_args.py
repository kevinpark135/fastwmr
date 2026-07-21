"""Command-line arguments shared by FastWMR training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


FASTSAC_BASELINE_TASK = "Isaac-Velocity-G1-FastSAC-Baseline-v0"
FASTWMR_TASK = "Isaac-Velocity-G1-FastWMR-v0"
TRAIN_TASKS = (FASTSAC_BASELINE_TASK, FASTWMR_TASK)


def build_train_parser() -> argparse.ArgumentParser:
    """Create the algorithm parser before IsaacLab adds launcher arguments."""

    parser = argparse.ArgumentParser(description="Train FastSAC or FastWMR on the Rough G1 task.")
    parser.add_argument("--task", choices=TRAIN_TASKS, default=FASTWMR_TASK)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--log-dir", default="logs/fastwmr")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--disable-final-checkpoint", action="store_true")
    parser.add_argument("--estimator-hidden-dim", type=int, default=256)
    parser.add_argument("--estimator-num-layers", type=int, default=1)
    parser.add_argument("--estimator-learning-rate", type=float, default=3e-4)
    parser.add_argument("--estimator-weight-decay", type=float, default=1e-3)
    parser.add_argument("--estimator-cache-steps", type=int, default=64)
    parser.add_argument("--sequence-batch-size", type=int, default=256)
    parser.add_argument("--burn-in-length", type=int, default=16)
    parser.add_argument("--learning-length", type=int, default=8)
    parser.add_argument("--require-episode-start", action="store_true")
    parser.add_argument("--disable-gradient-boundary-checks", action="store_true")
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
        "sequence_batch_size",
        "learning_length",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.random_action_steps < 0:
        raise ValueError("--random-action-steps must be non-negative.")
    if args.burn_in_length < 0:
        raise ValueError("--burn-in-length must be non-negative.")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative.")
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
