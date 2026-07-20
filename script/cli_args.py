"""Command-line arguments shared by FastWMR training scripts."""

from __future__ import annotations

import argparse


FASTSAC_BASELINE_TASK = "Isaac-Velocity-G1-FastSAC-Baseline-v0"


def build_train_parser() -> argparse.ArgumentParser:
    """Create the algorithm parser before IsaacLab adds launcher arguments."""

    parser = argparse.ArgumentParser(description="Train or smoke-test the FastSAC core baseline.")
    parser.add_argument("--task", default=FASTSAC_BASELINE_TASK)
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
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.random_action_steps < 0:
        raise ValueError("--random-action-steps must be non-negative.")
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
    if args.normalization_epsilon <= 0.0 or args.normalization_clip <= 0.0:
        raise ValueError("Observation normalization epsilon and clip must be positive.")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        raise ValueError("--episode-length-s must be positive when provided.")
