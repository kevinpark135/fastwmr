"""Launch a reproducible multi-checkpoint, multi-condition evaluation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    inspect_training_checkpoint,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EvaluationCondition,
    aggregate_evaluation_records,
    load_evaluation_record,
    training_seed_from_config,
    write_evaluation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastWMR stage-9 evaluation suite.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument(
        "--evaluation-seed",
        "--seed",
        dest="evaluation_seed",
        type=int,
        nargs="+",
        default=(42, 43, 44),
        help="Rollout seeds applied to every independently trained checkpoint.",
    )
    parser.add_argument(
        "--condition",
        nargs="+",
        choices=tuple(condition.value for condition in EvaluationCondition),
        default=tuple(condition.value for condition in EvaluationCondition),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/suite"))
    parser.add_argument("--minimum-training-seeds", type=int, default=3)
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.variant):
        parser.error("--checkpoint and --variant must be provided the same number of times.")
    if len(set(args.evaluation_seed)) != len(args.evaluation_seed):
        parser.error("--evaluation-seed values must be unique.")
    if args.minimum_training_seeds <= 0:
        parser.error("--minimum-training-seeds must be positive.")
    if args.steps <= 0 or args.num_envs <= 0 or min(args.evaluation_seed) < 0:
        parser.error("Evaluation steps, environments, and seeds must be non-negative/positive.")
    if any(not variant or Path(variant).name != variant for variant in args.variant):
        parser.error("Each --variant must be one non-empty path component.")

    experiments = []
    training_seeds_by_variant: dict[tuple[str, str], set[int]] = {}
    seen_training_runs: set[tuple[str, str, int]] = set()
    for checkpoint, variant in zip(args.checkpoint, args.variant, strict=True):
        metadata = inspect_training_checkpoint(checkpoint, map_location="cpu")
        training_seed = training_seed_from_config(metadata.config)
        group_key = (metadata.mode.value, variant)
        run_key = (*group_key, training_seed)
        if run_key in seen_training_runs:
            parser.error(
                f"Duplicate {metadata.mode.value}/{variant} checkpoint for "
                f"training seed {training_seed}."
            )
        seen_training_runs.add(run_key)
        training_seeds_by_variant.setdefault(group_key, set()).add(training_seed)
        experiments.append((checkpoint, variant, metadata, training_seed))

    for (mode, variant), training_seeds in training_seeds_by_variant.items():
        if len(training_seeds) < args.minimum_training_seeds:
            parser.error(
                f"{mode}/{variant} has {len(training_seeds)} training seeds; "
                f"at least {args.minimum_training_seeds} checkpoints are required."
            )

    output_directory = args.output_dir.expanduser().resolve()
    play_script = Path(__file__).with_name("play.py")
    result_paths: list[Path] = []
    for checkpoint, variant, metadata, training_seed in experiments:
        for condition in args.condition:
            for evaluation_seed in args.evaluation_seed:
                output = (
                    output_directory
                    / metadata.mode.value
                    / variant
                    / condition
                    / f"train_seed_{training_seed}"
                    / f"eval_seed_{evaluation_seed}.json"
                )
                command = [
                    sys.executable,
                    str(play_script),
                    "--checkpoint",
                    str(checkpoint),
                    "--variant",
                    variant,
                    "--condition",
                    condition,
                    "--seed",
                    str(evaluation_seed),
                    "--steps",
                    str(args.steps),
                    "--num-envs",
                    str(args.num_envs),
                    "--output",
                    str(output),
                    "--device",
                    args.device,
                    "--viz",
                    "none",
                ]
                subprocess.run(command, check=True)
                result_paths.append(output)

    records = [load_evaluation_record(path) for path in result_paths]
    rows = aggregate_evaluation_records(
        records,
        minimum_training_seeds=args.minimum_training_seeds,
        minimum_evaluation_seeds=len(args.evaluation_seed),
    )
    paths = write_evaluation_summary(output_directory, rows)
    print("evaluation_summary=" + ",".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
