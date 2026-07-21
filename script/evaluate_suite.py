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
    write_evaluation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastWMR stage-9 evaluation suite.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=(42, 43, 44))
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
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.variant):
        parser.error("--checkpoint and --variant must be provided the same number of times.")
    if len(set(args.seed)) < 3:
        parser.error("The stage-9 suite requires at least three unique seeds.")
    if args.steps <= 0 or args.num_envs <= 0 or min(args.seed) < 0:
        parser.error("Evaluation steps, environments, and seeds must be non-negative/positive.")
    if any(not variant or Path(variant).name != variant for variant in args.variant):
        parser.error("Each --variant must be one non-empty path component.")

    output_directory = args.output_dir.expanduser().resolve()
    play_script = Path(__file__).with_name("play.py")
    result_paths: list[Path] = []
    for checkpoint, variant in zip(args.checkpoint, args.variant, strict=True):
        metadata = inspect_training_checkpoint(checkpoint, map_location="cpu")
        for condition in args.condition:
            for seed in args.seed:
                output = (
                    output_directory
                    / metadata.mode.value
                    / variant
                    / condition
                    / f"seed_{seed}.json"
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
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--num-envs",
                    str(args.num_envs),
                    "--output",
                    str(output),
                    "--device",
                    args.device,
                    "--headless",
                ]
                subprocess.run(command, check=True)
                result_paths.append(output)

    records = [load_evaluation_record(path) for path in result_paths]
    rows = aggregate_evaluation_records(records, minimum_seeds=3)
    paths = write_evaluation_summary(output_directory, rows)
    print("evaluation_summary=" + ",".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
