"""Aggregate stage-9 JSON records into mean/std evaluation tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    aggregate_evaluation_records,
    load_evaluation_record,
    write_evaluation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FastWMR evaluation records.")
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/summary"))
    parser.add_argument("--minimum-training-seeds", type=int, default=3)
    parser.add_argument("--minimum-evaluation-seeds", type=int, default=1)
    args = parser.parse_args()
    records = [load_evaluation_record(path) for path in args.records]
    rows = aggregate_evaluation_records(
        records,
        minimum_training_seeds=args.minimum_training_seeds,
        minimum_evaluation_seeds=args.minimum_evaluation_seeds,
    )
    paths = write_evaluation_summary(args.output_dir, rows)
    print("evaluation_summary=" + ",".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
