"""Training entry point for FastWMR.

Planned responsibilities:
- Parse CLI args from ``cli_args.py``.
- Create the FastWMR or FastSAC-baseline IsaacLab task.
- Build the env wrapper, networks, transition replay, estimator rollout cache,
  and agent.
- Alternate rollout collection with learner updates.
- Save checkpoints and config snapshots under ``logs/fastwmr``.

This script is intentionally separate from environment config files so torch and
custom learner imports happen only during actual training.
"""
