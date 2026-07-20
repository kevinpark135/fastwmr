"""Evaluation entry point for FastWMR checkpoints.

Planned responsibilities:
- Load a trained actor/estimator checkpoint.
- Run the ``-Play`` task variant.
- Toggle perturbation ablations such as push, friction, and payload from CLI.
- Report rollout stability, command tracking, reconstruction quality, and
  per-environment hidden-state reset behavior.
"""
