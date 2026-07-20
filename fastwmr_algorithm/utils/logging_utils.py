"""Logging helpers for FastWMR experiments.

Planned responsibilities:
- Record estimator, critic, actor, entropy, and target-update metrics.
- Attach ablation tags such as estimator rollout length, control-feature mode,
  estimator freeze, gradient cutoff, and symmetry usage.
- Save config snapshots so each checkpoint can be traced back to the exact
  algorithm and environment settings.
"""
