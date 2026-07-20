# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level FastWMR agent configuration entry point.

This file will hold the IsaacLab-facing config object or config dictionary for
FastWMR runs. It should bridge task registration and the custom learner without
pulling torch networks into the environment config import path.

Planned responsibilities:
- Point training scripts at ``fastwmr_algorithm/config.py`` defaults.
- Expose key CLI-overridable fields: transition replay capacity, estimator
  rollout length, batch size, update count, discount, entropy settings, and
  control-feature ablations.
- Keep FastSAC-baseline and FastWMR settings close enough that differences are
  intentional and easy to log.
"""
