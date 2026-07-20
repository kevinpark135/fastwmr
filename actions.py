# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional action configuration helpers for FastWMR.

This file is reserved for lightweight action cfg objects, especially if delayed
actions, motor offsets, or joint-limit-aware action bounds need to be expressed
at the environment configuration level.

Planned responsibilities:
- Describe action-space choices that are safe to import during IsaacLab config
  parsing.
- Keep heavy runtime action implementations out of this file; put those in
  ``action_terms.py`` instead.
- Document whether the policy action is interpreted as target joint position,
  delta from default pose, or another PD-control command.
"""

