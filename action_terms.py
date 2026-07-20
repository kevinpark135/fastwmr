# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional runtime action terms for FastWMR.

This file is reserved for heavier action-term implementations that may touch
Isaac Sim/USD runtime objects. It is intentionally separated from
``actions.py`` so task registration can remain import-safe.

Planned responsibilities:
- Implement action delay, motor offset injection, or other partial-observability
  mechanisms that belong at the runtime action layer.
- Keep these terms compatible with replay logging so the learner can reconstruct
  exactly which action was applied at each timestep.

Import rule:
Only import this module from execution paths that already have a running
SimulationApp or a fully initialized IsaacLab environment.
"""

