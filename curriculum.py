# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Penalty curriculum for FastWMR rewards.

This module will implement the Holosoma-style curriculum used by the FastSAC
baseline: domain randomization is enabled at full strength from the beginning,
while selected penalty reward weights are increased as training becomes stable.

Planned responsibilities:
- Track average episode length or an equivalent stability metric.
- Expose penalty scale levels to ``rewards.py``.
- Provide parameters such as ``enabled`` and ``level_up_threshold`` for CLI and
  config overrides.

This file should not sample friction, push, payload, or other physical random
variables. Those belong in ``randomization.py``.
"""

