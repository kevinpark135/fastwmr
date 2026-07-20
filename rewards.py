# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for FastWMR G1 locomotion.

This file will adapt the existing G1 rewards toward the Holosoma/FastSAC
minimal-reward style: a small number of velocity-tracking, posture, contact, and
smoothness terms rather than a large hand-tuned reward suite.

Planned responsibilities:
- Reuse existing G1/IsaacLab reward helpers where possible.
- Keep the number of active terms small enough that failures are diagnosable.
- Multiply penalty terms by scales provided by ``curriculum.py`` instead of
  hard-coding a single fixed pressure from the start of training.

Important distinction:
The default curriculum should not gradually increase domain-randomization
strength. Randomization stays strong from the beginning; penalty weights become
stricter as average episode length improves.
"""

