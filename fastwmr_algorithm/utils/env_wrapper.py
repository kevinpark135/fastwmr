"""Environment adapter for FastWMR.

This wrapper will normalize the interface between IsaacLab ManagerBasedRLEnv and
the custom FastWMR learner.

Planned responsibilities:
- Split policy observations ``o_t`` from privileged reconstruction targets
  ``s_t``.
- Surface terminal, truncated, and final-observation information without losing
  reset-boundary semantics.
- Keep the learner independent from IsaacLab observation-group naming details.
"""

