"""Recurrent history encoder E_psi for FastWMR.

This module will implement the LSTM encoder that compresses raw observation
history into a recurrent context for world-state reconstruction.

Planned interface:
- ``forward_rollout``: update per-environment online hidden/cell states during
  data collection.
- ``forward_sequence``: re-infer hidden context from replayed raw observation
  sequences using the current estimator parameters.

Do not store hidden states in replay as training targets; they become stale when
the encoder changes.
"""

