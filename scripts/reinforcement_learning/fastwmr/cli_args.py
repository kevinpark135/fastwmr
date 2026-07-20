"""CLI argument helpers for FastWMR scripts.

Planned argument groups:
- task and logging paths
- FastSAC hyperparameters such as batch size, gamma, update count, and entropy
  target
- FastWMR hyperparameters such as estimator rollout length, transition replay
  capacity, hidden dimension, and control-feature mode
- ablation toggles for baseline, reconstruction-only control, estimator freeze,
  gradient cutoff, recent replay horizon, and symmetry augmentation
"""
