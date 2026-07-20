"""Optional symmetry augmentation utilities.

Holosoma uses left/right symmetry augmentation for FastSAC. In FastWMR this is
more delicate because the estimator is recurrent: mirrored samples should be
formed by mirroring the raw sequence and forwarding it through the LSTM, not by
mirroring only the final ``shat`` at SAC-loss time.

Initial recommendation:
Keep ``use_symmetry=False`` until the base estimator, replay, and SAC update
pipeline are stable.
"""

