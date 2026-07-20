"""Network modules for FastWMR."""

from .actor import TanhGaussianActor
from .critic import ScalarQNetwork, TargetTwinScalarCritic, TwinScalarCritic

__all__ = ["ScalarQNetwork", "TanhGaussianActor", "TargetTwinScalarCritic", "TwinScalarCritic"]
