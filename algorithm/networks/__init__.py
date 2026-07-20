"""Network modules for FastWMR."""

from .actor import TanhGaussianActor
from .critic import (
    C51QNetwork,
    ScalarQNetwork,
    TargetTwinC51Critic,
    TargetTwinScalarCritic,
    TwinC51Critic,
    TwinScalarCritic,
)

__all__ = [
    "C51QNetwork",
    "ScalarQNetwork",
    "TanhGaussianActor",
    "TargetTwinC51Critic",
    "TargetTwinScalarCritic",
    "TwinC51Critic",
    "TwinScalarCritic",
]
