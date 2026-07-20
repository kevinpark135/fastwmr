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
from .history_encoder import HistoryEncoder

__all__ = [
    "C51QNetwork",
    "HistoryEncoder",
    "ScalarQNetwork",
    "TanhGaussianActor",
    "TargetTwinC51Critic",
    "TargetTwinScalarCritic",
    "TwinC51Critic",
    "TwinScalarCritic",
]
