"""FastWMR estimator, SAC update, rollout, and learner orchestration modules."""

from .estimator_update import BurnInUnrollOutput, RecurrentSequenceEstimator, burn_in_and_unroll
from .fastwmr_agent import FastSACReplayUpdateLoop, FastWMRSequenceUpdateLoop, SequenceFeatureProcessor
from .sac_update import (
    ActorLossOutput,
    CriticLossOutput,
    EntropyTemperature,
    SACFeatureSource,
    SACTransitionBatch,
    SACUpdateMetrics,
    SACUpdater,
    compute_actor_loss,
    compute_critic_loss,
    compute_critic_target,
    compute_temperature_loss,
)

__all__ = [
    "ActorLossOutput",
    "BurnInUnrollOutput",
    "CriticLossOutput",
    "EntropyTemperature",
    "FastSACReplayUpdateLoop",
    "FastWMRSequenceUpdateLoop",
    "RecurrentSequenceEstimator",
    "SACFeatureSource",
    "SACTransitionBatch",
    "SACUpdateMetrics",
    "SACUpdater",
    "SequenceFeatureProcessor",
    "burn_in_and_unroll",
    "compute_actor_loss",
    "compute_critic_loss",
    "compute_critic_target",
    "compute_temperature_loss",
]
