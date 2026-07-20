"""FastWMR estimator, SAC update, rollout, and learner orchestration modules."""

from .estimator_update import BurnInUnrollOutput, RecurrentSequenceEstimator, burn_in_and_unroll
from .fastwmr_agent import FastSACReplayUpdateLoop, FastWMRSequenceUpdateLoop, SequenceFeatureProcessor
from .rollout_worker import FastSACRolloutCollector, RolloutStepResult
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
    "FastSACRolloutCollector",
    "FastWMRSequenceUpdateLoop",
    "RecurrentSequenceEstimator",
    "RolloutStepResult",
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
