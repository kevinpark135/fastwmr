"""Replay warm-up and update loops for FastSAC and sequence-based FastWMR."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..buffers import SequenceReplayBatch, TransitionReplayBuffer
from ..config import ReplayUpdateCfg, SequenceReplayCfg
from .sac_update import SACFeatureSource, SACTransitionBatch, SACUpdateMetrics, SACUpdater


SequenceFeatureProcessor = Callable[[SequenceReplayBatch], torch.Tensor]


class FastSACReplayUpdateLoop:
    """Ordinary transition sampling after random-action replay warm-up."""

    def __init__(
        self,
        replay: TransitionReplayBuffer,
        updater: SACUpdater,
        cfg: ReplayUpdateCfg,
        *,
        learner_device: torch.device | str,
    ) -> None:
        self.replay = replay
        self.updater = updater
        self.cfg = cfg
        self.learner_device = torch.device(learner_device)
        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def warming_up(self) -> bool:
        return self.environment_steps < self.cfg.random_action_steps

    @property
    def ready(self) -> bool:
        return (
            not self.warming_up
            and len(self.replay) >= self.cfg.minimum_replay_size
            and self.replay.can_sample(self.cfg.batch_size)
        )

    def advance_environment(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("Environment step increment must be positive.")
        self.environment_steps += steps

    @torch.no_grad()
    def select_actions(self, states: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """Use uniform bounded actions during warm-up, then the learned actor."""

        actor = self.updater.actor
        if states.shape[-1] != actor.input_dim:
            raise ValueError("Action-selection state dimension does not match the actor.")
        if self.warming_up:
            random_values = torch.rand((*states.shape[:-1], actor.action_dim), device=states.device)
            return actor.action_low + random_values * (actor.action_high - actor.action_low)
        return actor.act(states, deterministic=deterministic)

    def run_updates(self, *, generator: torch.Generator | None = None) -> list[SACUpdateMetrics]:
        if not self.ready:
            return []
        metrics: list[SACUpdateMetrics] = []
        for _ in range(self.cfg.num_updates):
            replay_batch = self.replay.sample(self.cfg.batch_size, generator=generator)
            batch = SACTransitionBatch.from_replay(
                replay_batch,
                feature_source=SACFeatureSource.POLICY_OBSERVATION,
            ).to(self.learner_device)
            metrics.append(self.updater.update(batch))
            self.gradient_steps += 1
        return metrics


class FastWMRSequenceUpdateLoop(FastSACReplayUpdateLoop):
    """Sequence replay that delegates current-estimator reconstruction.

    ``sequence_feature_processor`` must run burn-in/current-estimator inference,
    finish any estimator update for the full sequence, and return detached
    control features for the ``L + 1`` learning observations.
    """

    def __init__(
        self,
        replay: TransitionReplayBuffer,
        updater: SACUpdater,
        cfg: ReplayUpdateCfg,
        sequence_cfg: SequenceReplayCfg,
        sequence_feature_processor: SequenceFeatureProcessor,
        *,
        learner_device: torch.device | str,
    ) -> None:
        super().__init__(replay, updater, cfg, learner_device=learner_device)
        self.sequence_cfg = sequence_cfg
        self.sequence_feature_processor = sequence_feature_processor

    @property
    def ready(self) -> bool:
        return (
            not self.warming_up
            and len(self.replay) >= self.cfg.minimum_replay_size
            and self.replay.can_sample_sequences(
                self.sequence_cfg.batch_size,
                self.sequence_cfg.burn_in_length,
                self.sequence_cfg.learning_length,
                require_episode_start=self.sequence_cfg.require_episode_start,
            )
        )

    def run_updates(self, *, generator: torch.Generator | None = None) -> list[SACUpdateMetrics]:
        if not self.ready:
            return []
        metrics: list[SACUpdateMetrics] = []
        for _ in range(self.cfg.num_updates):
            sequence = self.replay.sample_sequences(
                self.sequence_cfg.batch_size,
                self.sequence_cfg.burn_in_length,
                self.sequence_cfg.learning_length,
                require_episode_start=self.sequence_cfg.require_episode_start,
                device=self.learner_device,
                generator=generator,
            )
            learning_features = self.sequence_feature_processor(sequence)
            batch = self._build_learning_batch(sequence, learning_features)
            metrics.append(self.updater.update(batch))
            self.gradient_steps += 1
        return metrics

    def _build_learning_batch(
        self,
        sequence: SequenceReplayBatch,
        learning_features: torch.Tensor,
    ) -> SACTransitionBatch:
        expected_shape = (
            sequence.batch_size,
            sequence.learning_length + 1,
            self.updater.actor.input_dim,
        )
        if learning_features.shape != expected_shape:
            raise ValueError(
                f"Sequence feature processor must return shape {expected_shape}, got {tuple(learning_features.shape)}."
            )
        if learning_features.requires_grad:
            raise ValueError("SAC learning features must be detached after estimator update.")
        if not torch.isfinite(learning_features).all():
            raise ValueError("SAC learning features must be finite.")
        return SACTransitionBatch(
            states=learning_features[:, :-1],
            actions=sequence.learning_actions,
            rewards=sequence.learning_rewards,
            next_states=learning_features[:, 1:],
            terminated=sequence.learning_terminated,
            truncated=sequence.learning_truncated,
        )
