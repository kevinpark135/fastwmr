"""End-to-end tests for FastWMR collection, reconstruction, and C51 SAC updates."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    C51SACUpdater,
    EntropyTemperature,
    EstimatorUpdater,
    FastWMREstimatorRuntime,
    FastWMRRolloutCollector,
    FastWMRSequenceFeatureProcessor,
    FastWMRSequenceUpdateLoop,
    WorldStateEstimator,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    EstimatorRolloutCache,
    EstimatorRolloutCacheSpec,
    ReplayBufferSpec,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_INTERFACE_CFG,
    DistributionalCriticCfg,
    ReplayUpdateCfg,
    SequenceReplayCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    HistoryEncoder,
    TargetTwinC51Critic,
    TanhGaussianActor,
    TwinC51Critic,
    WorldStateDecoder,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    IsaacLabEnvAdapter,
    RunningObservationNormalizer,
)


class _FastWMRObservationManager:
    def __init__(self, owner: "_RawFastWMREnv") -> None:
        self.owner = owner

    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor]:
        del update_history
        interface = DEFAULT_INTERFACE_CFG
        offsets = torch.arange(interface.policy_observation_dim, dtype=torch.float32) * 0.01
        policy = torch.sin(self.owner.state + offsets.unsqueeze(0))
        continuous_offsets = torch.arange(interface.continuous_target_dim, dtype=torch.float32) * 0.02
        continuous = torch.tanh(self.owner.state + continuous_offsets.unsqueeze(0))
        env_ids = torch.arange(self.owner.num_envs, dtype=torch.float32).unsqueeze(-1)
        contact_seed = self.owner.state.floor() + env_ids
        contacts = torch.cat(
            (
                torch.remainder(contact_seed, 2.0),
                torch.remainder(contact_seed + 1.0, 2.0),
            ),
            dim=-1,
        )
        return {
            "policy": policy,
            "privileged": torch.cat((continuous, contacts), dim=-1),
        }


class _RawFastWMREnv:
    def __init__(self) -> None:
        self.device = "cpu"
        self.num_envs = 2
        self.state = torch.zeros(self.num_envs, 1)
        self.observation_manager = _FastWMRObservationManager(self)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids.long()] = 0.0


class _FastWMRGymEnv:
    def __init__(self) -> None:
        self.unwrapped = _RawFastWMREnv()
        self.steps = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        del seed
        self.steps = 0
        self.unwrapped.state.zero_()
        return self.unwrapped.observation_manager.compute(), {}

    def step(self, actions: torch.Tensor):
        interface = DEFAULT_INTERFACE_CFG
        assert actions.shape == (self.unwrapped.num_envs, interface.action_dim)
        assert torch.isfinite(actions).all()
        self.steps += 1
        self.unwrapped.state += 1.0
        terminated = torch.tensor([self.steps % 3 == 0, False])
        truncated = torch.tensor([False, self.steps % 4 == 0])
        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.unwrapped._reset_idx(done_ids)
        observations = self.unwrapped.observation_manager.compute()
        rewards = 1.0 - 0.001 * actions.square().mean(dim=-1)
        return observations, rewards, terminated, truncated, {}

    def close(self) -> None:
        self.closed = True


def _integrated_pipeline() -> tuple[
    IsaacLabEnvAdapter,
    TransitionReplayBuffer,
    RunningObservationNormalizer,
    WorldStateEstimator,
    EstimatorUpdater,
    FastWMREstimatorRuntime,
    FastWMRSequenceFeatureProcessor,
    FastWMRSequenceUpdateLoop,
    FastWMRRolloutCollector,
]:
    interface = DEFAULT_INTERFACE_CFG
    env = IsaacLabEnvAdapter(_FastWMRGymEnv())
    replay = TransitionReplayBuffer(ReplayBufferSpec.fastwmr(capacity=64))
    rollout_cache = EstimatorRolloutCache(
        EstimatorRolloutCacheSpec.fastwmr(capacity_steps=8, num_envs=env.num_envs)
    )
    estimator = WorldStateEstimator(
        HistoryEncoder(interface.policy_observation_dim, hidden_dim=16),
        WorldStateDecoder(16, hidden_dim=16),
    )
    estimator_updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=1e-3),
    )
    runtime = FastWMREstimatorRuntime(estimator, num_envs=env.num_envs)
    normalizer = RunningObservationNormalizer(interface.policy_observation_dim)
    processor = FastWMRSequenceFeatureProcessor(
        estimator_updater,
        runtime,
        rollout_cache,
        observation_normalizer=normalizer,
    )

    actor = TanhGaussianActor(
        interface.actor_input_dim,
        interface.action_dim,
        cfg=TanhGaussianActorCfg(hidden_dim=16),
    )
    critic = TwinC51Critic(
        interface.critic_state_dim,
        interface.action_dim,
        cfg=DistributionalCriticCfg(
            hidden_dim=16,
            num_atoms=11,
            value_min=-5.0,
            value_max=5.0,
        ),
    )
    target_critic = TargetTwinC51Critic.from_online(critic)
    temperature = EntropyTemperature()
    sac_updater = C51SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=3e-4),
    )
    update_loop = FastWMRSequenceUpdateLoop(
        replay,
        sac_updater,
        ReplayUpdateCfg(
            random_action_steps=0,
            minimum_replay_size=4,
            batch_size=4,
            num_updates=1,
        ),
        SequenceReplayCfg(batch_size=1, burn_in_length=1, learning_length=2),
        processor,
        learner_device="cpu",
    )
    collector = FastWMRRolloutCollector(env, replay, update_loop)
    return (
        env,
        replay,
        normalizer,
        estimator,
        estimator_updater,
        runtime,
        processor,
        update_loop,
        collector,
    )


def test_integrated_collection_updates_estimator_runtime_and_c51_sac() -> None:
    torch.manual_seed(31)
    (
        env,
        replay,
        normalizer,
        estimator,
        estimator_updater,
        runtime,
        processor,
        update_loop,
        collector,
    ) = _integrated_pipeline()

    initial_feature = collector.reset(seed=31)
    results = [collector.collect_step() for _ in range(6)]
    transitions = replay.chronological()

    assert initial_feature.shape == (env.num_envs, DEFAULT_INTERFACE_CFG.control_feature_dim)
    assert not initial_feature.requires_grad
    assert replay.total_inserted == 12
    assert update_loop.gradient_steps > 0
    assert processor.updates == update_loop.gradient_steps
    assert sum(result.estimator_updates for result in results) == processor.updates
    assert estimator_updater.version == processor.updates
    assert runtime.estimator_version == estimator_updater.version
    assert runtime.rebuilds == processor.updates
    assert normalizer.samples_seen == env.num_envs * 7
    assert torch.equal(transitions.reset_boundaries, transitions.timesteps == 0)
    assert torch.any(transitions.final_observation_mask)
    assert torch.all(torch.isfinite(transitions.control_features))
    done_indices = transitions.final_observation_mask.nonzero(as_tuple=False).squeeze(-1)
    assert torch.all(
        transitions.final_observations[done_indices]
        != transitions.next_observations[done_indices]
    )
    assert all(
        torch.isfinite(metric)
        for result in results
        for update in result.updates
        for metric in update.__dict__.values()
    )
    assert all(parameter.grad is not None for parameter in estimator.parameters())
    env.close()


def test_replayed_sac_features_use_current_estimator_and_cut_its_gradients() -> None:
    torch.manual_seed(32)
    (
        env,
        replay,
        _normalizer,
        estimator,
        estimator_updater,
        runtime,
        processor,
        update_loop,
        collector,
    ) = _integrated_pipeline()
    collector.reset(seed=32)
    for _ in range(5):
        collector.collect_step()

    sequence = replay.sample_sequences(
        batch_size=1,
        burn_in_length=1,
        learning_length=2,
        device="cpu",
        generator=torch.Generator().manual_seed(32),
    )
    features = processor(sequence)
    current_reconstruction = estimator_updater.reconstruct_sequence(sequence)
    torch.testing.assert_close(
        features[..., -DEFAULT_INTERFACE_CFG.reconstruction_target_dim :],
        current_reconstruction,
    )
    assert not features.requires_grad
    assert runtime.estimator_version == estimator_updater.version

    for parameter in estimator.parameters():
        parameter.grad = None
    sac_batch = update_loop._build_learning_batch(sequence, features)
    update_loop.updater.update(sac_batch)
    assert all(parameter.grad is None for parameter in estimator.parameters())
    env.close()
