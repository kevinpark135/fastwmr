"""End-to-end tests for FastWMR collection, reconstruction, and C51 SAC updates."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    C51SACUpdater,
    EntropyTemperature,
    EstimatorUpdater,
    FastWMRAgent,
    FastWMREstimatorRuntime,
    FastWMRRolloutCollector,
    FastWMRSequenceFeatureProcessor,
    FastWMRSequenceUpdateLoop,
    GradientBoundaryError,
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
    assert update_loop.agent is not None
    assert update_loop.agent.update_steps == update_loop.gradient_steps
    assert all(parameter.grad is None for parameter in estimator.parameters())
    assert all(
        update.update_order == FastWMRAgent.UPDATE_ORDER
        and update.gradient_boundary.enabled
        and update.gradient_boundary.checks == 7
        for update in update_loop.last_agent_updates
    )
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


def test_agent_executes_optimizer_phases_in_declared_order() -> None:
    torch.manual_seed(33)
    (
        env,
        replay,
        _normalizer,
        estimator,
        estimator_updater,
        _runtime,
        _processor,
        update_loop,
        collector,
    ) = _integrated_pipeline()
    collector.reset(seed=33)
    for _ in range(4):
        collector.collect_step()
    sequence = replay.sample_sequences(
        batch_size=1,
        burn_in_length=1,
        learning_length=2,
        device="cpu",
        generator=torch.Generator().manual_seed(33),
    )
    agent = update_loop.agent
    assert agent is not None

    events: list[str] = []
    original_estimator = estimator_updater.update_sequence
    original_critic = update_loop.updater.update_critic
    original_actor = update_loop.updater.update_actor
    original_temperature = update_loop.updater.update_temperature
    original_target = update_loop.updater.update_target

    def update_estimator(sample):
        events.append("estimator")
        return original_estimator(sample)

    def update_critic(batch):
        events.append("critic")
        return original_critic(batch)

    def update_actor(states):
        events.append("actor")
        return original_actor(states)

    def update_temperature(log_probabilities):
        events.append("temperature")
        return original_temperature(log_probabilities)

    def update_target():
        events.append("target")
        return original_target()

    estimator_updater.update_sequence = update_estimator
    update_loop.updater.update_critic = update_critic
    update_loop.updater.update_actor = update_actor
    update_loop.updater.update_temperature = update_temperature
    update_loop.updater.update_target = update_target

    result = agent.update(sequence)

    assert tuple(events) == FastWMRAgent.UPDATE_ORDER
    assert result.update_order == FastWMRAgent.UPDATE_ORDER
    assert result.gradient_boundary.estimator_gradient_norm is not None
    assert result.gradient_boundary.estimator_gradient_norm > 0.0
    modules = (
        estimator,
        update_loop.updater.critic,
        update_loop.updater.actor,
        update_loop.updater.temperature,
        update_loop.updater.target_critic,
    )
    assert all(parameter.grad is None for module in modules for parameter in module.parameters())
    env.close()


def test_gradient_guard_detects_actor_to_estimator_leak_and_cleans_up() -> None:
    torch.manual_seed(34)
    (
        env,
        replay,
        _normalizer,
        estimator,
        _estimator_updater,
        _runtime,
        _processor,
        update_loop,
        collector,
    ) = _integrated_pipeline()
    collector.reset(seed=34)
    for _ in range(4):
        collector.collect_step()
    sequence = replay.sample_sequences(
        batch_size=1,
        burn_in_length=1,
        learning_length=2,
        device="cpu",
        generator=torch.Generator().manual_seed(34),
    )
    agent = update_loop.agent
    assert agent is not None
    original_actor = update_loop.updater.update_actor

    def leaking_actor_update(states):
        output = original_actor(states)
        parameter = next(estimator.parameters())
        parameter.grad = torch.ones_like(parameter)
        return output

    update_loop.updater.update_actor = leaking_actor_update

    with pytest.raises(
        GradientBoundaryError,
        match="actor update leaked a gradient into estimator",
    ):
        agent.update(sequence)

    modules = (
        estimator,
        update_loop.updater.critic,
        update_loop.updater.actor,
        update_loop.updater.temperature,
        update_loop.updater.target_critic,
    )
    assert all(parameter.grad is None for module in modules for parameter in module.parameters())
    env.close()


def test_agent_rejects_optimizer_parameter_overlap() -> None:
    (
        env,
        _replay,
        _normalizer,
        estimator,
        _estimator_updater,
        _runtime,
        processor,
        update_loop,
        _collector,
    ) = _integrated_pipeline()
    actor_parameters = list(update_loop.updater.actor.parameters())
    update_loop.updater.actor_optimizer = torch.optim.Adam(
        (*actor_parameters, next(estimator.parameters())),
        lr=3e-4,
    )

    with pytest.raises(ValueError, match="actor optimizer must own exactly"):
        FastWMRAgent(update_loop.updater, processor)
    env.close()
