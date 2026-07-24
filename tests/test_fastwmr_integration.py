"""End-to-end tests for FastWMR collection, reconstruction, and C51 SAC updates."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    C51SACUpdater,
    EMAControlEstimator,
    EntropyTemperature,
    EstimatorUpdater,
    FastWMRAgent,
    FastWMREstimatorRuntime,
    FastWMRRolloutCollector,
    FastWMRSequenceFeatureProcessor,
    FastWMRSequenceUpdateLoop,
    FastWMRV2EstimatorController,
    FastWMRV2UpdateLoop,
    GradientBoundaryError,
    TrainingMode,
    WorldStateEstimator,
    inspect_training_checkpoint,
    load_policy_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
    write_config_snapshot,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    EstimatorRolloutCache,
    EstimatorRolloutCacheSpec,
    ReplayBufferSpec,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ControlFeatureMode,
    DEFAULT_INTERFACE_CFG,
    DistributionalCriticCfg,
    FastWMRInterfaceCfg,
    FastWMRV2Cfg,
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


def _integrated_pipeline(
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    gradient_cutoff: bool = True,
    estimator_frozen: bool = False,
    num_updates: int = 1,
    validation_interval: int = 1,
    initial_validation_updates: int = 0,
    version: str = "v1",
    v2_cfg: FastWMRV2Cfg | None = None,
    normalizer_freeze_iteration: int | None = None,
) -> tuple[
    IsaacLabEnvAdapter,
    TransitionReplayBuffer,
    RunningObservationNormalizer,
    WorldStateEstimator,
    EstimatorUpdater,
    FastWMREstimatorRuntime,
    FastWMRSequenceFeatureProcessor | FastWMRV2EstimatorController,
    FastWMRSequenceUpdateLoop | FastWMRV2UpdateLoop,
    FastWMRRolloutCollector,
]:
    env = IsaacLabEnvAdapter(_FastWMRGymEnv())
    replay = TransitionReplayBuffer(ReplayBufferSpec.fastwmr(capacity=64, interface=interface))
    rollout_cache = EstimatorRolloutCache(
        EstimatorRolloutCacheSpec.fastwmr(
            capacity_steps=8,
            num_envs=env.num_envs,
            interface=interface,
        )
    )
    estimator = WorldStateEstimator(
        HistoryEncoder(interface.policy_observation_dim, hidden_dim=16),
        WorldStateDecoder(16, hidden_dim=16),
    )
    estimator_updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(estimator.parameters(), lr=1e-3),
        interface=interface,
    )
    normalizer = RunningObservationNormalizer(interface.policy_observation_dim)

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
    replay_cfg = ReplayUpdateCfg(
        random_action_steps=0,
        minimum_replay_size=4,
        batch_size=4,
        num_updates=num_updates,
    )
    sequence_cfg = SequenceReplayCfg(batch_size=1, burn_in_length=1, learning_length=2)
    if version == "v1":
        runtime = FastWMREstimatorRuntime(estimator, num_envs=env.num_envs)
        processor = FastWMRSequenceFeatureProcessor(
            estimator_updater,
            runtime,
            rollout_cache,
            interface=interface,
            observation_normalizer=normalizer,
            gradient_cutoff=gradient_cutoff,
            estimator_frozen=estimator_frozen,
        )
        update_loop = FastWMRSequenceUpdateLoop(
            replay,
            sac_updater,
            replay_cfg,
            sequence_cfg,
            processor,
            learner_device="cpu",
            verify_gradient_boundaries=gradient_cutoff,
            validation_interval=validation_interval,
            initial_validation_updates=initial_validation_updates,
            normalizer_freeze_iteration=normalizer_freeze_iteration,
        )
    elif version == "v2":
        resolved_v2_cfg = v2_cfg or FastWMRV2Cfg()
        control_estimator = copy.deepcopy(estimator)
        ema_estimator = EMAControlEstimator(
            estimator,
            control_estimator,
            tau=resolved_v2_cfg.control_estimator_tau,
        )
        runtime = FastWMREstimatorRuntime(control_estimator, num_envs=env.num_envs)
        processor = FastWMRV2EstimatorController(
            estimator_updater,
            ema_estimator,
            runtime,
            rollout_cache,
            cfg=resolved_v2_cfg,
            interface=interface,
            observation_normalizer=normalizer,
            estimator_frozen=estimator_frozen,
            validation_interval=validation_interval,
            initial_validation_updates=initial_validation_updates,
        )
        update_loop = FastWMRV2UpdateLoop(
            replay,
            sac_updater,
            replay_cfg,
            sequence_cfg,
            processor,
            learner_device="cpu",
            v2_cfg=resolved_v2_cfg,
            normalizer_freeze_iteration=normalizer_freeze_iteration,
        )
    else:
        raise ValueError(f"Unknown FastWMR test version {version!r}.")
    collector = FastWMRRolloutCollector(env, replay, update_loop, interface=interface)
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


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_fastwmr_collection_honors_normalizer_freeze_iteration(
    version: str,
) -> None:
    (
        env,
        _replay,
        normalizer,
        _estimator,
        _estimator_updater,
        _runtime,
        _processor,
        update_loop,
        collector,
    ) = _integrated_pipeline(
        version=version,
        normalizer_freeze_iteration=2,
    )

    collector.reset(seed=30)
    for _ in range(3):
        collector.collect_step()

    assert update_loop.normalization_frozen
    assert normalizer.samples_seen == env.num_envs * 3
    env.close()


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
    assert torch.all(torch.isfinite(transitions.reconstructions))
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


def test_update_bundle_rebuilds_runtime_once_and_periodicizes_validation() -> None:
    (
        env,
        _replay,
        _normalizer,
        _estimator,
        estimator_updater,
        runtime,
        _processor,
        update_loop,
        collector,
    ) = _integrated_pipeline(
        num_updates=4,
        validation_interval=3,
        initial_validation_updates=2,
    )
    collector.reset(seed=41)
    results = [collector.collect_step() for _ in range(3)]

    assert len(results[-1].updates) == 4
    assert estimator_updater.version == 4
    assert runtime.estimator_version == 4
    assert runtime.rebuilds == 1
    assert [
        update.gradient_boundary.enabled for update in update_loop.last_agent_updates
    ] == [True, True, False, True]
    env.close()


def test_v2_splits_sac_and_estimator_with_ema_gate_and_one_rebuild() -> None:
    v2_cfg = FastWMRV2Cfg(
        estimator_update_interval=8,
        estimator_updates_per_trigger=1,
        max_estimator_feature_age=100,
        stored_feature_replay_horizon=64,
        control_estimator_tau=0.25,
        reconstruction_gate_warmup_updates=2,
        reconstruction_gate_quality_threshold=1.0e9,
        reconstruction_gate_base_velocity_rmse_threshold=1.0e9,
        reconstruction_gate_contact_bce_threshold=1.0e9,
        reconstruction_gate_quality_patience=1,
        reconstruction_gate_validation_interval=1,
    )
    (
        env,
        replay,
        _normalizer,
        online_estimator,
        estimator_updater,
        runtime,
        controller,
        update_loop,
        collector,
    ) = _integrated_pipeline(
        version="v2",
        num_updates=4,
        v2_cfg=v2_cfg,
    )
    assert isinstance(controller, FastWMRV2EstimatorController)
    assert isinstance(update_loop, FastWMRV2UpdateLoop)
    initial_control_parameters = [
        parameter.detach().clone()
        for parameter in controller.ema_estimator.control_estimator.parameters()
    ]

    initial_feature = collector.reset(seed=42)
    assert torch.count_nonzero(
        initial_feature[:, DEFAULT_INTERFACE_CFG.policy_observation_dim :]
    ) == 0
    results = [collector.collect_step() for _ in range(3)]

    assert sum(len(result.updates) for result in results) == 8
    assert update_loop.gradient_steps == 8
    assert estimator_updater.version == 1
    assert controller.estimator_updates == 1
    assert controller.estimator_triggers == 1
    assert controller.control_estimator_version == 1
    assert controller.reconstruction_gate == pytest.approx(0.0)
    assert controller.gate_state.value == "ramping"
    assert controller.snapshot_active
    assert controller.snapshot_estimator_version == 1
    assert controller.snapshot_replay_resets == 1
    assert controller.online_estimator_frozen
    assert controller.gate_validation_checks == 1
    assert runtime.estimator_version == 1
    assert runtime.rebuilds == 1
    assert len(update_loop.last_estimator_updates) == 1
    assert update_loop.last_snapshot_replay_reset
    assert len(replay) == 0
    assert update_loop.last_rejected_features == 0
    for initial, online, control in zip(
        initial_control_parameters,
        online_estimator.parameters(),
        controller.ema_estimator.control_estimator.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(control, online.detach())

    online_after_trigger = [
        parameter.detach().clone() for parameter in online_estimator.parameters()
    ]
    collector.collect_step()
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            online_after_trigger,
            online_estimator.parameters(),
            strict=True,
        )
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in controller.ema_estimator.control_estimator.parameters()
    )
    newest = replay.chronological().reconstructions[-env.num_envs :]
    assert torch.count_nonzero(newest) > 0
    env.close()


def test_v2_qualifies_target_fields_and_never_closes_snapshot_gate(monkeypatch) -> None:
    v2_cfg = FastWMRV2Cfg(
        reconstruction_gate_warmup_updates=2,
        reconstruction_gate_quality_threshold=0.45,
        reconstruction_gate_base_velocity_rmse_threshold=0.65,
        reconstruction_gate_contact_bce_threshold=0.55,
        reconstruction_gate_quality_ema_decay=0.0,
        reconstruction_gate_quality_patience=1,
        reconstruction_gate_validation_interval=1,
    )
    pipeline = _integrated_pipeline(version="v2", v2_cfg=v2_cfg)
    env = pipeline[0]
    controller = pipeline[6]
    assert isinstance(controller, FastWMRV2EstimatorController)
    quality = torch.tensor(0.4)
    base_velocity_mse = torch.tensor(1.0)
    contact_bce = torch.tensor(0.5)
    monkeypatch.setattr(
        controller.estimator_updater,
        "evaluate_sequence",
        lambda _sequence: SimpleNamespace(
            metrics=SimpleNamespace(
                total_loss=quality,
                physical_field_losses={
                    "base_lin_vel_mse": base_velocity_mse,
                    "foot_contacts_bce": contact_bce,
                },
            )
        ),
    )
    controller.estimator_updates = 1

    controller.validate_reconstruction_gate(None)
    assert controller.gate_state.value == "closed"
    assert controller.gate_quality_failures == 1

    base_velocity_mse.fill_(0.36)
    controller.validate_reconstruction_gate(None)
    assert controller.gate_state.value == "ramping"
    controller.synchronize_control_estimator()
    assert controller.snapshot_active
    assert controller.consume_snapshot_activation()
    controller.advance_frozen_snapshot()
    assert controller.reconstruction_gate == pytest.approx(0.5)
    controller.advance_frozen_snapshot()
    assert controller.gate_state.value == "open"

    quality.fill_(1.0)
    base_velocity_mse.fill_(4.0)
    contact_bce.fill_(1.0)
    controller.validate_reconstruction_gate(None)
    assert controller.gate_state.value == "open"
    assert controller.reconstruction_gate == pytest.approx(1.0)
    assert controller.snapshot_estimator_version == controller.control_estimator_version
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
        features[
            ...,
            DEFAULT_INTERFACE_CFG.policy_observation_dim :
            DEFAULT_INTERFACE_CFG.policy_observation_dim
            + DEFAULT_INTERFACE_CFG.reconstruction_target_dim,
        ],
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

    def update_estimator(sample, **kwargs):
        events.append("estimator")
        return original_estimator(sample, **kwargs)

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


def test_checkpoint_resume_restores_models_optimizers_normalizer_and_counters(tmp_path) -> None:
    torch.manual_seed(35)
    (
        source_env,
        source_replay,
        source_normalizer,
        source_estimator,
        source_estimator_updater,
        source_runtime,
        _source_processor,
        source_loop,
        source_collector,
    ) = _integrated_pipeline(normalizer_freeze_iteration=2)
    source_collector.reset(seed=35)
    for _ in range(6):
        source_collector.collect_step()
    sequence = source_replay.sample_sequences(
        batch_size=1,
        burn_in_length=1,
        learning_length=2,
        device="cpu",
        generator=torch.Generator().manual_seed(35),
    )
    fixed_observations = torch.randn(3, DEFAULT_INTERFACE_CFG.policy_observation_dim)
    fixed_features = torch.randn(3, DEFAULT_INTERFACE_CFG.control_feature_dim)
    normalized_before = source_normalizer(fixed_observations)
    action_before = source_loop.updater.actor.act(fixed_features, deterministic=True)
    checkpoint_path = tmp_path / "checkpoints" / "step.pt"
    config = {"seed": 35, "sequence": {"burn_in": 1, "learning": 2}}

    save_training_checkpoint(
        checkpoint_path,
        mode=TrainingMode.FASTWMR,
        sac_updater=source_loop.updater,
        update_loop=source_loop,
        normalizer=source_normalizer,
        estimator_updater=source_estimator_updater,
        config=config,
    )
    snapshot_path = write_config_snapshot(tmp_path / "config_snapshot.json", config)

    (
        target_env,
        target_replay,
        target_normalizer,
        target_estimator,
        target_estimator_updater,
        target_runtime,
        _target_processor,
        target_loop,
        target_collector,
    ) = _integrated_pipeline()
    target_collector.reset(seed=351)
    for _ in range(4):
        target_collector.collect_step()
    assert target_replay.total_inserted > 0
    assert target_loop.sequence_feature_processor.updates > 0
    loaded = load_training_checkpoint(
        checkpoint_path,
        mode=TrainingMode.FASTWMR,
        sac_updater=target_loop.updater,
        update_loop=target_loop,
        normalizer=target_normalizer,
        estimator_updater=target_estimator_updater,
        runtime=target_runtime,
        rollout_cache=target_loop.sequence_feature_processor.rollout_cache,
        map_location="cpu",
    )

    assert loaded.config == config
    assert loaded.counters.environment_steps == source_loop.environment_steps
    assert loaded.counters.gradient_steps == source_loop.gradient_steps
    assert loaded.counters.estimator_version == source_estimator_updater.version
    assert target_loop.environment_steps == source_loop.environment_steps
    assert target_loop.gradient_steps == source_loop.gradient_steps
    assert target_loop.agent is not None and source_loop.agent is not None
    assert target_loop.agent.update_steps == source_loop.agent.update_steps
    assert target_estimator_updater.version == source_estimator_updater.version
    assert target_runtime.estimator_version == source_runtime.estimator_version
    assert target_runtime.environment_steps == 0
    assert target_runtime.rebuilds == 0
    assert torch.count_nonzero(target_runtime.state.hidden) == 0
    assert torch.count_nonzero(target_runtime.state.cell) == 0
    assert len(target_replay) == 0
    assert target_replay.total_inserted == 0
    assert target_loop.normalizer_freeze_iteration == 2
    assert target_loop.normalization_frozen
    assert len(target_loop.sequence_feature_processor.rollout_cache) == 0
    assert target_loop.sequence_feature_processor.updates == loaded.counters.agent_updates
    assert target_loop.sequence_feature_processor.last_estimator_update is None
    assert target_loop.sequence_feature_processor.last_runtime_rebuild is None
    torch.testing.assert_close(target_normalizer(fixed_observations), normalized_before)
    torch.testing.assert_close(
        target_loop.updater.actor.act(fixed_features, deterministic=True),
        action_before,
    )
    _assert_module_equal(source_estimator, target_estimator)
    _assert_module_equal(source_loop.updater.actor, target_loop.updater.actor)
    _assert_module_equal(source_loop.updater.critic, target_loop.updater.critic)
    _assert_module_equal(source_loop.updater.target_critic, target_loop.updater.target_critic)
    _assert_module_equal(source_loop.updater.temperature, target_loop.updater.temperature)
    assert snapshot_path.read_text(encoding="utf-8").endswith("\n")

    torch.manual_seed(350)
    source_update = source_loop.agent.update(sequence)
    torch.manual_seed(350)
    target_update = target_loop.agent.update(sequence)
    _assert_module_equal(source_estimator, target_estimator)
    _assert_module_equal(source_loop.updater.actor, target_loop.updater.actor)
    _assert_module_equal(source_loop.updater.critic, target_loop.updater.critic)
    _assert_module_equal(source_loop.updater.target_critic, target_loop.updater.target_critic)
    _assert_module_equal(source_loop.updater.temperature, target_loop.updater.temperature)
    torch.testing.assert_close(
        source_update.sac_update.critic_loss,
        target_update.sac_update.critic_loss,
    )
    assert source_update.estimator_update.metrics.total_loss == pytest.approx(
        target_update.estimator_update.metrics.total_loss
    )
    source_env.close()
    target_env.close()


def test_v2_checkpoint_restores_online_ema_scheduler_gate_and_policy_state(tmp_path) -> None:
    v2_cfg = FastWMRV2Cfg(
        estimator_update_interval=4,
        estimator_updates_per_trigger=1,
        stored_feature_replay_horizon=64,
        reconstruction_gate_warmup_updates=2,
        reconstruction_gate_quality_threshold=1.0e9,
        reconstruction_gate_base_velocity_rmse_threshold=1.0e9,
        reconstruction_gate_contact_bce_threshold=1.0e9,
        reconstruction_gate_quality_patience=1,
        reconstruction_gate_validation_interval=1,
    )
    source = _integrated_pipeline(
        version="v2",
        num_updates=4,
        v2_cfg=v2_cfg,
    )
    source_env = source[0]
    source_estimator_updater = source[4]
    source_runtime = source[5]
    source_controller = source[6]
    source_loop = source[7]
    source_collector = source[8]
    assert isinstance(source_controller, FastWMRV2EstimatorController)
    assert isinstance(source_loop, FastWMRV2UpdateLoop)
    source_collector.reset(seed=43)
    for _ in range(3):
        source_collector.collect_step()
    assert source_controller.snapshot_active
    assert source_controller.snapshot_replay_resets == 1

    checkpoint = save_training_checkpoint(
        tmp_path / "v2.pt",
        mode=TrainingMode.FASTWMR,
        sac_updater=source_loop.updater,
        update_loop=source_loop,
        normalizer=source[2],
        estimator_updater=source_estimator_updater,
        config={"arguments": {"fastwmr_version": "v2"}},
    )
    legacy_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    legacy_payload["format_version"] = 2
    legacy_payload.pop("fastwmr_representation_version")
    legacy_payload["architecture"].pop("actor_hidden_dim")
    legacy_payload["architecture"].pop("critic_hidden_dim")
    legacy_checkpoint = tmp_path / "legacy_v2.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    with pytest.raises(ValueError, match="predates qualified stationary estimator"):
        inspect_training_checkpoint(legacy_checkpoint)

    target = _integrated_pipeline(
        version="v2",
        num_updates=4,
        v2_cfg=v2_cfg,
    )
    target_env = target[0]
    target_controller = target[6]
    target_loop = target[7]
    assert isinstance(target_controller, FastWMRV2EstimatorController)
    assert isinstance(target_loop, FastWMRV2UpdateLoop)
    loaded = load_training_checkpoint(
        checkpoint,
        mode=TrainingMode.FASTWMR,
        sac_updater=target_loop.updater,
        update_loop=target_loop,
        normalizer=target[2],
        estimator_updater=target[4],
        runtime=target[5],
        rollout_cache=target_controller.rollout_cache,
        map_location="cpu",
    )

    assert loaded.counters.estimator_updates == source_controller.estimator_updates
    assert loaded.counters.estimator_triggers == source_controller.estimator_triggers
    assert loaded.counters.control_estimator_version == source_runtime.estimator_version
    assert target_loop.state_dict() == source_loop.state_dict()
    assert target_controller.reconstruction_gate == source_controller.reconstruction_gate
    assert target[5].estimator_version == source_runtime.estimator_version
    assert target[5].rebuilds == 0
    assert len(target[1]) == 0
    assert len(target_controller.rollout_cache) == 0
    _assert_module_equal(source[3], target[3])
    _assert_module_equal(
        source_controller.ema_estimator.control_estimator,
        target_controller.ema_estimator.control_estimator,
    )

    deployment_estimator = copy.deepcopy(target[3])
    load_policy_checkpoint(
        checkpoint,
        mode=TrainingMode.FASTWMR,
        actor=target_loop.updater.actor,
        estimator=deployment_estimator,
        normalizer=target[2],
        map_location="cpu",
    )
    _assert_module_equal(
        source_controller.ema_estimator.control_estimator,
        deployment_estimator,
    )
    source_env.close()
    target_env.close()


def test_checkpoint_rejects_mode_and_normalizer_mismatch(tmp_path) -> None:
    (
        env,
        _replay,
        normalizer,
        _estimator,
        estimator_updater,
        runtime,
        _processor,
        update_loop,
        _collector,
    ) = _integrated_pipeline()
    checkpoint_path = save_training_checkpoint(
        tmp_path / "fastwmr.pt",
        mode=TrainingMode.FASTWMR,
        sac_updater=update_loop.updater,
        update_loop=update_loop,
        normalizer=normalizer,
        estimator_updater=estimator_updater,
    )

    mismatched_mode_path = tmp_path / "mismatched_mode.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["mode"] = TrainingMode.FASTSAC.value
    torch.save(payload, mismatched_mode_path)

    with pytest.raises(ValueError, match="Checkpoint mode is 'fastsac'"):
        load_training_checkpoint(
            mismatched_mode_path,
            mode=TrainingMode.FASTWMR,
            sac_updater=update_loop.updater,
            update_loop=update_loop,
            normalizer=normalizer,
            estimator_updater=estimator_updater,
            runtime=runtime,
            rollout_cache=update_loop.sequence_feature_processor.rollout_cache,
            map_location="cpu",
        )

    with pytest.raises(ValueError, match="normalizer settings do not match"):
        load_training_checkpoint(
            checkpoint_path,
            mode=TrainingMode.FASTWMR,
            sac_updater=update_loop.updater,
            update_loop=update_loop,
            normalizer=None,
            estimator_updater=estimator_updater,
            runtime=runtime,
            rollout_cache=update_loop.sequence_feature_processor.rollout_cache,
            map_location="cpu",
        )
    env.close()


def test_policy_checkpoint_loads_actor_estimator_and_normalizer(tmp_path) -> None:
    source = _integrated_pipeline()
    target = _integrated_pipeline()
    source_env = source[0]
    target_env = target[0]
    source_normalizer = source[2]
    source_estimator_updater = source[4]
    source_loop = source[7]
    target_normalizer = target[2]
    target_estimator = target[3]
    target_loop = target[7]
    source_normalizer.update(torch.randn(4, source_normalizer.observation_dim))
    checkpoint = save_training_checkpoint(
        tmp_path / "policy.pt",
        mode=TrainingMode.FASTWMR,
        sac_updater=source_loop.updater,
        update_loop=source_loop,
        normalizer=source_normalizer,
        estimator_updater=source_estimator_updater,
        config={"arguments": {"hidden_dim": 16}},
    )

    metadata = inspect_training_checkpoint(checkpoint)
    result = load_policy_checkpoint(
        checkpoint,
        mode=TrainingMode.FASTWMR,
        actor=target_loop.updater.actor,
        estimator=target_estimator,
        normalizer=target_normalizer,
        map_location="cpu",
    )

    assert metadata.mode is TrainingMode.FASTWMR
    assert metadata.has_normalizer
    assert result.counters == metadata.counters
    _assert_module_equal(source_loop.updater.actor, target_loop.updater.actor)
    _assert_module_equal(source[3], target_estimator)
    _assert_module_equal(source_normalizer, target_normalizer)
    assert not target_loop.updater.actor.training
    assert not target_estimator.training
    assert not target_normalizer.training
    source_env.close()
    target_env.close()


def test_no_cutoff_routes_policy_gradients_into_estimator() -> None:
    torch.manual_seed(38)
    (
        env,
        _replay,
        _normalizer,
        _estimator,
        estimator_updater,
        runtime,
        processor,
        update_loop,
        collector,
    ) = _integrated_pipeline(gradient_cutoff=False)
    collector.reset(seed=38)
    for _ in range(5):
        collector.collect_step()

    assert update_loop.gradient_steps > 0
    assert estimator_updater.version == update_loop.gradient_steps * 2
    assert runtime.estimator_version == estimator_updater.version
    assert processor.last_estimator_update is not None
    assert update_loop.last_agent_updates
    boundary = update_loop.last_agent_updates[-1].gradient_boundary
    assert not boundary.cutoff_enabled
    assert boundary.policy_estimator_gradient_norm is not None
    assert boundary.policy_estimator_gradient_norm > 0.0
    env.close()


def test_frozen_estimator_is_evaluated_without_parameter_updates() -> None:
    torch.manual_seed(39)
    (
        env,
        _replay,
        _normalizer,
        estimator,
        estimator_updater,
        runtime,
        _processor,
        update_loop,
        collector,
    ) = _integrated_pipeline(estimator_frozen=True)
    initial_parameters = [parameter.detach().clone() for parameter in estimator.parameters()]
    collector.reset(seed=39)
    for _ in range(5):
        collector.collect_step()

    assert update_loop.gradient_steps > 0
    assert estimator_updater.version == 0
    assert runtime.estimator_version == 0
    assert runtime.rebuilds == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(initial_parameters, estimator.parameters(), strict=True)
    )
    boundary = update_loop.last_agent_updates[-1].gradient_boundary
    assert boundary.estimator_gradient_norm is None
    env.close()


def test_reconstruction_only_ablation_runs_integrated_update() -> None:
    interface = FastWMRInterfaceCfg(
        control_feature_mode=ControlFeatureMode.RECONSTRUCTION_ONLY
    )
    (
        env,
        replay,
        _normalizer,
        _estimator,
        _estimator_updater,
        _runtime,
        _processor,
        update_loop,
        collector,
    ) = _integrated_pipeline(interface=interface)
    initial_feature = collector.reset(seed=40)
    for _ in range(5):
        collector.collect_step()

    assert initial_feature.shape == (env.num_envs, interface.control_feature_dim)
    assert replay.spec.reconstruction_dim == interface.reconstruction_target_dim
    assert update_loop.updater.actor.input_dim == interface.control_feature_dim
    assert update_loop.gradient_steps > 0
    env.close()


def _assert_module_equal(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    assert source_state.keys() == target_state.keys()
    for name in source_state:
        torch.testing.assert_close(source_state[name], target_state[name])
