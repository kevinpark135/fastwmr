"""Train FastSAC or FastWMR on the shared Rough G1 environment."""

from __future__ import annotations

import copy
import time
import traceback
from functools import partial

from cli_args import (
    FASTSAC_BASELINE_TASK,
    FASTWMR_TASK,
    build_train_parser,
    validate_train_args,
)

from isaaclab.app import AppLauncher


PARSER = build_train_parser()
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
validate_train_args(ARGS)

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    C51SACUpdater,
    EMAControlEstimator,
    EntropyTemperature,
    EstimatorUpdater,
    FastSACReplayUpdateLoop,
    FastSACRolloutCollector,
    FastWMREstimatorRuntime,
    FastWMRRolloutCollector,
    FastWMRSequenceFeatureProcessor,
    FastWMRSequenceUpdateLoop,
    FastWMRV2EstimatorController,
    FastWMRV2UpdateLoop,
    SACUpdater,
    TrainingMode,
    WorldStateEstimator,
    augment_sequence_batch,
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
    DistributionalCriticCfg,
    FastWMRInterfaceCfg,
    FastWMRV2Cfg,
    ObservationNormalizationCfg,
    ReplayUpdateCfg,
    ScalarCriticCfg,
    SequenceReplayCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    HistoryEncoder,
    TargetTwinC51Critic,
    TargetTwinScalarCritic,
    TanhGaussianActor,
    TwinC51Critic,
    TwinScalarCritic,
    WorldStateDecoder,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EpisodeStatisticsTracker,
    IsaacLabEnvAdapter,
    RunningObservationNormalizer,
    TrainingMetricsLogger,
    estimator_metrics_dict,
    fastwmr_agent_metrics_dict,
    fastwmr_v2_metrics_dict,
    format_console_metrics,
    format_console_metrics_header,
    sac_metrics_dict,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.curriculum import (
    penalty_curriculum_state,
    terrain_curriculum_state,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@dataclass
class TrainingComponents:
    mode: TrainingMode
    replay: TransitionReplayBuffer
    updater: SACUpdater
    update_loop: FastSACReplayUpdateLoop
    collector: FastSACRolloutCollector | FastWMRRolloutCollector
    normalizer: RunningObservationNormalizer | None
    estimator_updater: EstimatorUpdater | None = None
    runtime: FastWMREstimatorRuntime | None = None
    rollout_cache: EstimatorRolloutCache | None = None


def _apply_rough_debug_overrides(cfg: object) -> None:
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 0
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
        cfg.scene.terrain.terrain_generator.curriculum = False


def _apply_penalty_curriculum_overrides(cfg: object) -> None:
    term = cfg.curriculum.penalty_weights
    if ARGS.disable_penalty_curriculum:
        cfg.curriculum.penalty_weights = None
        return
    term.params["scales"] = tuple(ARGS.penalty_scales)
    term.params["episode_length_thresholds"] = tuple(ARGS.penalty_length_thresholds)
    term.params["ema_decay"] = ARGS.penalty_ema_decay
    term.params["min_completed_episodes"] = ARGS.penalty_min_completed_episodes


def _build_sac_updater(
    state_dim: int,
    action_dim: int,
    device: torch.device,
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> SACUpdater:
    actor = TanhGaussianActor(
        state_dim,
        action_dim,
        cfg=TanhGaussianActorCfg(hidden_dim=ARGS.hidden_dim),
        action_low=action_low,
        action_high=action_high,
    ).to(device)
    if ARGS.critic_type == "c51":
        critic = TwinC51Critic(
            state_dim,
            action_dim,
            cfg=DistributionalCriticCfg(
                hidden_dim=ARGS.hidden_dim,
                num_atoms=ARGS.num_atoms,
                value_min=ARGS.value_min,
                value_max=ARGS.value_max,
            ),
        ).to(device)
        target_critic = TargetTwinC51Critic.from_online(critic)
        updater_type = C51SACUpdater
    else:
        critic = TwinScalarCritic(
            state_dim,
            action_dim,
            cfg=ScalarCriticCfg(hidden_dim=ARGS.hidden_dim),
        ).to(device)
        target_critic = TargetTwinScalarCritic.from_online(critic)
        updater_type = SACUpdater
    temperature = EntropyTemperature(ARGS.initial_temperature).to(device)
    optimizer_kwargs = {
        "lr": ARGS.learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": ARGS.weight_decay,
    }
    return updater_type(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), **optimizer_kwargs),
        critic_optimizer=torch.optim.Adam(critic.parameters(), **optimizer_kwargs),
        temperature_optimizer=torch.optim.Adam(
            temperature.parameters(),
            lr=ARGS.learning_rate,
            betas=(0.9, 0.95),
        ),
        discount=ARGS.discount,
        target_update_rate=ARGS.target_update_rate,
        target_entropy=ARGS.target_entropy,
    )


def _build_normalizer(
    observation_dim: int,
    device: torch.device,
) -> RunningObservationNormalizer | None:
    if ARGS.disable_observation_normalization:
        return None
    return RunningObservationNormalizer(
        observation_dim,
        ObservationNormalizationCfg(
            epsilon=ARGS.normalization_epsilon,
            clip=ARGS.normalization_clip,
        ),
    ).to(device)


def _build_components(
    raw_env: object,
    env: IsaacLabEnvAdapter,
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> TrainingComponents:
    observation_dim = int(raw_env.observation_space["policy"].shape[-1])
    action_dim = int(raw_env.unwrapped.action_manager.total_action_dim)
    device = env.device
    update_cfg = ReplayUpdateCfg(
        random_action_steps=ARGS.random_action_steps,
        minimum_replay_size=ARGS.minimum_replay_size,
        batch_size=ARGS.batch_size,
        num_updates=ARGS.num_updates,
    )
    normalizer = _build_normalizer(observation_dim, device)

    if ARGS.task == FASTSAC_BASELINE_TASK:
        replay = TransitionReplayBuffer(
            ReplayBufferSpec(
                capacity=ARGS.replay_capacity,
                observation_dim=observation_dim,
                action_dim=action_dim,
            ),
            storage_device=ARGS.replay_storage_device,
        )
        updater = _build_sac_updater(
            observation_dim,
            action_dim,
            device,
            action_low=action_low,
            action_high=action_high,
        )
        update_loop = FastSACReplayUpdateLoop(
            replay,
            updater,
            update_cfg,
            learner_device=device,
            observation_normalizer=normalizer,
        )
        return TrainingComponents(
            mode=TrainingMode.FASTSAC,
            replay=replay,
            updater=updater,
            update_loop=update_loop,
            collector=FastSACRolloutCollector(env, replay, update_loop),
            normalizer=normalizer,
        )

    if ARGS.task != FASTWMR_TASK:
        raise ValueError(f"Unsupported training task {ARGS.task!r}.")
    interface = FastWMRInterfaceCfg(
        control_feature_mode=ControlFeatureMode(ARGS.control_feature_mode)
    )
    privileged_dim = int(raw_env.observation_space["privileged"].shape[-1])
    actual_dimensions = (observation_dim, privileged_dim, action_dim)
    expected_dimensions = (
        interface.policy_observation_dim,
        interface.reconstruction_target_dim,
        interface.action_dim,
    )
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            f"FastWMR environment dimensions are {actual_dimensions}, expected {expected_dimensions}."
        )
    replay = TransitionReplayBuffer(
        ReplayBufferSpec.fastwmr(ARGS.replay_capacity, interface),
        storage_device=ARGS.replay_storage_device,
    )
    rollout_cache = EstimatorRolloutCache(
        EstimatorRolloutCacheSpec.fastwmr(
            ARGS.estimator_cache_steps,
            env.num_envs,
            interface,
        ),
        storage_device=device,
    )
    estimator = WorldStateEstimator(
        HistoryEncoder(
            observation_dim,
            hidden_dim=ARGS.estimator_hidden_dim,
            num_layers=ARGS.estimator_num_layers,
        ),
        WorldStateDecoder(
            ARGS.estimator_hidden_dim,
            hidden_dim=ARGS.estimator_hidden_dim,
        ),
    ).to(device)
    estimator_updater = EstimatorUpdater(
        estimator,
        torch.optim.Adam(
            estimator.parameters(),
            lr=ARGS.estimator_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=ARGS.estimator_weight_decay,
        ),
        interface=interface,
    )
    updater = _build_sac_updater(
        interface.control_feature_dim,
        action_dim,
        device,
        action_low=action_low,
        action_high=action_high,
    )
    sequence_cfg = SequenceReplayCfg(
        batch_size=ARGS.sequence_batch_size,
        burn_in_length=ARGS.burn_in_length,
        learning_length=ARGS.learning_length,
        require_episode_start=ARGS.require_episode_start,
        recent_transition_horizon=ARGS.recent_replay_horizon,
    )
    sequence_augmentation = (
        partial(augment_sequence_batch, interface=interface)
        if ARGS.use_symmetry
        else None
    )
    if ARGS.fastwmr_version == "v1":
        runtime = FastWMREstimatorRuntime(estimator, env.num_envs)
        processor = FastWMRSequenceFeatureProcessor(
            estimator_updater,
            runtime,
            rollout_cache,
            interface=interface,
            observation_normalizer=normalizer,
            gradient_cutoff=not ARGS.disable_gradient_cutoff,
            estimator_frozen=ARGS.freeze_estimator,
        )
        update_loop = FastWMRSequenceUpdateLoop(
            replay,
            updater,
            update_cfg,
            sequence_cfg,
            processor,
            learner_device=device,
            verify_gradient_boundaries=(
                not ARGS.disable_gradient_boundary_checks
                and not ARGS.disable_gradient_cutoff
            ),
            validation_interval=ARGS.validation_interval,
            initial_validation_updates=ARGS.initial_validation_updates,
            sequence_augmentation=sequence_augmentation,
        )
    else:
        v2_cfg = FastWMRV2Cfg(
            estimator_update_interval=ARGS.estimator_update_interval,
            estimator_updates_per_trigger=ARGS.estimator_updates_per_trigger,
            max_estimator_feature_age=(
                None
                if ARGS.disable_feature_age_filter
                else ARGS.max_estimator_feature_age
            ),
            stored_feature_replay_horizon=ARGS.stored_feature_replay_horizon,
            control_estimator_tau=ARGS.control_estimator_tau,
            reconstruction_gate_start_updates=ARGS.reconstruction_gate_start_updates,
            reconstruction_gate_warmup_updates=ARGS.reconstruction_gate_warmup_updates,
        )
        control_estimator = copy.deepcopy(estimator).to(device)
        ema_estimator = EMAControlEstimator(
            estimator,
            control_estimator,
            tau=v2_cfg.control_estimator_tau,
        )
        runtime = FastWMREstimatorRuntime(control_estimator, env.num_envs)
        controller = FastWMRV2EstimatorController(
            estimator_updater,
            ema_estimator,
            runtime,
            rollout_cache,
            cfg=v2_cfg,
            interface=interface,
            observation_normalizer=normalizer,
            estimator_frozen=ARGS.freeze_estimator,
            validation_interval=ARGS.validation_interval,
            initial_validation_updates=ARGS.initial_validation_updates,
        )
        update_loop = FastWMRV2UpdateLoop(
            replay,
            updater,
            update_cfg,
            sequence_cfg,
            controller,
            learner_device=device,
            v2_cfg=v2_cfg,
            sequence_augmentation=sequence_augmentation,
        )
    return TrainingComponents(
        mode=TrainingMode.FASTWMR,
        replay=replay,
        updater=updater,
        update_loop=update_loop,
        collector=FastWMRRolloutCollector(env, replay, update_loop, interface=interface),
        normalizer=normalizer,
        estimator_updater=estimator_updater,
        runtime=runtime,
        rollout_cache=rollout_cache,
    )


def _run_directory() -> Path:
    if ARGS.resume is not None:
        checkpoint_path = ARGS.resume.expanduser().resolve()
        if checkpoint_path.parent.name == "checkpoints":
            return checkpoint_path.parent.parent
        return checkpoint_path.parent
    run_name = ARGS.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = TrainingMode.FASTWMR if ARGS.task == FASTWMR_TASK else TrainingMode.FASTSAC
        run_name = f"{timestamp}_{mode.value}"
    return Path(ARGS.log_dir).expanduser().resolve() / run_name


def _checkpoint_config(components: TrainingComponents) -> dict[str, object]:
    return {
        "mode": components.mode.value,
        "arguments": vars(ARGS),
    }


def _save_checkpoint(
    path: Path,
    components: TrainingComponents,
    config: dict[str, object],
) -> Path:
    return save_training_checkpoint(
        path,
        mode=components.mode,
        sac_updater=components.updater,
        update_loop=components.update_loop,
        normalizer=components.normalizer,
        estimator_updater=components.estimator_updater,
        config=config,
    )


def run() -> None:
    torch.manual_seed(ARGS.seed)
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.seed = ARGS.seed
    if ARGS.episode_length_s is not None:
        cfg.episode_length_s = ARGS.episode_length_s
    if ARGS.rough_debug:
        _apply_rough_debug_overrides(cfg)
    _apply_penalty_curriculum_overrides(cfg)

    raw_env = gym.make(ARGS.task, cfg=cfg)
    env = IsaacLabEnvAdapter(raw_env)
    logger: TrainingMetricsLogger | None = None
    try:
        action_dim = int(raw_env.unwrapped.action_manager.total_action_dim)
        if ARGS.disable_joint_limit_action_bounds:
            action_low = torch.full((action_dim,), -1.0, device=env.device)
            action_high = torch.full((action_dim,), 1.0, device=env.device)
        else:
            action_bounds = env.joint_position_action_bounds(
                use_soft_limits=ARGS.use_soft_joint_limits
            )
            action_low = action_bounds.low
            action_high = action_bounds.high
        components = _build_components(
            raw_env,
            env,
            action_low=action_low,
            action_high=action_high,
        )
        run_directory = _run_directory()
        checkpoints_directory = run_directory / "checkpoints"
        config = _checkpoint_config(components)
        tensorboard_purge_step: int | None = None

        if ARGS.resume is not None:
            resumed = load_training_checkpoint(
                ARGS.resume,
                mode=components.mode,
                sac_updater=components.updater,
                update_loop=components.update_loop,
                normalizer=components.normalizer,
                estimator_updater=components.estimator_updater,
                runtime=components.runtime,
                rollout_cache=components.rollout_cache,
                map_location=env.device,
            )
            tensorboard_purge_step = resumed.counters.environment_steps + 1
            print(
                f"[{components.mode.value}] resumed {resumed.path} at "
                f"environment_step={resumed.counters.environment_steps} "
                f"gradient_step={resumed.counters.gradient_steps}"
            )
            initial_snapshot = run_directory / "config_snapshot.json"
            if not initial_snapshot.exists():
                write_config_snapshot(initial_snapshot, resumed.config)
            resume_snapshot = (
                run_directory
                / f"resume_config_snapshot_step_{resumed.counters.environment_steps:09d}.json"
            )
            write_config_snapshot(resume_snapshot, config)
        else:
            write_config_snapshot(run_directory / "config_snapshot.json", config)
        logger = TrainingMetricsLogger(
            run_directory,
            mode=components.mode.value,
            append=ARGS.resume is not None,
            tensorboard_purge_step=tensorboard_purge_step,
        )
        checkpoint_schedule = (
            f"every {ARGS.checkpoint_interval} steps"
            if ARGS.checkpoint_interval > 0
            else "periodic saving disabled"
        )
        print(f"[{components.mode.value}] metrics={logger.path}")
        print(f"[{components.mode.value}] tensorboard={logger.tensorboard_directory}")
        print(
            f"[{components.mode.value}] checkpoints={checkpoints_directory} "
            f"({checkpoint_schedule})"
        )
        print(format_console_metrics_header(components.mode.value))

        components.collector.reset(seed=ARGS.seed)
        episode_tracker = EpisodeStatisticsTracker(env.num_envs, device=env.device)
        generator = torch.Generator(device="cpu").manual_seed(ARGS.seed)
        initial_gradient_steps = components.update_loop.gradient_steps
        interval_reward_sum = 0.0
        interval_steps = 0
        interval_completed = 0
        interval_return_sum = 0.0
        interval_length_sum = 0
        last_sac_metrics = None
        last_agent_update = None
        last_estimator_update = None
        training_started = time.perf_counter()
        collected_steps = 0

        for local_step in range(ARGS.steps):
            result = components.collector.collect_step(generator=generator)
            collected_steps += 1
            interval_reward_sum += float(result.rewards.mean().item())
            interval_steps += 1
            completed = episode_tracker.update(
                result.rewards,
                result.terminated,
                result.truncated,
            )
            interval_completed += completed.count
            interval_return_sum += completed.return_sum
            interval_length_sum += completed.length_sum
            if result.updates:
                last_sac_metrics = result.updates[-1]
            if (
                isinstance(components.update_loop, FastWMRSequenceUpdateLoop)
                and components.update_loop.last_agent_updates
            ):
                last_agent_update = components.update_loop.last_agent_updates[-1]
            if (
                isinstance(components.update_loop, FastWMRV2UpdateLoop)
                and components.update_loop.last_estimator_updates
            ):
                last_estimator_update = components.update_loop.last_estimator_updates[-1]

            global_step = components.update_loop.environment_steps
            training_elapsed = time.perf_counter() - training_started
            wallclock_limit_reached = (
                ARGS.wallclock_limit_s is not None
                and training_elapsed >= ARGS.wallclock_limit_s
            )
            checkpoint_saved = (
                ARGS.checkpoint_interval > 0
                and global_step % ARGS.checkpoint_interval == 0
            )
            if checkpoint_saved:
                checkpoint_path = checkpoints_directory / f"step_{global_step:09d}.pt"
                _save_checkpoint(checkpoint_path, components, config)

            if (
                global_step % ARGS.log_interval == 0
                or local_step + 1 == ARGS.steps
                or wallclock_limit_reached
                or checkpoint_saved
            ):
                metrics: dict[str, int | float] = {
                    "rollout/reward_mean": interval_reward_sum / interval_steps,
                    "rollout/completed_episodes": interval_completed,
                    "replay/size": len(components.replay),
                    "replay/total_inserted": components.replay.total_inserted,
                    "replay/overwritten": max(
                        0,
                        components.replay.total_inserted - len(components.replay),
                    ),
                    "learner/environment_steps": global_step,
                    "learner/gradient_steps": components.update_loop.gradient_steps,
                    "learner/wallclock_seconds": training_elapsed,
                    "checkpoint/saved": int(checkpoint_saved),
                    "ablation/gradient_cutoff": int(not ARGS.disable_gradient_cutoff),
                    "ablation/estimator_frozen": int(ARGS.freeze_estimator),
                    "ablation/symmetry": int(ARGS.use_symmetry),
                    "ablation/reconstruction_only": int(
                        ARGS.control_feature_mode == "reconstruction_only"
                    ),
                }
                oldest_insertion_id = components.replay.oldest_insertion_id
                newest_insertion_id = components.replay.newest_insertion_id
                if oldest_insertion_id is not None and newest_insertion_id is not None:
                    metrics.update(
                        {
                            "replay/oldest_age": newest_insertion_id - oldest_insertion_id,
                            "replay/newest_insertion_id": newest_insertion_id,
                        }
                    )
                oldest_estimator_version = components.replay.oldest_estimator_version
                newest_estimator_version = components.replay.newest_estimator_version
                if oldest_estimator_version is not None and newest_estimator_version is not None:
                    metrics.update(
                        {
                            "replay/oldest_estimator_version": oldest_estimator_version,
                            "replay/newest_estimator_version": newest_estimator_version,
                        }
                    )
                if interval_completed:
                    metrics["episode/return_mean"] = interval_return_sum / interval_completed
                    metrics["episode/length_mean"] = interval_length_sum / interval_completed
                if last_sac_metrics is not None:
                    metrics.update(sac_metrics_dict(last_sac_metrics))
                if last_agent_update is not None:
                    metrics.update(fastwmr_agent_metrics_dict(last_agent_update))
                if last_estimator_update is not None:
                    metrics.update(estimator_metrics_dict(last_estimator_update))
                if isinstance(components.update_loop, FastWMRV2UpdateLoop):
                    metrics.update(fastwmr_v2_metrics_dict(components.update_loop))
                if components.runtime is not None:
                    metrics.update(
                        {
                            "runtime/hidden_norm": components.runtime.hidden_norm,
                            "runtime/rebuilds": components.runtime.rebuilds,
                            "runtime/estimator_version": components.runtime.estimator_version,
                        }
                    )
                if components.rollout_cache is not None:
                    metrics["estimator_cache/steps"] = len(components.rollout_cache)
                metrics.update(components.update_loop.drain_profile_metrics())
                if getattr(cfg.curriculum, "terrain_levels", None) is not None:
                    terrain_state = terrain_curriculum_state(raw_env.unwrapped)
                    if terrain_state is not None:
                        metrics.update(
                            {
                                f"curriculum/terrain_{name}": value
                                for name, value in terrain_state.items()
                            }
                        )
                curriculum_state = penalty_curriculum_state(raw_env.unwrapped)
                if curriculum_state is not None:
                    metrics.update(
                        {
                            f"curriculum/penalty_{name}": value
                            for name, value in curriculum_state.items()
                        }
                    )
                record = logger.log(global_step, metrics)
                print(format_console_metrics(record))
                interval_reward_sum = 0.0
                interval_steps = 0
                interval_completed = 0
                interval_return_sum = 0.0
                interval_length_sum = 0
            if wallclock_limit_reached:
                print(
                    f"[{components.mode.value}] wallclock_limit_reached="
                    f"{training_elapsed:.3f}s"
                )
                break

        if not ARGS.disable_final_checkpoint:
            final_step = components.update_loop.environment_steps
            checkpoint_path = checkpoints_directory / f"final_step_{final_step:09d}.pt"
            _save_checkpoint(checkpoint_path, components, config)
            print(f"[{components.mode.value}] final_checkpoint={checkpoint_path}")

        expected_insertions = collected_steps * ARGS.num_envs
        if components.replay.total_inserted != expected_insertions:
            raise AssertionError(
                f"Replay inserted {components.replay.total_inserted}, expected {expected_insertions}."
            )
        completed_updates = components.update_loop.gradient_steps - initial_gradient_steps
        print(
            f"[{components.mode.value}] PASS transitions={components.replay.total_inserted} "
            f"retained={len(components.replay)} new_gradient_steps={completed_updates} "
            f"normalizer_samples={components.normalizer.samples_seen if components.normalizer is not None else 0}"
        )
    finally:
        if logger is not None:
            logger.close()
        env.close()


def main() -> None:
    failed = False
    try:
        run()
    except BaseException:
        traceback.print_exc()
        failed = True
    finally:
        SIMULATION_APP.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
