"""Executable FastSAC baseline collector and learner for the Rough G1 task."""

from __future__ import annotations

import math
import traceback

from cli_args import FASTSAC_BASELINE_TASK, build_train_parser, validate_train_args

from isaaclab.app import AppLauncher


PARSER = build_train_parser()
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
validate_train_args(ARGS)

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    EntropyTemperature,
    FastSACReplayUpdateLoop,
    FastSACRolloutCollector,
    SACUpdater,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.buffers import (
    ReplayBufferSpec,
    TransitionReplayBuffer,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ReplayUpdateCfg,
    ScalarCriticCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    TargetTwinScalarCritic,
    TanhGaussianActor,
    TwinScalarCritic,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import IsaacLabEnvAdapter
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


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


def _build_updater(observation_dim: int, action_dim: int, device: torch.device) -> SACUpdater:
    actor_cfg = TanhGaussianActorCfg(hidden_dim=ARGS.hidden_dim)
    critic_cfg = ScalarCriticCfg(hidden_dim=ARGS.hidden_dim)
    actor = TanhGaussianActor(observation_dim, action_dim, cfg=actor_cfg).to(device)
    critic = TwinScalarCritic(observation_dim, action_dim, cfg=critic_cfg).to(device)
    target_critic = TargetTwinScalarCritic.from_online(critic)
    temperature = EntropyTemperature(ARGS.initial_temperature).to(device)
    optimizer_kwargs = {
        "lr": ARGS.learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": ARGS.weight_decay,
    }
    return SACUpdater(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        actor_optimizer=torch.optim.Adam(actor.parameters(), **optimizer_kwargs),
        critic_optimizer=torch.optim.Adam(critic.parameters(), **optimizer_kwargs),
        temperature_optimizer=torch.optim.Adam(temperature.parameters(), lr=ARGS.learning_rate, betas=(0.9, 0.95)),
        discount=ARGS.discount,
        target_update_rate=ARGS.target_update_rate,
        target_entropy=ARGS.target_entropy,
    )


def _metric_values(metric: object) -> tuple[float, ...]:
    names = ("critic_loss", "actor_loss", "temperature_loss", "temperature", "target_q_mean")
    return tuple(float(getattr(metric, name).item()) for name in names)


def run() -> None:
    if ARGS.task != FASTSAC_BASELINE_TASK:
        raise ValueError(f"Stage-2 runner currently supports only {FASTSAC_BASELINE_TASK!r}.")
    torch.manual_seed(ARGS.seed)
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.seed = ARGS.seed
    if ARGS.episode_length_s is not None:
        cfg.episode_length_s = ARGS.episode_length_s
    if ARGS.rough_debug:
        _apply_rough_debug_overrides(cfg)

    raw_env = gym.make(ARGS.task, cfg=cfg)
    env = IsaacLabEnvAdapter(raw_env)
    try:
        observation_dim = int(raw_env.observation_space["policy"].shape[-1])
        action_dim = int(raw_env.unwrapped.action_manager.total_action_dim)
        learner_device = env.device
        replay = TransitionReplayBuffer(
            ReplayBufferSpec(
                capacity=ARGS.replay_capacity,
                observation_dim=observation_dim,
                action_dim=action_dim,
            ),
            storage_device=ARGS.replay_storage_device,
        )
        updater = _build_updater(observation_dim, action_dim, learner_device)
        update_cfg = ReplayUpdateCfg(
            random_action_steps=ARGS.random_action_steps,
            minimum_replay_size=ARGS.minimum_replay_size,
            batch_size=ARGS.batch_size,
            num_updates=ARGS.num_updates,
        )
        update_loop = FastSACReplayUpdateLoop(
            replay,
            updater,
            update_cfg,
            learner_device=learner_device,
        )
        collector = FastSACRolloutCollector(env, replay, update_loop)
        collector.reset(seed=ARGS.seed)

        last_metric = None
        reward_sum = 0.0
        done_count = 0
        generator = torch.Generator(device="cpu").manual_seed(ARGS.seed)
        for step_index in range(ARGS.steps):
            result = collector.collect_step(generator=generator)
            reward_sum += float(result.rewards.mean().item())
            done_count += int((result.terminated | result.truncated).sum().item())
            if result.updates:
                last_metric = result.updates[-1]
                if not all(math.isfinite(value) for value in _metric_values(last_metric)):
                    raise FloatingPointError("FastSAC update produced a non-finite metric.")
            if (step_index + 1) % ARGS.log_interval == 0 or step_index + 1 == ARGS.steps:
                metric_text = "waiting-for-replay"
                if last_metric is not None:
                    values = _metric_values(last_metric)
                    metric_text = f"critic={values[0]:.4f} actor={values[1]:.4f} alpha={values[3]:.6f}"
                print(
                    f"[FastSAC] step={step_index + 1}/{ARGS.steps} replay={len(replay)} "
                    f"updates={update_loop.gradient_steps} reward={reward_sum / (step_index + 1):.4f} {metric_text}"
                )

        expected_insertions = ARGS.steps * ARGS.num_envs
        if replay.total_inserted != expected_insertions:
            raise AssertionError(f"Replay inserted {replay.total_inserted}, expected {expected_insertions} transitions.")
        if update_loop.gradient_steps == 0:
            raise AssertionError("Smoke run finished without a gradient update.")
        print(
            f"[FastSAC] PASS transitions={replay.total_inserted} retained={len(replay)} "
            f"gradient_steps={update_loop.gradient_steps} done={done_count}"
        )
    finally:
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
