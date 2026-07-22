"""Run fixed-budget nominal or OOD evaluation for a saved checkpoint."""

from __future__ import annotations

import traceback

from cli_args import (
    FASTSAC_BASELINE_PLAY_TASK,
    FASTWMR_PLAY_TASK,
    build_play_parser,
    validate_play_args,
)

from isaaclab.app import AppLauncher


PARSER = build_play_parser()
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
validate_play_args(ARGS)

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import math
import time
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.algorithm import (
    FastWMREstimatorRuntime,
    TrainingMode,
    WorldStateEstimator,
    inspect_training_checkpoint,
    load_policy_checkpoint,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    ControlFeatureMode,
    FastWMRInterfaceCfg,
    ObservationNormalizationCfg,
    TanhGaussianActorCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    HistoryEncoder,
    TanhGaussianActor,
    WorldStateDecoder,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    EvaluationCondition,
    EvaluationRecord,
    IsaacLabEnvAdapter,
    RunningObservationNormalizer,
    build_control_feature,
    training_seed_from_config,
    write_evaluation_record,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.randomization import (
    sample_apply_record_external_wrench,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class _OnlineCorrelation:
    """Accumulate Pearson correlation without retaining rollout tensors."""

    def __init__(self, interface: FastWMRInterfaceCfg) -> None:
        self.interface = interface
        self.statistics = {
            field.name: torch.zeros(6, dtype=torch.float64)
            for field in interface.reconstruction_layout.fields
        }
        self.statistics["overall"] = torch.zeros(6, dtype=torch.float64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        for field in self.interface.reconstruction_layout.fields:
            field_slice = self.interface.reconstruction_layout.field_slice(field.name)
            self._update_one(field.name, prediction[..., field_slice], target[..., field_slice])
        self._update_one("overall", prediction, target)

    def _update_one(self, name: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
        x = prediction.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        y = target.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        self.statistics[name] += torch.stack(
            (
                torch.tensor(float(x.numel()), dtype=torch.float64),
                x.sum(),
                y.sum(),
                x.square().sum(),
                y.square().sum(),
                (x * y).sum(),
            )
        )

    def correlations(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, values in self.statistics.items():
            count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = values.tolist()
            covariance = count * sum_xy - sum_x * sum_y
            variance = (count * sum_xx - sum_x**2) * (count * sum_yy - sum_y**2)
            output[name] = covariance / math.sqrt(variance) if variance > 1e-18 else 0.0
        return output


def _training_arguments(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        return {}
    arguments = config.get("arguments", {})
    return dict(arguments) if isinstance(arguments, dict) else {}


def _apply_condition(cfg: object, condition: EvaluationCondition) -> None:
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.penalty_weights = None
    cfg.observations.policy.enable_corruption = False
    if condition is EvaluationCondition.FRICTION_LOW:
        cfg.events.randomize_fastwmr_friction.params["friction_range"] = (0.05, 0.05)
    elif condition is EvaluationCondition.FRICTION_HIGH:
        cfg.events.randomize_fastwmr_friction.params["friction_range"] = (2.0, 2.0)
    elif condition is EvaluationCondition.PAYLOAD_HEAVY:
        cfg.events.randomize_fastwmr_payload.params["payload_mass_range"] = (10.0, 10.0)
    elif condition is EvaluationCondition.STRONG_PUSH:
        cfg.events.base_external_force_torque = EventTerm(
            func=sample_apply_record_external_wrench,
            mode="reset",
            params={
                "force_range": (-100.0, 100.0),
                "torque_range": (-20.0, 20.0),
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            },
        )


def _perturb_observation(
    observation: torch.Tensor,
    condition: EvaluationCondition,
    generator: torch.Generator,
) -> torch.Tensor:
    if condition is EvaluationCondition.OBSERVATION_NOISE:
        noise = torch.randn(
            observation.shape,
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        )
        return observation + ARGS.observation_noise_std * noise
    if condition is EvaluationCondition.OBSERVATION_MASKING:
        mask = torch.rand(
            observation.shape,
            device=observation.device,
            generator=generator,
        ) >= ARGS.observation_mask_probability
        return observation * mask
    return observation


def _default_output(
    mode: TrainingMode,
    condition: EvaluationCondition,
    training_seed: int,
) -> Path:
    return (
        Path("evaluations")
        / mode.value
        / ARGS.variant
        / condition.value
        / f"train_seed_{training_seed}"
        / f"eval_seed_{ARGS.seed}.json"
    )


def run() -> Path:
    metadata = inspect_training_checkpoint(ARGS.checkpoint, map_location="cpu")
    arguments = _training_arguments(metadata.config)
    training_seed = training_seed_from_config(metadata.config)
    expected_task = (
        FASTWMR_PLAY_TASK
        if metadata.mode is TrainingMode.FASTWMR
        else FASTSAC_BASELINE_PLAY_TASK
    )
    task = ARGS.task or expected_task
    if task != expected_task:
        raise ValueError(
            f"Checkpoint mode {metadata.mode.value!r} requires task {expected_task!r}."
        )
    condition = EvaluationCondition(ARGS.condition)
    torch.manual_seed(ARGS.seed)

    cfg = parse_env_cfg(task, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.seed = ARGS.seed
    _apply_condition(cfg, condition)
    raw_env = gym.make(task, cfg=cfg)
    env = IsaacLabEnvAdapter(raw_env)
    try:
        observation_dim = int(raw_env.observation_space["policy"].shape[-1])
        action_dim = int(raw_env.unwrapped.action_manager.total_action_dim)
        control_mode = ControlFeatureMode(
            str(arguments.get("control_feature_mode", "obs_and_reconstruction"))
        )
        interface = FastWMRInterfaceCfg(control_feature_mode=control_mode)
        actor_input_dim = (
            interface.control_feature_dim
            if metadata.mode is TrainingMode.FASTWMR
            else observation_dim
        )
        if bool(arguments.get("disable_joint_limit_action_bounds", False)):
            action_low = torch.full((action_dim,), -1.0, device=env.device)
            action_high = torch.full((action_dim,), 1.0, device=env.device)
        else:
            bounds = env.joint_position_action_bounds(
                use_soft_limits=bool(arguments.get("use_soft_joint_limits", False))
            )
            action_low, action_high = bounds.low, bounds.high
        actor = TanhGaussianActor(
            actor_input_dim,
            action_dim,
            cfg=TanhGaussianActorCfg(hidden_dim=int(arguments.get("hidden_dim", 768))),
            action_low=action_low,
            action_high=action_high,
        ).to(env.device)
        normalizer = None
        if metadata.has_normalizer:
            normalizer = RunningObservationNormalizer(
                observation_dim,
                ObservationNormalizationCfg(
                    epsilon=float(arguments.get("normalization_epsilon", 1e-5)),
                    clip=float(arguments.get("normalization_clip", 10.0)),
                ),
            ).to(env.device)

        estimator = None
        runtime = None
        correlations = None
        if metadata.mode is TrainingMode.FASTWMR:
            estimator_hidden_dim = int(arguments.get("estimator_hidden_dim", 256))
            estimator = WorldStateEstimator(
                HistoryEncoder(
                    observation_dim,
                    hidden_dim=estimator_hidden_dim,
                    num_layers=int(arguments.get("estimator_num_layers", 1)),
                ),
                WorldStateDecoder(estimator_hidden_dim, hidden_dim=estimator_hidden_dim),
            ).to(env.device)
            runtime = FastWMREstimatorRuntime(
                estimator,
                env.num_envs,
                estimator_version=metadata.counters.estimator_version,
            )
            correlations = _OnlineCorrelation(interface)
        load_policy_checkpoint(
            metadata.path,
            mode=metadata.mode,
            actor=actor,
            estimator=estimator,
            normalizer=normalizer,
            map_location=env.device,
        )

        observations, _ = env.reset(seed=ARGS.seed)
        reset_boundaries = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        returns = torch.zeros(env.num_envs, device=env.device)
        lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        steps_since_reset = torch.zeros_like(lengths)
        recovered = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        return_sum = 0.0
        length_sum = 0
        completed_episodes = 0
        terminated_episodes = 0
        tracking_linear_sum = 0.0
        tracking_yaw_sum = 0.0
        stable_samples = 0
        recovery_steps_sum = 0
        recovery_count = 0
        tilt_peak = 0.0
        hidden_norm_sum = 0.0
        generator = torch.Generator(device=env.device).manual_seed(ARGS.seed)
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        started = time.perf_counter()

        for _ in range(ARGS.steps):
            raw_policy = env.policy_observation(observations)
            policy = _perturb_observation(raw_policy, condition, generator)
            if runtime is not None:
                runtime_step = runtime.step(policy, reset_boundaries=reset_boundaries)
                privileged = env.privileged_observation(observations)
                assert correlations is not None
                correlations.update(runtime_step.reconstruction, privileged)
                hidden_norm_sum += runtime_step.hidden_norm
                feature = build_control_feature(
                    policy,
                    runtime_step.reconstruction,
                    cfg=interface,
                    normalizer=normalizer,
                )
            else:
                feature = normalizer(policy) if normalizer is not None else policy
            actions = actor.act(feature, deterministic=not ARGS.stochastic)
            step = env.step(actions)
            returns += step.rewards
            lengths += 1
            steps_since_reset += 1

            unwrapped = raw_env.unwrapped
            robot = unwrapped.scene["robot"]
            command = unwrapped.command_manager.get_command("base_velocity")
            linear_error = (robot.data.root_lin_vel_b.torch[:, :2] - command[:, :2]).norm(dim=-1)
            yaw_error = (robot.data.root_ang_vel_b.torch[:, 2] - command[:, 2]).abs()
            tracking_linear_sum += float(linear_error.sum().item())
            tracking_yaw_sum += float(yaw_error.sum().item())
            stable = (linear_error < 0.25) & (yaw_error < 0.25)
            stable_samples += int(stable.sum().item())
            newly_recovered = stable & ~recovered
            recovery_steps_sum += int(steps_since_reset[newly_recovered].sum().item())
            recovery_count += int(newly_recovered.sum().item())
            recovered |= stable
            tilt = robot.data.projected_gravity_b.torch[:, :2].norm(dim=-1)
            tilt_peak = max(tilt_peak, float(tilt.max().item()))

            done = step.terminated | step.truncated
            if torch.any(done):
                return_sum += float(returns[done].sum().item())
                length_sum += int(lengths[done].sum().item())
                completed_episodes += int(done.sum().item())
                terminated_episodes += int(step.terminated.sum().item())
                returns[done] = 0.0
                lengths[done] = 0
                steps_since_reset[done] = 0
                recovered[done] = False
            observations = step.observations
            reset_boundaries = done

        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        wallclock = time.perf_counter() - started
        samples = float(ARGS.steps * env.num_envs)
        return_mean = (
            return_sum / completed_episodes
            if completed_episodes
            else float(returns.mean().item())
        )
        length_mean = (
            length_sum / completed_episodes
            if completed_episodes
            else float(lengths.float().mean().item())
        )
        metrics = {
            "return_mean": return_mean,
            "episode_length_mean": length_mean,
            "fall_rate": terminated_episodes / float(env.num_envs + completed_episodes),
            "linear_tracking_error": tracking_linear_sum / samples,
            "yaw_tracking_error": tracking_yaw_sum / samples,
            "tracking_success_fraction": stable_samples / samples,
            "recovery_steps_mean": recovery_steps_sum / float(max(1, recovery_count)),
            "peak_tilt": tilt_peak,
            "sim_steps_per_second": ARGS.steps / wallclock,
            "environment_steps_per_second": samples / wallclock,
            "runtime_hidden_norm_mean": hidden_norm_sum / float(ARGS.steps),
        }
        record = EvaluationRecord(
            mode=metadata.mode.value,
            variant=ARGS.variant,
            condition=condition.value,
            training_seed=training_seed,
            evaluation_seed=ARGS.seed,
            checkpoint=str(metadata.path),
            checkpoint_environment_steps=metadata.counters.environment_steps,
            evaluation_steps=ARGS.steps,
            num_envs=env.num_envs,
            wallclock_seconds=wallclock,
            metrics=metrics,
            reconstruction_correlations=(
                correlations.correlations() if correlations is not None else {}
            ),
            metadata={
                "task": task,
                "control_feature_mode": control_mode.value,
                "deterministic": not ARGS.stochastic,
            },
        )
        output = (
            ARGS.output or _default_output(metadata.mode, condition, training_seed)
        ).expanduser()
        path = write_evaluation_record(output, record)
        print(
            f"[{metadata.mode.value}/{condition.value}] train_seed={training_seed} "
            f"eval_seed={ARGS.seed} output={path} "
            f"return={return_mean:.4f} fall_rate={metrics['fall_rate']:.4f} "
            f"tracking_error={metrics['linear_tracking_error']:.4f} "
            f"wallclock={wallclock:.3f}s"
        )
        return path
    finally:
        env.close()


def main() -> None:
    failed = False
    try:
        run()
    except BaseException:
        failed = True
        traceback.print_exc()
    finally:
        SIMULATION_APP.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
