"""Headless 1,000-step smoke runner for FastWMR and FastSAC tasks.

Run directly so the simulator lifecycle is isolated from the normal unit-test
process::

    python tests/task_smoke.py --steps 1000 --num-envs 16
"""

from __future__ import annotations

import argparse
import traceback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-dr", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()

# Isaac Sim must be launched before importing gym task implementations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.observations import (
    privileged_reconstruction_target,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.randomization import (
    FASTWMR_DR_BUFFER_WIDTHS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.rewards import (
    FASTSAC_REWARD_TERM_NAMES,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


TRAIN_TASKS = (
    "Isaac-Velocity-G1-FastWMR-v0",
    "Isaac-Velocity-G1-FastSAC-Baseline-v0",
)


def _apply_debug_overrides(cfg: object) -> None:
    """Keep the Rough task but make the smoke run cheap and deterministic."""

    cfg.seed = ARGS.seed
    cfg.scene.num_envs = ARGS.num_envs
    cfg.observations.policy.enable_corruption = False
    if not ARGS.full_dr:
        cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.curriculum.terrain_levels = None

    terrain = cfg.scene.terrain
    terrain.max_init_terrain_level = 0
    if terrain.terrain_generator is not None:
        terrain.terrain_generator.num_rows = 2
        terrain.terrain_generator.num_cols = 2
        terrain.terrain_generator.curriculum = False


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


def _check_observations(task_id: str, observations: dict[str, torch.Tensor]) -> None:
    expected_groups = {"policy", "privileged"} if "FastWMR" in task_id else {"policy"}
    if set(observations) != expected_groups:
        raise AssertionError(f"{task_id} returned groups {set(observations)}, expected {expected_groups}.")

    expected_dims = {"policy": 96, "privileged": 13}
    for group_name, tensor in observations.items():
        expected_shape = (ARGS.num_envs, expected_dims[group_name])
        if tensor.shape != expected_shape:
            raise AssertionError(f"{task_id}/{group_name} has shape {tensor.shape}, expected {expected_shape}.")
        if not tensor.dtype.is_floating_point:
            raise AssertionError(f"{task_id}/{group_name} must be floating point, got {tensor.dtype}.")
        _assert_finite(f"{task_id}/{group_name}", tensor)


def _check_dr_buffers(task_id: str, env: object) -> None:
    for attribute, width in FASTWMR_DR_BUFFER_WIDTHS.items():
        buffer = getattr(env, attribute, None)
        if not isinstance(buffer, torch.Tensor):
            raise AssertionError(f"{task_id} did not initialize env.{attribute}.")
        if buffer.shape != (ARGS.num_envs, width):
            raise AssertionError(
                f"{task_id}/env.{attribute} has shape {buffer.shape}, expected {(ARGS.num_envs, width)}."
            )
        if buffer.device != torch.device(env.device) or buffer.dtype != torch.float32:
            raise AssertionError(f"{task_id}/env.{attribute} has an invalid device or dtype.")
        _assert_finite(f"{task_id}/env.{attribute}", buffer)


def _check_reward_terms(task_id: str, env: object) -> None:
    reward_manager = env.reward_manager
    if tuple(reward_manager.active_terms) != FASTSAC_REWARD_TERM_NAMES:
        raise AssertionError(
            f"{task_id} reward terms are {tuple(reward_manager.active_terms)}, "
            f"expected {FASTSAC_REWARD_TERM_NAMES}."
        )
    _assert_finite(f"{task_id}/reward_terms", reward_manager._step_reward)


def _check_privileged_target(
    task_id: str,
    env: object,
    observations: dict[str, torch.Tensor],
) -> None:
    if "FastWMR" not in task_id:
        return

    expected = privileged_reconstruction_target(env)
    actual = observations["privileged"]
    if not torch.equal(actual, expected):
        max_error = (actual - expected).abs().max().item()
        raise AssertionError(
            f"{task_id}/privileged does not match the canonical 13D target; max error={max_error}."
        )


def _check_partial_wrench_recording(task_id: str, env: object) -> None:
    """Verify that a reset-style wrench update touches only selected environments."""

    if not ARGS.full_dr or ARGS.num_envs < 2:
        return

    wrench_cfg = env.event_manager.get_term_cfg("base_external_force_torque")
    selected = torch.arange(0, ARGS.num_envs, 2, device=env.device, dtype=torch.long)
    untouched = torch.arange(1, ARGS.num_envs, 2, device=env.device, dtype=torch.long)
    before = env.fastwmr_push_force_torques.clone()

    wrench_cfg.func(env, selected, **wrench_cfg.params)

    if not torch.equal(env.fastwmr_push_force_torques[untouched], before[untouched]):
        raise AssertionError(f"{task_id} partial wrench update modified unselected environments.")
    _check_dr_physics(task_id, env)


def _check_dr_physics(task_id: str, env: object) -> None:
    robot = env.scene["robot"]

    materials = wp.to_torch(robot.root_view.get_material_properties())
    recorded_friction = env.fastwmr_friction[:, 0].cpu()
    expected_friction = recorded_friction[:, None].expand_as(materials[..., 0])
    if not torch.allclose(materials[..., 0], expected_friction) or not torch.allclose(
        materials[..., 1], expected_friction
    ):
        raise AssertionError(f"{task_id} recorded friction does not match PhysX material properties.")

    payload_term = env.event_manager.get_term_cfg("randomize_fastwmr_payload").func
    pelvis_id = payload_term.body_ids.to(dtype=torch.long)
    actual_mass = robot.data.body_mass.torch[:, pelvis_id].squeeze(-1)
    nominal_mass = payload_term.default_mass[:, pelvis_id].squeeze(-1)
    if not torch.allclose(actual_mass - nominal_mass, env.fastwmr_payload_mass[:, 0]):
        raise AssertionError(f"{task_id} recorded payload does not match the applied pelvis mass.")

    if ARGS.full_dr:
        wrench_cfg = env.event_manager.get_term_cfg("base_external_force_torque")
        pelvis_id = wrench_cfg.params["asset_cfg"].body_ids[0]
        composer = robot.permanent_wrench_composer
        applied_force = wp.to_torch(composer.local_force_b)[:, pelvis_id]
        applied_torque = wp.to_torch(composer.local_torque_b)[:, pelvis_id]
        applied_wrench = torch.cat((applied_force, applied_torque), dim=-1)
        if not torch.allclose(applied_wrench, env.fastwmr_push_force_torques):
            raise AssertionError(f"{task_id} recorded wrench does not match the applied body-frame wrench.")


def _run_task(task_id: str) -> None:
    sim_utils.create_new_stage()
    cfg = parse_env_cfg(task_id, device=ARGS.device, num_envs=ARGS.num_envs)
    _apply_debug_overrides(cfg)
    env = gym.make(task_id, cfg=cfg)

    try:
        if env.unwrapped.action_manager.total_action_dim != 29:
            raise AssertionError(
                f"{task_id} resolved {env.unwrapped.action_manager.total_action_dim} actions instead of 29."
            )
        expected_action_shape = (ARGS.num_envs, 29)
        if env.action_space.shape != expected_action_shape:
            raise AssertionError(f"{task_id} action space is {env.action_space.shape}, expected {expected_action_shape}.")

        observations, _ = env.reset(seed=ARGS.seed)
        _check_observations(task_id, observations)
        _check_dr_buffers(task_id, env.unwrapped)
        _check_reward_terms(task_id, env.unwrapped)
        _check_privileged_target(task_id, env.unwrapped, observations)
        _check_dr_physics(task_id, env.unwrapped)
        _check_partial_wrench_recording(task_id, env.unwrapped)

        # Refresh once after the direct partial-reset probe so the returned
        # privileged observation reflects the newly recorded wrench.
        observations = env.unwrapped.observation_manager.compute()
        _check_observations(task_id, observations)
        _check_privileged_target(task_id, env.unwrapped, observations)

        with torch.inference_mode():
            for step in range(ARGS.steps):
                actions = 2.0 * torch.rand(expected_action_shape, device=env.unwrapped.device) - 1.0
                observations, rewards, terminated, truncated, _ = env.step(actions)

                _check_observations(task_id, observations)
                _check_privileged_target(task_id, env.unwrapped, observations)
                _assert_finite(f"{task_id}/reward", rewards)
                _check_reward_terms(task_id, env.unwrapped)
                if rewards.shape != (ARGS.num_envs,):
                    raise AssertionError(f"{task_id} reward shape is {rewards.shape}.")
                if terminated.shape != (ARGS.num_envs,) or terminated.dtype is not torch.bool:
                    raise AssertionError(f"{task_id} returned invalid terminated tensor {terminated.shape}.")
                if truncated.shape != (ARGS.num_envs,) or truncated.dtype is not torch.bool:
                    raise AssertionError(f"{task_id} returned invalid truncated tensor {truncated.shape}.")

                if (step + 1) % 250 == 0:
                    print(f"[SMOKE] {task_id}: {step + 1}/{ARGS.steps} steps")
    finally:
        env.close()
    print(f"[SMOKE] {task_id}: passed {ARGS.steps} steps")


def main() -> None:
    if ARGS.steps <= 0 or ARGS.num_envs <= 0:
        raise ValueError("--steps and --num-envs must be positive.")

    failed = False
    try:
        for task_id in TRAIN_TASKS:
            _run_task(task_id)
        print("[SMOKE] All FastWMR task smoke checks passed.")
    except BaseException:
        # SimulationApp.close() can otherwise hide the active traceback.
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
