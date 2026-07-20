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
    return parser.parse_args()


ARGS = _parse_args()

# Isaac Sim must be launched before importing gym task implementations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr  # noqa: F401
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

        with torch.inference_mode():
            for step in range(ARGS.steps):
                actions = 2.0 * torch.rand(expected_action_shape, device=env.unwrapped.device) - 1.0
                observations, rewards, terminated, truncated, _ = env.step(actions)

                _check_observations(task_id, observations)
                _assert_finite(f"{task_id}/reward", rewards)
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
