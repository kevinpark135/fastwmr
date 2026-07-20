# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization and privileged-buffer bookkeeping for FastWMR.

FastWMR needs randomization values both to perturb the simulator and to train
the estimator. Each randomization term should therefore perform three related
jobs:

1. Sample the randomized value.
2. Apply it to the simulator or delegate application to an IsaacLab event.
3. Record the value in ``env.fastwmr_*`` buffers for privileged observation and
   replay storage.

The buffers in this module are the sole source read by privileged observation
terms. Randomization functions added here must write the exact value they apply
to physics; independently resampling a reconstruction label is not allowed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


FASTWMR_FRICTION_ATTR = "fastwmr_friction"
FASTWMR_PAYLOAD_MASS_ATTR = "fastwmr_payload_mass"
FASTWMR_PUSH_FORCE_TORQUES_ATTR = "fastwmr_push_force_torques"

FASTWMR_DR_BUFFER_WIDTHS = {
    FASTWMR_FRICTION_ATTR: 1,
    FASTWMR_PAYLOAD_MASS_ATTR: 1,
    FASTWMR_PUSH_FORCE_TORQUES_ATTR: 6,
}
"""Canonical environment attributes and widths used by privileged targets."""


def _num_envs(env: "ManagerBasedEnv") -> int:
    if hasattr(env, "num_envs"):
        return int(env.num_envs)
    if hasattr(env, "scene") and hasattr(env.scene, "num_envs"):
        return int(env.scene.num_envs)
    return int(env.episode_length_buf.shape[0])


def _device(env: "ManagerBasedEnv") -> torch.device:
    if hasattr(env, "device"):
        return torch.device(env.device)
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.device
    return torch.device("cpu")


def _resolve_env_ids(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(_num_envs(env), device=device, dtype=torch.long)
    if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1:
        raise TypeError("env_ids must be a one-dimensional torch.Tensor or None.")
    ids = env_ids.to(device=device, dtype=torch.long)
    if ids.numel() > 0 and (ids.min() < 0 or ids.max() >= _num_envs(env)):
        raise IndexError("env_ids contains an environment index outside the valid range.")
    return ids


def _ensure_buffer(
    env: "ManagerBasedEnv",
    attribute: str,
    width: int,
    fill_value: float,
) -> torch.Tensor:
    expected_shape = (_num_envs(env), width)
    device = _device(env)
    buffer = getattr(env, attribute, None)
    if buffer is None:
        buffer = torch.full(expected_shape, fill_value, device=device, dtype=torch.float32)
        setattr(env, attribute, buffer)
        return buffer
    if not isinstance(buffer, torch.Tensor):
        raise TypeError(f"env.{attribute} must be a torch.Tensor.")
    if buffer.shape != expected_shape:
        raise ValueError(f"env.{attribute} must have shape {expected_shape}, got {tuple(buffer.shape)}.")
    if buffer.device != device:
        raise ValueError(f"env.{attribute} must be on {device}, got {buffer.device}.")
    if buffer.dtype != torch.float32:
        raise TypeError(f"env.{attribute} must have dtype torch.float32, got {buffer.dtype}.")
    return buffer


def initialize_fastwmr_dr_buffers(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    nominal_friction: float = 0.8,
) -> None:
    """Create or reset canonical per-environment DR record buffers.

    Friction starts at the task's nominal static coefficient. Payload stores
    additional mass in kilograms, so its neutral value is zero. The wrench
    target is also zero until an external force/torque event is active.
    """

    if not math.isfinite(nominal_friction) or nominal_friction < 0.0:
        raise ValueError("nominal_friction must be finite and non-negative.")

    device = _device(env)
    ids = _resolve_env_ids(env, env_ids, device)
    friction = _ensure_buffer(env, FASTWMR_FRICTION_ATTR, 1, nominal_friction)
    payload_mass = _ensure_buffer(env, FASTWMR_PAYLOAD_MASS_ATTR, 1, 0.0)
    push_force_torques = _ensure_buffer(env, FASTWMR_PUSH_FORCE_TORQUES_ATTR, 6, 0.0)

    friction[ids] = nominal_friction
    payload_mass[ids] = 0.0
    push_force_torques[ids] = 0.0


def _validate_range(value_range: tuple[float, float], name: str) -> None:
    if len(value_range) != 2 or not all(math.isfinite(value) for value in value_range):
        raise ValueError(f"{name} must contain two finite values.")
    if value_range[0] > value_range[1]:
        raise ValueError(f"{name} lower bound must not exceed its upper bound.")


def _asset_env_ids(env: "ManagerBasedEnv", env_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    return _resolve_env_ids(env, env_ids, device).to(dtype=torch.int32)


def _single_body_id(asset_cfg: SceneEntityCfg, name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
        raise ValueError(f"{name} requires asset_cfg to resolve exactly one body.")
    return torch.tensor(asset_cfg.body_ids, device=device, dtype=torch.int32)


class randomize_and_record_friction(ManagerTermBase):
    """Apply one bucketed friction coefficient per environment and record it."""

    def __init__(self, cfg, env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        selects_all_bodies = self.asset_cfg.body_ids == slice(None) or (
            isinstance(self.asset_cfg.body_ids, list) and len(self.asset_cfg.body_ids) == self.asset.num_bodies
        )
        if not selects_all_bodies:
            raise ValueError("FastWMR friction randomization must target every robot body.")

        friction_range = cfg.params["friction_range"]
        _validate_range(friction_range, "friction_range")
        num_buckets = int(cfg.params["num_buckets"])
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive.")
        self.friction_buckets = torch.linspace(friction_range[0], friction_range[1], num_buckets, device="cpu")

        manager_name = env.sim.physics_manager.__name__.lower()
        if "physx" not in manager_name or manager_name == "ovphysxmanager":
            raise RuntimeError("FastWMR exact friction recording currently requires the PhysX backend.")

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        restitution: float,
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Assign one material row to all shapes of each selected environment."""

        import warp as wp  # Imported lazily so pure config tests do not require Kit startup.

        if not math.isfinite(restitution) or restitution < 0.0:
            raise ValueError("restitution must be finite and non-negative.")
        ids_cpu = _asset_env_ids(env, env_ids, torch.device("cpu"))
        bucket_ids = torch.randint(0, num_buckets, (ids_cpu.numel(),), device="cpu")
        friction = self.friction_buckets[bucket_ids]
        material_rows = torch.stack(
            (friction, friction, torch.full_like(friction, restitution)),
            dim=-1,
        )

        materials = wp.to_torch(self.asset.root_view.get_material_properties())
        materials[ids_cpu] = material_rows.unsqueeze(1)
        self.asset.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32),
            wp.from_torch(ids_cpu, dtype=wp.int32),
        )

        buffer = _ensure_buffer(env, FASTWMR_FRICTION_ATTR, 1, 0.8)
        buffer[ids_cpu.to(buffer.device, dtype=torch.long), 0] = friction.to(buffer.device)


class randomize_and_record_payload_mass(ManagerTermBase):
    """Add a sampled pelvis payload relative to nominal mass and record it."""

    def __init__(self, cfg, env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        _validate_range(cfg.params["payload_mass_range"], "payload_mass_range")
        self.body_ids = _single_body_id(self.asset_cfg, "payload randomization", self.asset.device)
        self.default_mass: torch.Tensor | None = None
        self.default_inertia: torch.Tensor | None = None

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        payload_mass_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
        min_mass: float,
    ) -> None:
        if not math.isfinite(min_mass) or min_mass <= 0.0:
            raise ValueError("min_mass must be finite and positive.")
        if self.default_mass is None:
            self.default_mass = self.asset.data.body_mass.torch.clone()
            self.default_inertia = self.asset.data.body_inertia.torch.clone()
        assert self.default_inertia is not None

        ids = _asset_env_ids(env, env_ids, self.asset.device)
        ids_long = ids.to(dtype=torch.long)
        body_ids_long = self.body_ids.to(dtype=torch.long)
        payload = torch.empty(ids.numel(), device=self.asset.device).uniform_(*payload_mass_range)
        nominal_mass = self.default_mass[ids_long[:, None], body_ids_long].squeeze(-1)
        randomized_mass = (nominal_mass + payload).clamp_min(min_mass)
        applied_payload = randomized_mass - nominal_mass

        self.asset.set_masses_index(
            masses=randomized_mass.unsqueeze(-1),
            body_ids=self.body_ids,
            env_ids=ids,
        )
        nominal_inertia = self.default_inertia[ids_long[:, None], body_ids_long]
        randomized_inertia = nominal_inertia * (randomized_mass / nominal_mass).view(-1, 1, 1)
        self.asset.set_inertias_index(
            inertias=randomized_inertia,
            body_ids=self.body_ids,
            env_ids=ids,
        )

        buffer = _ensure_buffer(env, FASTWMR_PAYLOAD_MASS_ATTR, 1, 0.0)
        buffer[ids_long.to(buffer.device), 0] = applied_payload.to(buffer.device)


def sample_apply_record_external_wrench(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Apply one body-frame pelvis wrench per episode and record the exact 6D value."""

    _validate_range(force_range, "force_range")
    _validate_range(torque_range, "torque_range")
    asset = env.scene[asset_cfg.name]
    ids = _asset_env_ids(env, env_ids, asset.device)
    body_ids = _single_body_id(asset_cfg, "external wrench randomization", asset.device)
    size = (ids.numel(), 1, 3)
    forces = torch.empty(size, device=asset.device).uniform_(*force_range)
    torques = torch.empty(size, device=asset.device).uniform_(*torque_range)
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=ids,
    )

    buffer = _ensure_buffer(env, FASTWMR_PUSH_FORCE_TORQUES_ATTR, 6, 0.0)
    wrench = torch.cat((forces[:, 0], torques[:, 0]), dim=-1)
    buffer[ids.to(buffer.device, dtype=torch.long)] = wrench.to(buffer.device)
