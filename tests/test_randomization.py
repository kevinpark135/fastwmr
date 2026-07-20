"""Tests for FastWMR domain-randomization record buffers."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.fastwmr_env_cfg import (
    G1FastWMREnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.observations import (
    privileged_friction,
    privileged_payload_mass,
    privileged_push_force_torque,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.randomization import (
    FASTWMR_DR_BUFFER_WIDTHS,
    FASTWMR_FRICTION_ATTR,
    FASTWMR_PAYLOAD_MASS_ATTR,
    FASTWMR_PUSH_FORCE_TORQUES_ATTR,
    initialize_fastwmr_dr_buffers,
    randomize_and_record_friction,
    randomize_and_record_payload_mass,
    sample_apply_record_external_wrench,
)


def _env(num_envs: int = 4) -> SimpleNamespace:
    return SimpleNamespace(num_envs=num_envs, device="cpu")


def test_startup_event_initializes_canonical_dr_buffers() -> None:
    env = _env()

    initialize_fastwmr_dr_buffers(env, None, nominal_friction=0.8)

    assert FASTWMR_DR_BUFFER_WIDTHS == {
        FASTWMR_FRICTION_ATTR: 1,
        FASTWMR_PAYLOAD_MASS_ATTR: 1,
        FASTWMR_PUSH_FORCE_TORQUES_ATTR: 6,
    }
    assert torch.equal(env.fastwmr_friction, torch.full((4, 1), 0.8))
    assert torch.equal(env.fastwmr_payload_mass, torch.zeros(4, 1))
    assert torch.equal(env.fastwmr_push_force_torques, torch.zeros(4, 6))
    for attribute, width in FASTWMR_DR_BUFFER_WIDTHS.items():
        buffer = getattr(env, attribute)
        assert buffer.shape == (4, width)
        assert buffer.dtype == torch.float32
        assert buffer.device.type == "cpu"


def test_partial_initialization_resets_only_requested_environments() -> None:
    env = _env()
    initialize_fastwmr_dr_buffers(env, None)
    env.fastwmr_friction.fill_(1.5)
    env.fastwmr_payload_mass.fill_(3.0)
    env.fastwmr_push_force_torques.fill_(7.0)

    initialize_fastwmr_dr_buffers(env, torch.tensor([1, 3]), nominal_friction=0.6)

    assert torch.equal(env.fastwmr_friction[:, 0], torch.tensor([1.5, 0.6, 1.5, 0.6]))
    assert torch.equal(env.fastwmr_payload_mass[:, 0], torch.tensor([3.0, 0.0, 3.0, 0.0]))
    assert torch.equal(env.fastwmr_push_force_torques[[0, 2]], torch.full((2, 6), 7.0))
    assert torch.equal(env.fastwmr_push_force_torques[[1, 3]], torch.zeros(2, 6))


def test_privileged_observations_read_the_initialized_buffers() -> None:
    env = _env(num_envs=2)
    initialize_fastwmr_dr_buffers(env, None)
    env.fastwmr_friction.copy_(torch.tensor([[0.3], [1.2]]))
    env.fastwmr_payload_mass.copy_(torch.tensor([[2.0], [4.0]]))
    env.fastwmr_push_force_torques.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6))

    assert torch.equal(privileged_friction(env), env.fastwmr_friction)
    assert torch.equal(privileged_payload_mass(env), env.fastwmr_payload_mass)
    assert torch.equal(privileged_push_force_torque(env), env.fastwmr_push_force_torques)


def test_buffer_initialization_rejects_invalid_contracts() -> None:
    env = _env()
    env.fastwmr_friction = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="must have shape"):
        initialize_fastwmr_dr_buffers(env, None)

    with pytest.raises(ValueError, match="non-negative"):
        initialize_fastwmr_dr_buffers(_env(), None, nominal_friction=-0.1)
    with pytest.raises(IndexError, match="outside"):
        initialize_fastwmr_dr_buffers(_env(), torch.tensor([4]))


def test_environment_registers_dr_buffer_startup_event() -> None:
    cfg = G1FastWMREnvCfg()

    assert cfg.events.initialize_fastwmr_dr_buffers.func is initialize_fastwmr_dr_buffers
    assert cfg.events.initialize_fastwmr_dr_buffers.mode == "startup"
    assert cfg.events.initialize_fastwmr_dr_buffers.params == {"nominal_friction": 0.8}


def test_environment_registers_exact_sample_apply_record_events() -> None:
    cfg = G1FastWMREnvCfg()

    assert cfg.events.physics_material is None
    assert cfg.events.add_base_mass is None
    assert cfg.events.push_robot is None
    assert cfg.events.randomize_fastwmr_friction.func is randomize_and_record_friction
    assert cfg.events.randomize_fastwmr_friction.mode == "startup"
    assert cfg.events.randomize_fastwmr_friction.params["friction_range"] == (0.2, 1.5)
    assert cfg.events.randomize_fastwmr_payload.func is randomize_and_record_payload_mass
    assert cfg.events.randomize_fastwmr_payload.mode == "startup"
    assert cfg.events.randomize_fastwmr_payload.params["payload_mass_range"] == (-5.0, 5.0)
    assert cfg.events.base_external_force_torque.func is sample_apply_record_external_wrench
    assert cfg.events.base_external_force_torque.mode == "reset"
    assert cfg.events.base_external_force_torque.params["force_range"] == (-50.0, 50.0)
    assert cfg.events.base_external_force_torque.params["torque_range"] == (-10.0, 10.0)
