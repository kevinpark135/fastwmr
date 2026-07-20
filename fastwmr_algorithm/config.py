"""Shared FastWMR interface and algorithm configuration.

This module deliberately has no IsaacLab or torch imports. Environment configs,
networks, replay buffers, and tests can therefore agree on tensor layouts
without starting the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TargetKind(str, Enum):
    """Loss family used for a reconstructed target field."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class ControlFeatureMode(str, Enum):
    """Features exposed to both the actor and the primary twin critics."""

    OBS_AND_RECONSTRUCTION = "obs_and_reconstruction"
    RECONSTRUCTION_ONLY = "reconstruction_only"


@dataclass(frozen=True)
class TensorFieldSpec:
    """A named, fixed-width slice in a concatenated tensor."""

    name: str
    width: int
    kind: TargetKind = TargetKind.CONTINUOUS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tensor field names must not be empty.")
        if self.width <= 0:
            raise ValueError(f"Field {self.name!r} must have a positive width, got {self.width}.")


@dataclass(frozen=True)
class TensorLayoutSpec:
    """Ordered tensor fields with deterministic names, widths, and slices."""

    fields: tuple[TensorFieldSpec, ...]

    def __post_init__(self) -> None:
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError(f"Tensor field names must be unique, got {names}.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def dim(self) -> int:
        return sum(field.width for field in self.fields)

    def field_slice(self, name: str) -> slice:
        """Return the last-dimension slice assigned to ``name``."""

        offset = 0
        for field in self.fields:
            next_offset = offset + field.width
            if field.name == name:
                return slice(offset, next_offset)
            offset = next_offset
        raise KeyError(f"Unknown tensor field {name!r}; expected one of {self.names}.")

    def select_kind(self, kind: TargetKind) -> "TensorLayoutSpec":
        return TensorLayoutSpec(tuple(field for field in self.fields if field.kind is kind))


def make_policy_observation_layout(action_dim: int) -> TensorLayoutSpec:
    """Build the deployable proprioceptive observation layout ``o_t``.

    Joint position, joint velocity, and previous action all follow the action
    manager's 29-DoF ordering. Base linear velocity and terrain scans are
    intentionally absent because they are not assumed available on hardware.
    """

    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    return TensorLayoutSpec(
        (
            TensorFieldSpec("base_ang_vel", 3),
            TensorFieldSpec("projected_gravity", 3),
            TensorFieldSpec("velocity_command", 3),
            TensorFieldSpec("joint_pos", action_dim),
            TensorFieldSpec("joint_vel", action_dim),
            TensorFieldSpec("previous_action", action_dim),
        )
    )


MVP_RECONSTRUCTION_LAYOUT = TensorLayoutSpec(
    (
        TensorFieldSpec("base_lin_vel", 3),
        TensorFieldSpec("friction", 1),
        TensorFieldSpec("payload_mass", 1),
        TensorFieldSpec("push_force_torque", 6),
        TensorFieldSpec("foot_contacts", 2, TargetKind.DISCRETE),
    )
)
"""MVP privileged target ``s_t``: 11 continuous values and 2 contact bits."""


@dataclass(frozen=True)
class FastWMRInterfaceCfg:
    """Canonical dimensions and feature-routing choices for FastWMR."""

    action_dim: int = 29
    reconstruction_layout: TensorLayoutSpec = MVP_RECONSTRUCTION_LAYOUT
    control_feature_mode: ControlFeatureMode = ControlFeatureMode.OBS_AND_RECONSTRUCTION

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.reconstruction_layout.dim <= 0:
            raise ValueError("The reconstruction target must contain at least one value.")

    @property
    def policy_observation_layout(self) -> TensorLayoutSpec:
        return make_policy_observation_layout(self.action_dim)

    @property
    def policy_observation_dim(self) -> int:
        return self.policy_observation_layout.dim

    @property
    def reconstruction_target_dim(self) -> int:
        return self.reconstruction_layout.dim

    @property
    def continuous_target_dim(self) -> int:
        return self.reconstruction_layout.select_kind(TargetKind.CONTINUOUS).dim

    @property
    def discrete_target_dim(self) -> int:
        return self.reconstruction_layout.select_kind(TargetKind.DISCRETE).dim

    @property
    def control_feature_dim(self) -> int:
        if self.control_feature_mode is ControlFeatureMode.RECONSTRUCTION_ONLY:
            return self.reconstruction_target_dim
        return self.policy_observation_dim + self.reconstruction_target_dim

    @property
    def actor_input_dim(self) -> int:
        return self.control_feature_dim

    @property
    def critic_state_dim(self) -> int:
        return self.control_feature_dim

    @property
    def critic_input_dim(self) -> int:
        return self.critic_state_dim + self.action_dim


DEFAULT_INTERFACE_CFG = FastWMRInterfaceCfg()
"""Default 29-DoF contract: 96D policy obs, 13D target, 109D feature."""


@dataclass(frozen=True)
class FastWMRAlgoCfg:
    """MVP algorithm settings that do not belong to the IsaacLab task config.

    SAC uses ordinary transition replay. Recurrent hidden/cell tensors are
    rollout-only state, and estimator updates consume a separate recent rollout
    cache. Exact replay-time recurrent re-inference remains a later research
    branch rather than an MVP claim.
    """

    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG
    discount: float = 0.97
    target_update_rate: float = 0.005
    initial_temperature: float = 0.001
    target_entropy: float = 0.0
    estimator_hidden_dim: int = 256
    estimator_num_layers: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}.")
        if not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("target_update_rate must be in (0, 1].")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive.")
        if self.estimator_hidden_dim <= 0 or self.estimator_num_layers <= 0:
            raise ValueError("Estimator hidden dimensions and layer count must be positive.")
