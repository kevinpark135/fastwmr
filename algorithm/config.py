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
class TanhGaussianActorCfg:
    """FastSAC stochastic actor architecture and exploration bounds.

    ``log_std_max=0`` is the FastSAC-specific choice that caps the pre-tanh
    action standard deviation at one. The three policy widths follow the
    official implementation as ``hidden_dim``, ``hidden_dim / 2``, and
    ``hidden_dim / 4``.
    """

    hidden_dim: int = 512
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    use_layer_norm: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be positive and divisible by four.")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max.")


DEFAULT_ACTOR_CFG = TanhGaussianActorCfg()


@dataclass(frozen=True)
class ScalarCriticCfg:
    """Scalar FastSAC critic used before the C51 extension is introduced."""

    hidden_dim: int = 768
    use_layer_norm: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be positive and divisible by four.")


DEFAULT_CRITIC_CFG = ScalarCriticCfg()


@dataclass(frozen=True)
class ObservationNormalizationCfg:
    """Running observation normalization used by rollout and replay learning."""

    enabled: bool = True
    epsilon: float = 1e-5
    clip: float | None = 10.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("Observation normalization epsilon must be positive.")
        if self.clip is not None and self.clip <= 0.0:
            raise ValueError("Observation normalization clip must be positive when provided.")


DEFAULT_OBSERVATION_NORMALIZATION_CFG = ObservationNormalizationCfg()


@dataclass(frozen=True)
class JointLimitActionBoundsCfg:
    """FastSAC symmetric action scaling derived from robot joint limits."""

    enabled: bool = True
    use_soft_limits: bool = False


DEFAULT_JOINT_LIMIT_ACTION_BOUNDS_CFG = JointLimitActionBoundsCfg()


@dataclass(frozen=True)
class ReplayUpdateCfg:
    """FastSAC collection warm-up and replay update schedule."""

    random_action_steps: int = 10
    minimum_replay_size: int = 8192
    batch_size: int = 8192
    num_updates: int = 8

    def __post_init__(self) -> None:
        if self.random_action_steps < 0:
            raise ValueError("random_action_steps must be non-negative.")
        if self.minimum_replay_size <= 0 or self.batch_size <= 0 or self.num_updates <= 0:
            raise ValueError("Replay sizes and num_updates must be positive.")
        if self.minimum_replay_size < self.batch_size:
            raise ValueError("minimum_replay_size must be at least batch_size.")


DEFAULT_REPLAY_UPDATE_CFG = ReplayUpdateCfg()


@dataclass(frozen=True)
class SequenceReplayCfg:
    """FastWMR R2D2-style replay window dimensions."""

    batch_size: int = 256
    burn_in_length: int = 16
    learning_length: int = 8
    require_episode_start: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.learning_length <= 0:
            raise ValueError("Sequence batch_size and learning_length must be positive.")
        if self.burn_in_length < 0:
            raise ValueError("burn_in_length must be non-negative.")


DEFAULT_SEQUENCE_REPLAY_CFG = SequenceReplayCfg()


@dataclass(frozen=True)
class FastWMRAlgoCfg:
    """MVP algorithm settings that do not belong to the IsaacLab task config.

    FastSAC uses ordinary transition sampling. FastWMR additionally samples raw
    boundary-safe sequences and reconstructs learning-time recurrent context
    with the current estimator. Hidden/cell tensors remain runtime values and
    are never persisted in replay.
    """

    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG
    actor: TanhGaussianActorCfg = DEFAULT_ACTOR_CFG
    critic: ScalarCriticCfg = DEFAULT_CRITIC_CFG
    observation_normalization: ObservationNormalizationCfg = DEFAULT_OBSERVATION_NORMALIZATION_CFG
    joint_limit_action_bounds: JointLimitActionBoundsCfg = DEFAULT_JOINT_LIMIT_ACTION_BOUNDS_CFG
    replay_update: ReplayUpdateCfg = DEFAULT_REPLAY_UPDATE_CFG
    sequence_replay: SequenceReplayCfg = DEFAULT_SEQUENCE_REPLAY_CFG
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
