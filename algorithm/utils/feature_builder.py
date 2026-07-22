"""Single routing point for actor and primary-critic control features."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..config import ControlFeatureMode, DEFAULT_INTERFACE_CFG, FastWMRInterfaceCfg


ObservationNormalizer = Callable[[torch.Tensor], torch.Tensor]


def _check_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension.")
    if tensor.shape[-1] != expected:
        raise ValueError(f"{name} must end in dimension {expected}, got shape {tuple(tensor.shape)}.")


def build_control_feature(
    policy_observation: torch.Tensor,
    reconstructed_state: torch.Tensor,
    *,
    cfg: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    normalizer: ObservationNormalizer | None = None,
    detach_reconstruction: bool = True,
    reconstruction_gate: float = 1.0,
) -> torch.Tensor:
    """Build ``x_t`` while enforcing the estimator gradient cutoff.

    The default is ``concat(normalize(o_t), stop_grad(shat_t))``. The privileged
    target ``s_t`` is intentionally not accepted by this API, preventing it from
    leaking into deployable actor or primary-critic features.
    """

    _check_last_dim(policy_observation, cfg.policy_observation_dim, "policy_observation")
    _check_last_dim(reconstructed_state, cfg.reconstruction_target_dim, "reconstructed_state")
    if not 0.0 <= reconstruction_gate <= 1.0:
        raise ValueError("reconstruction_gate must be in [0, 1].")
    if policy_observation.shape[:-1] != reconstructed_state.shape[:-1]:
        raise ValueError(
            "policy_observation and reconstructed_state batch shapes must match, got "
            f"{policy_observation.shape[:-1]} and {reconstructed_state.shape[:-1]}."
        )

    normalized_observation = normalizer(policy_observation) if normalizer is not None else policy_observation
    if normalized_observation.shape != policy_observation.shape:
        raise ValueError("The observation normalizer must preserve tensor shape.")

    routed_reconstruction = (
        reconstructed_state.detach() if detach_reconstruction else reconstructed_state
    )
    routed_reconstruction = reconstruction_gate * routed_reconstruction
    if cfg.control_feature_mode is ControlFeatureMode.RECONSTRUCTION_ONLY:
        feature = routed_reconstruction
    else:
        feature = torch.cat((normalized_observation, routed_reconstruction), dim=-1)

    _check_last_dim(feature, cfg.control_feature_dim, "control_feature")
    return feature


def build_critic_input(
    control_feature: torch.Tensor,
    action: torch.Tensor,
    *,
    cfg: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
) -> torch.Tensor:
    """Validate and concatenate the primary critic input ``(x_t, a_t)``."""

    _check_last_dim(control_feature, cfg.control_feature_dim, "control_feature")
    _check_last_dim(action, cfg.action_dim, "action")
    if control_feature.shape[:-1] != action.shape[:-1]:
        raise ValueError(
            f"control_feature and action batch shapes differ: {control_feature.shape[:-1]} vs {action.shape[:-1]}."
        )
    critic_input = torch.cat((control_feature, action), dim=-1)
    _check_last_dim(critic_input, cfg.critic_input_dim, "critic_input")
    return critic_input
