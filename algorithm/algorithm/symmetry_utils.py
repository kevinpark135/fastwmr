"""Left/right sequence symmetry for the 29-DoF G1 FastWMR contract.

FastWMR mirrors raw recurrent inputs and privileged targets before estimator
inference. It never mirrors only a cached reconstruction, which would skip the
LSTM's temporal dynamics.
"""

from __future__ import annotations

from dataclasses import fields

import torch

from ..buffers import SequenceReplayBatch
from ..config import ControlFeatureMode, DEFAULT_INTERFACE_CFG, FastWMRInterfaceCfg


G1_ACTION_MIRROR_PERMUTATION = (
    1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13,
    16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27,
)
G1_ACTION_MIRROR_SIGNS = (
    1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
)


def _signed_permutation(
    tensor: torch.Tensor,
    permutation: tuple[int, ...],
    signs: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    if tensor.shape[-1] != len(permutation) or len(permutation) != len(signs):
        raise ValueError(f"{name} must end in dimension {len(permutation)}.")
    indices = torch.tensor(permutation, device=tensor.device, dtype=torch.long)
    sign_tensor = tensor.new_tensor(signs)
    return tensor.index_select(-1, indices) * sign_tensor


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """Mirror normalized G1 joint actions in resolved 29-DoF order."""

    return _signed_permutation(
        action,
        G1_ACTION_MIRROR_PERMUTATION,
        G1_ACTION_MIRROR_SIGNS,
        "action",
    )


def mirror_policy_observation(
    observation: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
) -> torch.Tensor:
    """Mirror one policy observation without changing its canonical layout."""

    if observation.shape[-1] != interface.policy_observation_dim:
        raise ValueError(
            f"observation must end in dimension {interface.policy_observation_dim}."
        )
    layout = interface.policy_observation_layout
    mirrored = observation.clone()
    mirrored[..., layout.field_slice("base_ang_vel")] *= observation.new_tensor((-1, 1, -1))
    mirrored[..., layout.field_slice("projected_gravity")] *= observation.new_tensor((1, -1, 1))
    mirrored[..., layout.field_slice("velocity_command")] *= observation.new_tensor((1, -1, -1))
    for name in ("joint_pos", "joint_vel", "previous_action"):
        field_slice = layout.field_slice(name)
        mirrored[..., field_slice] = mirror_action(observation[..., field_slice])
    return mirrored


def mirror_reconstruction_target(
    target: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
) -> torch.Tensor:
    """Mirror continuous world-state fields and swap foot contacts."""

    if target.shape[-1] != interface.reconstruction_target_dim:
        raise ValueError(f"target must end in dimension {interface.reconstruction_target_dim}.")
    layout = interface.reconstruction_layout
    mirrored = target.clone()
    mirrored[..., layout.field_slice("base_lin_vel")] *= target.new_tensor((1, -1, 1))
    push_slice = layout.field_slice("push_force_torque")
    mirrored[..., push_slice] *= target.new_tensor((1, -1, 1, -1, 1, -1))
    contacts = target[..., layout.field_slice("foot_contacts")]
    mirrored[..., layout.field_slice("foot_contacts")] = contacts.flip(-1)
    return mirrored


def mirror_control_feature(
    feature: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
) -> torch.Tensor:
    """Mirror a stored diagnostic feature while preserving feature mode."""

    if feature.shape[-1] != interface.control_feature_dim:
        raise ValueError(f"feature must end in dimension {interface.control_feature_dim}.")
    if interface.control_feature_mode is ControlFeatureMode.RECONSTRUCTION_ONLY:
        return mirror_reconstruction_target(feature, interface=interface)
    observation = feature[..., : interface.policy_observation_dim]
    reconstruction = feature[..., interface.policy_observation_dim :]
    return torch.cat(
        (
            mirror_policy_observation(observation, interface=interface),
            mirror_reconstruction_target(reconstruction, interface=interface),
        ),
        dim=-1,
    )


def mirror_sequence_batch(
    sequence: SequenceReplayBatch,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
) -> SequenceReplayBatch:
    """Mirror every physical field of one recurrent replay batch."""

    unchanged = {
        field.name: getattr(sequence, field.name).clone()
        for field in fields(sequence)
        if field.name
        not in {
            "observations",
            "privileged_states",
            "stored_control_features",
            "actions",
            "burn_in_length",
            "learning_length",
        }
    }
    return SequenceReplayBatch(
        observations=mirror_policy_observation(sequence.observations, interface=interface),
        privileged_states=mirror_reconstruction_target(
            sequence.privileged_states,
            interface=interface,
        ),
        stored_control_features=mirror_control_feature(
            sequence.stored_control_features,
            interface=interface,
        ),
        actions=mirror_action(sequence.actions),
        **unchanged,
        burn_in_length=sequence.burn_in_length,
        learning_length=sequence.learning_length,
    )


def augment_sequence_batch(
    sequence: SequenceReplayBatch,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
) -> SequenceReplayBatch:
    """Append a mirrored raw sequence batch for a separate recurrent forward."""

    mirrored = mirror_sequence_batch(sequence, interface=interface)
    tensor_names = (
        "observations",
        "privileged_states",
        "stored_control_features",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "episode_ids",
        "env_ids",
        "timesteps",
        "reset_boundaries",
        "insertion_ids",
    )
    return SequenceReplayBatch(
        **{
            name: torch.cat((getattr(sequence, name), getattr(mirrored, name)), dim=0)
            for name in tensor_names
        },
        burn_in_length=sequence.burn_in_length,
        learning_length=sequence.learning_length,
    )
