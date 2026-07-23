"""Canonical physical-to-learning transforms for FastWMR reconstruction."""

from __future__ import annotations

import torch

from ..config import (
    DEFAULT_INTERFACE_CFG,
    DEFAULT_RECONSTRUCTION_NORMALIZATION_CFG,
    FastWMRInterfaceCfg,
    ReconstructionNormalizationCfg,
)


def reconstruction_center_and_scale(
    reference: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    cfg: ReconstructionNormalizationCfg = DEFAULT_RECONSTRUCTION_NORMALIZATION_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return broadcastable center and scale tensors in reconstruction order."""

    if reference.shape[-1] != interface.reconstruction_target_dim:
        raise ValueError(
            "Reconstruction reference width does not match the interface contract."
        )
    layout = interface.reconstruction_layout
    expected_widths = {
        "base_lin_vel": 3,
        "friction": 1,
        "payload_mass": 1,
        "push_force_torque": 6,
        "foot_contacts": 2,
    }
    for name, width in expected_widths.items():
        field = next((item for item in layout.fields if item.name == name), None)
        if field is None or field.width != width:
            raise ValueError(
                f"Reconstruction normalization requires {name!r} with width {width}."
            )

    center = reference.new_zeros(interface.reconstruction_target_dim)
    scale = reference.new_ones(interface.reconstruction_target_dim)
    center[layout.field_slice("friction")] = cfg.friction_center
    scale[layout.field_slice("base_lin_vel")] = cfg.base_velocity_scale
    scale[layout.field_slice("friction")] = cfg.friction_scale
    scale[layout.field_slice("payload_mass")] = cfg.payload_mass_scale
    wrench = layout.field_slice("push_force_torque")
    scale[wrench.start : wrench.start + 3] = cfg.force_scale
    scale[wrench.start + 3 : wrench.stop] = cfg.torque_scale
    return center, scale


def normalize_reconstruction(
    physical_state: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    cfg: ReconstructionNormalizationCfg = DEFAULT_RECONSTRUCTION_NORMALIZATION_CFG,
) -> torch.Tensor:
    """Map physical privileged targets into the estimator/control feature space."""

    center, scale = reconstruction_center_and_scale(
        physical_state,
        interface=interface,
        cfg=cfg,
    )
    return (physical_state - center) / scale


def denormalize_reconstruction(
    normalized_state: torch.Tensor,
    *,
    interface: FastWMRInterfaceCfg = DEFAULT_INTERFACE_CFG,
    cfg: ReconstructionNormalizationCfg = DEFAULT_RECONSTRUCTION_NORMALIZATION_CFG,
) -> torch.Tensor:
    """Map estimator output back to physical units for diagnostics and evaluation."""

    center, scale = reconstruction_center_and_scale(
        normalized_state,
        interface=interface,
        cfg=cfg,
    )
    return normalized_state * scale + center
