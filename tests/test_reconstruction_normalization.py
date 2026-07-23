"""Tests for the fixed FastWMR reconstruction learning space."""

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    denormalize_reconstruction,
    normalize_reconstruction,
)


def test_reconstruction_normalization_matches_physical_scales_and_round_trips() -> None:
    physical = torch.tensor(
        [
            2.0,
            -4.0,
            1.0,
            1.5,
            10.0,
            50.0,
            -100.0,
            25.0,
            10.0,
            -20.0,
            5.0,
            1.0,
            0.0,
        ]
    )

    normalized = normalize_reconstruction(physical)

    torch.testing.assert_close(
        normalized,
        torch.tensor(
            [
                1.0,
                -2.0,
                0.5,
                1.0,
                2.0,
                1.0,
                -2.0,
                0.5,
                1.0,
                -2.0,
                0.5,
                1.0,
                0.0,
            ]
        ),
    )
    torch.testing.assert_close(denormalize_reconstruction(normalized), physical)
