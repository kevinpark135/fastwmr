"""Tests for the FastWMR continuous and discrete decoder heads."""

import pytest
import torch
import torch.nn.functional as F

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.config import (
    DEFAULT_INTERFACE_CFG,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.networks import (
    DecoderOutput,
    WorldStateDecoder,
)


def test_default_decoder_matches_reconstruction_contract() -> None:
    cfg = DEFAULT_INTERFACE_CFG
    decoder = WorldStateDecoder(input_dim=256)
    encoded_history = torch.randn(8, 256)

    output = decoder(encoded_history)

    assert output.continuous.shape == (8, cfg.continuous_target_dim)
    assert output.discrete_logits.shape == (8, cfg.discrete_target_dim)
    assert output.reconstruction.shape == (8, cfg.reconstruction_target_dim)
    assert decoder.output_dim == cfg.reconstruction_target_dim
    assert torch.all(output.discrete_probabilities >= 0.0)
    assert torch.all(output.discrete_probabilities <= 1.0)
    torch.testing.assert_close(
        output.reconstruction[..., : cfg.continuous_target_dim],
        output.continuous,
    )
    torch.testing.assert_close(
        output.reconstruction[..., cfg.continuous_target_dim :],
        output.discrete_probabilities,
    )


def test_decoder_preserves_sequence_dimensions() -> None:
    decoder = WorldStateDecoder(
        input_dim=5,
        continuous_dim=3,
        discrete_dim=2,
        hidden_dim=7,
    )

    output = decoder(torch.randn(4, 6, 5))

    assert output.continuous.shape == (4, 6, 3)
    assert output.discrete_logits.shape == (4, 6, 2)
    assert output.reconstruction.shape == (4, 6, 5)


def test_mse_and_bce_losses_train_their_respective_heads() -> None:
    decoder = WorldStateDecoder(
        input_dim=5,
        continuous_dim=3,
        discrete_dim=2,
        hidden_dim=7,
    )
    encoded_history = torch.randn(4, 6, 5, requires_grad=True)
    continuous_target = torch.randn(4, 6, 3)
    discrete_target = torch.randint(0, 2, (4, 6, 2), dtype=torch.float32)

    output = decoder(encoded_history)
    continuous_loss = F.mse_loss(output.continuous, continuous_target)
    discrete_loss = F.binary_cross_entropy_with_logits(
        output.discrete_logits,
        discrete_target,
    )
    (continuous_loss + discrete_loss).backward()

    assert encoded_history.grad is not None
    assert torch.isfinite(encoded_history.grad).all()
    assert all(parameter.grad is not None for parameter in decoder.continuous_head.parameters())
    assert all(parameter.grad is not None for parameter in decoder.discrete_head.parameters())


def test_decoder_heads_do_not_share_parameters() -> None:
    decoder = WorldStateDecoder(input_dim=5, continuous_dim=3, discrete_dim=2, hidden_dim=7)

    continuous_parameters = {id(parameter) for parameter in decoder.continuous_head.parameters()}
    discrete_parameters = {id(parameter) for parameter in decoder.discrete_head.parameters()}

    assert continuous_parameters.isdisjoint(discrete_parameters)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 0}, "input_dim"),
        ({"input_dim": 3, "continuous_dim": 0}, "continuous_dim"),
        ({"input_dim": 3, "discrete_dim": 0}, "discrete_dim"),
        ({"input_dim": 3, "hidden_dim": 0}, "hidden_dim"),
    ],
)
def test_invalid_decoder_dimensions_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldStateDecoder(**kwargs)


def test_invalid_decoder_input_is_rejected() -> None:
    decoder = WorldStateDecoder(input_dim=5, continuous_dim=3, discrete_dim=2)

    with pytest.raises(TypeError, match="floating dtype"):
        decoder(torch.ones(2, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="encoded_history must have shape"):
        decoder(torch.randn(5))
    with pytest.raises(ValueError, match="encoded_history must have shape"):
        decoder(torch.randn(2, 4))


def test_decoder_output_rejects_mismatched_leading_shapes() -> None:
    with pytest.raises(ValueError, match="batch shapes must match"):
        DecoderOutput(
            continuous=torch.randn(2, 3),
            discrete_logits=torch.randn(3, 2),
        )
