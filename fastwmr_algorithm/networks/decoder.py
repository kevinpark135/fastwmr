"""World-state decoder D_psi for FastWMR.

The decoder maps encoder features or hidden states to reconstructed world-state
components.

Planned heads:
- continuous head trained with MSE
- discrete/contact head trained with BCE
- optional latent regularization term used by the estimator update

The decoder should serve reconstruction only. Actor and critic losses must not
backpropagate through it.
"""

