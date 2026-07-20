"""FastWMR actor network pi_theta.

The actor consumes the shared control feature produced only by
``utils.feature_builder``. Its default input is ``concat(norm(o), detach(shat))``;
reconstruction-only input is an explicit ablation. It should implement the
FastSAC stochastic tanh-Gaussian policy with joint-limit-aware action bounds.

Planned details:
- LayerNorm in the policy MLP.
- Pre-tanh log standard deviation capped according to the FastSAC recipe.
- Reparameterized sampling for SAC actor updates.
- No gradient flow into the estimator input.
"""
