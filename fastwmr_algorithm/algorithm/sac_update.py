"""FastSAC update over the shared detached FastWMR control feature.

Both actor and twin critics consume ``x_t`` produced by ``feature_builder``.
The standard transition mini-batch does not contain recurrent hidden state.

Planned losses:
- critic target using average target Q and entropy term
- twin critic loss averaged across the transition batch
- actor loss using average Q1/Q2 and reparameterized actions
- entropy temperature auto-tuning
- target critic soft update

Ground-truth privileged ``s_t`` must never enter this module. If actor/critic
gradients reach the estimator, the WMR gradient-cutoff design is broken.
"""
