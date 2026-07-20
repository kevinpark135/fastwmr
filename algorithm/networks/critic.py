"""FastWMR twin critic networks Q_phi1 and Q_phi2.

The primary critics receive ``(x_t, a_t)``, where ``x_t`` is exactly the same
detached control feature received by the actor. Ground-truth privileged target
``s_t`` is never a primary-critic input.

Planned details:
- twin critics plus target critics
- average-Q targets instead of clipped double-Q min by default
- LayerNorm for humanoid-task stability
- C51 distributional critic head for the full FastSAC version

For an initial prototype, a scalar-Q implementation may be easier, but it should
be labeled as a SAC-style FastWMR prototype rather than a full FastSAC clone.
"""
