"""Optional recent-rollout cache for online recurrent context rebuilding.

This cache keeps ordered current-rollout ``(o_t, s_t)`` samples and episode
boundaries. Long-lived estimator training sequences are sampled from transition
replay; this cache is not an off-policy dataset and does not store recurrent
state.
"""
