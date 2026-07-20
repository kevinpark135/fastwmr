"""Recent rollout cache used only for online estimator reconstruction updates.

This cache keeps ordered current-rollout ``(o_t, s_t)`` samples and episode
boundaries long enough to train the estimator. It is separate from SAC replay,
is not a long-lived off-policy dataset, and does not store recurrent state.
"""
