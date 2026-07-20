"""Online estimator reconstruction update from the recent rollout cache.

The MVP starts with one-step/chunked current-rollout updates and separates this
data from SAC replay. The decoder predicts the fixed continuous and discrete
target slices, optimized with MSE and BCE respectively. Burn-in and exact
replay-time current-estimator re-inference are later research extensions.
"""
