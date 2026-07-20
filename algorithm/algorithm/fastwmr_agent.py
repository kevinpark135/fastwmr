"""Top-level FastWMR learner and optimizer coordinator.

Planned update order:
1. update the estimator from its current rollout cache
2. sample ordinary transitions from SAC replay
3. critic update over stored detached control features
4. actor update
5. entropy-temperature update
6. target-critic soft update

Estimator, critic, actor, and entropy temperature have separate optimizer state.
This is the main place to keep ablation switches explicit and logged.
"""
