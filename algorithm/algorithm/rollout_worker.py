"""Environment collection loop for FastWMR.

Planned flow:
1. Read current policy observation ``o_t`` and privileged target ``s_t``.
2. Update per-environment ``h_roll/c_roll`` with the current estimator.
3. Decode reconstructed state ``shat_t``.
4. Build ``x_t = concat(norm(o_t), stop_grad(shat_t))`` through the shared
   feature builder and sample an action from ``pi_theta(x_t)``.
5. Step the environment, append ``(o_t, s_t)`` to the estimator rollout cache,
   and append the detached control transition to ordinary SAC replay.
6. Reset hidden state only for terminated or truncated environments, then
   detach it at rollout chunk boundaries.
"""
