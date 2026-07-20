# FastWMR G1 IsaacLab Task

This directory is the staging area for the FastWMR implementation described in
`pdf/FastWMR.pdf`, `pdf/directory.pdf`, and `pdf/roadmap.pdf`.

The current files are mostly structure and ownership markers. Implementation
will be added in layers:

1. Environment/task layer: observations, rewards, randomization, curriculum, and
   baseline task registration.
2. Algorithm layer: transition and boundary-safe sequence replay, recurrent
   world-state estimator, decoder, actor, critic, and FastSAC update.
3. Script layer: training, evaluation, CLI overrides, logging, and ablations.

## Update Log

- Initial scaffold: created the FastWMR file tree and documented each file's
  intended responsibility without implementing training code.
- Interface contract: fixed the G1 29-DoF action layout, 96D policy observation,
  13D privileged target, done/bootstrap semantics, rollout recurrent-state
  shape, and detached 109D actor/critic control feature.
- Task smoke gate: registered FastWMR and policy-only FastSAC training/play
  tasks, enabled G1 contact reporters, and completed 1,000 finite random-action
  steps for both training tasks with 16 environments.

## Verification

Run lightweight contract and registry tests with ``pytest -q tests``. Run the
isolated Isaac Sim smoke gate with:

```bash
python tests/task_smoke.py --steps 1000 --num-envs 16
```
