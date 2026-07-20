# FastWMR G1 IsaacLab Task

This directory is the staging area for the FastWMR implementation described in
`pdf/FastWMR.pdf`, `pdf/directory.pdf`, and `pdf/roadmap.pdf`.

Implementation is progressing in independently verified layers:

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
- FastSAC core gate: connected Rough G1 collection to transition replay and the
  scalar SAC learner, including random-action warm-up, replay wraparound,
  reset-safe final observations, and finite actor/critic/temperature updates.

## Verification

Run lightweight contract and registry tests with ``pytest -q tests``. Run the
isolated Isaac Sim smoke gate with:

```bash
python tests/task_smoke.py --steps 1000 --num-envs 16
```

Run the compact FastSAC learner gate with:

```bash
python script/train.py --viz none --device cuda:0 --num-envs 16 --steps 12 \
  --replay-capacity 64 --random-action-steps 1 --minimum-replay-size 32 \
  --batch-size 32 --num-updates 1 --hidden-dim 64 --rough-debug
```
