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
- FastSAC normalization: added checkpointable running observation statistics;
  replay remains raw while rollout actions and learner batches share the same
  current normalized representation.
- FastSAC action bounds: derive symmetric per-joint tanh scales from resolved G1
  limits, default positions, and IsaacLab action scaling so zero action remains
  the configured default pose.
- FastSAC C51 critic: added independent online/target categorical twin critics,
  entropy-aware Bellman projection, cross-entropy critic updates, and mean-Q
  actor updates. The default support is 101 atoms over ``[-20, 20]``.
- FastWMR DR records: added startup-managed per-environment friction, payload,
  and 6D external-wrench buffers with fixed shapes, partial-env reset support,
  and privileged-observation wiring.
- FastWMR exact DR path: replaced opaque built-in samples with environment-level
  sample/apply/record events for friction, additive pelvis payload, and episodic
  body-frame pelvis wrench; smoke tests compare records against PhysX tensors.

## Verification

Run lightweight contract and registry tests with ``pytest -q tests``. Run the
isolated Isaac Sim smoke gate with:

```bash
python tests/task_smoke.py --steps 1000 --num-envs 16
```

Add ``--full-dr`` to retain external wrench randomization and verify every DR
record directly against the applied PhysX material, mass, and wrench tensors.

Run the compact FastSAC learner gate with:

```bash
python script/train.py --viz none --device cuda:0 --num-envs 16 --steps 12 \
  --replay-capacity 64 --random-action-steps 1 --minimum-replay-size 32 \
  --batch-size 32 --num-updates 1 --hidden-dim 64 --rough-debug
```

The learner uses the C51 critic by default. Pass ``--critic-type scalar`` for
the scalar FastSAC ablation.

## References and Attribution

The FastSAC actor/critic topology, observation normalization, joint-limit-aware
action scaling, and C51 update in this repository were informed by the official
[Holosoma](https://github.com/amazon-far/holosoma) implementation. In
particular, the categorical twin critic follows Holosoma's independent
per-head target projection and averages the two expected Q-values only for the
actor objective. This repository is an independent FastWMR implementation and
does not vendor Holosoma source code.

Please cite Holosoma using the metadata in its official
[CITATION.cff](https://github.com/amazon-far/holosoma/blob/main/CITATION.cff):

```bibtex
@software{holosoma,
  author = {{Amazon FAR} and Pieter Abbeel and Juyue Chen and Rocky Duan and
            Alejandro Escontrela and Manan Gandhi and Samuel Gundry and
            Xiaoyu Huang and Angjoo Kanazawa and Tomasz Lewicki and Jiaman Li and
            Karen Liu and Clay Rosenthal and Younggyo Seo and Carlo Sferrazza and
            Guanya Shi and Linda Shih and Jonathan Tseng and Zhen Wu and
            Lujie Yang and Brent Yi and Yuanhang Zhang},
  title  = {Holosoma},
  url    = {https://github.com/amazon-far/holosoma}
}
```
