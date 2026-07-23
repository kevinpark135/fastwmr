# FastWMR

FastWMR is an Isaac Lab research implementation for proprioception-only velocity
locomotion with the Unitree G1 29-DoF humanoid. It extends FastSAC with a
recurrent world-state estimator trained to reconstruct simulator-only targets
from observation history.

The actor never receives privileged state directly. Privileged quantities are
used only as estimator reconstruction targets during training; control uses the
deployable proprioceptive observation and the estimator's reconstruction.

## Features

- FastWMR and FastSAC baseline tasks on the same G1 rough-terrain environment.
- A canonical 96D proprioceptive policy observation and 13D privileged world-state
  reconstruction target.
- Boundary-safe sequence replay with episode-aware sampling, recurrent burn-in,
  and a separate estimator rollout cache.
- A two-timescale FastWMR v2 learner with stored SAC features, a low-frequency
  online estimator, an EMA control estimator, reconstruction gating, and
  confidence-aware feature freshness without transition rejection. Strict-current
  FastWMR v1 remains available as a reference mode.
- Recurrent history encoder with continuous regression and discrete contact
  decoding heads.
- Tanh-Gaussian actor, automatic entropy temperature optimization, and twin C51
  distributional critics, with scalar twin critics available for ablation.
- Observation normalization and joint-limit-aware policy action bounds.
- Recorded domain randomization for friction, payload, and external force/torque,
  together with terrain, corruption, and penalty curricula.
- Checkpoint/resume support, JSONL metrics, deterministic checkpoint evaluation,
  multi-seed robustness suites, and reconstruction-correlation reporting.
- Experiment controls for reconstruction-only features, estimator freezing,
  gradient cutoff, recent replay, symmetry augmentation, critic type, and reward
  curriculum.

## Repository Structure

```text
fastwmr/
├── algorithm/
│   ├── algorithm/       # SAC/FastWMR updates, rollout workers, and checkpoints
│   ├── buffers/         # Transition replay and estimator rollout cache
│   ├── networks/        # Actor, critics, recurrent encoder, and decoder
│   └── utils/           # Normalization, bounds, logging, evaluation, and adapters
├── agents/              # Isaac Lab PPO reference configurations
├── script/              # Training, playback, and evaluation entry points
├── tests/               # Unit, integration, contract, and task smoke tests
├── fastwmr_env_cfg.py   # FastWMR G1 environment configuration
├── baseline_env_cfg.py  # Shared-environment FastSAC baseline
├── observations.py      # Policy observations and privileged estimator targets
├── rewards.py           # Minimal locomotion reward specification
├── randomization.py     # Sample, apply, and record domain randomization
└── curriculum.py        # Terrain and penalty schedules
```

## Quick Start

### Setup

Install Isaac Lab and activate its Python environment first. This repository is
an Isaac Lab task configuration package and should be checked out at its package
location:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab

git clone https://github.com/kevinpark135/fastwmr.git \
  "$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fastwmr"

cd "$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fastwmr"
```

The package registers four Gymnasium task IDs:

| Purpose | Task ID |
| --- | --- |
| FastWMR training | `Isaac-Velocity-G1-FastWMR-v0` |
| FastSAC baseline training | `Isaac-Velocity-G1-FastSAC-Baseline-v0` |
| FastWMR evaluation | `Isaac-Velocity-G1-FastWMR-Play-v0` |
| FastSAC baseline evaluation | `Isaac-Velocity-G1-FastSAC-Baseline-Play-v0` |

### Training

Train FastWMR with the default C51 critic:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastWMR-v0 \
  --run-name g1_fastwmr \
  --viz none
```

The default `--fastwmr-version v2` runs eight reconstruction-replay SAC updates
per estimator trigger, performs one sequence estimator update, synchronizes the
EMA control estimator, and rebuilds recurrent runtime state once. Replay keeps
raw observations and ungated normalized reconstructions; each learner minibatch
is rebuilt with the current observation normalizer, reconstruction gate, and a
freshness confidence. Stale estimator outputs are masked without removing their
raw transitions from SAC replay. The main controls are
`--estimator-update-interval`, `--estimator-updates-per-trigger`,
`--control-estimator-tau`, `--max-estimator-feature-age`,
`--fresh-reconstruction-fraction`, and the reconstruction-gate options.

Run the strict-current v1 reference path with:

```bash
python script/train.py \
  --fastwmr-version v1 \
  --run-name g1_fastwmr_v1 \
  --viz none
```

Train the FastSAC baseline under the shared environment, reward, action, and
randomization configuration:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastSAC-Baseline-v0 \
  --run-name g1_fastsac_baseline \
  --viz none
```

#### FastWMR-specific options

These options configure the estimator and sequence learner used by the FastWMR
task. They are separate from the shared FastSAC replay, critic, and environment
options.

| Option | Default | Purpose |
| --- | --- | --- |
| `--fastwmr-version {v1,v2}` | `v2` | Select the strict-current reference learner or the two-timescale learner. |
| `--estimator-hidden-dim` | `256` | Set the recurrent estimator and decoder hidden width. |
| `--estimator-num-layers` | `1` | Set the recurrent encoder depth. |
| `--estimator-learning-rate` | `3e-4` | Set the estimator optimizer learning rate. |
| `--estimator-weight-decay` | `1e-3` | Set estimator Adam weight decay. |
| `--estimator-cache-steps` | `64` | Set the per-environment recurrent rollout-cache length. |
| `--sequence-batch-size` | `256` | Set the number of replay sequences sampled per estimator update. |
| `--burn-in-length` | `32` | Set recurrent warm-up steps excluded from estimator loss. |
| `--learning-length` | `8` | Set sequence steps included in estimator loss. |
| `--require-episode-start` | disabled | Restrict sampled sequences to episode starts. |
| `--episode-start-fraction` | `0.25` | Guarantee this minimum exact-context fraction in estimator batches. |
| `--recent-replay-horizon` | `200000` | Restrict estimator sequence sampling to recent transitions. |

FastWMR v2 adds the following two-timescale controls:

| Option | Default | Purpose |
| --- | --- | --- |
| `--estimator-update-interval` | `8` | Trigger estimator learning after this many SAC updates. |
| `--estimator-updates-per-trigger` | `1` | Run this many sequence updates per estimator trigger. |
| `--max-estimator-feature-age` | auto | Set the estimator-version age that receives non-zero reconstruction confidence. |
| `--disable-feature-age-filter` | disabled | Treat every stored reconstruction as fresh; intended for ablation. |
| `--fresh-reconstruction-fraction` | `0.5` | Reserve this fresh-feature fraction when the reconstruction gate is fully open. |
| `--stored-feature-replay-horizon` | unset | Optionally restrict SAC transition replay; unset uses the full buffer. |
| `--control-estimator-tau` | `0.005` | Set the EMA update rate for the control estimator. |
| `--reconstruction-gate-start-updates` | `0` | Require at least this many estimator updates before quality can open the gate. |
| `--reconstruction-gate-warmup-updates` | `200` | Ramp the gate after its quality checks pass. |
| `--reconstruction-gate-quality-threshold` | `0.45` | Open the gate below this normalized validation-loss EMA. |
| `--reconstruction-gate-close-threshold` | `0.55` | Close the gate above this EMA, leaving hysteresis around the open threshold. |
| `--reconstruction-gate-quality-ema-decay` | `0.9` | Smooth independently sampled gate-validation losses. |
| `--reconstruction-gate-quality-patience` | `3` | Require this many consecutive EMA threshold passes. |
| `--reconstruction-gate-validation-interval` | `8` | Validate gate quality every N estimator attempts in every gate state. |

FastWMR control features contain normalized proprioception, 13 reconstructed
world-state values, and one reconstruction-confidence value. The confidence is
the product of the global quality gate and the per-transition freshness mask.
When confidence is zero, only the reconstruction is masked; the transition's
proprioception, action, reward, and bootstrap state remain in SAC replay.

The actor and critic use separate width controls: `--actor-hidden-dim 512` and
`--critic-hidden-dim 768`. The legacy `--hidden-dim` option still overrides
both widths for reproducing older runs.

FastWMR diagnostics and ablations are controlled with
`--validation-interval`, `--initial-validation-updates`,
`--disable-gradient-boundary-checks`, `--control-feature-mode`,
`--freeze-estimator`, and `--use-symmetry`. The
`--disable-gradient-cutoff` ablation is available only with
`--fastwmr-version v1`.

Use `--num-envs`, `--steps`, `--wallclock-limit-s`, and the replay/update options
to set the collection and learner budgets. Run `python script/train.py --help`
for the complete configuration surface.

Each run writes its resolved configuration, metrics, and checkpoints beneath:

```text
<IsaacLab>/logs/fastwmr/<run-name>/
├── config_snapshot.json
├── metrics.jsonl
├── tensorboard/
│   └── events.out.tfevents...
└── checkpoints/
```

The default checkpoint interval is 50 environment iterations. Override it with
`--checkpoint-interval`, or use `--log-dir` to replace the IsaacLab-level log
root. Monitor all runs with:

```bash
tensorboard --logdir ~/IsaacLab/logs/fastwmr --port 6006
```

### Resume Training

Resume FastWMR from a versioned checkpoint while restoring model, optimizer,
normalization, estimator, and learner-counter state:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastWMR-v0 \
  --resume ~/IsaacLab/logs/fastwmr/g1_fastwmr/checkpoints/step_000001000.pt \
  --steps 1000 \
  --viz none
```

Replay contents, the estimator rollout cache, and recurrent runtime state are
ephemeral and restart empty after resume.

### Evaluation

Run a deterministic nominal evaluation for one checkpoint:

```bash
python script/play.py \
  --checkpoint ~/IsaacLab/logs/fastwmr/g1_fastwmr/checkpoints/final_step_000001000.pt \
  --condition nominal_rough \
  --viz none
```

The evaluator infers FastWMR or FastSAC mode from the checkpoint. Available
conditions are `nominal_rough`, `friction_low`, `friction_high`, `payload_heavy`,
`strong_push`, `observation_noise`, and `observation_masking`.

Run the reproducible robustness matrix with at least three independently trained
checkpoints per variant. Repeat `--checkpoint` and `--variant` as a pair; checkpoints
that belong to the same comparison group use the same variant name:

```bash
python script/evaluate_suite.py \
  --checkpoint ~/IsaacLab/logs/fastwmr/g1_fastwmr_seed42/checkpoints/final_step_000001000.pt \
  --variant primary \
  --checkpoint ~/IsaacLab/logs/fastwmr/g1_fastwmr_seed43/checkpoints/final_step_000001000.pt \
  --variant primary \
  --checkpoint ~/IsaacLab/logs/fastwmr/g1_fastwmr_seed44/checkpoints/final_step_000001000.pt \
  --variant primary \
  --evaluation-seed 100 101 102
```

The training seed is read from each checkpoint. Individual records are written to
`<condition>/train_seed_<N>/eval_seed_<N>.json`, so checkpoints from the same
variant cannot overwrite one another. Metrics are averaged across evaluation seeds
within each training seed first; the reported mean and standard deviation are then
computed across independent training seeds. Aggregated JSON, CSV, and Markdown
summaries are written under `evaluations/suite/` by default.
Evaluation records include return, survival/fall rate, linear and yaw tracking
RMSE, push recovery, action-rate RMS, mechanical power, and physical estimator
RMSE/correlation for every reconstructed field.

### Representation Diagnostics

The observation normalizer can be frozen without disabling normalization:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastWMR-v0 \
  --num-envs 64 \
  --steps 1000 \
  --seed 42 \
  --max-estimator-feature-age 256 \
  --normalizer-freeze-iteration 128 \
  --run-name fastwmr_v2_normalizer_freeze \
  --viz none
```

At iteration 128, statistics stop changing while rollout and learner inputs
continue to use the frozen transform. Compare that run against a v2 controller
whose reconstruction remains gated off:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastWMR-v0 \
  --num-envs 64 \
  --steps 1000 \
  --seed 42 \
  --max-estimator-feature-age 256 \
  --reconstruction-gate-start-updates 100000 \
  --run-name fastwmr_v2_gate0 \
  --viz none
```

Then run the strict-current reference, which rebuilds learning features with
the current estimator instead of using stored v2 features:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastWMR-v0 \
  --fastwmr-version v1 \
  --num-envs 64 \
  --steps 1000 \
  --seed 42 \
  --run-name fastwmr_v1_strict_current \
  --viz none
```

Run these experiments sequentially to avoid GPU contention. Pin
`episode/return_mean`, `sac/policy_entropy`, `normalizer/frozen`,
`normalizer/samples_seen`, `v2/reconstruction_gate`,
`v2/gate_quality_ema`, `v2/gate_state`, `sac/q_gap_mean`,
`sac/c51_lower_endpoint_mass`, `sac/c51_upper_endpoint_mass`,
`sac/c51_distribution_entropy`, `sac/policy_action_saturation_fraction`, and
`estimator/context_exact_fraction` in TensorBoard. For confidence-aware replay,
also pin `replay/full_transition_count`, `replay/fresh_reconstruction_count`,
`replay/stale_reconstruction_count`, `replay/sampled_fresh_fraction`, and
`representation/confidence_mean`.

### Ablations

The main FastWMR ablations are exposed directly by the training entry point:

```bash
# Reconstructed state only as the actor/critic control feature
python script/train.py --control-feature-mode reconstruction_only --viz none

# Freeze the estimator while training the controller
python script/train.py --freeze-estimator --viz none

# Remove the estimator-to-controller gradient cutoff in strict v1
python script/train.py --fastwmr-version v1 --disable-gradient-cutoff --viz none

# Treat every stored v2 reconstruction as fresh
python script/train.py --disable-feature-age-filter --viz none

# Restrict sequence sampling to recent replay and enable symmetry augmentation
python script/train.py --recent-replay-horizon 131072 --use-symmetry --viz none
```

## References and Attribution

### Research papers

- Younggyo Seo, Carmelo Sferrazza, Juyue Chen, Guanya Shi, Rocky Duan, and
  Pieter Abbeel. [Learning Sim-to-Real Humanoid Locomotion in 15 Minutes
  (FastSAC)](pdf/FastSAC.pdf), 2025. This is the primary reference for the
  FastSAC actor, critic, replay, update, normalization, action-scaling, and
  minimal-reward design.
- Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin,
  and Pieter Abbeel. [FastTD3: Simple, Fast, and Capable Reinforcement Learning
  for Humanoid Control](pdf/FastTD3.pdf), 2025. This informs the high-throughput
  off-policy humanoid training setup and distributional critic design.
- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.
  [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
  with a Stochastic Actor](pdf/SAC.pdf), 2018. This provides the underlying SAC
  objective, entropy regularization, and temperature optimization.
- Wandong Sun, Long Chen, Yongbo Su, Baoshi Cao, Yang Liu, and Zongwu Xie.
  [Learning Humanoid Locomotion with World Model Reconstruction
  (WMR)](pdf/WMR.pdf), 2025. This is the primary reference for explicit world
  reconstruction, privileged estimator targets, recurrent state estimation,
  and the detached estimator-policy training boundary.

### Software reference

The FastSAC actor/critic topology, observation normalization, joint-limit-aware
action scaling, C51 update, and minimal locomotion reward implementation were
also informed by the official
[Holosoma](https://github.com/amazon-far/holosoma) repository. In particular,
the categorical twin critic follows Holosoma's independent per-head target
projection and averages the two expected Q-values only for the actor objective.
This repository is an independent FastWMR implementation and does not vendor
Holosoma source code.

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
