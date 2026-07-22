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
  --device cuda:0 \
  --viz none
```

Train the FastSAC baseline under the shared environment, reward, action, and
randomization configuration:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastSAC-Baseline-v0 \
  --run-name g1_fastsac_baseline \
  --device cuda:0 \
  --viz none
```

Use `--num-envs`, `--steps`, `--wallclock-limit-s`, and the replay/update options
to set the collection and learner budgets. Run `python script/train.py --help`
for the complete configuration surface.

Each run writes its resolved configuration, metrics, and checkpoints beneath:

```text
logs/fastwmr/<run-name>/
├── config_snapshot.json
├── metrics.jsonl
└── checkpoints/
```

### Resume Training

Resume FastWMR from a versioned checkpoint while restoring model, optimizer,
normalization, estimator, and learner-counter state:

```bash
python script/train.py \
  --task Isaac-Velocity-G1-FastWMR-v0 \
  --resume logs/fastwmr/g1_fastwmr/checkpoints/step_000001000.pt \
  --steps 1000 \
  --device cuda:0 \
  --viz none
```

Replay contents, the estimator rollout cache, and recurrent runtime state are
ephemeral and restart empty after resume.

### Evaluation

Run a deterministic nominal evaluation for one checkpoint:

```bash
python script/play.py \
  --checkpoint logs/fastwmr/g1_fastwmr/checkpoints/final_step_000001000.pt \
  --condition nominal_rough \
  --device cuda:0 \
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
  --checkpoint logs/fastwmr/g1_fastwmr_seed42/checkpoints/final_step_000001000.pt \
  --variant primary \
  --checkpoint logs/fastwmr/g1_fastwmr_seed43/checkpoints/final_step_000001000.pt \
  --variant primary \
  --checkpoint logs/fastwmr/g1_fastwmr_seed44/checkpoints/final_step_000001000.pt \
  --variant primary \
  --evaluation-seed 100 101 102 \
  --device cuda:0
```

The training seed is read from each checkpoint. Individual records are written to
`<condition>/train_seed_<N>/eval_seed_<N>.json`, so checkpoints from the same
variant cannot overwrite one another. Metrics are averaged across evaluation seeds
within each training seed first; the reported mean and standard deviation are then
computed across independent training seeds. Aggregated JSON, CSV, and Markdown
summaries are written under `evaluations/suite/` by default.

### Ablations

The main FastWMR ablations are exposed directly by the training entry point:

```bash
# Reconstructed state only as the actor/critic control feature
python script/train.py --control-feature-mode reconstruction_only --viz none

# Freeze the estimator while training the controller
python script/train.py --freeze-estimator --viz none

# Remove the estimator-to-controller gradient cutoff
python script/train.py --disable-gradient-cutoff --viz none

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
