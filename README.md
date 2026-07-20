# FastWMR G1 IsaacLab Task

This directory is the staging area for the FastWMR implementation described in
`pdf/FastWMR.pdf`, `pdf/directory.pdf`, and `pdf/roadmap.pdf`.

Implementation is progressing in independently verified layers:

1. Environment/task layer: observations, rewards, randomization, curriculum, and
   baseline task registration.
2. Algorithm layer: transition and boundary-safe sequence replay, recurrent
   world-state estimator, decoder, actor, critic, and FastSAC update.
3. Script layer: training, evaluation, CLI overrides, logging, and ablations.

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
