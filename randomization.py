# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization and privileged-buffer bookkeeping for FastWMR.

FastWMR needs randomization values both to perturb the simulator and to train
the estimator. Each randomization term should therefore perform three related
jobs:

1. Sample the randomized value.
2. Apply it to the simulator or delegate application to an IsaacLab event.
3. Record the value in ``env.fastwmr_*`` buffers for privileged observation and
   replay storage.

Planned buffers:
- ``env.fastwmr_friction``
- ``env.fastwmr_push_force_torques``
- ``env.fastwmr_payload_mass``
- any later stiffness, damping, gravity, or motor-offset targets

Implementation note:
Some IsaacLab built-in events do not expose the sampled value after applying it.
For those cases, this module may need a documented approximation: sample from
the same distribution for the estimator target while the built-in event handles
the physical application.
"""

