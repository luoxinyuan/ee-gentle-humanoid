# EE Gentle Humanoid Training

This repository contains the IsaacLab training code used for low-level GentleHumanoid policies and high-level hierarchical EE/root compliance policies.

## Installation

1. Create a Conda environment.

```bash
conda create -n gentle python=3.10
conda activate gentle
```

2. Install PyTorch.

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

3. Install Isaac Sim. Ubuntu 22.04 is recommended.

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Optional smoke test
isaacsim
```

4. Install Isaac Lab.

```bash
cd <where-you-want-to-install-IsaacLab>
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none
```

5. Install this repository.

```bash
cd <repo-root>
pip install -e .
```

## Low-Level Training

Low-level policies are trained with `train.sh`. Edit the variables at the top of the script before running:

```bash
PROJECT="luoxinyuan-duke-university/gentle_humanoid"
export CUDA_VISIBLE_DEVICES=0,1,2,4
MASTER_PORT=29507
NPROC=4
```

Then select one or more `run_pipeline` calls near the bottom of `train.sh` and run:

```bash
bash train.sh
```

The low-level pipeline has three stages:

```text
train    -> 4B frames
adapt    -> 1B frames
finetune -> 2B frames
```

Common low-level task configs:

```text
G1/G1_gentle
  3-point root/wrist policy with the base GentleHumanoid compliance setup.

G1/G1_gentle_3kp
  Explicit 3-keypoint root/wrist low-level config.

G1/G1_gentle_5kp
  5-keypoint low-level config with feet command/target observations.

G1/G1_gentle_3kp_stiff
  3-keypoint non-compliant/stiff config with max_force=30 and net_force_limit=30.

G1/G1_gentle_3kp_stiff_rot
  3-keypoint stiff config with additional EE orientation tracking reward.

G1/G1_gentle_limmt_force30_safe_default
  LIMMT dataset, max external force 30N, default safe force range [5, 15].

G1/G1_gentle_5kp_limmt_force30_safe_default
  5-keypoint LIMMT dataset, max external force 30N, default safe force range [5, 15].
```

Example:

```bash
run_pipeline "G1/G1_gentle_limmt_force30_safe_default" "gentle" "limmt_full_force30_safe_default"
```

## High-Level End-to-End Training

Use `train_hl.sh` for a single-stage high-level PPO policy with privileged observations directly available to the policy.

Edit these variables in `train_hl.sh`:

```bash
CUDA_VISIBLE_DEVICES="0,1"
MASTER_PORT="29508"
NPROC="2"

ALGO="root_ppo"
TASK="G1/G1_hl_ee_compliance_pos_delta"
LOW_RUN_PATH="luoxinyuan-duke-university/gentle_humanoid/gentle_3kp_stiff_finetune_limmt_full_force30"
RUN_NAME="..."
```

Run:

```bash
bash train_hl.sh
```

Common high-level end-to-end tasks:

```text
G1/G1_hl_root_hold
  High-level root command policy for holding root position.

G1/G1_hl_force_resist
  Root force-resist behavior.

G1/G1_hl_force_walk
  Root force-follow/walk behavior.

G1/G1_hl_ee_compliance_pos_delta
  EE compliance with 6D residual xyz EE action.
```

## High-Level Teacher-Student Training

Use `train_hl_teacher_student.sh` for the deployable high-level teacher-student pipeline.

Edit these variables in the script:

```bash
CUDA_VISIBLE_DEVICES="2,3"
MASTER_PORT="29502"
NPROC="2"

TASK="G1/G1_hl_ee_y_compliance_pos_delta_student"
LOW_RUN_PATH="luoxinyuan-duke-university/gentle_humanoid/gentle_3kp_stiff_finetune_limmt_full_force30"
RUN_NAME="..."
```

Run:

```bash
bash train_hl_teacher_student.sh
```

The high-level teacher-student pipeline currently follows the low-level frame budget:

```text
teacher  -> 4B frames
adapt    -> 1B frames
finetune -> 2B frames
```

The current teacher-student script uses the force-estimator algorithms:

```text
root_student_force_ppo
root_student_force_ppo_adapt
root_student_force_ppo_finetune
```

These estimate `hl_force_priv` from deployable student observations and feed the predicted force information directly into the student actor.

Common high-level student tasks:

```text
G1/G1_hl_ee_compliance_pos_delta_student
  Isotropic EE compliance with 6D residual xyz EE action.

G1/G1_hl_ee_y_compliance_pos_delta_student
  Directional EE compliance: y direction compliant, x/z directions stiff.
  Current setting:
    stiffness: [600, 200, 600]
    max_offset: [0.05, 0.25, 0.05]

G1/G1_hl_force_walk_feet_student
  Root + feet high-level action space for force-walk behavior.
```

## Notes

- `task.action.low_policy.run_path` controls which frozen low-level policy is loaded during high-level training.
- High-level EE `pos_delta` action uses 6 dimensions: left/right EE xyz residuals.
- `cfg/task/G1/hl/force/net_pull_ee.yaml` controls the EE net-pull force sampling used by high-level EE compliance tasks.
- `cfg/task/G1/hl/task/ee_compliance.yaml` controls the high-level EE compliance reward and regularization overrides.
