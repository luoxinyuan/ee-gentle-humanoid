#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
MASTER_PORT="${MASTER_PORT:-29502}"
NPROC="${NPROC:-3}"

ALGO="root_ppo"
TASK="G1/G1_hl_force_walk"
LOW_RUN_PATH="${LOW_RUN_PATH:-${LOW_PROJECT_PATH}/gentle_3kp_stiff_finetune_limmt_full_force30}"
ROOT_DAMPING="${ROOT_DAMPING:-200}"
DAMPING_TAG="${ROOT_DAMPING//./p}"
# 命名方式: task_stiff_lowlevel_priv
RUN_NAME="force_walk_vel_delta_3kp_priv_B${DAMPING_TAG}"
RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES

cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
  task="$TASK"
  algo="$ALGO"
  task.action.low_policy.run_path="$LOW_RUN_PATH"
  task.reward.root_hold.root_force_velocity_tracking.damping="$ROOT_DAMPING"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
