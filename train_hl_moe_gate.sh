#!/usr/bin/env bash
set -euo pipefail

HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="4,5,6,7"
MASTER_PORT="29502"
NPROC="4"

TASK="G1/G1_hl_ee_xyz_learned_moe_200_600_force_b_student"
ALGO="root_student_force_learned_moe_ppo"
TOTAL_FRAMES="500_000_000"
RUN_ID="ee_xyz_learned_moe_200_600_gate_v2_$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES

cmd=(torchrun
  --nproc_per_node="$NPROC"
  --master_port="$MASTER_PORT"
  scripts/train.py
  task="$TASK"
  algo="$ALGO"
  total_frames="$TOTAL_FRAMES"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
