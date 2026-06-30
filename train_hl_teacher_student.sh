#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid_high_level"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="6,7"
MASTER_PORT="29501"
NPROC="2"

TASK="G1/G1_hl_ee_compliance_pos_delta_student"
LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_3kp_stiff_finetune_limmt_full_force30"
RUN_NAME="hl_ee_compliance_pos_delta015_stiff200_3kp_student_simple_amass_limmt_full_verystiff"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TEACHER_RUN_ID="${RUN_NAME}_teacher_${TIMESTAMP}"
ADAPT_RUN_ID="${RUN_NAME}_adapt_${TIMESTAMP}"
FINETUNE_RUN_ID="${RUN_NAME}_finetune_${TIMESTAMP}"

export CUDA_VISIBLE_DEVICES

run_stage() {
  local algo="$1"
  local run_id="$2"
  local checkpoint_path="${3:-}"

  local cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
    task="$TASK"
    algo="$algo"
    task.action.low_policy.run_path="$LOW_RUN_PATH"
    wandb.project="$HL_WANDB_PROJECT"
    wandb.id="$run_id"
  )

  if [[ -n "$checkpoint_path" ]]; then
    cmd+=(checkpoint_path="$checkpoint_path")
  fi

  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

run_stage \
  "root_student_ppo" \
  "$TEACHER_RUN_ID"

run_stage \
  "root_student_ppo_adapt" \
  "$ADAPT_RUN_ID" \
  "run:${HL_PROJECT_PATH}/${TEACHER_RUN_ID}"

run_stage \
  "root_student_ppo_finetune" \
  "$FINETUNE_RUN_ID" \
  "run:${HL_PROJECT_PATH}/${ADAPT_RUN_ID}"
