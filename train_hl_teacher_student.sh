#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid_high_level"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="4,5,6,7"
MASTER_PORT="29501"
NPROC="4"
LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_3kp_stiff_finetune_limmt_full_force30"
ROOT_DAMPING="1000"
DAMPING_TAG="${ROOT_DAMPING//./p}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES

run_stage() {
  local task="$1"
  local algo="$2"
  local run_id="$3"
  local total_frames="$4"
  local checkpoint_path="${5:-}"

  local cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
    task="$task"
    algo="$algo"
    total_frames="$total_frames"
    task.action.low_policy.run_path="$LOW_RUN_PATH"
    task.reward.root_hold.root_force_velocity_tracking.damping="$ROOT_DAMPING"
    wandb.project="$HL_WANDB_PROJECT"
    wandb.id="$run_id"
  )

  if [[ -n "$checkpoint_path" ]]; then
    cmd+=(checkpoint_path="$checkpoint_path")
  fi

  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

run_pipeline() {
  local task="$1"
  local run_name="$2"

  local teacher_run_id="${run_name}_teacher_${TIMESTAMP}"
  local adapt_run_id="${run_name}_adapt_${TIMESTAMP}"

  run_stage \
    "$task" \
    "root_student_force_ppo" \
    "$teacher_run_id" \
    "4000_000_000"

  run_stage \
    "$task" \
    "root_student_force_ppo_adapt" \
    "$adapt_run_id" \
    "1000_000_000" \
    "run:${HL_PROJECT_PATH}/${teacher_run_id}"
}

LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_3kp_stiff_finetune_limmt_full_force30"
run_pipeline "G1/G1_hl_force_walk_student" "force_resist_vel_delta_3kp_stu_B${DAMPING_TAG}"

LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_5kp_finetune_limmt_full_stiff30"
run_pipeline "G1/G1_hl_force_resist_5kp_student" "force_resist_vel_delta_5kp_stu_B${DAMPING_TAG}"
