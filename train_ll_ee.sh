#!/usr/bin/env bash
set -euo pipefail

PROJECT="luoxinyuan-duke-university/gentle_humanoid"
TASK="G1/G1_gentle_3kp_ee_range"
TRAIN_RUN="gentle_3kp_ee_range_train"
ADAPT_RUN="gentle_3kp_ee_range_adapt"
FINETUNE_RUN="gentle_3kp_ee_range_finetune"
CUDA_DEVICES="0,1,2,3"
MASTER_PORT="29501"
NPROC="4"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

run_stage() {
  local stage="$1"
  local run_id="$2"
  shift 2

  local cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
    task="$TASK" "+exp=$stage" wandb.id="$run_id" "$@")
  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

run_stage train "$TRAIN_RUN"
run_stage adapt "$ADAPT_RUN" checkpoint_path="run:${PROJECT}/${TRAIN_RUN}"
run_stage finetune "$FINETUNE_RUN" checkpoint_path="run:${PROJECT}/${ADAPT_RUN}"
