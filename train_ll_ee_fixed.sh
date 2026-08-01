#!/usr/bin/env bash
set -euo pipefail

PROJECT="luoxinyuan-duke-university/gentle_humanoid"
TASK="G1/G1_gentle_3kp_ee_net_pull_force_b"
TRAIN_RUN="gentle_3kp_ee_fixed200_train"
ADAPT_RUN="gentle_3kp_ee_fixed200_adapt"
FINETUNE_RUN="gentle_3kp_ee_fixed200_finetune"
CUDA_DEVICES="4,5,6,7"
MASTER_PORT="29503"
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
