#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
PROJECT="luoxinyuan-duke-university/gentle_humanoid"
export CUDA_VISIBLE_DEVICES=0,1,2
MASTER_PORT=29501
NPROC=3
SCRIPT="scripts/train.py"

run_pipeline() {
  local TASK="$1" TAG="$2" SUFFIX="$3"

  local ID_TRAIN="${TAG}_train_${SUFFIX}"
  local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
  local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

  # ---------- TRAIN ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=train
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"
}

# run_pipeline "G1/G1_gentle" "gentle" "test111"
# run_pipeline "G1/G1_gentle" "gentle" "3point_amass_limmt_full_stiff30"
# run_pipeline "G1/G1_gentle_3kp" "gentle_3kp" "limmt_full_stiff30"
# run_pipeline "G1/G1_gentle_5kp" "gentle_5kp" "limmt_full_stiff30"
# run_pipeline "G1/G1_gentle_3kp_stiff" "gentle_3kp_stiff" "limmt_lafan_4to1_force30"
run_pipeline "G1/G1_gentle_3kp_stiff_bm_nogait" "gentle_3kp_stiff_bm_nogait" "bm_nogait"
# run_pipeline "G1/ablation/G1_gentle_3kp_floor_loco" "gentle_3kp_floor_loco" "compliance_force30"
# run_pipeline "G1/ablation/G1_gentle_3kp_floor_loco_friction_narrow" "gentle_3kp_floor_loco_friction_narrow" "compliance_force30"
# run_pipeline "G1/G1_gentle_3kp_stiff_rot" "3kp_stiff_rot" "limmt_full_force30_ee_rot05"
# run_pipeline "G1/G1_gentle_3kp_very_stiff_level1" "gentle_3kp_verystiff" "level1_sim2real"
# run_pipeline "G1/G1_gentle_limmt_force30_safe_default" "gentle" "limmt_full_force30_safe_default"
# run_pipeline "G1/G1_gentle_5kp_limmt_force30_safe_default" "gentle_5kp" "limmt_full_force30_safe_default"

# run_pipeline "G1/G1_no_force" "noforce" "motion_tracking_RL"
# run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"
