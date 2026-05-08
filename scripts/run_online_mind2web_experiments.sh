#!/bin/bash
# Sequential online-Mind2Web ablations: baseline / in-task / visual-skill / both.
# 100 tasks each, max_steps=30. The agent early-exits via STOP after
# blocked_exit_streak=2 consecutive blocked-state detections so doomed
# tasks (CAPTCHA, access denied, cloudflare interstitial, ...) don't
# burn the full 30 steps.

set -euo pipefail

cd /u/yhuang48/GUIMem

mkdir -p outputs/slurm outputs/online_mind2web

set -a
source /u/yhuang48/GUIMem/.env
set +a

CONFIGS=(
  "configs/experiments/online_exp1_baseline.yaml"
  "configs/experiments/online_exp2_in_task.yaml"
  "configs/experiments/online_exp3_visual_skill.yaml"
  "configs/experiments/online_exp4_both.yaml"
)

LABELS=(
  "online_exp1_baseline"
  "online_exp2_in_task"
  "online_exp3_visual_skill"
  "online_exp4_both"
)

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  label="${LABELS[$i]}"

  echo "============================================================"
  echo "Task $((i + 1)): ${label}"
  echo "Config: ${config}"
  echo "============================================================"

  python evaluate.py --config "${config}"

  echo ""
  echo "${label} done."
done
