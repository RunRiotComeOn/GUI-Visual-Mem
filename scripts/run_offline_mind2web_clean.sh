#!/bin/bash
# Run the "clean" offline mind2web ablations: no ground-truth previous_actions injection.
# Mirrors run_offline_mind2web_experiments_v4.sbatch but uses the *_clean configs and
# skips the visual-skill mining step (reuses the existing skill store).

set -euo pipefail

cd /u/yhuang48/GUIMem

mkdir -p outputs/slurm outputs/offline

set -a
source /u/yhuang48/GUIMem/.env
set +a

CONFIGS=(
  "configs/experiments/exp1_baseline_v4_clean.yaml"
  "configs/experiments/exp2_in_task_v4_clean.yaml"
  "configs/experiments/exp3_visual_skill_v4_clean.yaml"
  "configs/experiments/exp4_both_v4_clean.yaml"
)

PRED_FILES=(
  "outputs/offline/exp1_baseline_v4_clean_predictions_100.jsonl"
  "outputs/offline/exp2_in_task_v4_clean_predictions_100.jsonl"
  "outputs/offline/exp3_visual_skill_v4_clean_predictions_100.jsonl"
  "outputs/offline/exp4_both_v4_clean_predictions_100.jsonl"
)

LABELS=(
  "exp1_baseline_v4_clean"
  "exp2_in_task_v4_clean"
  "exp3_visual_skill_v4_clean"
  "exp4_both_v4_clean"
)

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  pred_file="${PRED_FILES[$i]}"
  label="${LABELS[$i]}"

  echo "============================================================"
  echo "Task $((i + 1)): ${label}"
  echo "Config: ${config}"
  echo "Predictions: ${pred_file}"
  echo "============================================================"

  python evaluate.py --config "${config}"

  echo ""
  echo "Evaluating ${label}"
  python evaluation/offline_evaluation/Multimodal-Mind2Web/eval_guimem.py \
    --pred_file "${pred_file}" --label "${label}"
done
